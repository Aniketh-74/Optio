"""Tool schemas and tool results: the input cost nobody puts in their model.

Three stages over the two places tool use spends tokens.

**Schemas are billed on every single turn.** A tool definition is ordinary
input. An agent carrying a dozen MCP-connected tools pays for all twelve
descriptions before the user has said anything, and pays again on every step of
the loop. Anthropic's published figure for deferring tool loading is an **85%
token reduction with MCP-evaluation accuracy rising 49% -> 74%** -- the largest
single number in the public cost literature, and one of the few where the
saving and the accuracy gain point the same way. Full deferral needs a
discovery round trip and so belongs to the caller's agent design (ADR-016); the
part that is a pure request transform, and therefore ours, is making the
schemas that *do* get sent cost less.

**Results are billed on every turn after the one that fetched them.** A tool
returning a 20k-token payload has not spent 20k tokens, it has raised the price
of every subsequent step by 20k for the rest of the conversation. That is the
quadratic tax with a large constant, and capping it is the cheapest structural
fix available here.

Deliberately absent: any attempt to rewrite a schema's *shape*. Reordering
properties, collapsing enums, or inferring that a parameter is unused would all
save tokens and would all eventually send a provider a schema the caller did
not write. These stages only remove text that carries no instruction to the
model, or truncate text that has already served its purpose.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from optio_optimize.similarity import overlap_ratio, words
from optio_optimize.stages.base import Fidelity, Stage, StageResult

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest, Message

#: JSON Schema keys that are documentation for humans reading the schema and
#: carry no instruction to the model. ``title`` is defined by JSON Schema as
#: annotation only; ``$schema``, ``$id`` and ``$comment`` are metadata about the
#: document rather than about the value it describes.
#:
#: Conspicuously *not* here: ``description``, ``examples``, ``default`` and
#: ``enum``. Every one of those is read by the model and shapes tool selection
#: and argument construction. They are the largest keys in a typical schema and
#: stripping them would show the best token numbers in this file, which is
#: exactly why the line is drawn on whether the model uses the text rather than
#: on how much of it there is.
NON_SEMANTIC_KEYS = frozenset({"title", "$schema", "$id", "$comment"})

#: How deep to walk a schema. Tool schemas nest through ``properties`` and
#: ``items`` and are occasionally self-referential through ``$ref``; a bounded
#: walk cannot hang on one. Ten levels is far past any hand-written schema.
MAX_SCHEMA_DEPTH = 10

#: Default ceiling on a single tool result, in tokens. Generous on purpose: the
#: cost this stage targets is a runaway payload, not an ordinary one, and a cap
#: tight enough to bite on normal results would trade a large invisible saving
#: for a small visible breakage.
DEFAULT_MAX_TOOL_RESULT_TOKENS = 2000

#: Appended to a truncated tool result. The model must be told, or it reasons
#: over a payload it believes is complete -- which turns a cost optimization
#: into a correctness bug that presents as the model inventing the missing rows.
_TRUNCATION_NOTICE = "\n\n[truncated by optio_optimize: {dropped} of {total} characters omitted]"

#: Relevance below which a tool is judged unrelated to the conversation.
#: Matches :data:`~optio_optimize.stages.retrieval.MIN_RELEVANCE`'s posture --
#: low enough that only a tool sharing essentially no vocabulary with the
#: request is a candidate.
MIN_TOOL_RELEVANCE = 0.05

#: Never prune below this many tools. A model left with one tool will use it
#: whether or not it fits, which is a worse failure than carrying the schemas.
MIN_KEPT_TOOLS = 3


def _tool_tokens(tools: tuple[dict[str, Any], ...], ctx: StageContext, model: str) -> int:
    """Tokens a tool set costs, serialized the way ``count_request`` serializes it."""
    return sum(
        ctx.counter.count_text(json.dumps(tool, separators=(",", ":")), model) for tool in tools
    )


def _strip(node: object, depth: int = 0) -> object:
    """Return ``node`` with annotation-only keys removed, bounded in depth."""
    if depth >= MAX_SCHEMA_DEPTH:
        return node
    if isinstance(node, dict):
        return {
            key: _strip(value, depth + 1)
            for key, value in node.items()
            if key not in NON_SEMANTIC_KEYS
        }
    if isinstance(node, list):
        return [_strip(item, depth + 1) for item in node]
    return node


def _strip_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Return one tool schema with annotation-only keys removed."""
    return cast("dict[str, Any]", _strip(tool))


class MinifyToolsStage(Stage):
    """Remove annotation-only keys from tool schemas.

    Strips :data:`NON_SEMANTIC_KEYS` wherever they appear in a tool definition.
    Nothing the model reads is removed -- see that constant's own note for
    where the line sits and why it is drawn on *use* rather than on size.

    ``SHAPED`` rather than ``IDENTICAL`` for the reason
    :class:`~optio_optimize.stages.retrieval.DeduplicateStage` gives: the text
    sent genuinely differs, and this package does not claim byte-identical
    output for a transform it has not proved that strictly.
    ``structured_output`` was labelled lossless on exactly this kind of
    reasoning and the live A/B suite caught it.

    The saving is real but modest on a well-written schema and large on a
    generated one -- OpenAPI-derived and MCP-bridged tools routinely carry a
    ``title`` on every property, which is one wasted key per parameter per tool
    per turn.
    """

    fidelity = Fidelity.SHAPED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "minify_tools"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Strip annotation-only keys from every tool schema."""
        if not request.tools:
            return self.declines(request)

        stripped = tuple(_strip_tool(tool) for tool in request.tools)
        if stripped == request.tools:
            return self.declines(request)

        before = _tool_tokens(request.tools, ctx, request.model)
        after = _tool_tokens(stripped, ctx, request.model)
        saved = max(0, before - after)
        if saved == 0:
            # The keys were present but cost nothing measurable. Rewriting the
            # caller's schemas for no gain is pure risk.
            return self.declines(request)

        return StageResult(
            request=request.with_tools(stripped),
            saved_input_tokens=saved,
            note=f"stripped schema annotations from {len(stripped)} tool(s)",
        )


class CapToolResultsStage(Stage):
    """Bound how many tokens one tool result may add to the conversation.

    A tool result is not paid for once. It enters the message list and is
    resent on every subsequent turn, so a single oversized payload raises the
    price of the entire remaining run. This caps each ``tool``-role message at
    :data:`DEFAULT_MAX_TOOL_RESULT_TOKENS` and appends an explicit notice
    saying how much was dropped.

    **The notice is not decoration.** A silently truncated payload is worse
    than an expensive one: the model reasons over what it can see, concludes
    the list of rows it was given is the whole list, and answers confidently
    from a fragment. Telling it the payload was cut converts a wrong answer
    into a request for the rest, which is a failure the caller can act on.

    ``SHAPED``: information is removed, but only from a payload the caller's
    own tool produced and only above a ceiling they set. It is not ``ALTERED``
    because nothing is invented and nothing is paraphrased -- the same
    distinction :class:`~optio_optimize.stages.history.TrimHistoryStage` draws
    against summarization.

    **Where this is the wrong stage.** A tool whose entire job is to hand the
    model a large document to work over -- fetch-and-summarize, transcript
    analysis -- is a legitimate large payload, and capping it removes the task
    rather than its waste. Raise the ceiling or disable this stage for those
    workloads; the field-guide alternative of returning a summary plus a
    ``fetch_full(id)`` tool is a change to the caller's tool design and so sits
    outside what this package can do for them (ADR-016).
    """

    fidelity = Fidelity.SHAPED

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS) -> None:
        """Build the stage.

        Args:
            max_tokens: Ceiling per tool result.

        Raises:
            ValueError: If the ceiling is not positive. Setup fails loudly
                (§4.2); a zero ceiling would erase every tool result and
                present as the agent losing its ability to use tools at all.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "cap_tool_results"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Truncate any tool result exceeding the ceiling."""
        capped: list[Message] = []
        saved = 0
        for message in request.messages:
            if message.role != "tool":
                capped.append(message)
                continue
            tokens = ctx.counter.count_text(message.content, request.model)
            if tokens <= self.max_tokens:
                capped.append(message)
                continue
            trimmed = self._truncate(message.content, tokens)
            saved += tokens - ctx.counter.count_text(trimmed, request.model)
            capped.append(message.with_content(trimmed))

        if saved <= 0:
            return self.declines(request)
        return StageResult(
            request=request.with_messages(tuple(capped)),
            saved_input_tokens=saved,
            note=f"capped oversized tool result(s), {saved} tokens",
        )

    def _truncate(self, content: str, tokens: int) -> str:
        """Cut ``content`` to roughly the ceiling and append the notice.

        Cuts by character proportion rather than by re-tokenizing in a loop.
        The result is approximate, which is correct here: the ceiling exists to
        stop a runaway payload, and spending several tokenizer passes to land
        exactly on it would cost more than the imprecision does.
        """
        keep = max(1, int(len(content) * self.max_tokens / tokens))
        dropped = len(content) - keep
        return content[:keep] + _TRUNCATION_NOTICE.format(dropped=dropped, total=len(content))


class PruneToolsStage(Stage):
    """Drop tools that share no vocabulary with the conversation.

    The same cheap lexical test
    :class:`~optio_optimize.stages.retrieval.PruneRetrievalStage` applies to
    retrieved chunks, applied to tool descriptions: a tool whose name and
    description overlap essentially nothing in the recent conversation is
    unlikely to be the one the model reaches for, and it is being paid for on
    every turn regardless.

    ``ALTERED``, and the only stage in this module that is. The others remove
    text the model does not read or truncate a payload with a notice; this one
    removes a *capability*. If the judgment is wrong the model cannot call the
    tool it needed, and what the caller sees is not an error but an agent that
    inexplicably declines to do something it could do yesterday. Off by
    default, and subject to ADR-015's evidence bar like every other ``ALTERED``
    stage.

    Two guards, both load-bearing:

    * **A tool already called in this conversation is never pruned.** Its
      presence in the history is direct evidence the model wants it, which
      beats any lexical score this stage could compute. Dropping it would also
      leave a ``tool_calls`` entry in the history referring to a tool no longer
      declared, which some providers reject outright.
    * **Never prune below :data:`MIN_KEPT_TOOLS`.** A model holding one tool
      will use it whether or not it fits the task.
    """

    fidelity = Fidelity.ALTERED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "prune_tools"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Drop tools with no lexical connection to the conversation."""
        if len(request.tools) <= MIN_KEPT_TOOLS:
            return self.declines(request)

        query = words(self._recent_text(request))
        if not query:
            return self.declines(request)

        called = self._already_called(request)
        # Kept by index rather than by object identity: two tools can compare
        # equal, and identity would then drop both or neither depending on
        # which one the set happened to hold.
        scored = [
            (index, self._score(tool, query, called)) for index, tool in enumerate(request.tools)
        ]
        keep = {index for index, score in scored if score >= MIN_TOOL_RELEVANCE}
        if len(keep) < MIN_KEPT_TOOLS:
            # Below the floor: keep the best-scoring set rather than the
            # threshold-passing one, so the floor is a floor and not a veto.
            ranked = sorted(scored, key=lambda row: row[1], reverse=True)
            keep = {index for index, _ in ranked[:MIN_KEPT_TOOLS]}
        if len(keep) == len(request.tools):
            return self.declines(request)

        # Preserve the caller's ordering: tool order is stable across turns and
        # therefore part of what a provider's prefix cache matches on.
        ordered = tuple(tool for index, tool in enumerate(request.tools) if index in keep)
        saved = _tool_tokens(request.tools, ctx, request.model) - _tool_tokens(
            ordered, ctx, request.model
        )
        return StageResult(
            request=request.with_tools(ordered),
            saved_input_tokens=max(0, saved),
            note=f"pruned {len(request.tools) - len(ordered)} of {len(request.tools)} tool(s)",
        )

    @staticmethod
    def _recent_text(request: LLMRequest) -> str:
        """Conversation text this stage scores tools against.

        Only non-system messages. A system prompt describes the agent's whole
        job, so scoring against it makes every tool look relevant -- which
        would make this stage silently never fire, the "looks configured, does
        nothing" trap ``config.py`` exists to prevent.
        """
        return " ".join(m.content for m in request.messages if m.role != "system")

    @staticmethod
    def _already_called(request: LLMRequest) -> set[str]:
        """Names of tools this conversation has already invoked."""
        names: set[str] = set()
        for message in request.messages:
            if message.name:
                names.add(message.name)
            calls = message.extra.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if isinstance(call, dict):
                    function = call.get("function")
                    if isinstance(function, dict) and isinstance(function.get("name"), str):
                        names.add(function["name"])
        return names

    @staticmethod
    def _score(tool: dict[str, Any], query: set[str], called: set[str]) -> float:
        """Relevance of one tool to the conversation, on ``[0, 1]``.

        A tool already called scores ``1.0``: history beats lexical guessing.
        """
        name, description = _tool_identity(tool)
        if name and name in called:
            return 1.0
        return overlap_ratio(f"{name} {description}", query)


def _tool_identity(tool: dict[str, Any]) -> tuple[str, str]:
    """Extract ``(name, description)`` from either tool schema shape.

    Providers disagree: OpenAI nests under ``function``, Anthropic puts both at
    the top level. Reading both means this stage does not need a provider flag
    to work, and an unrecognized shape yields empty strings rather than an
    exception -- a tool this function cannot read simply scores zero on
    lexical overlap and is protected by :data:`MIN_KEPT_TOOLS` instead.
    """
    function = tool.get("function")
    source: dict[str, Any] = function if isinstance(function, dict) else tool
    name = source.get("name")
    description = source.get("description")
    return (
        name if isinstance(name, str) else "",
        description if isinstance(description, str) else "",
    )


__all__ = [
    "DEFAULT_MAX_TOOL_RESULT_TOKENS",
    "MAX_SCHEMA_DEPTH",
    "MIN_KEPT_TOOLS",
    "MIN_TOOL_RELEVANCE",
    "NON_SEMANTIC_KEYS",
    "CapToolResultsStage",
    "MinifyToolsStage",
    "PruneToolsStage",
]
