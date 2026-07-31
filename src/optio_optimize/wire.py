"""Translating a normalized request into each provider's own shape.

:class:`~optio_optimize.types.LLMRequest` is provider-neutral by design
(ADR-013), which means *something* has to convert it at the boundary, and that
conversion is where this package's most expensive bug lived. The live
benchmark's OpenAI adapter did not forward ``request.tools``. Nothing failed:
the call succeeded, the report printed, and ``minify_tools`` claimed 3,240
saved tokens while the provider billed byte-identical totals in both arms
because the field the stage had rewritten was dropped on the floor. A whole
live run measured nothing and said so confidently.

ADR-017 adds a second place that has to make the same translation -- batch
submission builds a request body as plain JSON rather than SDK keyword
arguments -- so the translation lives here, once, and both call sites share it.
Two independent copies of this logic is precisely how the same field goes
missing from one of them.

What is deliberately *not* centralized: the final ``create(...)`` call. The
live adapters pass explicit keyword arguments rather than unpacking a dict,
because the SDKs' ``create`` methods are heavily overloaded and
``**dict[str, object]`` type-checks only while the package is absent -- the
unpacked version passed CI, where ``openai`` is not installed, and failed the
moment anyone ran a live benchmark. So this module returns the *parts* that are
easy to forget (messages, tools, system blocks) and each call site assembles
them in the form its transport needs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from optio_optimize.types import LLMResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from optio_optimize.types import LLMRequest, Message

#: Anthropic's only cache-control type. Named so the three places that emit it
#: -- system blocks, marked turns, and the adapter's raw-param path -- cannot
#: drift apart on a literal. Copy it at each use: a shared mutable dict living
#: on several request bodies is a bug waiting for someone to edit one of them.
EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}

#: Request fields that are deliberately never put on the wire, with the reason.
#: Checked by a test against :class:`~optio_optimize.types.LLMRequest`'s own
#: field list, so adding a field to the request type fails the suite until
#: someone decides which side of this line it belongs on. That is the same
#: guard :func:`~optio_optimize.cache.request_key` applies to the cache key, and
#: for the same reason: a new field that is silently not sent is invisible.
UNSENT_FIELDS: dict[str, str] = {
    "extra": "provider transport details, merged by each adapter rather than named here",
}
# `thinking_budget` was excused here until 2026-07-30, on the grounds that its
# shape is provider-specific. That was true and was not a reason to drop it: a
# caller who set the field had it silently discarded, and reasoning tokens bill
# at *completion* rates, so the excuse hid the most expensive tokens in a
# request. ADR-018 splits the idea into the two shapes the vendors actually
# accept -- `thinking_budget` (Anthropic, a token count) and `reasoning_effort`
# (OpenAI, a category) -- rather than inventing a conversion between them.

#: The key under which an adapter stashes a caller's original message param on
#: :attr:`~optio_optimize.types.Message.extra`, so anything this package does not
#: model -- image blocks, ``tool_use``/``tool_result``, provider extensions --
#: rides through and is restored verbatim.
#:
#: Named here rather than privately in each adapter because it is a contract
#: *between* modules: both adapters write it and :func:`optio_optimize.cache
#: .request_key` now reads it. It was two identical private literals until
#: 2026-07-30, which is the arrangement this module's docstring warns about.
RAW_CONTENT_KEY = "_raw"


def is_text_block(block: Any) -> bool:
    """Is ``block`` a text content block?

    Handles both shapes the SDKs use: a plain dict, and a pydantic block object
    echoed back from a previous response. Shared rather than reimplemented per
    caller for this module's founding reason -- two subtly different readings of
    one wire shape is how the readings come to disagree. Here the disagreement
    would be silent and expensive: a block wrongly judged *text* contributes
    nothing to the cache key (ADR-022).
    """
    if isinstance(block, dict):
        return block.get("type") == "text"
    return bool(getattr(block, "type", "") == "text")


def block_text(block: Any) -> str:
    """The ``text`` of one text block, whichever shape it arrived in."""
    if isinstance(block, dict):
        return str(block.get("text", ""))
    return str(getattr(block, "text", ""))


def as_block_dict(block: Any) -> dict[str, Any] | None:
    """A content block as a plain dict, or ``None`` if it cannot be rendered as one.

    The same ``model_dump`` normalization :func:`canonical_block` performs, kept
    separate because one caller wants a hashable string and the other wants to
    edit a field and put the block back.
    """
    if isinstance(block, dict):
        return dict(block)
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        dumped = dump()
        return dict(dumped) if isinstance(dumped, dict) else None
    return None


def _raw_content_blocks(message: Message) -> list[Any] | None:
    """The caller's original content block list, when there is one."""
    raw = message.extra.get(RAW_CONTENT_KEY)
    if not isinstance(raw, dict):
        return None
    content = raw.get("content")
    return content if isinstance(content, list) else None


def tool_result_payload(block: Any) -> str | None:
    """The text a ``tool_result`` block carries, or ``None`` for other blocks.

    Anthropic accepts either a string or a list of blocks as a tool result's
    ``content``; both are handled, and a list contributes only its text.
    """
    data = as_block_dict(block)
    if data is None or data.get("type") != "tool_result":
        return None
    content = data.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block_text(b) for b in content if is_text_block(b))
    return ""


def tool_result_payloads(message: Message) -> list[str]:
    """Every ``tool_result`` payload this message carries, in order.

    Empty for a message that carries none, including one whose tool result is a
    plain ``role="tool"`` message -- that shape keeps its payload in
    ``Message.content`` and needs no unwrapping.

    Exists because ``cap_tool_results`` could not see this content at all: a
    tool result reaching :func:`wrap_anthropic_client` is a block inside a
    ``role="user"`` turn, and the text derivation deliberately contributes
    nothing for a non-text block. An 8,001-token payload measured as 0 tokens
    and went to the wire uncapped (ADR-032).
    """
    blocks = _raw_content_blocks(message)
    if blocks is None:
        return []
    payloads = [tool_result_payload(block) for block in blocks]
    return [payload for payload in payloads if payload is not None]


def with_capped_tool_results(message: Message, cap: Callable[[str], str]) -> Message:
    """A copy of ``message`` with each ``tool_result`` payload passed through ``cap``.

    ``cap`` owns the truncation policy -- the ceiling and the notice belong to
    the stage -- while the shape of the thing being truncated stays here.
    ADR-022's rule: a second, subtly different reading of the same wire shape is
    the divergence this module exists to prevent.

    The caller's own dict is never edited. A new raw param is built, which
    ADR-016 requires independently of whether mutating in place would happen to
    reach the wire (it would: ``_param_from_message`` returns the raw param
    untouched when the derived text is unchanged, and for a ``tool_result``
    block it is ``""`` on both sides).

    A capped block's ``content`` is written back as a **string**. Anthropic
    accepts that shape, and preserving a truncated list's block structure would
    mean deciding which of several blocks absorbed the cut.
    """
    blocks = _raw_content_blocks(message)
    if blocks is None:
        return message
    raw = message.extra[RAW_CONTENT_KEY]
    if not isinstance(raw, dict):
        return message

    changed = False
    rebuilt: list[Any] = []
    for block in blocks:
        payload = tool_result_payload(block)
        if payload is None:
            rebuilt.append(block)
            continue
        capped = cap(payload)
        if capped == payload:
            rebuilt.append(block)
            continue
        data = as_block_dict(block)
        if data is None:
            rebuilt.append(block)
            continue
        data["content"] = capped
        rebuilt.append(data)
        changed = True

    if not changed:
        return message
    return replace(
        message,
        extra={**message.extra, RAW_CONTENT_KEY: {**raw, "content": rebuilt}},
    )


def canonical_block(block: Any) -> str:
    """A stable string identifying one content block, for hashing.

    Normalizes SDK objects through ``model_dump`` so that a block built as a
    dict and the same block echoed back as a pydantic object produce the same
    string. Without that, a conversation replayed from stored dicts could never
    hit a cache entry written from a live response's own objects.

    ``sort_keys`` makes the result independent of key order, and ``default=repr``
    keeps a stray non-JSON value (a ``datetime``, a bytes payload) from raising.

    This deliberately does **not** swallow exceptions. A payload nothing here can
    render is a payload whose identity is unknown, and the caller
    (:func:`~optio_optimize.cache.request_key`) is reached only from stage
    ``before`` hooks, which the pipeline guards per stage: raising means the cache
    stage is skipped for this request, so the call goes to the provider
    unoptimized. That is a lost cache hit. Returning a constant instead would be
    a *shared key* for payloads whose contents differ, which is the wrong answer
    this whole ADR is about.
    """
    if isinstance(block, dict):
        return json.dumps(block, sort_keys=True, separators=(",", ":"), default=repr)
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), sort_keys=True, separators=(",", ":"), default=repr)
    return repr(block)


def openai_messages(request: LLMRequest) -> list[dict[str, Any]]:
    """Build OpenAI's ``messages`` array.

    ``tool_calls`` and ``tool_call_id`` are lifted out of ``extra`` explicitly
    rather than forwarded wholesale: OpenAI rejects a ``tool`` message with no
    preceding ``tool_calls``, so dropping them silently -- which the first
    version of the live adapter did -- makes every tool-calling workload fail
    with a 400 regardless of what any stage did.

    Args:
        request: The request, after every stage has run.

    Returns:
        Plain dicts. The caller casts to the SDK's ``TypedDict`` if it needs to;
        a batch submission serializes them as JSON instead.
    """
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            entry["name"] = message.name
        if "tool_calls" in message.extra:
            entry["tool_calls"] = message.extra["tool_calls"]
        if "tool_call_id" in message.extra:
            entry["tool_call_id"] = message.extra["tool_call_id"]
        messages.append(entry)
    return messages


def openai_tools(request: LLMRequest) -> list[dict[str, Any]] | None:
    """Return the tool schemas, or ``None`` when there are none.

    ``None`` rather than ``[]``: an empty array is a different request from an
    absent field, and OpenAI rejects the empty one.
    """
    return list(request.tools) or None


def anthropic_system_and_turns(
    request: LLMRequest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into Anthropic's ``system`` blocks and ``messages`` turns.

    Anthropic takes the system prompt as a separate top-level parameter rather
    than a message role, and this is where the library's ``cacheable`` marker
    becomes a real API field -- the entire difference between a ~90% input
    discount and none.

    **The marker is honoured on turns too, not only on system blocks.**
    ``PrefixCacheStage`` marks the *last message of the stable prefix*, which is
    a system message only in the shortest conversations; once there are three or
    more turns the boundary lands on a user or assistant message. An earlier
    version of this function emitted ``cache_control`` for system blocks alone,
    so on any real conversation the marker was computed, placed, reported in the
    savings ledger -- and silently dropped on the way to the wire. Caught by the
    Anthropic adapter's first test run.

    A breakpoint on a turn caches everything above it, the system prompt
    included, so this is strictly better than marking the system block alone.
    Anthropic requires block-shaped content to carry ``cache_control``, so a
    marked turn's string content is promoted to a one-element text block;
    unmarked turns keep the plain string they arrived with.

    Returns:
        ``(system_blocks, turns)``. Either may be empty.
    """
    system_blocks: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            block: dict[str, Any] = {"type": "text", "text": message.content}
            if message.cacheable:
                block["cache_control"] = dict(EPHEMERAL_CACHE_CONTROL)
            system_blocks.append(block)
        elif message.role == "tool":
            _append_tool_result(turns, message)
        elif message.extra.get("tool_calls"):
            turns.append(_assistant_with_tool_calls(message))
        elif message.cacheable:
            turns.append(
                {
                    "role": message.role,
                    "content": [
                        {
                            "type": "text",
                            "text": message.content,
                            "cache_control": dict(EPHEMERAL_CACHE_CONTROL),
                        }
                    ],
                }
            )
        else:
            turns.append({"role": message.role, "content": message.content})
    return system_blocks, turns


def _append_tool_result(turns: list[dict[str, Any]], message: Message) -> None:
    """Add one ``tool_result`` block, merging into the previous user turn.

    Anthropic requires alternating roles, so parallel tool calls -- two
    ``role="tool"`` messages in a row -- must become **one** user turn carrying
    two blocks, not two user turns, which is a 400 (ADR-025).
    """
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": message.extra.get("tool_call_id", ""),
        "content": message.content,
    }
    if message.cacheable:
        block["cache_control"] = dict(EPHEMERAL_CACHE_CONTROL)
    if turns and turns[-1]["role"] == "user" and isinstance(turns[-1]["content"], list):
        turns[-1]["content"].append(block)
        return
    turns.append({"role": "user", "content": [block]})


def _assistant_with_tool_calls(message: Message) -> dict[str, Any]:
    """Translate an assistant turn proposing tool calls into ``tool_use`` blocks.

    The neutral shape is OpenAI's -- ``arguments`` a JSON *string* -- and
    Anthropic's ``input`` is an object, so it is parsed rather than forwarded.
    Any narration alongside the call is kept as a leading text block: a model may
    explain itself before calling, and dropping that loses content the caller was
    billed for and the model may refer back to.

    Raises:
        ValueError: If ``arguments`` is not valid JSON. Neither alternative is
            safe -- dropping the block orphans the ``tool_result`` answering it,
            and ``{}`` sends a call the model never made -- so this reaches the
            pipeline's per-stage fail-open and the request goes unoptimized.
    """
    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    for call in message.extra["tool_calls"]:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        raw = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tool_calls: arguments for {function.get('name', '?')!r} are not valid JSON, "
                f"so this call cannot be represented in Anthropic's shape"
            ) from exc
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "input": arguments,
            }
        )
    if message.cacheable and content:
        content[-1]["cache_control"] = dict(EPHEMERAL_CACHE_CONTROL)
    return {"role": "assistant", "content": content}


def anthropic_tools(request: LLMRequest) -> list[dict[str, Any]] | None:
    """Return tool schemas in Anthropic's shape, or ``None``."""
    return [as_anthropic_tool(tool) for tool in request.tools] or None


def as_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one tool schema into Anthropic's shape.

    OpenAI nests ``name``/``description``/``parameters`` under ``function``;
    Anthropic puts ``name``/``description``/``input_schema`` at the top level. A
    schema already in Anthropic's shape passes through, so a caller who writes
    for either provider gets the same behaviour.

    Args:
        tool: A tool schema in either provider's shape.

    Returns:
        The schema as Anthropic expects it.
    """
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
    }


def openai_body(request: LLMRequest, model: str) -> dict[str, Any]:
    """Build a complete OpenAI chat-completions request body as JSON.

    Used by batch submission, which writes request envelopes to a JSONL file and
    therefore has no SDK method to pass keyword arguments to. The live adapter
    does *not* use this -- see the module docstring on why it assembles keyword
    arguments explicitly instead -- so the two must be kept in step by the test
    that compares them field for field.

    ``None`` values are omitted rather than sent: the batch endpoint validates
    each envelope against the same schema as the synchronous one, and an
    explicit ``"temperature": null`` is a validation error rather than a default.

    Args:
        request: The request, after every stage has run.
        model: Model to bill against -- not necessarily ``request.model``, which
            the routing stage may already have changed.

    Returns:
        A JSON-serializable body.
    """
    body: dict[str, Any] = {"model": model, "messages": openai_messages(request)}
    if request.max_tokens is not None:
        body["max_completion_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.response_format is not None:
        body["response_format"] = request.response_format
    # `reasoning_effort`, never a value derived from `thinking_budget`. OpenAI
    # takes a category and Anthropic a token count, and translating between
    # them means inventing thresholds that depend on the model (ADR-018).
    if request.reasoning_effort is not None:
        body["reasoning_effort"] = request.reasoning_effort
    tools = openai_tools(request)
    if tools is not None:
        body["tools"] = tools
    if request.stop:
        body["stop"] = list(request.stop)
    return body


def anthropic_body(
    request: LLMRequest,
    model: str,
    *,
    default_max_tokens: int = 1024,
) -> dict[str, Any]:
    """Build a complete Anthropic messages request body as JSON.

    Args:
        request: The request, after every stage has run.
        model: Model to bill against.
        default_max_tokens: Used when the caller set none. Anthropic requires
            the field, so there is no "omit it" option the way there is on
            OpenAI -- a value has to be invented, and inventing it visibly here
            beats burying it in an adapter.

    Returns:
        A JSON-serializable body.
    """
    system_blocks, turns = anthropic_system_and_turns(request)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens or default_max_tokens,
        "messages": turns,
    }
    if system_blocks:
        body["system"] = system_blocks
    if request.temperature is not None:
        body["temperature"] = request.temperature
    # `is not None` rather than truthiness: budget 0 means "do not think", which
    # is a real instruction, while absence means the caller said nothing at all.
    # Collapsing the two would silently disable reasoning for everyone who left
    # the field alone -- the absence-is-not-zero rule, on the field where
    # getting it backwards is most expensive.
    if request.thinking_budget is not None:
        body["thinking"] = {"type": "enabled", "budget_tokens": request.thinking_budget}
    tools = anthropic_tools(request)
    if tools is not None:
        body["tools"] = tools
    if request.stop:
        body["stop_sequences"] = list(request.stop)
    return body


def response_from_anthropic_message(message: Any) -> LLMResponse:
    """Normalize an Anthropic message into this package's response model.

    ``input_tokens`` is reported by Anthropic *excluding* both cache reads and
    cache writes, so **both** are added back to make the field mean what it
    means everywhere else here: total prompt tokens, of which some were
    discounted and some carried a premium. Getting that wrong in one place would
    make batched, synchronous and adapter totals silently incomparable.

    Writes were omitted at first, and the omission was not symmetric: it dropped
    the most expensive band of prompt tokens from the total, so a cached call
    reported a fraction of its real cost and every prefix-cache saving derived
    from it came out too high. On the first turn of the run behind this
    package's 53.7% figure it reported 200 tokens against a true 4,805.

    Attribute access rather than isinstance narrowing, because the two callers
    receive structurally identical objects from different SDK code paths --
    a batch result entry's ``.message`` and a live ``messages.create`` return.

    Args:
        message: An ``anthropic.types.Message`` or a batch result's message.

    Returns:
        The normalized response.
    """
    usage = getattr(message, "usage", None)
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
    # The one-hour band, which the provider has always reported and this package
    # did not read until ADR-021. A one-hour write costs 2x base input against
    # 1.25x for five minutes, so folding the two together under-bills the more
    # expensive one by 37.5% -- the same direction as the omission above, and the
    # reason both are read here rather than one being inferred from the other.
    creation = getattr(usage, "cache_creation", None) if usage else None
    written_1h = min(written, int(getattr(creation, "ephemeral_1h_input_tokens", 0) or 0))
    text = "".join(
        str(getattr(block, "text", ""))
        for block in getattr(message, "content", [])
        if getattr(block, "type", "") == "text"
    )
    return LLMResponse(
        content=text,
        input_tokens=(int(getattr(usage, "input_tokens", 0)) + cached + written) if usage else 0,
        output_tokens=int(getattr(usage, "output_tokens", 0)) if usage else 0,
        cached_input_tokens=cached,
        cache_write_tokens=written,
        cache_write_1h_tokens=written_1h,
        model=str(getattr(message, "model", "")),
        finish_reason=getattr(message, "stop_reason", None),
    )
