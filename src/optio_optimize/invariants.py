"""Rules the library must not break, checked outside the request path.

Two rule sets, because the two useful kinds of rule have different shapes.

**Absolute rules** are true of any single request and are enforced by the
provider: violating one is a rejected call. They were never the problem.

**Preservation rules** are true of any *rewrite* and are enforced by nobody.
``trim_history`` dropped the user's task out of agent conversations for weeks;
providers accept a conversation with no user message, so nothing failed, the
model inferred a task from the surrounding tool results, and the wrong answer
was attributed to the model. 1,304 tests missed it because every fixture used
the same tidy alternating-chat shape as the code's own mental model.

The distinction matters because the rule that would have caught it is not
expressible about one request. A caller whose history legitimately begins with
an assistant message is not malformed, and the library is right to send it
unchanged. The library is wrong to *create* one. Only the before-and-after pair
shows the difference.

**This never runs on the request path.** It is per-message work on every call,
for a guarantee the pipeline's fail-open provides another way, and SC-5's
overhead budget is not there to be spent on self-checking. Its two callers are
the test suite and the real-agent probe.

**A violation never carries prompt content** -- only a rule name, an index and
a role. Violations are printed and reach terminal scrollback and CI logs, so
§10's rule applies exactly as it does to the fail-open guard, which logs an
exception's type and never its message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from optio_optimize.types import LLMRequest, Message

#: A ``tool`` message with no assistant ``tool_calls`` before it. Rejected by
#: every major provider -- the same class of defect that made ``fan_out`` fail
#: all 12 live calls over a missing ``"json"`` literal.
TOOL_RESULT_UNPAIRED = "tool_result_unpaired"

#: A ``tool`` message whose ``tool_call_id`` matches no preceding call.
TOOL_RESULT_UNMATCHED_ID = "tool_result_unmatched_id"

#: An empty message with no ``tool_calls`` attached. An assistant message with
#: empty content *and* a tool call is the most common shape in a real agent
#: loop and is valid; empty and unattached is not.
EMPTY_CONTENT_UNATTACHED = "empty_content_unattached"

#: Nothing but system messages. There is no question to answer.
NO_ANSWERABLE_MESSAGE = "no_answerable_message"

#: The library removed the caller's last user message. In a chat the first user
#: turn is a stale question; in an agent loop it is *the task*, and every
#: message after it is the agent's own tool traffic. Providers accept a
#: conversation with no user message, so this fails silently: the model infers
#: a task from the evidence and answers a question nobody asked. Measured on
#: 2026-07-29 -- the trimmed arm cost *more* than the correct one, because a
#: model that has lost the question writes longer.
LAST_USER_MESSAGE_DROPPED = "last_user_message_dropped"

#: The library removed a system message. ``PrefixCacheStage`` depends on the
#: system prompt being present on every call.
SYSTEM_PROMPT_DROPPED = "system_prompt_dropped"

#: Surviving messages came back in a different relative order. No stage claims
#: to reorder messages -- ``reorder_context`` moves blocks *inside* one
#: message's content.
MESSAGE_ORDER_CHANGED = "message_order_changed"

#: The library added a tool the caller did not supply. Stages may remove and
#: minify; inventing a capability is not something any of them claim.
TOOLS_ADDED = "tools_added"

#: The library removed a tool the conversation shows the agent already calling.
#: ``PruneToolsStage``'s stated promise; unverified against a real loop until
#: this rule existed.
CALLED_TOOL_REMOVED = "called_tool_removed"


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule.

    Attributes:
        rule: One of this module's rule-name constants.
        message_index: Position of the offending message, when one message is
            responsible. ``None`` for rules about the request as a whole.
        role: Role of the offending message, when there is one. Safe to print:
            a role is a fixed vocabulary, never caller text.
    """

    rule: str
    message_index: int | None = None
    role: str | None = None


def check(request: LLMRequest) -> tuple[Violation, ...]:
    """Return every absolute rule ``request`` breaks.

    Args:
        request: The request as it would be sent.

    Returns:
        Violations in message order; empty when the request is well formed.
    """
    messages = request.messages
    violations: list[Violation] = []

    if not any(m.role != "system" for m in messages):
        violations.append(Violation(NO_ANSWERABLE_MESSAGE))

    called_ids: set[str] = set()
    for index, message in enumerate(messages):
        called_ids.update(_call_ids(message))

        if message.role == "tool":
            violations.extend(_check_tool_result(messages, index, message, called_ids))
        elif not message.content and not _call_ids(message):
            violations.append(Violation(EMPTY_CONTENT_UNATTACHED, index, message.role))

    return tuple(violations)


def _check_tool_result(
    messages: Sequence[Message],
    index: int,
    message: Message,
    called_ids: set[str],
) -> list[Violation]:
    """Check one ``tool`` message against the calls that precede it."""
    violations: list[Violation] = []

    # Walk back past any run of sibling tool results to the assistant that
    # issued them: parallel tool calls put several results in a row, and only
    # the first is adjacent to its call.
    cursor = index - 1
    while cursor >= 0 and messages[cursor].role == "tool":
        cursor -= 1
    if cursor < 0 or not _call_ids(messages[cursor]):
        violations.append(Violation(TOOL_RESULT_UNPAIRED, index, message.role))
        return violations

    result_id = message.extra.get("tool_call_id")
    if isinstance(result_id, str) and result_id not in called_ids:
        violations.append(Violation(TOOL_RESULT_UNMATCHED_ID, index, message.role))
    return violations


def _tool_call_entries(message: Message) -> tuple[Any, ...]:
    """Raw ``tool_calls`` entries, wherever the caller's adapter stashed them.

    Two conventions exist in this package and both are correct for their site.
    :func:`~optio_optimize.wire.openai_messages` lifts ``tool_calls`` to
    ``extra["tool_calls"]`` because it builds a body from parts. The framework
    adapters keep the caller's whole message param under ``extra["_raw"]``
    instead, so they can hand back untouched anything this package does not
    model.

    Reading only the first convention made every rule here misfire on real
    agent traffic: a tool-calling assistant message looked like empty content
    with nothing attached, and every tool result that followed looked orphaned.
    Twenty violations on a run where the library had done nothing wrong. A
    checker that cries wolf gets switched off, which would have cost more than
    the rules are worth.
    """
    direct = message.extra.get("tool_calls")
    if isinstance(direct, (list, tuple)):
        return tuple(direct)
    raw = message.extra.get("_raw")
    if isinstance(raw, dict):
        nested = raw.get("tool_calls")
        if isinstance(nested, (list, tuple)):
            return tuple(nested)
    return ()


def _call_ids(message: Message) -> frozenset[str]:
    """Tool-call ids a message carries.

    A message with none returns an empty set, which is also how "this is not an
    assistant tool-call message" is expressed.
    """
    return frozenset(
        str(entry["id"])
        for entry in _tool_call_entries(message)
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def _identity(message: Message) -> tuple[Any, ...]:
    """A hashable stand-in for a message, for set membership across a rewrite.

    Content is included because a stage that rewrites text has changed the
    message; role and name because two messages can share text. Deliberately
    not object identity: stages return new frozen objects rather than mutating,
    so ``is`` comparison finds nothing.
    """
    return (message.role, message.content, message.name)


def check_transform(original: LLMRequest, sent: LLMRequest) -> tuple[Violation, ...]:
    """Return every preservation rule broken between ``original`` and ``sent``.

    The rules here cannot be expressed about a single request. A caller whose
    history legitimately opens with an assistant message is not malformed, and
    the library is right to pass it through untouched -- it is wrong to
    *create* one. Only the pair distinguishes those.

    Args:
        original: The request as the caller supplied it.
        sent: The request after every stage ran.

    Returns:
        Violations; empty when the rewrite preserved everything it must.
    """
    violations: list[Violation] = []
    violations.extend(_check_survivors(original, sent))
    violations.extend(_check_order(original, sent))
    violations.extend(_check_tools(original, sent))
    return tuple(violations)


def _check_survivors(original: LLMRequest, sent: LLMRequest) -> list[Violation]:
    """The system prompt and the last user turn must both survive."""
    violations: list[Violation] = []

    # Presence, not identity -- the same correction the system rule below
    # already carries, for the same reason and against different stages.
    # `deduplicate` and `prune_retrieval` are both default-on and both legally
    # rewrite the *last* user turn: one strips repeated context out of it, the
    # other drops retrieved passages from it. An identity check calls each of
    # those a dropped question, so the rule would fire on ordinary traffic while
    # the library behaved correctly, and the next real violation would be
    # ignored along with the noise.
    #
    # What this rule protects is that the model is still being asked something.
    # A rewritten question satisfies that; no question at all does not.
    last_user = _last_index(original.messages, "user")
    if last_user is not None and not any(m.role == "user" for m in sent.messages):
        violations.append(Violation(LAST_USER_MESSAGE_DROPPED, last_user, "user"))

    # Presence, not identity. What this protects is PrefixCacheStage's
    # dependency on *a* system prompt being there every call -- and editing one
    # is legitimate and default-on: StructuredOutputStage appends an
    # instruction, ConcisionStage used to. An identity check called every one of
    # those a dropped prompt, which on a real agent run fired on every single
    # call while the library was behaving correctly.
    if any(m.role == "system" for m in original.messages) and not any(
        m.role == "system" for m in sent.messages
    ):
        violations.append(Violation(SYSTEM_PROMPT_DROPPED, None, "system"))
    return violations


def _check_order(original: LLMRequest, sent: LLMRequest) -> list[Violation]:
    """Messages that survived must appear in their original relative order.

    Only survivors are compared. Inserting a message -- ``TrimHistoryStage``'s
    elision marker, ``StructuredOutputStage``'s system message -- is allowed,
    and so is rewriting one, which makes it a non-survivor under
    :func:`_identity` and simply drops out of the comparison.

    **A subsequence walk, not an index lookup.** The obvious implementation
    builds ``{identity: index}`` and compares positions, and it is wrong,
    because message identities are not unique: in an agent loop every
    tool-calling assistant message is ``("assistant", "", None)`` and every
    terse tool result carries the same text. A dict keeps only the last index
    of each, so an untouched request reads as reordered. Caught by this
    module's own "an untouched request passes" test on the first run.

    Walking instead means two byte-identical messages are interchangeable,
    which is also the right answer: swapping them changes nothing a provider
    or a model could observe.
    """
    messages = original.messages
    cursor = 0
    for message in sent.messages:
        identity = _identity(message)
        found = next(
            (i for i in range(cursor, len(messages)) if _identity(messages[i]) == identity),
            None,
        )
        if found is not None:
            cursor = found + 1
        elif any(_identity(m) == identity for m in messages[:cursor]):
            # Present in the original, but only *before* where we have already
            # walked to -- so it moved backwards past a message that survived.
            return [Violation(MESSAGE_ORDER_CHANGED)]
        # Otherwise it is not in the original at all: an inserted message,
        # which is allowed and does not advance the cursor.
    return []


def _check_tools(original: LLMRequest, sent: LLMRequest) -> list[Violation]:
    """Tools may be removed or minified, never added; called ones must stay."""
    violations: list[Violation] = []
    before, after = _tool_names(original.tools), _tool_names(sent.tools)

    if after - before:
        violations.append(Violation(TOOLS_ADDED))
    for name in _called_tool_names(original.messages) & before:
        if name not in after:
            violations.append(Violation(CALLED_TOOL_REMOVED))
    return violations


def _tool_names(tools: Iterable[dict[str, Any]]) -> frozenset[str]:
    """Tool names, reading both the OpenAI-nested and Anthropic-flat shapes."""
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        source = function if isinstance(function, dict) else tool
        name = source.get("name")
        if isinstance(name, str):
            names.add(name)
    return frozenset(names)


def _called_tool_names(messages: Sequence[Message]) -> frozenset[str]:
    """Names of tools the conversation shows the agent already invoking."""
    names: set[str] = set()
    for message in messages:
        for entry in _tool_call_entries(message):
            if not isinstance(entry, dict):
                continue
            function = entry.get("function")
            name = function.get("name") if isinstance(function, dict) else entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return frozenset(names)


def _last_index(messages: Sequence[Message], role: str) -> int | None:
    """Index of the last message with ``role``, or ``None``."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == role:
            return index
    return None
