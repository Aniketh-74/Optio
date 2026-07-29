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
    from collections.abc import Sequence

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


def _call_ids(message: Message) -> frozenset[str]:
    """Tool-call ids a message carries, in either provider's shape.

    OpenAI puts them in ``extra["tool_calls"]`` as dicts with an ``id``.
    A message with none returns an empty set, which is also how "this is not
    an assistant tool-call message" is expressed.
    """
    raw = message.extra.get("tool_calls")
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(
        str(entry["id"])
        for entry in raw
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
