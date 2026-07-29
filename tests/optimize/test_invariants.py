"""Rules the library must not break, and the two shapes they come in.

Absolute rules are enforced by the provider: violating one is a 400. They are
cheap to state and were never the problem.

Preservation rules are enforced by nobody. That is why a stage could drop the
user's task for weeks while 1,304 tests passed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from optio_optimize import LLMRequest, Message
from optio_optimize.invariants import (
    CALLED_TOOL_REMOVED,
    EMPTY_CONTENT_UNATTACHED,
    LAST_USER_MESSAGE_DROPPED,
    MESSAGE_ORDER_CHANGED,
    NO_ANSWERABLE_MESSAGE,
    SYSTEM_PROMPT_DROPPED,
    TOOL_RESULT_UNMATCHED_ID,
    TOOL_RESULT_UNPAIRED,
    TOOLS_ADDED,
    check,
    check_transform,
)

pytestmark = pytest.mark.optimize

_SEARCH = {"type": "function", "function": {"name": "search", "description": "d"}}
_LOOKUP = {"type": "function", "function": {"name": "lookup", "description": "d"}}


def _request(*messages: Message) -> LLMRequest:
    return LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)


def _calls(*ids: str) -> Message:
    """An assistant message carrying tool calls, as a real agent emits it."""
    return Message(
        role="assistant",
        content="",
        extra={"tool_calls": [{"id": i, "type": "function"} for i in ids]},
    )


def _result(call_id: str, text: str = "ok") -> Message:
    return Message(role="tool", content=text, extra={"tool_call_id": call_id})


def _rules(request: LLMRequest) -> set[str]:
    return {v.rule for v in check(request)}


def _transform_rules(original: LLMRequest, sent: LLMRequest) -> set[str]:
    return {v.rule for v in check_transform(original, sent)}


def _agent_loop(steps: int = 6) -> LLMRequest:
    """System prompt, one task, then nothing but tool traffic."""
    messages: list[Message] = [
        Message(role="system", content="You are a support agent."),
        Message(role="user", content="Refund the damaged widget."),
    ]
    for step in range(steps):
        messages.append(_calls(f"c{step}"))
        messages.append(_result(f"c{step}"))
    return LLMRequest(model="gpt-4o", messages=tuple(messages), temperature=0.0)


def _called_tool_loop() -> LLMRequest:
    """An agent loop that has already called `search`, so every rule can fire."""
    return LLMRequest(
        model="gpt-4o",
        messages=(
            Message(role="system", content="You are a support agent."),
            Message(role="user", content="Refund the damaged widget."),
            Message(
                role="assistant",
                content="",
                extra={"tool_calls": [{"id": "c1", "function": {"name": "search"}}]},
            ),
            _result("c1"),
        ),
        temperature=0.0,
    )


class TestAWellFormedRequestPasses:
    def test_a_plain_chat(self):
        assert (
            check(
                _request(
                    Message(role="system", content="terse"),
                    Message(role="user", content="hi"),
                )
            )
            == ()
        )

    def test_a_real_agent_loop(self):
        assert (
            check(
                _request(
                    Message(role="system", content="terse"),
                    Message(role="user", content="do the thing"),
                    _calls("c1"),
                    _result("c1"),
                )
            )
            == ()
        )

    def test_parallel_tool_calls(self):
        assert (
            check(
                _request(
                    Message(role="user", content="do two things"),
                    _calls("c1", "c2"),
                    _result("c1"),
                    _result("c2"),
                )
            )
            == ()
        )

    def test_a_call_still_awaiting_its_result(self):
        # Mid-loop this is normal: the call has been made, the result has not
        # come back. Only the tool-result-to-call direction is checkable.
        assert (
            check(
                _request(
                    Message(role="user", content="go"),
                    _calls("c1"),
                )
            )
            == ()
        )


class TestAbsoluteRules:
    def test_a_tool_result_with_no_preceding_call_is_flagged(self):
        request = _request(
            Message(role="system", content="terse"),
            Message(role="user", content="hi"),
            _result("c1"),
        )
        assert TOOL_RESULT_UNPAIRED in _rules(request)

    def test_a_tool_result_whose_id_was_never_called_is_flagged(self):
        request = _request(
            Message(role="user", content="hi"),
            _calls("c1"),
            _result("c9"),
        )
        assert TOOL_RESULT_UNMATCHED_ID in _rules(request)

    def test_empty_content_without_tool_calls_is_flagged(self):
        request = _request(
            Message(role="user", content="hi"),
            Message(role="assistant", content=""),
        )
        assert EMPTY_CONTENT_UNATTACHED in _rules(request)

    def test_empty_content_with_tool_calls_is_fine(self):
        # The single most common real-agent message shape, and the one no
        # fixture in this repo used before 2026-07-29.
        assert check(_request(Message(role="user", content="hi"), _calls("c1"))) == ()

    def test_a_system_only_request_is_flagged(self):
        request = _request(Message(role="system", content="terse"))
        assert NO_ANSWERABLE_MESSAGE in _rules(request)

    def test_an_empty_request_is_flagged(self):
        assert NO_ANSWERABLE_MESSAGE in _rules(_request())


class TestViolationsCarryNoPromptContent:
    def test_no_violation_repeats_message_text(self):
        # Violations are printed and can reach CI logs. Section 10's rule --
        # the same one that makes the fail-open guard log an exception's type
        # and never its message -- applies here.
        secret = "the customer's confidential question"
        request = _request(
            Message(role="system", content=secret),
            Message(role="assistant", content=""),
        )
        rendered = " ".join(f"{v.rule}{v.message_index}{v.role}" for v in check(request))
        assert secret not in rendered
        assert all(secret not in str(v) for v in check(request))

    def test_a_violation_locates_the_problem(self):
        request = _request(Message(role="user", content="hi"), _result("c1"))
        violation = next(v for v in check(request) if v.rule == TOOL_RESULT_UNPAIRED)
        assert violation.message_index == 1
        assert violation.role == "tool"


# --------------------------------------------------------------------------
# Preservation rules: the half nobody enforces.
# --------------------------------------------------------------------------


class TestPreservationRules:
    def test_an_untouched_request_passes(self):
        request = _agent_loop()
        assert check_transform(request, request) == ()

    def test_an_ordinary_trim_passes(self):
        # Dropping middle turns is what trim_history is *for*. The rule must
        # not fire on correct behaviour, or it will be switched off.
        original = _agent_loop(steps=6)
        kept = original.messages[:2] + original.messages[-4:]
        assert check_transform(original, original.with_messages(kept)) == ()

    def test_dropping_the_last_user_message_is_flagged(self):
        # The 2026-07-29 defect, exactly: system + tool traffic, no task.
        original = _agent_loop()
        stripped = tuple(m for m in original.messages if m.role != "user")
        assert LAST_USER_MESSAGE_DROPPED in _transform_rules(
            original, original.with_messages(stripped)
        )

    def test_dropping_the_system_prompt_is_flagged(self):
        original = _agent_loop()
        stripped = tuple(m for m in original.messages if m.role != "system")
        assert SYSTEM_PROMPT_DROPPED in _transform_rules(original, original.with_messages(stripped))

    def test_reordering_surviving_messages_is_flagged(self):
        original = _agent_loop(steps=2)
        reversed_ = tuple(reversed(original.messages))
        assert MESSAGE_ORDER_CHANGED in _transform_rules(
            original, original.with_messages(reversed_)
        )

    def test_an_inserted_elision_marker_does_not_count_as_reordering(self):
        # TrimHistoryStage inserts a system marker declaring the gap. New
        # messages are allowed; reordering surviving ones is not.
        original = _agent_loop(steps=6)
        marker = Message(role="system", content="[earlier turns omitted]")
        kept = (*original.messages[:2], marker, *original.messages[-4:])
        assert check_transform(original, original.with_messages(kept)) == ()

    def test_rewriting_a_message_is_not_reordering(self):
        # cap_tool_results truncates content in place. The rewritten message is
        # a new object with new text and must not read as "removed".
        original = _agent_loop(steps=2)
        messages = list(original.messages)
        messages[3] = messages[3].with_content("truncated...")
        assert check_transform(original, original.with_messages(tuple(messages))) == ()

    def test_adding_a_tool_is_flagged(self):
        original = replace(_agent_loop(), tools=(_SEARCH,))
        widened = replace(original, tools=(_SEARCH, _LOOKUP))
        assert TOOLS_ADDED in _transform_rules(original, widened)

    def test_pruning_an_uncalled_tool_passes(self):
        original = replace(_agent_loop(steps=0), tools=(_SEARCH, _LOOKUP))
        assert check_transform(original, replace(original, tools=(_SEARCH,))) == ()

    def test_removing_a_tool_the_agent_already_called_is_flagged(self):
        # PruneToolsStage's stated promise, never checked against a real loop.
        original = replace(_called_tool_loop(), tools=(_SEARCH, _LOOKUP))
        assert CALLED_TOOL_REMOVED in _transform_rules(
            original, replace(original, tools=(_LOOKUP,))
        )


def _damaged_drop_user(original: LLMRequest) -> LLMRequest:
    return original.with_messages(tuple(m for m in original.messages if m.role != "user"))


def _damaged_drop_system(original: LLMRequest) -> LLMRequest:
    return original.with_messages(tuple(m for m in original.messages if m.role != "system"))


def _damaged_reorder(original: LLMRequest) -> LLMRequest:
    return original.with_messages(tuple(reversed(original.messages)))


def _damaged_add_tool(original: LLMRequest) -> LLMRequest:
    return replace(original, tools=(*original.tools, _LOOKUP))


def _damaged_remove_called_tool(original: LLMRequest) -> LLMRequest:
    return replace(original, tools=())


#: Each preservation rule paired with a transform that must trigger it. A rule
#: with no entry here has no proof it can fire; a rule whose entry does not fire
#: is broken. The two tests below catch those two failures separately.
_DAMAGE: dict[str, Any] = {
    LAST_USER_MESSAGE_DROPPED: _damaged_drop_user,
    SYSTEM_PROMPT_DROPPED: _damaged_drop_system,
    MESSAGE_ORDER_CHANGED: _damaged_reorder,
    TOOLS_ADDED: _damaged_add_tool,
    CALLED_TOOL_REMOVED: _damaged_remove_called_tool,
}


class TestEveryPreservationRuleCanFail:
    """A rule that cannot fail is not a rule."""

    @pytest.mark.parametrize("rule", sorted(_DAMAGE))
    def test_the_rule_fires_on_a_transform_that_breaks_it(self, rule):
        original = replace(_called_tool_loop(), tools=(_SEARCH,))
        damaged = _DAMAGE[rule](original)
        assert rule in _transform_rules(original, damaged), (
            f"{rule} did not fire on a transform built specifically to break it"
        )

    def test_every_preservation_constant_has_a_damage_case(self):
        # The other direction: a rule added to the module with no proof it can
        # fire. Reads the module rather than a hand-kept list, so adding a
        # constant and forgetting its damage case fails here.
        from optio_optimize import invariants

        declared = {
            getattr(invariants, name)
            for name in dir(invariants)
            if name.isupper() and isinstance(getattr(invariants, name), str)
        }
        absolute = {
            TOOL_RESULT_UNPAIRED,
            TOOL_RESULT_UNMATCHED_ID,
            EMPTY_CONTENT_UNATTACHED,
            NO_ANSWERABLE_MESSAGE,
        }
        assert (declared - absolute) == set(_DAMAGE)


class TestAgainstTheRealStage:
    def test_the_trim_history_fix_holds_under_the_rule(self):
        # Not a synthetic transform: the actual stage, on the actual shape that
        # broke. If someone removes the task-anchor floor, this fails.
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.history import TrimHistoryStage
        from optio_optimize.tokens import HeuristicCounter

        original = _agent_loop(steps=10)
        ctx = StageContext(config=OptimizeConfig(recent_turns=4), counter=HeuristicCounter())
        result = TrimHistoryStage().before(original, ctx)

        assert check_transform(original, result.request) == ()
        assert check(result.request) == ()
