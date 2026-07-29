"""Rules the library must not break, and the two shapes they come in.

Absolute rules are enforced by the provider: violating one is a 400. They are
cheap to state and were never the problem.

Preservation rules are enforced by nobody. That is why a stage could drop the
user's task for weeks while 1,304 tests passed.
"""

from __future__ import annotations

import pytest

from optio_optimize import LLMRequest, Message
from optio_optimize.invariants import (
    EMPTY_CONTENT_UNATTACHED,
    NO_ANSWERABLE_MESSAGE,
    TOOL_RESULT_UNMATCHED_ID,
    TOOL_RESULT_UNPAIRED,
    check,
)

pytestmark = pytest.mark.optimize


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
