"""``cap_tool_results`` must see the shape Anthropic callers actually send (ADR-032).

The stage bounds a single tool result because an oversized payload is re-billed
on every later turn. On the live Sonnet 4.5 suite it saved **7,831 tokens on
``mcp_agent``** -- the second-largest stage saving there, behind only
``trim_history``.

``mcp_agent`` builds ``Message(role="tool", content=payload)`` directly. A user
reaching this package through ``wrap_anthropic_client`` sends the Anthropic wire
shape, where a tool result is a ``tool_result`` **block** inside a ``role="user"``
turn. Measured on both, with an 8,001-token payload::

    bench shape (role='tool')   content_tokens=8001   saved=5981
    adapter shape               content_tokens=   0   saved=   0

Zero for two independent reasons: the role filter skips it, **and**
``_text_from_content`` contributes nothing for a non-text block, so
``message.content`` is ``""`` and there is nothing to measure. The payload lives
in ``extra[RAW_CONTENT_KEY]``, where the stage never looked.

ADR-022 settled this principle for the sibling case -- a cache key reading only
``Message.content`` hashed two different images identically -- and it was never
carried to the stage whose whole purpose is bounding the largest non-text
payload there is.
"""

from __future__ import annotations

from typing import Any

import pytest

from optio_optimize.adapters.anthropic import _message_from_param, _param_from_message
from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.tools import CapToolResultsStage
from optio_optimize.tokens import default_counter
from optio_optimize.types import LLMRequest, Message
from optio_optimize.wire import RAW_CONTENT_KEY

pytestmark = pytest.mark.optimize

MODEL = "claude-sonnet-4-5"
HUGE = "row data " * 4_000


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=default_counter())


def _adapter_message(payload: str = HUGE, *, blocks: int = 1) -> Message:
    """A tool result in the shape `wrap_anthropic_client` receives."""
    return _message_from_param(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"t{n}", "content": payload}
                for n in range(blocks)
            ],
        }
    )


def _request(*messages: Message) -> LLMRequest:
    return LLMRequest(
        model=MODEL,
        messages=(Message(role="user", content="go"), *messages),
        temperature=0.0,
    )


def _payload_of(message: Message, index: int = 0) -> str:
    raw = message.extra[RAW_CONTENT_KEY]
    assert isinstance(raw, dict)
    content = raw["content"]
    assert isinstance(content, list)
    block: Any = content[index]
    return str(block["content"])


class TestTheAdapterShapeIsCapped:
    def test_an_oversized_block_is_truncated(self) -> None:
        """The whole point: 8,001 tokens reached the wire uncapped."""
        result = CapToolResultsStage().before(_request(_adapter_message()), _ctx())

        assert len(_payload_of(result.request.messages[1])) < len(HUGE)

    def test_a_saving_is_booked(self) -> None:
        result = CapToolResultsStage().before(_request(_adapter_message()), _ctx())

        assert result.saved_input_tokens > 0
        assert result.note

    def test_the_saving_matches_what_was_removed(self) -> None:
        """ADR-024: the ledger records tokens the provider will not be billed for.

        Measured against the block payload, not against ``message.content`` --
        which is ``""`` here and never moved.
        """
        counter = default_counter()
        before_tokens = counter.count_text(HUGE, MODEL)

        result = CapToolResultsStage().before(_request(_adapter_message()), _ctx())

        after_tokens = counter.count_text(_payload_of(result.request.messages[1]), MODEL)
        assert result.saved_input_tokens == before_tokens - after_tokens

    def test_a_small_result_is_left_alone(self) -> None:
        small = _adapter_message("ok")

        result = CapToolResultsStage().before(_request(small), _ctx())

        assert result.saved_input_tokens == 0
        assert _payload_of(result.request.messages[1]) == "ok"

    def test_every_block_in_one_turn_is_capped(self) -> None:
        """A single Anthropic user turn may carry several tool results.

        That is exactly why the adapter cannot translate one param into one
        ``role="tool"`` message, and why the stage handles the block list.
        """
        result = CapToolResultsStage().before(_request(_adapter_message(blocks=3)), _ctx())

        message = result.request.messages[1]
        assert all(len(_payload_of(message, n)) < len(HUGE) for n in range(3))


class TestTheBlockDiscriminatorIsExact:
    """Pinned directly, because the stage path hides a wrong answer.

    A block wrongly judged a tool result yields an empty payload, which caps to
    itself and rebuilds nothing -- so "treat every block as a tool_result"
    survived every end-to-end test here until these were added.
    """

    def test_a_text_block_is_not_a_tool_result(self) -> None:
        from optio_optimize.wire import tool_result_payload

        assert tool_result_payload({"type": "text", "text": "hello"}) is None

    def test_a_tool_use_block_is_not_a_tool_result(self) -> None:
        from optio_optimize.wire import tool_result_payload

        assert tool_result_payload({"type": "tool_use", "id": "t1", "input": {}}) is None

    def test_a_tool_result_block_is_one(self) -> None:
        from optio_optimize.wire import tool_result_payload

        assert tool_result_payload({"type": "tool_result", "content": "rows"}) == "rows"

    def test_a_text_only_turn_reports_no_payloads(self) -> None:
        from optio_optimize.wire import tool_result_payloads

        message = _message_from_param(
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        )

        assert tool_result_payloads(message) == []


class TestOtherBlocksInTheSameTurnAreUntouched:
    """An Anthropic turn mixes blocks freely, and only one kind is ours."""

    @staticmethod
    def _mixed() -> Message:
        return _message_from_param(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "here is the result"},
                    {"type": "tool_result", "tool_use_id": "t0", "content": HUGE},
                ],
            }
        )

    def test_the_text_block_survives(self) -> None:
        result = CapToolResultsStage().before(_request(self._mixed()), _ctx())

        raw = result.request.messages[1].extra[RAW_CONTENT_KEY]
        assert isinstance(raw, dict)
        blocks = raw["content"]
        assert blocks[0] == {"type": "text", "text": "here is the result"}

    def test_the_tool_result_beside_it_is_still_capped(self) -> None:
        result = CapToolResultsStage().before(_request(self._mixed()), _ctx())

        assert len(_payload_of(result.request.messages[1], 1)) < len(HUGE)

    def test_no_block_is_dropped(self) -> None:
        """Dropping one silently would orphan a ``tool_use`` and 400 the call."""
        result = CapToolResultsStage().before(_request(self._mixed()), _ctx())

        raw = result.request.messages[1].extra[RAW_CONTENT_KEY]
        assert isinstance(raw, dict)
        assert len(raw["content"]) == 2


class TestAnUnderCeilingPayloadIsNeverRewritten:
    """Tested at the policy function, not through the stage.

    An over-eager cap makes the result *longer* -- the truncation notice is
    appended -- so ``saved`` goes negative, the stage declines, and the original
    request comes back looking untouched. The end-to-end assertion passes while
    the policy is wrong.
    """

    def test_cap_returns_short_text_identically(self) -> None:
        stage = CapToolResultsStage()

        assert stage._cap("ok", _ctx(), MODEL) == "ok"

    def test_cap_truncates_text_over_the_ceiling(self) -> None:
        stage = CapToolResultsStage()

        capped = stage._cap(HUGE, _ctx(), MODEL)

        assert len(capped) < len(HUGE)
        assert "truncated by optio_optimize" in capped

    def test_a_payload_at_the_ceiling_is_left_alone(self) -> None:
        stage = CapToolResultsStage(max_tokens=50)
        counter = default_counter()
        text = "word " * 40
        assert counter.count_text(text, MODEL) <= 50

        assert stage._cap(text, _ctx(), MODEL) == text


class TestTheCallersRequestIsNotMutated:
    def test_the_original_message_is_untouched(self) -> None:
        """ADR-016: never edit an object we were handed.

        Capping in place would happen to reach the wire, because
        ``_param_from_message`` returns the raw param when the derived text is
        unchanged -- and it would be a side effect on the caller's own dict.
        """
        original = _adapter_message()

        CapToolResultsStage().before(_request(original), _ctx())

        assert _payload_of(original) == HUGE


class TestItSurvivesTheRoundTrip:
    def test_the_capped_payload_reaches_the_wire(self) -> None:
        """The fix is worthless if the adapter sends the original anyway."""
        result = CapToolResultsStage().before(_request(_adapter_message()), _ctx())

        param = _param_from_message(result.request.messages[1])

        assert isinstance(param, dict)
        blocks = param["content"]
        assert isinstance(blocks, list)
        assert len(str(blocks[0]["content"])) < len(HUGE)

    def test_the_block_keeps_its_tool_use_id(self) -> None:
        """A ``tool_result`` orphaned from its ``tool_use`` is a 400."""
        result = CapToolResultsStage().before(_request(_adapter_message()), _ctx())

        param = _param_from_message(result.request.messages[1])

        assert isinstance(param, dict)
        assert param["content"][0]["tool_use_id"] == "t0"
        assert param["content"][0]["type"] == "tool_result"

    def test_the_turn_keeps_its_role(self) -> None:
        result = CapToolResultsStage().before(_request(_adapter_message()), _ctx())

        param = _param_from_message(result.request.messages[1])

        assert isinstance(param, dict)
        assert param["role"] == "user"


class TestTheNeutralPathIsUnchanged:
    def test_a_role_tool_message_is_still_capped(self) -> None:
        """`mcp_agent`'s 7,831 tokens must not regress."""
        neutral = Message(role="tool", content=HUGE, extra={"tool_call_id": "t1"})

        result = CapToolResultsStage().before(_request(neutral), _ctx())

        assert result.saved_input_tokens > 0
        assert len(result.request.messages[1].content) < len(HUGE)

    def test_an_ordinary_user_turn_is_untouched(self) -> None:
        plain = Message(role="user", content=HUGE)

        result = CapToolResultsStage().before(_request(plain), _ctx())

        assert result.saved_input_tokens == 0
