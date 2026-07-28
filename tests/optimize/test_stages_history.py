"""TrimHistoryStage: bounding conversation history to a recent window.

``multi_turn_chat`` (bench/workloads.py) is the workload this exists for: a
conversation that resends its whole history every step, so cost grows with
the square of its length unless something bounds what gets sent.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.history import TrimHistoryStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(*, recent_turns: int = 6) -> StageContext:
    return StageContext(
        config=OptimizeConfig(recent_turns=recent_turns), counter=HeuristicCounter()
    )


def _chat(turns: int, *, system: bool = True) -> LLMRequest:
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content="You are terse."))
    for turn in range(turns):
        messages.append(Message(role="user", content=f"question {turn}"))
        messages.append(Message(role="assistant", content=f"answer {turn}"))
    return LLMRequest(model="gpt-4o", messages=tuple(messages), temperature=0.0)


class TestItDropsAgedOutHistory:
    def test_history_beyond_the_window_is_removed(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=10)  # system + 20 turn-messages
        ctx = _ctx(recent_turns=6)

        result = stage.before(request, ctx)

        assert len(result.request.messages) == 1 + 6
        assert result.saved_input_tokens > 0

    def test_the_system_message_always_survives(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=10)
        ctx = _ctx(recent_turns=6)

        result = stage.before(request, ctx)

        assert result.request.messages[0].role == "system"

    def test_the_most_recent_turns_are_the_ones_kept(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=10)
        ctx = _ctx(recent_turns=4)

        result = stage.before(request, ctx)

        kept = result.request.messages[1:]
        assert kept[0].content == "question 8"
        assert kept[-1].content == "answer 9"

    def test_multiple_system_messages_all_survive(self) -> None:
        stage = TrimHistoryStage()
        rest = _chat(turns=10, system=False).messages
        messages = (
            Message(role="system", content="rule one"),
            Message(role="system", content="rule two"),
            *rest,
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)
        ctx = _ctx(recent_turns=4)

        result = stage.before(request, ctx)

        assert result.request.messages[0].content == "rule one"
        assert result.request.messages[1].content == "rule two"
        assert len(result.request.messages) == 2 + 4


class TestItDeclinesWhenThereIsNothingToTrim:
    def test_history_at_or_under_the_window_is_untouched(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=2)  # 4 turn-messages, under the default window of 6
        ctx = _ctx(recent_turns=6)

        result = stage.before(request, ctx)

        assert result.request.messages == request.messages
        assert result.saved_input_tokens == 0
        assert not result.short_circuited

    def test_a_system_only_request_is_untouched(self) -> None:
        stage = TrimHistoryStage()
        request = LLMRequest(
            model="gpt-4o",
            messages=(Message(role="system", content="hello"),),
            temperature=0.0,
        )
        ctx = _ctx(recent_turns=6)

        result = stage.before(request, ctx)

        assert result.request.messages == request.messages


class TestItNeverOrphansAToolResult:
    """A ``tool`` message with no preceding tool_calls assistant message is an
    invalid request on every major provider -- the same class of defect that
    made ``fan_out`` fail all 12 live calls over a missing ``"json"`` literal.
    """

    def test_a_cut_landing_on_a_tool_message_is_pushed_back_to_the_assistant(self) -> None:
        stage = TrimHistoryStage()
        # Naive cut (5 messages, keep 3) lands exactly on the tool result,
        # which would orphan it from the assistant message before it.
        messages = (
            Message(role="system", content="sys"),
            Message(role="user", content="user0"),
            Message(role="assistant", content="calling fetch_record(0)"),
            Message(role="tool", content="record 0: ok", name="fetch_record"),
            Message(role="user", content="user1"),
            Message(role="assistant", content="answer1"),
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)
        ctx = _ctx(recent_turns=3)

        result = stage.before(request, ctx)

        kept = result.request.messages
        assert kept[1].role == "assistant"
        assert kept[1].content == "calling fetch_record(0)"
        assert kept[2].role == "tool"
        # Only the leading user turn -- entirely before the tool exchange --
        # was safe to drop.
        assert [m.content for m in messages if m not in kept] == ["user0"]

    def test_parallel_tool_results_all_move_together(self) -> None:
        stage = TrimHistoryStage()
        messages = (
            Message(role="system", content="sys"),
            Message(role="user", content="user0"),
            Message(role="assistant", content="calling two tools"),
            Message(role="tool", content="result a", name="tool_a"),
            Message(role="tool", content="result b", name="tool_b"),
            Message(role="user", content="user1"),
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)
        ctx = _ctx(recent_turns=2)  # naive cut would land inside the pair of tool results

        result = stage.before(request, ctx)

        kept = result.request.messages
        assert kept[1].content == "calling two tools"
        assert [m.content for m in kept if m.role == "tool"] == ["result a", "result b"]

    def test_no_safe_cut_point_means_no_trim(self) -> None:
        """The whole history is one tool exchange: trimming nothing is correct."""
        stage = TrimHistoryStage()
        messages = (
            Message(role="system", content="sys"),
            Message(role="assistant", content="calling two tools"),
            Message(role="tool", content="result a"),
            Message(role="tool", content="result b"),
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)
        ctx = _ctx(recent_turns=1)

        result = stage.before(request, ctx)

        assert result.request.messages == request.messages
        assert result.saved_input_tokens == 0

    def test_a_cut_landing_on_a_safe_boundary_does_not_over_extend(self) -> None:
        """When the naive cut is already safe, nothing extra is kept."""
        stage = TrimHistoryStage()
        messages = (
            Message(role="system", content="sys"),
            Message(role="user", content="user0"),
            Message(role="assistant", content="calling fetch_record(0)"),
            Message(role="tool", content="record 0: ok"),
            Message(role="assistant", content="final answer 0"),
            Message(role="user", content="user1"),
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)
        ctx = _ctx(recent_turns=2)  # naive cut lands on "user1"'s predecessor, a safe boundary

        result = stage.before(request, ctx)

        kept = result.request.messages
        assert len(kept) == 1 + 2  # system + exactly the requested window, no extension needed


class TestItIntegratesWithThePipeline:
    def test_repeated_calls_keep_the_prompt_bounded(self) -> None:
        """The shape multi_turn_chat exercises: history that grows every step."""
        from optio_optimize import LLMResponse, Optimizer

        optimizer = Optimizer(trim_history=True, exact_cache=False, prefix_cache=False)
        seen_sizes: list[int] = []

        def provider(request: LLMRequest) -> LLMResponse:
            seen_sizes.append(len(request.messages))
            return LLMResponse(
                content="ok", input_tokens=100, output_tokens=5, model=request.model
            )

        history: list[Message] = [Message(role="system", content="You are terse.")]
        for turn in range(20):
            history.append(Message(role="user", content=f"question {turn}"))
            optimizer.call(
                LLMRequest(model="gpt-4o", messages=tuple(history), temperature=0.0),
                provider,
            )
            history.append(Message(role="assistant", content=f"answer {turn}"))

        ceiling = 1 + OptimizeConfig().recent_turns
        # Unbounded, the last call would have carried 41 messages. Trimming
        # holds every call to system + the recent-turn window.
        assert max(seen_sizes) <= ceiling
        assert seen_sizes[-1] <= ceiling
