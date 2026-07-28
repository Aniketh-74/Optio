"""SummarizeHistoryStage: replacing aged-out history with a summary."""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.summarize import SummarizeHistoryStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(*, recent_turns: int = 4) -> StageContext:
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


class TestNoSummarizerConfigured:
    def test_always_declines_with_no_summarizer(self) -> None:
        stage = SummarizeHistoryStage()  # default: summarizer=None
        request = _chat(turns=10)

        result = stage.before(request, _ctx())

        assert not result.short_circuited
        assert result.request.messages == request.messages
        assert result.note == ""


class TestWithASummarizer:
    def test_replaces_aged_out_history_with_a_summary_message(self) -> None:
        stage = SummarizeHistoryStage(summarizer=lambda text: f"summary of {len(text)} chars")
        request = _chat(turns=10)

        result = stage.before(request, _ctx(recent_turns=4))

        messages = result.request.messages
        assert messages[0].role == "system"  # original system message
        assert messages[1].role == "system"  # the summary, inserted as one
        assert "summary of" in messages[1].content
        # system + summary + the 4 kept turn-messages.
        assert len(messages) == 1 + 1 + 4

    def test_the_most_recent_turns_survive_verbatim(self) -> None:
        stage = SummarizeHistoryStage(summarizer=lambda text: "summary")
        request = _chat(turns=10)

        result = stage.before(request, _ctx(recent_turns=4))

        kept = result.request.messages[2:]  # after system + summary
        assert kept[0].content == "question 8"
        assert kept[-1].content == "answer 9"

    def test_declines_when_history_is_already_within_the_window(self) -> None:
        stage = SummarizeHistoryStage(summarizer=lambda text: "summary")
        request = _chat(turns=1)  # 2 turn-messages, under a window of 4

        result = stage.before(request, _ctx(recent_turns=4))

        assert result.request.messages == request.messages
        assert result.note == ""

    def test_an_empty_summary_is_treated_as_a_decline(self) -> None:
        stage = SummarizeHistoryStage(summarizer=lambda text: "")
        request = _chat(turns=10)

        result = stage.before(request, _ctx())

        assert result.request.messages == request.messages
        assert result.note == ""

    def test_the_summarizer_receives_only_the_dropped_turns(self) -> None:
        seen: list[str] = []

        def spy(text: str) -> str:
            seen.append(text)
            return "summary"

        stage = SummarizeHistoryStage(summarizer=spy)
        request = _chat(turns=10)

        stage.before(request, _ctx(recent_turns=4))

        assert seen  # the summarizer was actually called
        assert "question 9" not in seen[0]  # the most recent turn was never dropped
        assert "question 0" in seen[0]  # the oldest turn was

    def test_a_raising_summarizer_propagates_for_the_pipelines_guard_to_catch(self) -> None:
        """Stages are permitted to raise (stages/base.py); the pipeline's own
        guard absorbs it. This stage must not add a second, redundant guard.
        """

        def broken(text: str) -> str:
            raise RuntimeError("summarizer is down")

        stage = SummarizeHistoryStage(summarizer=broken)

        with pytest.raises(RuntimeError, match="summarizer is down"):
            stage.before(_chat(turns=10), _ctx())


class TestToolCallSafety:
    """Shares TrimHistoryStage's invariant: never orphan a tool result."""

    def test_a_cut_landing_on_a_tool_message_is_pushed_back_to_the_assistant(self) -> None:
        stage = SummarizeHistoryStage(summarizer=lambda text: "summary")
        messages = (
            Message(role="system", content="sys"),
            Message(role="user", content="user0"),
            Message(role="assistant", content="calling fetch_record(0)"),
            Message(role="tool", content="record 0: ok"),
            Message(role="user", content="user1"),
            Message(role="assistant", content="answer1"),
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)

        result = stage.before(request, _ctx(recent_turns=3))

        kept = result.request.messages
        # system + summary + [assistant(tool_calls), tool, user1, answer1]
        assert kept[2].role == "assistant"
        assert kept[2].content == "calling fetch_record(0)"
        assert kept[3].role == "tool"

    def test_no_safe_cut_point_means_no_summary(self) -> None:
        stage = SummarizeHistoryStage(summarizer=lambda text: "summary")
        messages = (
            Message(role="system", content="sys"),
            Message(role="assistant", content="calling two tools"),
            Message(role="tool", content="result a"),
            Message(role="tool", content="result b"),
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)

        result = stage.before(request, _ctx(recent_turns=1))

        assert result.request.messages == request.messages
        assert result.note == ""
