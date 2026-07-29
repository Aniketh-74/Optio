"""TrimHistoryStage: bounding conversation history to a recent window.

``multi_turn_chat`` (bench/workloads.py) is the workload this exists for: a
conversation that resends its whole history every step, so cost grows with
the square of its length unless something bounds what gets sent.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.errors import OptimizeConfigError
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.history import TrimHistoryStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(
    *,
    recent_turns: int = 6,
    compact_at_tokens: int | None = None,
    anchor_turns: int = 0,
) -> StageContext:
    return StageContext(
        config=OptimizeConfig(
            recent_turns=recent_turns,
            compact_at_tokens=compact_at_tokens,
            anchor_turns=anchor_turns,
        ),
        counter=HeuristicCounter(),
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

        # system + the anchored first question + an elision marker + the window.
        assert len(result.request.messages) == 1 + 1 + 1 + 6
        assert result.saved_input_tokens > 0

    def test_the_opening_question_is_never_dropped(self) -> None:
        # In a chat this is an old question; in an agent loop it is the task,
        # and every message after it is the agent's own tool traffic. A real
        # Agents SDK run (scripts/real_agent_run.py) sent a conversation with
        # no user message at all -- the provider accepted it, and the model
        # answered a question it had to guess at.
        stage = TrimHistoryStage()

        result = stage.before(_chat(turns=10), _ctx(recent_turns=6))

        kept = result.request.messages
        assert kept[1].content == "question 0"

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
        assert kept[0].content == "question 0"  # anchored: the task
        assert kept[-1].content == "answer 9"
        assert [m.content for m in kept][-4:] == [
            "question 8",
            "answer 8",
            "question 9",
            "answer 9",
        ]

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
        assert len(result.request.messages) == 2 + 1 + 1 + 4


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
        # "user0" is the task and is now anchored, so the only cut available
        # is between it and the tool exchange -- which is not safe, because it
        # would orphan the tool result. Trimming nothing is the right answer.
        assert kept == messages
        assert result.saved_input_tokens == 0

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
        assert kept[1].content == "user0"  # the anchored task
        assert "calling two tools" in [m.content for m in kept]
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
        # system + anchored task + elision + exactly the requested window.
        assert len(kept) == 1 + 1 + 1 + 2
        assert kept[-2:] == messages[-2:]


class TestItIntegratesWithThePipeline:
    def test_repeated_calls_keep_the_prompt_bounded(self) -> None:
        """The shape multi_turn_chat exercises: history that grows every step."""
        from optio_optimize import LLMResponse, Optimizer

        optimizer = Optimizer(trim_history=True, exact_cache=False, prefix_cache=False)
        seen_sizes: list[int] = []

        def provider(request: LLMRequest) -> LLMResponse:
            seen_sizes.append(len(request.messages))
            return LLMResponse(content="ok", input_tokens=100, output_tokens=5, model=request.model)

        history: list[Message] = [Message(role="system", content="You are terse.")]
        for turn in range(20):
            history.append(Message(role="user", content=f"question {turn}"))
            optimizer.call(
                LLMRequest(model="gpt-4o", messages=tuple(history), temperature=0.0),
                provider,
            )
            history.append(Message(role="assistant", content=f"answer {turn}"))

        # system + the anchored opening question + an elision marker + window.
        # The constant matters far less than the property: unbounded, the last
        # call would have carried 41 messages, and the anchor adds a fixed two
        # rather than a growing number -- which is the whole claim.
        ceiling = 1 + 1 + 1 + OptimizeConfig().recent_turns
        assert max(seen_sizes) <= ceiling
        assert seen_sizes[-1] <= ceiling


class TestAppendThenCompact:
    """``compact_at_tokens``: hold trimming until the prompt is worth cutting.

    The mechanism is about *when* the prompt head moves, not what it says.
    Trimming every turn moves the start of the message list every turn, so a
    provider prefix cache matches nothing; leaving the conversation to append
    keeps the head byte-stable and bills all of it at the cached rate. The
    published guidance is that the second wins in almost every case.

    These tests pin the behaviour, not the guidance. Whether it is *worth*
    turning on is a live question the simulator cannot answer -- it matches a
    growing prefix by exact string comparison and has already been wrong about
    this exact stage once, predicting cost up 34.8% where the live API measured
    it down 8.4%.
    """

    def test_a_short_prompt_is_left_alone(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=20)
        ctx = _ctx(compact_at_tokens=1_000_000)

        assert stage.before(request, ctx).request.messages == request.messages

    def test_it_still_cuts_once_the_threshold_is_crossed(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=20)
        ctx = _ctx(compact_at_tokens=1)

        result = stage.before(request, ctx)
        assert len(result.request.messages) < len(request.messages)
        assert result.saved_input_tokens > 0

    def test_without_a_threshold_it_trims_every_turn_as_before(self) -> None:
        # The default is unchanged: this option adds a mode, it does not
        # replace the shipped behaviour.
        stage = TrimHistoryStage()
        request = _chat(turns=20)

        result = stage.before(request, _ctx())
        assert len(result.request.messages) < len(request.messages)

    def test_the_prompt_sawtooths_rather_than_growing_without_bound(self) -> None:
        """The property that makes this safe: it still bounds the prompt.

        Append-then-compact holds more tokens *between* compactions, which is
        the trade. What it must not do is stop bounding the conversation at
        all -- that would turn a cost option into an unbounded context leak.
        """
        from optio_optimize import LLMResponse, Optimizer

        optimizer = Optimizer(
            trim_history=True,
            compact_at_tokens=600,
            exact_cache=False,
            prefix_cache=False,
            detect_unstable_prefix=False,
        )
        sizes: list[int] = []

        def provider(request: LLMRequest) -> LLMResponse:
            sizes.append(len(request.messages))
            return LLMResponse(content="ok", input_tokens=100, output_tokens=5, model=request.model)

        history: list[Message] = [Message(role="system", content="You are terse. " * 40)]
        for turn in range(40):
            history.append(Message(role="user", content=f"question {turn} " * 10))
            optimizer.call(
                LLMRequest(model="gpt-4o", messages=tuple(history), temperature=0.0), provider
            )
            history.append(Message(role="assistant", content=f"answer {turn} " * 10))

        assert max(sizes) < 40, "the prompt grew without bound; compaction never fired"
        assert max(sizes) > 1 + OptimizeConfig().recent_turns, (
            "it compacted on every turn, which is the mode this option exists to avoid"
        )

    def test_a_non_positive_threshold_is_rejected(self) -> None:
        with pytest.raises(OptimizeConfigError, match="must be positive"):
            OptimizeConfig(compact_at_tokens=0)


class TestAnchoredTrimming:
    """``anchor_turns``: keep the oldest turns, drop the middle.

    Motivated by two of this project's own measurements pointing the same way.
    The provider's cached region is ``system + oldest turns`` -- 87% of
    ``multi_turn_chat``'s prompt was served from cache before trimming touched
    it -- and the recall audit found load-bearing facts stated in the first
    exchange and never repeated, of which plain trimming recovered 0 of 4. A
    front cut discards the cheapest and the most valuable context at once.

    Live on ``multi_turn_chat_long``: sliding saved 26.3% of cost with 25/50
    replies unchanged; anchoring saved 16.8% with **50/50** unchanged.
    """

    def test_the_oldest_turns_survive_the_cut(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=20)
        original = request.messages

        result = stage.before(request, _ctx(recent_turns=6, anchor_turns=2))
        kept = result.request.messages

        assert original[1] in kept, "the opening user message was dropped"
        assert original[2] in kept, "the opening reply was dropped"

    def test_the_recent_turns_still_survive(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=20)

        result = stage.before(request, _ctx(recent_turns=6, anchor_turns=2))

        assert request.messages[-1] in result.request.messages

    def test_the_middle_is_what_goes(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=20)

        result = stage.before(request, _ctx(recent_turns=6, anchor_turns=2))

        # question 5 sits in neither the opening pair nor the last six messages.
        assert not any("question 5" in m.content for m in result.request.messages)

    def test_the_gap_is_declared_rather_than_hidden(self) -> None:
        # A model shown the opening of a conversation followed by a much later
        # exchange, with no sign anything is missing, will try to reconcile the
        # jump. One line is cheap insurance.
        stage = TrimHistoryStage()

        result = stage.before(_chat(turns=20), _ctx(recent_turns=6, anchor_turns=2))

        assert any("omitted" in m.content for m in result.request.messages)

    def test_even_a_plain_slide_declares_its_gap_now(self) -> None:
        # This used to assert the opposite. A pure front cut leaves a
        # conversation that reads as though it started later, which needs no
        # marker -- but there is no longer such a thing as a pure front cut,
        # because the opening turn is always anchored, so there is always a
        # gap between it and the window.
        stage = TrimHistoryStage()

        result = stage.before(_chat(turns=20), _ctx(recent_turns=6, anchor_turns=0))

        assert any("omitted" in m.content for m in result.request.messages)

    def test_it_still_saves_tokens(self) -> None:
        stage = TrimHistoryStage()

        anchored = stage.before(_chat(turns=20), _ctx(recent_turns=6, anchor_turns=2))
        slid = stage.before(_chat(turns=20), _ctx(recent_turns=6, anchor_turns=0))

        assert anchored.saved_input_tokens > 0
        assert anchored.saved_input_tokens < slid.saved_input_tokens, (
            "anchoring keeps more than sliding, so it must claim less"
        )

    def test_a_conversation_too_short_to_anchor_is_left_alone(self) -> None:
        stage = TrimHistoryStage()
        request = _chat(turns=4)  # 8 history messages, window 6 + anchor 2

        assert stage.before(request, _ctx(recent_turns=6, anchor_turns=2)).note == ""

    def test_the_default_is_still_a_plain_sliding_window(self) -> None:
        # Anchoring changes what every caller sends. One good measurement on
        # one workload is not grounds for that (ADR-016).
        assert OptimizeConfig().anchor_turns == 0

    def test_a_negative_anchor_is_rejected(self) -> None:
        with pytest.raises(OptimizeConfigError, match="cannot be negative"):
            OptimizeConfig(anchor_turns=-1)


class TestTheTaskIsNotHistory:
    """The bug a real agent found, and no chat workload could.

    In a chat, the first user turn is an old question that has been answered.
    In an agent loop it is *the task*, and everything after it is the agent's
    own tool traffic -- so a sliding window whose oldest entry is the task
    drops the only statement of what the model is supposed to do.

    It fails silently: providers accept a conversation with no user message at
    all. The live run (scripts/real_agent_run.py) got a markdown dump of order
    fields instead of the two-sentence answer requested, and paid more for it.
    """

    @staticmethod
    def _agent_loop(steps: int) -> LLMRequest:
        """System prompt, one task, then nothing but tool traffic."""
        messages: list[Message] = [
            Message(role="system", content="You are a support agent."),
            Message(role="user", content="Refund the damaged blue widget for alice@example.com."),
        ]
        for step in range(steps):
            messages.append(Message(role="assistant", content=f"calling tool {step}"))
            messages.append(Message(role="tool", content=f"result {step}", name=f"tool_{step}"))
        return LLMRequest(model="gpt-4o", messages=tuple(messages), temperature=0.0)

    def test_the_task_survives_a_long_tool_loop(self) -> None:
        result = TrimHistoryStage().before(self._agent_loop(steps=10), _ctx(recent_turns=4))

        kept = result.request.messages
        assert any(m.role == "user" for m in kept), (
            "the conversation went to the provider with no user message at all; "
            "the model has to guess what it was asked"
        )
        assert kept[1].content.startswith("Refund the damaged")

    def test_it_still_trims(self) -> None:
        # The fix must not turn the stage off -- anchoring one turn is not the
        # same as declining, and a stage that silently stops saving is the
        # failure mode config.py warns about.
        request = self._agent_loop(steps=10)
        result = TrimHistoryStage().before(request, _ctx(recent_turns=4))

        assert result.saved_input_tokens > 0
        assert len(result.request.messages) < len(request.messages)

    def test_the_gap_is_declared(self) -> None:
        result = TrimHistoryStage().before(self._agent_loop(steps=10), _ctx(recent_turns=4))

        assert any("omitted" in m.content for m in result.request.messages)

    def test_no_tool_result_is_orphaned(self) -> None:
        # Anchoring moves the cut point, which is exactly where the orphan
        # hazard lives: a tool message whose assistant call was dropped is an
        # invalid request on every major provider.
        kept = TrimHistoryStage().before(self._agent_loop(steps=10), _ctx(recent_turns=4))
        messages = kept.request.messages
        for index, message in enumerate(messages):
            if message.role == "tool":
                assert index > 0, "a tool result led the conversation"
                assert messages[index - 1].role in {"assistant", "tool"}

    def test_an_agent_loop_with_no_user_turn_is_left_alone_at_the_front(self) -> None:
        # Not every caller starts with a user message. When the first turn is
        # not one, there is no task to anchor and the plain window applies.
        messages = (
            Message(role="system", content="sys"),
            *[Message(role="assistant", content=f"step {i}") for i in range(10)],
        )
        request = LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)

        result = TrimHistoryStage().before(request, _ctx(recent_turns=4))

        assert result.request.messages[1].content == "step 6"
