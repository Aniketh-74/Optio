"""Trimming must price the output it buys (ADR-026).

The first full live Anthropic benchmark showed ``trim_history`` -- on by default
-- **increasing** total cost. Isolated, three arms, one wall-clock window:

======================== ========= ========= ==========
arm                      input     output    cost
======================== ========= ========= ==========
A optimizer off          20,610    379       $0.02251
B default                19,494    **1,096** **-11.0%**
C default minus trimming 20,610    **376**   +0.1%
======================== ========= ========= ==========

It saved 1,116 input tokens and bought 717 output tokens, and output bills at 5x
input on Haiku.

**And the effect is provider-dependent, which is the whole difficulty.** The same
workloads through the same code:

===================== ==================== ======== ========================= ========
workload              gpt-4o-mini output   cost     claude-haiku-4-5 output   cost
===================== ==================== ======== ========================= ========
multi_turn_chat       147 -> 91  (-38%)    -0.4%    379 -> 1,096  (+189%)     -11.0%
multi_turn_chat_long  672 -> 560 (-17%)    +19.9%   997 -> 5,693  (+471%)     +10.9%
===================== ==================== ======== ========================= ========

Trimming makes GPT-4o-mini terser and Haiku far more verbose. So there is no
constant to encode: a figure fitted to Haiku forfeits a real 19.9% win on
OpenAI, and one fitted to OpenAI reproduces the loss. The gate is priced from the
model's own rates and then superseded by observation.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.history import TrimHistoryStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx(**overrides: object) -> StageContext:
    return StageContext(
        config=OptimizeConfig(**overrides),  # type: ignore[arg-type]
        counter=HeuristicCounter(),
    )


def _chat(turns: int, *, words: int = 8, model: str = "claude-haiku-4-5") -> LLMRequest:
    """A conversation of ``turns`` exchanges with turns of a chosen size."""
    messages = [Message(role="system", content="You are a helpful assistant.")]
    for index in range(turns):
        messages.append(Message(role="user", content=f"question {index} " + "word " * words))
        messages.append(Message(role="assistant", content=f"answer {index} " + "word " * words))
    return LLMRequest(model=model, messages=tuple(messages), temperature=0.0)


class TestASmallSavingIsNotWorthTheRisk:
    def test_a_short_conversation_is_left_alone(self) -> None:
        """The measured loss, as the assertion that used to fail.

        Twelve turns of ordinary chat saves about 99 input tokens per request.
        On Haiku that has to clear 100 risk-tokens x 5 = 500 before it is worth
        buying, so the stage declines -- which is what removes the -11.0%.
        """
        result = TrimHistoryStage().before(_chat(8), _ctx())

        assert result.note == ""
        assert result.saved_input_tokens == 0

    def test_a_large_saving_still_trims(self) -> None:
        # The long workload wins on both providers (+19.9% / +10.9%) and must
        # keep winning: a gate that blocked this would fix nothing and cost a
        # third of the package's headline saving.
        request = _chat(60, words=120)

        result = TrimHistoryStage().before(request, _ctx())

        assert result.saved_input_tokens > 0
        assert len(result.request.messages) < len(request.messages)


class TestTheThresholdComesFromTheModelsOwnRates:
    def test_a_cheaper_output_ratio_trims_sooner(self) -> None:
        """The ratio is 4 on gpt-4o-mini and 5 on Haiku, so the bar differs.

        One rule, no per-vendor branch: the same conversation can be worth
        trimming on a model whose output is relatively cheap and not on one
        where it is dear.
        """
        stage = TrimHistoryStage()
        borderline = 22

        haiku = stage.before(_chat(borderline, model="claude-haiku-4-5"), _ctx())
        gpt = TrimHistoryStage().before(_chat(borderline, model="gpt-4o-mini"), _ctx())

        # Never the other way round: gpt-4o-mini's output premium is smaller.
        assert gpt.saved_input_tokens >= haiku.saved_input_tokens

    def test_an_unpriced_model_does_not_assume_parity(self) -> None:
        """Output costs more than input on every model in PRICING.

        Treating them as equal would drop the threshold to the risk figure
        alone, which is the flattering direction -- it trims more.
        """
        stage = TrimHistoryStage()

        unknown = stage.before(_chat(8, model="some-new-model"), _ctx())

        assert unknown.saved_input_tokens == 0


class TestObservationSupersedesTheConstant:
    def test_a_model_that_does_not_inflate_starts_trimming(self) -> None:
        """The bootstrap figure is a worst case, not a belief about all models.

        Once the stage has seen trimmed and untrimmed replies of the same
        length, it knows trimming costs nothing here and the threshold collapses
        -- which is what keeps a terse model from being penalised by a number
        measured on a verbose one.
        """
        stage = TrimHistoryStage()
        ctx = _ctx()
        request = _chat(12)

        for _ in range(5):
            stage.after(request, LLMResponse(content="x", output_tokens=30), ctx)
            stage._trimmed_output.append(30)

        result = stage.before(request, ctx)

        assert result.saved_input_tokens > 0

    def test_a_model_that_inflates_keeps_declining(self) -> None:
        # The Haiku case, learned rather than assumed: replies after trimming
        # run far longer, so even a decent input saving stays a bad trade.
        stage = TrimHistoryStage()
        ctx = _ctx()
        request = _chat(20, words=40)

        for _ in range(5):
            stage._untrimmed_output.append(30)
            stage._trimmed_output.append(400)

        assert stage.before(request, ctx).saved_input_tokens == 0

    def test_the_two_groups_are_recorded_separately(self) -> None:
        """One running mean cannot tell "we trimmed" from "the questions got harder".

        The declined requests are the control the production path otherwise
        lacks, which is the difference between this and the proxy ADR-018
        rejected.
        """
        stage = TrimHistoryStage()
        ctx = _ctx()

        stage.before(_chat(8), ctx)  # declines: small saving
        stage.after(_chat(8), LLMResponse(content="x", output_tokens=25), ctx)

        assert stage._untrimmed_output == [25]
        assert stage._trimmed_output == []


class TestNoHypotheticalIsBooked:
    def test_declining_reports_no_negative_saving(self) -> None:
        """ADR-024, one commit old, and written for exactly this temptation.

        Booking the output the stage *avoided* buying would be counting a
        hypothesis -- there is no control arm in production to measure it
        against.
        """
        result = TrimHistoryStage().before(_chat(8), _ctx())

        assert result.saved_output_tokens == 0
        assert result.saved_input_tokens == 0

    def test_the_note_explains_the_decline(self) -> None:
        # A silent decline is indistinguishable from a stage that never ran, and
        # this one now declines on traffic it used to trim.
        stage = TrimHistoryStage()
        stage.before(_chat(8), _ctx())

        assert stage.last_decline_reason
        assert "output" in stage.last_decline_reason.lower()
