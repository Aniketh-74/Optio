"""``adaptive_max_tokens`` could set a ceiling the provider rejects (ADR-037).

Every model has a maximum on completion tokens, it is **not** the context
window, and exceeding it is a hard 400 before any generation::

    max_tokens: 1000000 > 64000, which is the maximum allowed number of
    output tokens for claude-haiku-4-5-20251001

The caps this package prices range over a factor of four -- 32,000 on
``claude-opus-4-1`` against 128,000 on ``claude-opus-5`` -- and nothing here
modelled them.

That is reachable from :class:`AdaptiveMaxTokensStage`, which sets
``max_tokens`` on requests that carried **none**. Its ceiling is
``max(FLOOR_TOKENS, p95 x 2)``, raised further to
``thinking_budget + ANSWER_HEADROOM_TOKENS`` when a reasoning budget is set. On
a 32,000-cap model an observed p95 of 16,001 yields 32,002: over the cap,
rejected, and fail-open then re-sends the request unoptimized at full price.
The stage that exists to lower cost would be raising it, which is what ADR-013
rule 1 forbids.

The guard is deliberately narrow. It clamps only a ceiling **this package
chose**, only downward, and only to a hard provider limit -- so it is not the
"substitute our guess for the caller's instruction" the stage rightly refuses
when ``max_tokens`` was set by the caller.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig, max_output_tokens_for
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.output import (
    ANSWER_HEADROOM_TOKENS,
    MIN_OBSERVATIONS,
    AdaptiveMaxTokensStage,
)
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(model: str, **overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": model,
        "messages": (Message(role="user", content="hi"),),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


def _response(output_tokens: int, model: str) -> LLMResponse:
    return LLMResponse(
        content="x",
        input_tokens=10,
        output_tokens=output_tokens,
        model=model,
        finish_reason="stop",
    )


def _trained(model: str, observed: int) -> tuple[AdaptiveMaxTokensStage, StageContext]:
    """A stage with enough observations at ``observed`` tokens to impose a ceiling."""
    stage = AdaptiveMaxTokensStage()
    ctx = _ctx()
    for _ in range(MIN_OBSERVATIONS):
        stage.after(_request(model), _response(observed, model), ctx)
    return stage, ctx


class TestTheCeilingNeverExceedsTheProvidersCap:
    def test_a_long_workload_on_a_32k_cap_model_is_clamped(self) -> None:
        """The reachable case in today's table.

        p95 of 16,001 doubles to 32,002 against ``claude-opus-4-1``'s cap of
        32,000. Two tokens over is the same 400 as a million over.
        """
        model = "claude-opus-4-1"
        stage, ctx = _trained(model, 16_001)

        result = stage.before(_request(model), ctx)

        assert result.request.max_tokens == 32_000
        assert result.request.max_tokens <= max_output_tokens_for(model)

    def test_a_reasoning_budget_cannot_push_the_ceiling_over_the_cap(self) -> None:
        """The other route in: the headroom floor raises the ceiling.

        ``max(ceiling, thinking_budget + ANSWER_HEADROOM_TOKENS)`` is applied
        after the percentile, so a large budget lifts the ceiling past a cap
        that the percentile alone would have respected.
        """
        model = "claude-opus-4-1"
        stage, ctx = _trained(model, 400)
        budget = 32_000 - (ANSWER_HEADROOM_TOKENS // 2)

        result = stage.before(_request(model, thinking_budget=budget), ctx)

        assert result.request.max_tokens == 32_000

    @pytest.mark.parametrize(
        ("model", "cap"),
        [("claude-opus-4-1", 32_000), ("claude-haiku-4-5", 64_000), ("claude-opus-5", 128_000)],
    )
    def test_the_clamp_follows_the_model(self, model: str, cap: int) -> None:
        """A single constant would be wrong for two of these three.

        The same mistake ADR-036 named: one vendor's number applied globally.
        Here it would be one *model's* number applied across a table whose
        values differ fourfold.
        """
        stage, ctx = _trained(model, 200_000)

        result = stage.before(_request(model), ctx)

        assert result.request.max_tokens == cap


class TestTheGuardDoesNotOverreach:
    def test_a_ceiling_under_the_cap_is_left_alone(self) -> None:
        """The guard is a clamp, not a target. It must never *raise* a ceiling."""
        model = "claude-haiku-4-5"
        stage, ctx = _trained(model, 300)

        result = stage.before(_request(model), ctx)

        assert result.request.max_tokens == 600

    def test_an_unmeasured_model_is_not_clamped_to_a_guess(self) -> None:
        """Absence is not zero and not a default (ADR-027, ADR-036).

        A model missing from the table gets the ceiling the observations
        support. Inventing a cap here would truncate replies on every model
        this package has not probed.
        """
        model = "some-future-model"
        stage, ctx = _trained(model, 50_000)

        result = stage.before(_request(model), ctx)

        assert max_output_tokens_for(model) is None
        assert result.request.max_tokens == 100_000

    def test_the_callers_own_ceiling_is_still_never_touched(self) -> None:
        """Even one above the cap.

        The stage's standing rule is that an explicit ``max_tokens`` is the
        caller's instruction. A request that will 400 because the *caller* asked
        for too much is theirs to fix, and silently rewriting it would hide a
        mistake rather than surface it -- ADR-001's posture, applied to the one
        case where this package could technically intervene.
        """
        model = "claude-opus-4-1"
        stage, ctx = _trained(model, 400)

        result = stage.before(_request(model, max_tokens=99_999), ctx)

        assert result.request.max_tokens == 99_999

    def test_a_budget_at_or_over_the_cap_makes_the_stage_decline(self) -> None:
        """Clamping would swap one 400 for another.

        Anthropic rejects a ``max_tokens`` at or below ``thinking.budget_tokens``
        as well as one above the cap. When the caller's budget is itself at the
        cap, every legal ceiling is also an illegal one, so there is nothing
        this stage can set. Declining leaves the caller's own request to fail
        visibly rather than substituting one that fails for a reason they did
        not write.
        """
        model = "claude-opus-4-1"
        stage, ctx = _trained(model, 400)

        result = stage.before(_request(model, thinking_budget=32_000), ctx)

        assert result.request.max_tokens is None
        assert result.saved_output_tokens == 0

    def test_the_saving_is_never_negative_after_clamping(self) -> None:
        """``baseline = actual + saved`` breaks if a stage reports a negative here.

        The clamp lowers the ceiling, so the expected saving computed against
        the *unclamped* ceiling would overstate; computed against a larger
        ceiling than was imposed it could go negative. Neither is allowed.
        """
        model = "claude-opus-4-1"
        stage, ctx = _trained(model, 16_001)

        result = stage.before(_request(model), ctx)

        assert result.saved_output_tokens >= 0
