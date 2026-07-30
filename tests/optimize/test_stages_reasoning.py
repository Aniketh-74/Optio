"""ReasoningBudgetStage, and the two stages it is not allowed to collide with.

Reasoning tokens are the most expensive tokens in a request -- billed at the
completion rate, which is 4-5x input on every model in ``PRICING``, and on a
reasoning model the thinking trace routinely runs several times the length of
the visible answer. ADR-018 makes controlling them ``Fidelity.ALTERED`` anyway,
for a reason worth restating here because these tests exist to enforce it: a
lower budget is free on easy questions and wrong on hard ones, and *hard* is
precisely why someone chose a reasoning model. The failure leaves no trace --
a truncated thinking trace still produces a confident, well-formed answer, and
the savings report measures tokens.

So the tests below are not about how much this stage saves. They are about the
three promises that make it safe enough to exist at all:

* it never raises a budget and never invents one (ADR-018's own words),
* it never lowers below a figure this workload has actually been observed to
  stay under, and
* it cannot double-count against ``chain_of_draft`` or truncate an answer
  through ``adaptive_max_tokens`` -- the two interaction hazards ADR-018
  records, each with a named test here.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages import build_stages
from optio_optimize.stages.base import Fidelity, StageContext
from optio_optimize.stages.output import (
    ANSWER_HEADROOM_TOKENS,
    MIN_OBSERVATIONS,
    MIN_THINKING_BUDGET,
    REASONING_CEILING_MULTIPLIER,
    AdaptiveMaxTokensStage,
    ChainOfDraftStage,
    ReasoningBudgetStage,
)
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx(**config_overrides: object) -> StageContext:
    return StageContext(config=OptimizeConfig(**config_overrides), counter=HeuristicCounter())  # type: ignore[arg-type]


def _request(**overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": "claude-opus-4",
        "messages": (Message(role="user", content="hi"),),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


def _response(output_tokens: int, **overrides: object) -> LLMResponse:
    defaults: dict[str, object] = {
        "content": "x",
        "input_tokens": 10,
        "output_tokens": output_tokens,
        "model": "claude-opus-4",
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)  # type: ignore[arg-type]


def _observe(
    stage: AdaptiveMaxTokensStage | ReasoningBudgetStage,
    ctx: StageContext,
    output_tokens: int,
    *,
    count: int = MIN_OBSERVATIONS,
    thinking_budget: int | None = 32_000,
    served_from: str | None = None,
) -> None:
    """Feed the stage a run of completed exchanges.

    ``thinking_budget`` is the budget the *sent* request carried, which is what
    decides whether the exchange says anything about reasoning length at all.
    """
    sent = _request(thinking_budget=thinking_budget)
    reply = _response(output_tokens, served_from=served_from)
    for _ in range(count):
        stage.after(sent, reply, ctx)


class TestItNeverSpendsMoreThanTheCallerAsked:
    """ADR-018: "never raises a budget, only lowers it, and never sets one
    where the caller set none." Both halves are cost rules before they are
    correctness rules -- a cost-reduction library that raises a budget has
    increased the bill, which is the outcome ADR-013's rule 1 forbids."""

    def test_it_declines_when_the_caller_set_no_budget(self) -> None:
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 2_000)

        result = stage.before(_request(), ctx)

        assert result.request.thinking_budget is None
        assert result.note == ""

    def test_it_never_raises_a_budget_the_caller_set_low(self) -> None:
        # Observed use is 8k, so the derived ceiling is 16k -- far above the
        # 1024 the caller chose. Clamping *up* to it would spend money nobody
        # asked to spend.
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 8_000)

        result = stage.before(_request(thinking_budget=MIN_THINKING_BUDGET), ctx)

        assert result.request.thinking_budget == MIN_THINKING_BUDGET
        assert result.note == ""

    def test_it_lowers_a_budget_far_above_anything_observed(self) -> None:
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 2_000)

        result = stage.before(_request(thinking_budget=32_000), ctx)

        assert result.request.thinking_budget == int(2_000 * REASONING_CEILING_MULTIPLIER)
        assert "32000" in result.note.replace(",", "")

    def test_it_never_lowers_below_the_providers_own_minimum(self) -> None:
        # Anthropic rejects `budget_tokens` under 1024 outright, so a ceiling
        # derived from tiny observed replies is not a smaller budget, it is a
        # 400 and a fail-open call at full price.
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 10)

        result = stage.before(_request(thinking_budget=32_000), ctx)

        assert result.request.thinking_budget == MIN_THINKING_BUDGET

    def test_it_declines_before_enough_observations_exist(self) -> None:
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 2_000, count=MIN_OBSERVATIONS - 1)

        result = stage.before(_request(thinking_budget=32_000), ctx)

        assert result.request.thinking_budget == 32_000
        assert result.note == ""

    def test_it_claims_no_saving_it_cannot_measure(self) -> None:
        """The saving lives entirely in the tail this stage prevents.

        ``output_tokens`` bundles thinking with the answer -- no provider in
        ``PRICING`` reports them apart -- so what a lowered budget avoids is
        unknowable from here. A zero and a note is the honest report;
        ``chain_of_draft`` took the same decision for the same reason, and it
        is also what keeps the two from crediting the same tokens twice.
        """
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 2_000)

        result = stage.before(_request(thinking_budget=32_000), ctx)

        assert result.saved_input_tokens == 0
        assert result.saved_output_tokens == 0
        assert result.note, "the ceiling it imposed should still be visible"

    def test_it_declares_itself_altered(self) -> None:
        assert ReasoningBudgetStage().fidelity is Fidelity.ALTERED
        assert ReasoningBudgetStage().lossy
        assert "reasoning_budget" in OptimizeConfig(reasoning_budget=True).lossy_enabled


class TestWhatCountsAsAnObservation:
    def test_a_call_that_did_no_thinking_is_not_an_observation(self) -> None:
        """Otherwise the ceiling is derived from the wrong population.

        A request sent without a budget produces a short ordinary reply. Folding
        those in drags the p95 down and the ceiling lands *below* what real
        reasoning calls need -- the stage would then bind on every one of them,
        which is the exact failure it is built to avoid.
        """
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 50, thinking_budget=None)

        result = stage.before(_request(thinking_budget=32_000), ctx)

        assert result.request.thinking_budget == 32_000
        assert result.note == ""

    def test_a_cached_reply_is_not_an_observation(self) -> None:
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 2_000, served_from="exact_cache")

        assert stage.before(_request(thinking_budget=32_000), ctx).note == ""

    def test_the_observation_history_stays_bounded(self) -> None:
        # Section 11's memory rule: this list lives for the process lifetime.
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True)
        _observe(stage, ctx, 2_000, count=1_500)

        assert len(stage._lengths) <= 1_000


class TestTheChainOfDraftOverlap:
    """ADR-018 hazard 1, and rule 5 of ``stages/__init__.py`` generalized.

    Both stages target reasoning verbosity. Rule 5 exists because
    ``minify_tools`` and ``prune_tools`` would otherwise each claim the same
    removed tokens and the report would overstate the pair. Here the resolution
    is stronger than an ordering: the two never both act.
    """

    def test_it_declines_while_chain_of_draft_is_enabled(self) -> None:
        stage = ReasoningBudgetStage()
        ctx = _ctx(reasoning_budget=True, chain_of_draft=True)
        _observe(stage, ctx, 2_000)

        result = stage.before(_request(thinking_budget=32_000), ctx)

        assert result.request.thinking_budget == 32_000
        assert result.note == ""

    def test_neither_stage_can_credit_the_others_tokens(self) -> None:
        ctx = _ctx(reasoning_budget=True, chain_of_draft=True)
        reasoning = ReasoningBudgetStage()
        _observe(reasoning, ctx, 2_000)
        request = _request(thinking_budget=32_000)

        first = reasoning.before(request, ctx)
        second = ChainOfDraftStage().before(first.request, ctx)

        # Neither claims a completion token, so no arrangement of the two can
        # count one twice -- the property rule 5 protects by ordering alone.
        assert first.saved_output_tokens == 0
        assert second.saved_output_tokens == 0


class TestTheAdaptiveMaxTokensInteraction:
    """ADR-018 hazard 2: "a low budget plus a tight completion ceiling can
    exhaust the ceiling on thinking and truncate before the answer begins.
    Ordering and a floor are required, not optional."

    On Anthropic this is not merely a truncation risk, it is a 400: the API
    requires ``max_tokens`` to exceed ``thinking.budget_tokens``. So a ceiling
    derived from observed *total* output, applied to a request carrying a
    reasoning budget, can turn a working call into a failed one -- fail-open
    then re-sends it unoptimized at full price, and the report shows a saving
    for a stage that cost money.
    """

    def test_a_ceiling_is_never_set_at_or_below_the_reasoning_budget(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        _observe(stage, ctx, 300)

        result = stage.before(_request(thinking_budget=4_000), ctx)

        ceiling = result.request.max_tokens
        assert ceiling is not None
        assert ceiling > 4_000, "a ceiling under the budget is rejected outright by Anthropic"

    def test_the_ceiling_leaves_room_for_the_answer_above_the_budget(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        _observe(stage, ctx, 300)

        result = stage.before(_request(thinking_budget=4_000), ctx)

        ceiling = result.request.max_tokens
        assert ceiling is not None
        # Exceeding the budget by one token is legal and useless: thinking would
        # consume the ceiling and generation stop before the answer began.
        assert ceiling - 4_000 >= ANSWER_HEADROOM_TOKENS

    def test_a_request_without_a_budget_is_unaffected(self) -> None:
        # The floor must not quietly raise the ceiling on the ordinary case.
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        _observe(stage, ctx, 300, thinking_budget=None)

        result = stage.before(_request(), ctx)

        assert result.request.max_tokens == int(300 * 2.0)

    def test_the_pair_in_sequence_produces_a_request_the_provider_accepts(self) -> None:
        ctx = _ctx(reasoning_budget=True)
        reasoning = ReasoningBudgetStage()
        adaptive = AdaptiveMaxTokensStage()
        _observe(reasoning, ctx, 2_000)
        _observe(adaptive, ctx, 2_000)

        sent = adaptive.before(reasoning.before(_request(thinking_budget=32_000), ctx).request, ctx)

        budget = sent.request.thinking_budget
        ceiling = sent.request.max_tokens
        assert budget is not None
        assert ceiling is not None
        assert ceiling > budget

    def test_lowering_the_budget_first_is_what_tightens_the_ceiling(self) -> None:
        """Why ``reasoning_budget`` precedes ``adaptive_max_tokens``.

        The ceiling has to clear whatever budget will actually be sent. Run
        second, it clears the reduced one; run first, it is floored to the
        caller's original 32k and bounds nothing.
        """
        ctx = _ctx(reasoning_budget=True)
        reasoning = ReasoningBudgetStage()
        _observe(reasoning, ctx, 2_000)
        request = _request(thinking_budget=32_000)

        adaptive_after = AdaptiveMaxTokensStage()
        _observe(adaptive_after, ctx, 2_000)
        lowered = reasoning.before(request, ctx).request
        after = adaptive_after.before(lowered, ctx).request.max_tokens

        adaptive_before = AdaptiveMaxTokensStage()
        _observe(adaptive_before, ctx, 2_000)
        before = adaptive_before.before(request, ctx).request.max_tokens

        assert after is not None and before is not None
        assert after < before


class TestTheRegistryPlacesItCorrectly:
    def test_it_is_absent_unless_enabled(self) -> None:
        names = [stage.name for stage in build_stages(OptimizeConfig())]
        assert "reasoning_budget" not in names

    def test_it_runs_before_adaptive_max_tokens(self) -> None:
        names = [stage.name for stage in build_stages(OptimizeConfig(reasoning_budget=True))]

        assert "reasoning_budget" in names
        assert names.index("reasoning_budget") < names.index("adaptive_max_tokens")

    def test_it_runs_after_the_cache_lookups(self) -> None:
        # Rule 1: a hit makes every later stage's work wasted, and this stage
        # changes a field the exact-cache key includes.
        names = [stage.name for stage in build_stages(OptimizeConfig(reasoning_budget=True))]

        assert names.index("exact_cache") < names.index("reasoning_budget")
