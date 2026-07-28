"""The benchmark must not flatter the thing it measures.

A benchmark is a claim generator, so its failure mode is not crashing — it is
producing a number that is wrong in the library's favour. Every test here
targets that: the adversarial workloads must report no gain, a stage promising
identical output must deliver it, and metrics that a simulator cannot know must
come back ``None`` rather than as a plausible fabrication.

Three real defects found by running this suite during development, each now
pinned by a test below:

* ``structured_output`` was labelled lossless and diverged on every JSON
  response. It appends an instruction to the prompt, so the model answers
  differently — which is the point, and is not losslessness.
* Throughput was reported as ``+11,951%`` because "did the provider take real
  time" was inferred from the clock, and our own tokenizer ran inside the timed
  region.
* The simulator exact-matched cache prefixes, so a growing conversation never
  hit and prefix caching looked worthless.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench import WORKLOADS, SimulatedProvider, compare
from optio_optimize.bench.providers import SpendGuard
from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import Fidelity

pytestmark = pytest.mark.optimize

#: Only the stages that promise byte-identical output.
IDENTICAL_ONLY = OptimizeConfig(structured_output=False, adaptive_max_tokens=False)


class TestIdenticalStagesReallyAreIdentical:
    """The promise ``Fidelity.IDENTICAL`` makes, checked on every workload."""

    @pytest.mark.parametrize("name", sorted(WORKLOADS))
    def test_no_response_diverges(self, name: str) -> None:
        result = compare(WORKLOADS[name], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o")

        if not result.quality.is_interpretable:
            pytest.skip("sampled workload: divergence would measure model randomness")
        assert result.quality.divergent == 0, (
            f"{name}: {result.quality.divergent} response(s) changed while every "
            "active stage promised identical output"
        )


class TestTheSuiteReportsItsOwnLimits:
    """Workloads this library cannot help must show that plainly."""

    def test_unique_prompts_save_nothing(self) -> None:
        result = compare(
            WORKLOADS["unique_questions"], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o"
        )

        assert result.optimized.provider_calls == result.baseline.provider_calls
        assert result.total_token_reduction == pytest.approx(0.0, abs=0.01)

    def test_sampled_requests_are_never_cached(self) -> None:
        """Caching a sampled call would replace variety with a frozen answer."""
        result = compare(
            WORKLOADS["sampled_creative"], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o"
        )

        assert result.cache_lookups == 0, "the cache considered a sampled request"
        assert result.optimized.provider_calls == 8

    def test_a_sampled_baseline_marks_quality_uninterpretable(self) -> None:
        result = compare(
            WORKLOADS["sampled_creative"], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o"
        )

        assert not result.quality.is_interpretable


class TestMetricsRefuseToFabricate:
    """Figures a simulator cannot know must be absent, not invented."""

    def test_output_reduction_is_absent_without_a_live_model(self) -> None:
        result = compare(WORKLOADS["tool_loop"], SimulatedProvider(), model="gpt-4o")
        assert result.output_token_reduction is None

    def test_latency_and_throughput_absent_without_modelled_latency(self) -> None:
        result = compare(WORKLOADS["tool_loop"], SimulatedProvider(), model="gpt-4o")

        assert result.latency_change is None
        assert result.throughput_change is None

    def test_they_appear_once_latency_is_modelled(self) -> None:
        provider = SimulatedProvider(latency_ms=1.0)
        result = compare(
            WORKLOADS["retry_storm"], SimulatedProvider(latency_ms=1.0), model="gpt-4o"
        )

        assert provider.models_latency
        assert result.throughput_change is not None

    def test_an_unpriced_model_yields_no_cost_figure(self) -> None:
        result = compare(WORKLOADS["tool_loop"], SimulatedProvider(), model="not-a-real-model")
        assert result.cost_reduction is None


class TestSavingsAreRealAndAttributed:
    def test_the_retry_storm_avoids_all_but_one_call(self) -> None:
        result = compare(
            WORKLOADS["retry_storm"], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o"
        )

        assert result.optimized.provider_calls == 1
        assert result.true_cache_hit_rate == pytest.approx(14 / 15, abs=0.01)

    def test_cost_falls_at_least_as_fast_as_tokens(self) -> None:
        """Prefix caching lowers price without lowering count.

        Cost reduction below token reduction would mean the optimized arm
        somehow paid more per token, which no stage can cause.
        """
        for name in ("retry_storm", "tool_loop", "fan_out"):
            result = compare(WORKLOADS[name], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o")
            tokens = result.total_token_reduction
            cost = result.cost_reduction
            assert tokens is not None and cost is not None
            assert cost >= tokens - 0.001, f"{name}: cost fell slower than tokens"

    def test_prefix_caching_pays_where_no_tokens_are_saved(self) -> None:
        """The multi-turn case: identical token count, materially lower bill.

        Worth its own test because it is the result most easily lost. An earlier
        simulator exact-matched marked prefixes, so a growing conversation never
        hit, and this workload reported zero benefit from the largest lossless
        saving the library offers.
        """
        result = compare(
            WORKLOADS["multi_turn_chat"], SimulatedProvider(), IDENTICAL_ONLY, model="gpt-4o"
        )

        assert result.total_token_reduction == pytest.approx(0.0, abs=0.01)
        assert result.optimized.cached_input_tokens > 0, "no prefix cache hits at all"
        cost = result.cost_reduction
        assert cost is not None and cost > 0.20, f"expected a real discount, got {cost:.1%}"


class TestStageFidelityIsDeclaredHonestly:
    def test_reshaping_stages_do_not_claim_identical_output(self) -> None:
        from optio_optimize.stages.output import AdaptiveMaxTokensStage, StructuredOutputStage

        for stage in (StructuredOutputStage(), AdaptiveMaxTokensStage()):
            assert stage.fidelity is Fidelity.SHAPED, (
                f"{stage.name} changes the reply and must not claim identical output"
            )
            assert not stage.lossy, "reshaping preserves content; it is not content loss"

    def test_cache_stages_claim_identical_output(self) -> None:
        from optio_optimize.stages.caching import ExactCacheStage, PrefixCacheStage

        for stage in (ExactCacheStage(), PrefixCacheStage()):
            assert stage.fidelity is Fidelity.IDENTICAL


class TestTheSpendGuardStopsBeforeSpending:
    def test_it_refuses_a_call_that_would_breach_the_cap(self) -> None:
        guard = SpendGuard(cap_usd=0.01)
        guard.record(0.009)

        with pytest.raises(RuntimeError, match="spend cap reached"):
            guard.check(0.005)

    def test_it_allows_calls_within_the_cap(self) -> None:
        guard = SpendGuard(cap_usd=1.0)
        guard.check(0.5)
        guard.record(0.5)
        guard.check(0.4)

    def test_a_non_positive_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SpendGuard(cap_usd=0.0)
