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
from optio_optimize.stages.base import Fidelity, StageContext
from optio_optimize.stages.retrieval import PruneRetrievalStage
from optio_optimize.tokens import HeuristicCounter

pytestmark = pytest.mark.optimize

#: Only the stages that promise byte-identical output. trim_history, dedup and
#: pruning are SHAPED (Phase 2, ADR-013) -- on by default, but they change what
#: the prompt says, not just its price, so they are excluded here for the same
#: reason structured_output and adaptive_max_tokens are.
IDENTICAL_ONLY = OptimizeConfig(
    structured_output=False,
    adaptive_max_tokens=False,
    trim_history=False,
    deduplicate=False,
    prune_retrieval=False,
)


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

    def test_avoiding_calls_always_lowers_cost(self) -> None:
        """Fewer calls must cost less, on any provider.

        Deliberately *not* "cost falls at least as fast as tokens", which was
        the first version of this test and is false under automatic prefix
        caching: there the baseline's prompt tokens are already discounted, so
        a call the cache avoids was a cheap call, and cost falls more slowly
        than count. Both behaviours are correct; only the weaker claim holds
        across providers.
        """
        for style in ("automatic", "explicit"):
            for name in ("retry_storm", "tool_loop", "fan_out"):
                result = compare(
                    WORKLOADS[name],
                    SimulatedProvider(prefix_cache_style=style),
                    IDENTICAL_ONLY,
                    model="gpt-4o",
                )
                tokens = result.total_token_reduction
                cost = result.cost_reduction
                assert tokens is not None and cost is not None
                assert tokens > 0.5, f"{name}/{style}: expected substantial token savings"
                assert cost > 0.5, f"{name}/{style}: tokens fell {tokens:.1%} but cost {cost:.1%}"

    def test_explicit_caching_beats_tokens_alone(self) -> None:
        """Where the marker is required, cost falls faster than count.

        The mechanism worth having a test for: no tokens are avoided at all,
        and the bill still drops by roughly a third, because the same tokens are
        billed at the cached rate.
        """
        result = compare(
            WORKLOADS["multi_turn_chat"],
            SimulatedProvider(prefix_cache_style="explicit"),
            IDENTICAL_ONLY,
            model="gpt-4o",
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


class TestPrefixCachingIsCreditedToTheRightParty:
    """The library must not claim a discount the provider grants for free.

    The single worst measurement error this project made: the simulator modelled
    only Anthropic-style explicit caching, so ``multi_turn_chat`` reported a
    36.3% saving from our prefix marker. A live OpenAI run measured **-1.8%** --
    OpenAI caches any prefix over ~1024 tokens automatically, so the baseline arm
    was already getting the discount and the marker added nothing.

    Attributing a provider feature to your own library is how a benchmark
    collapses on first contact with reality. These tests pin the distinction.
    """

    def test_automatic_caching_gives_our_marker_no_credit(self) -> None:
        result = compare(
            WORKLOADS["multi_turn_chat"],
            SimulatedProvider(prefix_cache_style="automatic"),
            IDENTICAL_ONLY,
            model="gpt-4o",
        )

        cost = result.cost_reduction
        assert cost is not None
        assert cost < 0.05, (
            f"claimed {cost:.1%} on a provider that caches automatically; the "
            "baseline arm gets the same discount, so our contribution is ~0"
        )

    def test_explicit_caching_is_where_the_marker_pays(self) -> None:
        result = compare(
            WORKLOADS["multi_turn_chat"],
            SimulatedProvider(prefix_cache_style="explicit"),
            IDENTICAL_ONLY,
            model="gpt-4o",
        )

        cost = result.cost_reduction
        assert cost is not None
        assert cost > 0.20, (
            f"only {cost:.1%} on a provider that caches nothing without a marker; "
            "this is the case the stage exists for"
        )

    def test_arms_start_from_a_cold_cache(self) -> None:
        """Cache state must not leak from the baseline into the optimized arm.

        Both arms share one provider. Without a reset the baseline warms the
        prefix cache and the optimized arm inherits the hits, which shows up as
        a saving this library did not cause.
        """
        provider = SimulatedProvider(prefix_cache_style="automatic")
        result = compare(WORKLOADS["rag_queries"], provider, IDENTICAL_ONLY, model="gpt-4o")

        assert result.baseline.cached_input_tokens == result.optimized.cached_input_tokens, (
            "arms saw different provider-cache state, so the comparison is biased"
        )


class TestPruneRetrievalActuallyPrunes:
    """``rag_queries`` cannot prove prune_retrieval works, only that it's safe.

    Every chunk in ``rag_queries`` is about the same topic, so nothing ever
    scores below the relevance floor and the stage always declines --
    measured at 0 tokens saved, both simulated and live
    (docs/optimize-benchmarks.md). A correct zero on a workload with nothing
    to prune is not evidence the stage does anything. ``rag_queries_noisy``
    mixes in one genuinely irrelevant chunk specifically to test that.
    """

    def test_the_irrelevant_chunk_is_dropped_from_every_request(self) -> None:
        stage = PruneRetrievalStage()
        ctx = StageContext(config=OptimizeConfig(), counter=HeuristicCounter())

        for request in WORKLOADS["rag_queries_noisy"].requests():
            result = stage.before(request, ctx)
            sent = result.request.messages[-1].content
            assert "office parking" not in sent, "the irrelevant chunk survived pruning"

    def test_every_relevant_chunk_survives_in_every_request(self) -> None:
        stage = PruneRetrievalStage()
        ctx = StageContext(config=OptimizeConfig(), counter=HeuristicCounter())

        for request in WORKLOADS["rag_queries_noisy"].requests():
            result = stage.before(request, ctx)
            sent = result.request.messages[-1].content
            assert "Quarterly revenue" in sent, "a relevant chunk was dropped, not just the noise"
            # All six relevant chunks share the same body text, so counting
            # occurrences (rather than checking each [doc N] tag) is what
            # actually proves none of them were pruned.
            assert sent.count("Quarterly revenue") == 6

    def test_the_stage_reports_real_savings_on_this_workload(self) -> None:
        result = compare(
            WORKLOADS["rag_queries_noisy"],
            SimulatedProvider(),
            OptimizeConfig(
                exact_cache=False,
                prefix_cache=False,
                structured_output=False,
                adaptive_max_tokens=False,
                trim_history=False,
                deduplicate=False,
            ),
            model="gpt-4o",
        )

        assert result.stage_savings.get("prune_retrieval", 0) > 0
