"""The benchmark must price the cache writes it causes (ADR-027 follow-on).

Found while chasing why ``prefix_cache`` reported zero provider cache reads on
every live Anthropic workload. Two separate defects sat behind that question,
and the second is the serious one:

1. ``ArmResult`` tracked cache **reads** and not **writes**, so a report showing
   ``reads 0`` could not distinguish "the breakpoint was ignored" (reads 0,
   writes 0) from "the prefix changes between calls" (reads 0, writes N).
   ``docs/optimize-benchmarks.md`` names those as different bugs whose totals
   cannot tell them apart.

2. ``ABResult.cost_usd`` priced every prompt token at the base or cached rate
   and **never at the write premium** -- so the arm that places breakpoints was
   billed 1.25x tokens at 1.0x. That is the identical asymmetry ADR-021 landed a
   whole accounting change to remove from ``SavingsReport``, reproduced inside
   the benchmark that measures it, and it flatters the optimized arm every time
   ``prefix_cache`` writes.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.metrics import ABResult, ArmResult, QualityResult

pytestmark = pytest.mark.optimize


def _arms(**optimized: int) -> ABResult:
    baseline = ArmResult(name="baseline", requests=1, provider_calls=1, input_tokens=10_000)
    arm = ArmResult(name="optimized", requests=1, provider_calls=1, input_tokens=10_000)
    for field, value in optimized.items():
        setattr(arm, field, value)
    return ABResult(
        workload="w",
        model="claude-haiku-4-5",
        baseline=baseline,
        optimized=arm,
        quality=QualityResult(),
    )


class TestWritesAreTracked:
    def test_an_arm_records_cache_writes(self) -> None:
        assert ArmResult(name="x").cache_write_tokens == 0

    def test_reads_and_writes_are_separate_fields(self) -> None:
        """Reads-0 means two different things and only writes tell them apart.

        Reads 0 / writes 0 is a breakpoint the provider ignored -- wrong floor,
        wrong placement. Reads 0 / writes N is a breakpoint that worked and a
        prefix that changed before it could be read.
        """
        arm = ArmResult(name="x", cached_input_tokens=0, cache_write_tokens=4_000)

        assert arm.cached_input_tokens == 0
        assert arm.cache_write_tokens == 4_000


class TestWritesArePricedAtTheirPremium:
    def test_a_write_costs_more_than_a_plain_input_token(self) -> None:
        """Haiku 4.5: 1.00 base, 1.25 write. The arm that writes must pay it."""
        plain = _arms()
        writing = _arms(cache_write_tokens=8_000)

        plain_cost = plain.cost_usd(plain.optimized)
        writing_cost = writing.cost_usd(writing.optimized)

        assert plain_cost is not None and writing_cost is not None
        assert writing_cost > plain_cost

    def test_the_premium_matches_the_published_rate(self) -> None:
        # 2,000 base at $1.00 + 8,000 written at $1.25.
        result = _arms(cache_write_tokens=8_000)

        cost = result.cost_usd(result.optimized)

        assert cost == pytest.approx((2_000 * 1.00 + 8_000 * 1.25) / 1_000_000)

    def test_a_read_is_still_discounted(self) -> None:
        result = _arms(cached_input_tokens=8_000)

        cost = result.cost_usd(result.optimized)

        assert cost == pytest.approx((2_000 * 1.00 + 8_000 * 0.10) / 1_000_000)

    def test_reads_and_writes_together_add_up(self) -> None:
        # The realistic shape: write once, read thereafter.
        result = _arms(cached_input_tokens=6_000, cache_write_tokens=3_000)

        cost = result.cost_usd(result.optimized)

        expected = (1_000 * 1.00 + 6_000 * 0.10 + 3_000 * 1.25) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_an_unpriced_model_still_returns_none(self) -> None:
        baseline = ArmResult(name="baseline", input_tokens=100)
        result = ABResult(
            workload="w",
            model="mystery-model",
            baseline=baseline,
            optimized=baseline,
            quality=QualityResult(),
        )

        assert result.cost_usd(result.optimized) is None

    def test_the_flattering_direction_is_the_one_that_was_wrong(self) -> None:
        """Before this, a writing arm was billed 1.25x tokens at 1.0x.

        Stated as a regression guard rather than a rate check: the defect always
        made the optimized arm look cheaper than it was, which is the direction
        this project treats as serious.
        """
        result = _arms(cache_write_tokens=8_000)

        cost = result.cost_usd(result.optimized)
        naive = (10_000 * 1.00) / 1_000_000

        assert cost is not None and cost > naive
