"""Cascade pays by cost-weighted escalation, not by count (ADR-034).

The first live cascade run, Haiku 4.5 -> Sonnet 4.5, eight requests::

    escalation rate: 50.0%          <- the reported statistic
    all-expensive    $0.007074
    total spend      $0.008861
    net saving      -$0.001787      -25.3%

Break-even comes straight off the rate card. An accepted-cheap request pays C
instead of E and saves ``E - C``; an escalated one pays ``C + E`` and loses C.
Setting those equal gives ``r = 1 - C/E`` -- **66.7%** for this pair. At 50%
escalation the run should have made money.

It lost money because ``escalation_rate`` counts requests and the bill weights
them. The four escalated requests were **92% of the baseline spend**; the four
that passed were 8%. That correlation is structural: a request is likelier to
fail a verifier when it is long, carries tools, or demands a schema, and every
one of those also makes it expensive. So a count-weighted rate flatters cascade
on essentially any real workload -- in the flattering direction, which this
project treats as the serious one.
"""

from __future__ import annotations

import pytest

from optio_optimize.cascade import CascadeStats, break_even_escalation_rate

pytestmark = pytest.mark.optimize


class TestBreakEvenComesFromTheRateCard:
    @pytest.mark.parametrize(
        ("expensive", "cheap", "rate"),
        [
            ("claude-sonnet-4-5", "claude-haiku-4-5", 1 - 1 / 3),
            ("claude-opus-5", "claude-haiku-4-5", 1 - 1 / 5),
            ("claude-opus-5", "claude-sonnet-5", 1 - 2 / 5),
        ],
    )
    def test_it_is_one_minus_the_price_ratio(self, expensive: str, cheap: str, rate: float) -> None:
        assert break_even_escalation_rate(expensive, cheap) == pytest.approx(rate)

    def test_the_pair_from_the_live_run(self) -> None:
        """66.7%, against which 50% looked safe."""
        assert break_even_escalation_rate("claude-sonnet-4-5", "claude-haiku-4-5") == pytest.approx(
            0.667, abs=0.001
        )

    def test_a_wider_price_gap_tolerates_more_escalation(self) -> None:
        narrow = break_even_escalation_rate("claude-opus-5", "claude-sonnet-5")
        wide = break_even_escalation_rate("claude-opus-5", "claude-haiku-4-5")

        assert narrow is not None and wide is not None
        assert wide > narrow

    def test_an_unpriced_model_gives_none_not_a_guess(self) -> None:
        assert break_even_escalation_rate("claude-sonnet-4-5", "some-future-model") is None
        assert break_even_escalation_rate("some-future-model", "claude-haiku-4-5") is None

    def test_a_cheap_model_that_is_not_cheaper_gives_none(self) -> None:
        """Routing to something dearer is a configuration error, not a rate."""
        assert break_even_escalation_rate("claude-haiku-4-5", "claude-opus-5") is None


def _stats(
    *, cheap_in: int, cheap_out: int, acc_in: int, acc_out: int, esc_in: int, esc_out: int
) -> CascadeStats:
    return CascadeStats(
        attempted=8,
        escalated=4,
        cheap_input_tokens=cheap_in,
        cheap_output_tokens=cheap_out,
        accepted_cheap_input_tokens=acc_in,
        accepted_cheap_output_tokens=acc_out,
        escalated_input_tokens=esc_in,
        escalated_output_tokens=esc_out,
    )


class TestTheCostWeightedRateIsTheOneToRead:
    def test_it_is_reported(self) -> None:
        stats = _stats(
            cheap_in=4_000, cheap_out=400, acc_in=200, acc_out=20, esc_in=3_800, esc_out=380
        )

        cost = stats.cost_summary("claude-sonnet-4-5", "claude-haiku-4-5")

        assert cost is not None
        assert cost.cost_weighted_escalation_rate is not None

    def test_it_exceeds_the_count_rate_when_big_requests_escalate(self) -> None:
        """The live run's shape: the expensive requests are the ones that fail."""
        stats = _stats(
            cheap_in=4_000, cheap_out=400, acc_in=200, acc_out=20, esc_in=3_800, esc_out=380
        )

        cost = stats.cost_summary("claude-sonnet-4-5", "claude-haiku-4-5")

        assert cost is not None
        assert stats.escalation_rate == pytest.approx(0.5)
        assert cost.cost_weighted_escalation_rate is not None
        assert cost.cost_weighted_escalation_rate > 0.9

    def test_it_falls_below_the_count_rate_when_small_requests_escalate(self) -> None:
        """The favourable shape, and the reason this is not a constant offset."""
        stats = _stats(
            cheap_in=4_000, cheap_out=400, acc_in=3_800, acc_out=380, esc_in=200, esc_out=20
        )

        cost = stats.cost_summary("claude-sonnet-4-5", "claude-haiku-4-5")

        assert cost is not None
        assert cost.cost_weighted_escalation_rate is not None
        assert cost.cost_weighted_escalation_rate < 0.5

    def test_past_break_even_the_net_saving_is_negative(self) -> None:
        """The two numbers must agree, which is the whole claim.

        If the cost-weighted rate exceeds break-even, the measured net saving
        has to be negative -- otherwise the formula is wrong, not the report.
        """
        stats = _stats(
            cheap_in=4_000, cheap_out=400, acc_in=200, acc_out=20, esc_in=3_800, esc_out=380
        )

        cost = stats.cost_summary("claude-sonnet-4-5", "claude-haiku-4-5")
        break_even = break_even_escalation_rate("claude-sonnet-4-5", "claude-haiku-4-5")

        assert cost is not None and break_even is not None
        assert cost.cost_weighted_escalation_rate is not None
        assert cost.cost_weighted_escalation_rate > break_even
        assert cost.net_saving_usd < 0

    def test_below_break_even_the_net_saving_is_positive(self) -> None:
        stats = _stats(
            cheap_in=4_000, cheap_out=400, acc_in=3_800, acc_out=380, esc_in=200, esc_out=20
        )

        cost = stats.cost_summary("claude-sonnet-4-5", "claude-haiku-4-5")
        break_even = break_even_escalation_rate("claude-sonnet-4-5", "claude-haiku-4-5")

        assert cost is not None and break_even is not None
        assert cost.cost_weighted_escalation_rate is not None
        assert cost.cost_weighted_escalation_rate < break_even
        assert cost.net_saving_usd > 0

    def test_no_baseline_means_no_rate(self) -> None:
        """ADR-028's posture: a ratio over nothing is a division, not a measurement."""
        stats = CascadeStats(attempted=0)

        cost = stats.cost_summary("claude-sonnet-4-5", "claude-haiku-4-5")

        assert cost is not None
        assert cost.cost_weighted_escalation_rate is None
