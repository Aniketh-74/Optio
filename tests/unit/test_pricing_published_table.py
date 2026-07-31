"""Every row of the published price list, and the one that changes date (ADR-031).

ADR-029 left seven currently-served models unpriced because nobody here had read
their rates off the vendor's page. The page has now been supplied. These tests
pin it row by row, so a future edit that drifts from the source fails here
rather than in someone's billing dashboard.

The row that needed an architectural decision::

    Claude Sonnet 5 (through Aug 31, 2026)    2 / 10
    Claude Sonnet 5 (from Sep 1, 2026)        3 / 15

A ``dict[str, ModelPrice]`` cannot hold that. Whichever single number is
written is wrong on one side of 2026-09-01 and wrong by 50% -- larger than most
savings this library reports. It is not a prediction: it is a published, dated
commitment on the vendor's own page, as auditable as the row above it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from optio.lanes.cost.pricing import ModelPrice, StaticPricingProvider

#: The published table, 2026-07-31. Base input and output, USD per million.
PUBLISHED = [
    ("claude-fable-5", 10.00, 50.00),
    ("claude-mythos-5", 10.00, 50.00),
    ("claude-opus-5", 5.00, 25.00),
    ("claude-opus-4-8", 5.00, 25.00),
    ("claude-opus-4-7", 5.00, 25.00),
    ("claude-opus-4-6", 5.00, 25.00),
    ("claude-opus-4-5", 5.00, 25.00),
    ("claude-opus-4-1", 15.00, 75.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-sonnet-4-5", 3.00, 15.00),
    ("claude-haiku-4-5", 1.00, 5.00),
    ("claude-haiku-3-5", 0.80, 4.00),
]

#: Ids `models.list` returned on 2026-07-31, in the form the API reports back.
SERVED = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
]


@pytest.fixture
def provider() -> StaticPricingProvider:
    return StaticPricingProvider()


class TestThePublishedTable:
    @pytest.mark.parametrize(("model", "base", "output"), PUBLISHED)
    def test_the_row_matches_the_vendors_page(
        self, provider: StaticPricingProvider, model: str, base: float, output: float
    ) -> None:
        price = provider.price_for(model)

        assert price is not None, model
        assert (price.input_per_million, price.output_per_million) == (base, output)

    @pytest.mark.parametrize("model", SERVED)
    def test_every_served_model_is_priced(
        self, provider: StaticPricingProvider, model: str
    ) -> None:
        """The gap ADR-029 opened deliberately, now closed with sourced data.

        Ten of these eleven returned ``None`` before ADR-029's lookup fix, and
        seven still did after it because no rate had been read off the page.
        """
        assert provider.price_for(model) is not None, model

    def test_opus_4_5_is_not_opus_4s_price(self, provider: StaticPricingProvider) -> None:
        """ADR-029's 3x overstatement, confirmed against the source.

        Opus 4.5 is $5/$25 and was being priced from Opus 4's $15/$75 row. The
        error was real, not merely suspected.
        """
        opus_45 = provider.price_for("claude-opus-4-5")
        opus_4 = provider.price_for("claude-opus-4")

        assert opus_45 is not None and opus_4 is not None
        assert opus_45.input_per_million == 5.00
        assert opus_4.input_per_million == 15.00


class TestSonnet5ChangesPriceOnAKnownDate:
    def test_todays_promotional_rate_applies_now(self) -> None:
        provider = StaticPricingProvider(today=lambda: dt.date(2026, 7, 31))

        price = provider.price_for("claude-sonnet-5")

        assert price == ModelPrice(2.00, 10.00)

    def test_the_last_day_of_the_promotion_still_gets_it(self) -> None:
        """ "through Aug 31" is inclusive; an off-by-one here is a 50% error."""
        provider = StaticPricingProvider(today=lambda: dt.date(2026, 8, 31))

        price = provider.price_for("claude-sonnet-5")

        assert price == ModelPrice(2.00, 10.00)

    def test_the_first_day_of_the_new_rate_gets_it(self) -> None:
        provider = StaticPricingProvider(today=lambda: dt.date(2026, 9, 1))

        price = provider.price_for("claude-sonnet-5")

        assert price == ModelPrice(3.00, 15.00)

    def test_it_stays_at_the_new_rate_afterwards(self) -> None:
        provider = StaticPricingProvider(today=lambda: dt.date(2027, 3, 14))

        price = provider.price_for("claude-sonnet-5")

        assert price == ModelPrice(3.00, 15.00)

    def test_a_dated_snapshot_of_it_follows_the_schedule(self) -> None:
        """The schedule attaches to the model, not to the literal string.

        The API reports a dated id back, so a schedule keyed only on the alias
        would silently serve the wrong side of the boundary for half the
        lookups -- the same failure mode ADR-029's matcher fixed.
        """
        provider = StaticPricingProvider(today=lambda: dt.date(2026, 9, 1))

        assert provider.price_for("claude-sonnet-5-20260601") == ModelPrice(3.00, 15.00)

    def test_no_other_model_is_date_dependent(self) -> None:
        """The map holds the rare published exception, not a price history."""
        early = StaticPricingProvider(today=lambda: dt.date(2020, 1, 1))
        late = StaticPricingProvider(today=lambda: dt.date(2030, 1, 1))

        for model, _, _ in PUBLISHED:
            assert early.price_for(model) == late.price_for(model), model

    def test_the_default_provider_reads_the_real_calendar(self) -> None:
        """Not captured at import: a long-running process must not go stale."""
        provider = StaticPricingProvider()

        assert provider.price_for("claude-sonnet-5") is not None


class TestTheTwoTablesAgree:
    @pytest.mark.parametrize(("model", "base", "output"), PUBLISHED)
    def test_core_and_optimizer_report_the_same_rate(
        self, provider: StaticPricingProvider, model: str, base: float, output: float
    ) -> None:
        """Two tables, sixteen models, one set of numbers.

        They hold different columns -- core prices for signals, the optimizer
        also carries cache bands -- but they may not disagree about a rate. A
        future edit to one now fails here instead of diverging quietly.
        """
        from optio_optimize.config import pricing_for

        core = provider.price_for(model)
        optimizer = pricing_for(model)

        assert core is not None and optimizer is not None, model
        assert core.input_per_million == optimizer.input_usd_per_m
        assert core.output_per_million == optimizer.output_usd_per_m
