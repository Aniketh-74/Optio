"""The optimizer's cache bands, against the published table (ADR-031).

Every one of the sixteen published rows has the identical structure::

    5-minute write = 1.25x base    1-hour write = 2.00x base    cache hit = 0.10x base

``optio_optimize``'s table derived its cache columns from exactly those
multipliers. The derivation was correct and it was an *assumption*; it is now
sourced. That matters because ADR-021 landed a whole accounting change after a
**missing** write rate understated a live measurement by 3.6 points, and a wrong
multiplier would be the same class of error with no symptom.

The rates are written out per row rather than computed. A multiplier holding
across every model today is a fact about today's price list, not a law, and the
first model to break the pattern would be silently mispriced by a formula and
visibly wrong in a table.
"""

from __future__ import annotations

import datetime as dt

import pytest

from optio_optimize.config import CHEAP_COUNTERPART, PRICING, pricing_for

pytestmark = pytest.mark.optimize

#: model -> (base, 5m write, 1h write, cache hit, output), USD per million,
#: transcribed from the published table of 2026-07-31.
PUBLISHED = [
    ("claude-fable-5", 10.00, 12.50, 20.00, 1.00, 50.00),
    ("claude-mythos-5", 10.00, 12.50, 20.00, 1.00, 50.00),
    ("claude-opus-5", 5.00, 6.25, 10.00, 0.50, 25.00),
    ("claude-opus-4-8", 5.00, 6.25, 10.00, 0.50, 25.00),
    ("claude-opus-4-7", 5.00, 6.25, 10.00, 0.50, 25.00),
    ("claude-opus-4-6", 5.00, 6.25, 10.00, 0.50, 25.00),
    ("claude-opus-4-5", 5.00, 6.25, 10.00, 0.50, 25.00),
    ("claude-opus-4-1", 15.00, 18.75, 30.00, 1.50, 75.00),
    ("claude-sonnet-4-6", 3.00, 3.75, 6.00, 0.30, 15.00),
    ("claude-sonnet-4-5", 3.00, 3.75, 6.00, 0.30, 15.00),
    ("claude-haiku-4-5", 1.00, 1.25, 2.00, 0.10, 5.00),
    ("claude-haiku-3-5", 0.80, 1.00, 1.60, 0.08, 4.00),
]


class TestEveryBandMatchesThePage:
    @pytest.mark.parametrize(("model", "base", "write5m", "write1h", "hit", "output"), PUBLISHED)
    def test_the_row_is_transcribed_correctly(
        self,
        model: str,
        base: float,
        write5m: float,
        write1h: float,
        hit: float,
        output: float,
    ) -> None:
        pricing = pricing_for(model)

        assert pricing is not None, model
        assert pricing.input_usd_per_m == base
        assert pricing.output_usd_per_m == output
        assert pricing.cached_input_usd_per_m == hit
        assert pricing.cache_write_usd_per_m == write5m
        assert pricing.cache_write_1h_usd_per_m == write1h

    @pytest.mark.parametrize(("model", "base", "write5m", "write1h", "hit", "output"), PUBLISHED)
    def test_the_published_multiples_hold_for_this_row(
        self,
        model: str,
        base: float,
        write5m: float,
        write1h: float,
        hit: float,
        output: float,
    ) -> None:
        """Sixteen for sixteen, which is why the derivation was safe.

        Stated as a separate check from the transcription so a future row that
        genuinely breaks the pattern fails *here* -- a signal to look at the
        page -- rather than being quietly rounded into the multiplier.
        """
        assert write5m == pytest.approx(base * 1.25)
        assert write1h == pytest.approx(base * 2.00)
        assert hit == pytest.approx(base * 0.10)

    def test_no_anthropic_row_is_missing_a_band(self) -> None:
        """A missing write rate bills 1.25x tokens at 1.0x, silently."""
        for model, pricing in PRICING.items():
            if not model.startswith("claude"):
                continue
            assert pricing.cached_input_usd_per_m is not None, model
            assert pricing.cache_write_usd_per_m is not None, model
            assert pricing.cache_write_1h_usd_per_m is not None, model


class TestSonnet5ChangesPriceOnAKnownDate:
    def test_the_promotional_bands_apply_now(self) -> None:
        pricing = pricing_for("claude-sonnet-5", today=lambda: dt.date(2026, 7, 31))

        assert pricing is not None
        assert (pricing.input_usd_per_m, pricing.output_usd_per_m) == (2.00, 10.00)
        assert pricing.cache_write_usd_per_m == 2.50
        assert pricing.cache_write_1h_usd_per_m == 4.00
        assert pricing.cached_input_usd_per_m == 0.20

    def test_the_last_promotional_day_is_inclusive(self) -> None:
        pricing = pricing_for("claude-sonnet-5", today=lambda: dt.date(2026, 8, 31))

        assert pricing is not None
        assert pricing.input_usd_per_m == 2.00

    def test_the_new_bands_apply_from_september(self) -> None:
        pricing = pricing_for("claude-sonnet-5", today=lambda: dt.date(2026, 9, 1))

        assert pricing is not None
        assert (pricing.input_usd_per_m, pricing.output_usd_per_m) == (3.00, 15.00)
        assert pricing.cache_write_usd_per_m == 3.75
        assert pricing.cache_write_1h_usd_per_m == 6.00
        assert pricing.cached_input_usd_per_m == 0.30

    def test_a_dated_snapshot_follows_the_schedule(self) -> None:
        pricing = pricing_for("claude-sonnet-5-20260601", today=lambda: dt.date(2026, 9, 1))

        assert pricing is not None
        assert pricing.input_usd_per_m == 3.00


class TestRoutingTargetsAreRealAndCheaper:
    def test_every_side_of_every_pair_is_priced(self) -> None:
        for expensive, cheap in CHEAP_COUNTERPART.items():
            assert pricing_for(expensive) is not None, expensive
            assert pricing_for(cheap) is not None, f"{expensive} -> {cheap}"

    def test_routing_always_moves_to_something_cheaper(self) -> None:
        """Checkable against the table now, rather than asserted."""
        for expensive, cheap in CHEAP_COUNTERPART.items():
            costly, thrifty = pricing_for(expensive), pricing_for(cheap)
            assert costly is not None and thrifty is not None
            assert thrifty.input_usd_per_m < costly.input_usd_per_m, f"{expensive} -> {cheap}"
            assert thrifty.output_usd_per_m < costly.output_usd_per_m, f"{expensive} -> {cheap}"

    def test_the_current_anthropic_families_have_a_counterpart(self) -> None:
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5"):
            assert model in CHEAP_COUNTERPART, model
