"""The optimizer's price table must resolve the ids the API actually reports.

The sibling defect to ADR-029's. Core `optio` inferred prices it did not have;
``optio_optimize.PRICING`` had the opposite failure and it is the mild one --
exact ``.get()`` only, so of the eleven ids ``models.list`` returned on
2026-07-31, **ten came back unpriced** and produced no cost figures at all.

That still matters, because the API reports the *dated* id back on every
response while callers write the alias. A table reached only by exact match
therefore misses half its lookups even for models it carries: a user on
``claude-sonnet-4-5-20250929`` gets no savings figures from the package whose
headline output is savings figures.

The lookup rule is deliberately the same as core's, and deliberately *not*
imported from it: ``optio_optimize`` has no dependency on ``optio`` and ADR-013
exists to keep it that way.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import CHEAP_COUNTERPART, PRICING, pricing_for

pytestmark = pytest.mark.optimize

#: What `models.list` returned on 2026-07-31.
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


class TestADatedIdResolvesThroughItsAlias:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
        ],
    )
    def test_the_dated_form_prices_the_same_as_the_alias(self, model: str) -> None:
        alias = model.rsplit("-", 1)[0]

        assert pricing_for(model) is pricing_for(alias)

    def test_the_alias_still_works(self) -> None:
        assert pricing_for("claude-haiku-4-5") is PRICING["claude-haiku-4-5"]


class TestAVersionBumpIsADifferentModel:
    @pytest.mark.parametrize(
        "model", ["claude-opus-4-9", "claude-opus-6", "claude-sonnet-4-7", "claude-haiku-4-6"]
    )
    def test_a_later_opus_does_not_inherit_an_earlier_rate(self, model: str) -> None:
        """The same rule as core's, for the same reason (ADR-029).

        These were four served-but-unpriced models when the test was written;
        ADR-031 supplied the page and priced them. The rule is unchanged, so
        the cases moved to generations that do not exist yet.
        """
        assert pricing_for(model) is None

    @pytest.mark.parametrize(
        "model", ["claude-opus-4-10", "claude-sonnet-4-50", "claude-haiku-4-51"]
    )
    def test_a_longer_version_number_does_not_inherit_a_shorter_ones_rate(self, model: str) -> None:
        """The shape the cases above cannot reach, and the one that actually leaks.

        ``claude-opus-4-9`` shares no prefix with any priced id, so a bare
        ``startswith`` gets it right by accident. ``claude-opus-4-10`` **does**
        start with ``claude-opus-4-1`` -- so without the four-digit
        discriminator in ``_SAME_MODEL_SUFFIX`` a tenth release would silently
        bill at the first one's rate.

        Found by mutation: removing the suffix check from ``_row_for`` left the
        whole pricing suite green. The equivalent check in ``_limit_for`` was
        pinned by four tests; this one, older and more load-bearing, by none.
        """
        assert pricing_for(model) is None

    def test_an_unknown_model_is_none_not_a_guess(self) -> None:
        assert pricing_for("some-future-model") is None

    def test_an_empty_model_is_none(self) -> None:
        assert pricing_for("") is None


class TestTheCacheRatesAreConsistent:
    @pytest.mark.parametrize("model", ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"])
    def test_every_anthropic_row_carries_all_four_rates(self, model: str) -> None:
        """A missing write rate bills 1.25x tokens at 1.0x.

        That is not hypothetical: the Haiku row shipped without one and the
        measurement it existed to price reported 53.7% where the truth was
        50.1% -- an error in the direction that flatters this library.
        """
        pricing = PRICING[model]

        assert pricing.cached_input_usd_per_m is not None
        assert pricing.cache_write_usd_per_m is not None
        assert pricing.cache_write_1h_usd_per_m is not None

    @pytest.mark.parametrize("model", ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"])
    def test_the_published_multiples_hold(self, model: str) -> None:
        # Anthropic's published premiums: reads 0.1x, 5-minute writes 1.25x,
        # one-hour writes 2.0x.
        pricing = PRICING[model]
        base = pricing.input_usd_per_m

        assert pricing.cached_input_usd_per_m == pytest.approx(base * 0.10)
        assert pricing.cache_write_usd_per_m == pytest.approx(base * 1.25)
        assert pricing.cache_write_1h_usd_per_m == pytest.approx(base * 2.00)


class TestNothingPointsAtAModelThatDoesNotExist:
    @pytest.mark.parametrize("fictional", ["claude-opus-4", "claude-sonnet-4", "claude-haiku-4"])
    def test_the_404_ids_are_gone_from_the_table(self, fictional: str) -> None:
        """All three return ``404 not_found_error``.

        They were family keys, not model ids, and one of them being the
        benchmark's default model meant no live Anthropic run had ever
        completed a single call.
        """
        assert fictional not in PRICING

    def test_every_routing_target_is_a_model_we_can_price(self) -> None:
        """It mapped Opus and Sonnet to ``claude-haiku-4``, which 404s.

        Benchmark-only -- production routing reads ``config.cheap_model``, which
        the caller sets -- but a suite that cannot route cannot measure routing.
        """
        for expensive, cheap in CHEAP_COUNTERPART.items():
            assert pricing_for(cheap) is not None, f"{expensive} -> {cheap}"

    def test_every_routing_source_is_a_model_we_can_price(self) -> None:
        for expensive in CHEAP_COUNTERPART:
            assert pricing_for(expensive) is not None

    def test_routing_always_moves_to_something_cheaper(self) -> None:
        for expensive, cheap in CHEAP_COUNTERPART.items():
            costly, thrifty = pricing_for(expensive), pricing_for(cheap)
            assert costly is not None and thrifty is not None
            assert thrifty.input_usd_per_m < costly.input_usd_per_m, f"{expensive} -> {cheap}"


class TestCoverageOfWhatTheApiServes:
    def test_every_served_model_is_priced(self) -> None:
        """Now genuinely everything, by the id the API reports back.

        This asserted four of eleven when ADR-029 wrote it, because seven had
        no sourced rate. ADR-031 closed that with the published page, so the
        assertion is the strong one: a caller on any model Anthropic serves
        gets dollar figures rather than ``None``.
        """
        unpriced = [model for model in SERVED if pricing_for(model) is None]

        assert unpriced == []
