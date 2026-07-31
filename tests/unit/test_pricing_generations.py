"""A price may not be inferred across model generations (ADR-029).

``StaticPricingProvider.price_for`` documented itself as "matched exactly first
and then by longest prefix", for "vendor-prefixed ids and dated snapshots". It
was implemented as ``if name in normalised`` -- substring containment -- over a
table whose Anthropic rows were ``claude-opus-4``, ``claude-sonnet-4`` and
``claude-haiku-4``, none of which is a model id the API will serve.

Measured against the eleven ids ``models.list`` returned on 2026-07-31, five
distinct Opus generations collapsed onto one row. Anthropic cut Opus list
pricing at 4.5, so a million input and two hundred thousand output tokens on
``claude-opus-4-5`` reported **$30.00 against a $10.00 bill** -- 3x over,
silently, with nothing to distinguish an inferred number from a looked-up one.

Containment also priced anything merely *containing* a known name:
``not-really-gpt-4o-at-all-v2`` came back at gpt-4o's rate.

The sibling of ``tests/unit/test_pricing.py``, whose founding rule this extends:
"a fabricated cost is worse than a missing one -- a budget policy cannot tell an
invented number from a real one, and would gate real money on it." An inferred
one is fabricated; it just looks up a real row to do it.
"""

from __future__ import annotations

import logging

import pytest

from optio.lanes.cost.pricing import StaticPricingProvider, cost_of


@pytest.fixture
def provider() -> StaticPricingProvider:
    return StaticPricingProvider()


class TestADatedSnapshotIsTheSameModel:
    """The documented intent, which must survive the fix."""

    def test_an_openai_dated_snapshot_resolves(self, provider: StaticPricingProvider) -> None:
        price = provider.price_for("gpt-4o-2024-11-20")

        assert price is not None
        assert price.input_per_million == 2.50

    def test_a_compact_date_resolves(self, provider: StaticPricingProvider) -> None:
        price = provider.price_for("claude-sonnet-4-5-20250929")

        assert price is not None
        assert price.input_per_million == 3.00

    def test_a_vendor_prefix_is_stripped(self, provider: StaticPricingProvider) -> None:
        assert provider.price_for("openai/gpt-4o") == provider.price_for("gpt-4o")

    def test_a_dotted_vendor_prefix_is_stripped(self, provider: StaticPricingProvider) -> None:
        assert provider.price_for("anthropic.claude-3-haiku") == provider.price_for(
            "claude-3-haiku"
        )


class TestAVersionBumpIsADifferentModel:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-9",
            "claude-opus-6",
            "claude-sonnet-4-7",
            "claude-sonnet-6",
            "claude-haiku-5",
            "claude-haiku-4-6",
        ],
    )
    def test_an_unpriced_generation_returns_none(
        self, provider: StaticPricingProvider, model: str
    ) -> None:
        """A generation the table does not carry gets ``None``, not a neighbour.

        When this test was written these were the seven models Anthropic served
        and nobody here had a published rate for; ADR-031 supplied the page and
        priced all seven. The *rule* is what mattered and it is unchanged, so
        the cases moved to generations that do not exist yet -- which is where
        the next instance of this bug would appear.
        """
        assert provider.price_for(model) is None

    def test_opus_4_5_is_not_priced_as_opus_4(self, provider: StaticPricingProvider) -> None:
        """The 3x case, as an assertion.

        Opus 4.5 has its own row now. What must never come back is $15/$75.
        """
        price = provider.price_for("claude-opus-4-5-20251101")

        assert price is not None
        assert price.input_per_million != 15.00

    def test_the_three_times_overstatement_is_gone(self) -> None:
        billed = cost_of("claude-opus-4-5-20251101", 1_000_000, 200_000)

        assert billed == pytest.approx(10.00)


class TestContainmentIsNotAMatch:
    def test_a_name_merely_containing_a_known_model_is_unpriced(
        self, provider: StaticPricingProvider
    ) -> None:
        assert provider.price_for("not-really-gpt-4o-at-all-v2") is None

    def test_a_derived_model_is_unpriced(self, provider: StaticPricingProvider) -> None:
        assert provider.price_for("some-vendor/claude-opus-4-distilled-mini") is None

    def test_gpt_4o_mini_is_never_priced_as_gpt_4o(self, provider: StaticPricingProvider) -> None:
        """The case the original docstring called out, still held."""
        mini = provider.price_for("gpt-4o-mini")

        assert mini is not None
        assert mini.input_per_million == 0.15


class TestTheModelsAnthropicActuallyServes:
    @pytest.mark.parametrize(
        ("model", "input_rate", "output_rate"),
        [
            ("claude-opus-4-5", 5.00, 25.00),
            ("claude-opus-4-1", 15.00, 75.00),
            ("claude-sonnet-4-5", 3.00, 15.00),
            ("claude-haiku-4-5", 1.00, 5.00),
        ],
    )
    def test_a_sourced_row_is_asserted_not_inferred(
        self, provider: StaticPricingProvider, model: str, input_rate: float, output_rate: float
    ) -> None:
        price = provider.price_for(model)

        assert price is not None
        assert (price.input_per_million, price.output_per_million) == (input_rate, output_rate)

    def test_haiku_4_5_agrees_with_the_optimizer_table(
        self, provider: StaticPricingProvider
    ) -> None:
        """Two tables, one model, one price.

        Core prices for signals and ``optio_optimize`` prices with cache rates
        for savings. They may hold different columns; they may not disagree
        about a rate.
        """
        from optio_optimize.config import PRICING

        core = provider.price_for("claude-haiku-4-5")
        optimizer = PRICING["claude-haiku-4-5"]

        assert core is not None
        assert core.input_per_million == optimizer.input_usd_per_m
        assert core.output_per_million == optimizer.output_usd_per_m


class TestAnUnpricedModelSaysSo:
    def test_an_unknown_model_is_logged(
        self, provider: StaticPricingProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silence would read as a broken cost lane rather than a missing row."""
        with caplog.at_level(logging.WARNING):
            provider.price_for("claude-vega-7")

        assert "claude-vega-7" in caplog.text

    def test_the_message_points_at_the_way_out(
        self, provider: StaticPricingProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            provider.price_for("claude-vega-7")

        assert "PricingProvider" in caplog.text

    def test_it_warns_once_per_model_not_once_per_call(
        self, provider: StaticPricingProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ADR-004: a pricing gap may never become a log flood."""
        with caplog.at_level(logging.WARNING):
            for _ in range(50):
                provider.price_for("claude-vega-7")

        assert caplog.text.count("claude-vega-7") == 1

    def test_each_unknown_model_gets_its_own_warning(
        self, provider: StaticPricingProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            provider.price_for("claude-vega-7")
            provider.price_for("claude-vega-8")

        assert "claude-vega-7" in caplog.text
        assert "claude-vega-8" in caplog.text

    def test_a_known_model_logs_nothing(
        self, provider: StaticPricingProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            provider.price_for("gpt-4o")

        assert caplog.text == ""
