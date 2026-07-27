"""Model pricing (M2-1).

The acceptance criterion: known models priced, unknown model produces **no
signal** and never raises. A fabricated cost is worse than a missing one --
a budget policy cannot tell an invented number from a real one, and would gate
real money on it.
"""

from __future__ import annotations

import pytest

from optio.lanes.cost.pricing import (
    DEFAULT_PROVIDER,
    PRICING_TABLE_VERSION,
    ModelPrice,
    PricingProvider,
    StaticPricingProvider,
    cost_of,
    known_models,
)


class TestPriceArithmetic:
    def test_cost_is_per_million_tokens(self) -> None:
        # Prices are stored in the published unit so the table stays auditable
        # against the vendor's page.
        price = ModelPrice(input_per_million=2.50, output_per_million=10.00)
        assert price.cost(1_000_000, 0) == pytest.approx(2.50)
        assert price.cost(0, 1_000_000) == pytest.approx(10.00)

    def test_input_and_output_are_priced_separately(self) -> None:
        price = ModelPrice(input_per_million=2.50, output_per_million=10.00)
        assert price.cost(1_000_000, 1_000_000) == pytest.approx(12.50)

    def test_zero_tokens_cost_zero(self) -> None:
        # A real zero, distinct from "unknown" -- it must survive to the span.
        assert ModelPrice(2.50, 10.00).cost(0, 0) == 0.0


class TestModelResolution:
    def test_known_model_is_priced(self) -> None:
        assert cost_of("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)

    def test_lookup_is_case_insensitive(self) -> None:
        assert cost_of("GPT-4O", 1_000_000, 0) == cost_of("gpt-4o", 1_000_000, 0)

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert cost_of("  gpt-4o  ", 1_000_000, 0) == pytest.approx(2.50)

    def test_dated_snapshots_resolve_to_the_base_model(self) -> None:
        # Frameworks report ids like "gpt-4o-2024-11-20"; requiring a row per
        # snapshot would make every vendor release a silent pricing gap.
        assert cost_of("gpt-4o-2024-11-20", 1_000_000, 0) == pytest.approx(2.50)

    def test_vendor_prefixed_ids_resolve(self) -> None:
        assert cost_of("openai/gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
        assert cost_of("anthropic.claude-3-haiku-v1", 1_000_000, 0) == pytest.approx(0.25)

    def test_longest_match_wins(self) -> None:
        # The dangerous case: gpt-4o-mini priced as gpt-4o would over-report by
        # ~17x, and the number would look entirely plausible.
        mini = cost_of("gpt-4o-mini", 1_000_000, 0)
        full = cost_of("gpt-4o", 1_000_000, 0)

        assert mini == pytest.approx(0.15)
        assert full == pytest.approx(2.50)
        assert mini != full

    def test_haiku_is_not_priced_as_sonnet(self) -> None:
        assert cost_of("claude-3-5-haiku", 1_000_000, 0) == pytest.approx(0.80)
        assert cost_of("claude-3-5-sonnet", 1_000_000, 0) == pytest.approx(3.00)


class TestUnknownModelsProduceNoSignal:
    def test_unknown_model_returns_none(self) -> None:
        assert cost_of("some-model-we-have-never-seen", 1000, 1000) is None

    def test_unknown_model_does_not_raise(self) -> None:
        # Pricing runs on the hot path; raising here would reach the agent.
        cost_of("", 1000, 1000)
        cost_of("???", 1000, 1000)

    def test_empty_model_id_returns_none(self) -> None:
        assert cost_of("", 1000, 1000) is None

    @pytest.mark.parametrize(
        ("input_tokens", "output_tokens"),
        [(-1, 0), (0, -1), (-5, -5)],
    )
    def test_negative_token_counts_return_none(self, input_tokens: int, output_tokens: int) -> None:
        # A negative count means the framework reported something we do not
        # understand. Pricing it would invent a number.
        assert cost_of("gpt-4o", input_tokens, output_tokens) is None

    def test_none_is_distinguishable_from_zero_cost(self) -> None:
        # The whole reason unknown returns None: a real zero must not look like
        # a failure, and a failure must not look like a free run.
        assert cost_of("gpt-4o", 0, 0) == 0.0
        assert cost_of("unknown-model", 0, 0) is None


class TestPluggableProvider:
    def test_custom_provider_is_used(self) -> None:
        # Negotiated rates, self-hosted models, or a vendor we have not added.
        class FlatRateProvider:
            def price_for(self, model: str) -> ModelPrice | None:
                return ModelPrice(1.00, 1.00)

        assert cost_of("anything-at-all", 1_000_000, 0, FlatRateProvider()) == (pytest.approx(1.00))

    def test_a_provider_returning_none_yields_no_signal(self) -> None:
        class NeverKnowsProvider:
            def price_for(self, model: str) -> ModelPrice | None:
                return None

        assert cost_of("gpt-4o", 1000, 1000, NeverKnowsProvider()) is None

    def test_static_provider_satisfies_the_protocol(self) -> None:
        assert isinstance(DEFAULT_PROVIDER, PricingProvider)

    def test_provider_can_be_built_from_a_custom_table(self) -> None:
        provider = StaticPricingProvider({"my-model": ModelPrice(5.00, 5.00)})

        assert provider.price_for("my-model") == ModelPrice(5.00, 5.00)
        assert provider.price_for("gpt-4o") is None


class TestTable:
    def test_table_is_not_empty(self) -> None:
        assert len(known_models()) > 10

    def test_table_covers_the_major_vendors(self) -> None:
        models = known_models()
        assert any(m.startswith("gpt-") for m in models)
        assert any(m.startswith("claude-") for m in models)
        assert any(m.startswith("gemini-") for m in models)

    def test_every_price_is_positive(self) -> None:
        for model in known_models():
            price = DEFAULT_PROVIDER.price_for(model)
            assert price is not None
            assert price.input_per_million > 0
            assert price.output_per_million > 0

    def test_output_is_never_cheaper_than_input(self) -> None:
        # Universal across vendors today. If this ever fails it is far more
        # likely a transposed pair in the table than a genuine vendor change.
        for model in known_models():
            price = DEFAULT_PROVIDER.price_for(model)
            assert price is not None
            assert price.output_per_million >= price.input_per_million, model

    def test_version_is_recorded(self) -> None:
        # So a support conversation can establish which table priced a run.
        assert PRICING_TABLE_VERSION
        assert DEFAULT_PROVIDER.version == PRICING_TABLE_VERSION

    def test_repr_reports_model_count(self) -> None:
        assert "models=" in repr(DEFAULT_PROVIDER)
