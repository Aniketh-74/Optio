"""Model pricing (M2-1).

Cost has to be computed **on the hot path**, so pricing is a static in-memory
table rather than a network lookup. That is a deliberate trade: a table goes
stale, but a pricing API call would add a network round trip to every LLM step,
breaking the latency budget (SC-5) and adding a failure mode to the critical
path. A stale price produces a slightly wrong number; a hanging HTTP call
produces a slow agent.

**Unknown models produce no signal.** Not a guess, not a zero, not the price of
a similar-looking model. A fabricated cost is worse than a missing one: a
downstream budget policy cannot tell an invented number from a real one, and
would gate real money on it (ADR-004, ``docs/signals.md``).

Prices are USD per **million** tokens, matching how vendors publish them, and
are converted once at lookup. Storing them in the published unit keeps the table
auditable against the vendor's page -- the numbers in the source should be the
numbers a reviewer sees quoted.

Updating the table is a routine maintenance task, not an architectural change:
edit ``_PRICES``, bump ``PRICING_TABLE_VERSION``, note the date. Users who need
prices we do not carry -- negotiated rates, self-hosted models, a vendor we have
not added -- supply their own :class:`PricingProvider` instead of waiting on us.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

_log: Final = logging.getLogger("optio")

#: Bumped whenever ``_PRICES`` changes. Emitted nowhere yet; it exists so a
#: support conversation can establish which table a given run was priced with.
PRICING_TABLE_VERSION: Final = "2026-07-27"

#: Divisor converting published per-million-token prices to per-token.
_PER_MILLION: Final = 1_000_000.0


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Published price for one model, in USD per million tokens.

    Attributes:
        input_per_million: Price per million input (prompt) tokens.
        output_per_million: Price per million output (completion) tokens.
    """

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute the cost of one call.

        Args:
            input_tokens: Prompt tokens consumed.
            output_tokens: Completion tokens produced.

        Returns:
            Cost in USD.
        """
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / _PER_MILLION


#: Published list prices, USD per million tokens, as of PRICING_TABLE_VERSION.
#:
#: Keys are matched case-insensitively, exactly first and then by longest
#: prefix, so a dated snapshot id like ``gpt-4o-2024-11-20`` resolves to the
#: ``gpt-4o`` entry without needing its own row. Prefix matching is deliberately
#: last-resort and longest-first: ``gpt-4o-mini`` must never be priced as
#: ``gpt-4o``, which costs substantially more.
_PRICES: Final[dict[str, ModelPrice]] = {
    # OpenAI
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4-turbo": ModelPrice(10.00, 30.00),
    "gpt-4": ModelPrice(30.00, 60.00),
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50),
    "o1": ModelPrice(15.00, 60.00),
    "o1-mini": ModelPrice(1.10, 4.40),
    "o3-mini": ModelPrice(1.10, 4.40),
    # Anthropic
    "claude-opus-4": ModelPrice(15.00, 75.00),
    "claude-sonnet-4": ModelPrice(3.00, 15.00),
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00),
    "claude-3-opus": ModelPrice(15.00, 75.00),
    "claude-3-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-haiku": ModelPrice(0.25, 1.25),
    # Google
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00),
    "gemini-1.5-flash": ModelPrice(0.075, 0.30),
}


@runtime_checkable
class PricingProvider(Protocol):
    """Resolves a model identifier to its price.

    Implement this to price models the built-in table does not carry:
    negotiated enterprise rates, self-hosted models, or a vendor we have not
    added yet. A provider returning ``None`` means "I do not know this model",
    which produces no cost signal rather than a guess.
    """

    def price_for(self, model: str) -> ModelPrice | None:
        """Return the price for a model.

        Args:
            model: Model identifier as reported by the framework.

        Returns:
            The price, or ``None`` when the model is unknown.
        """
        ...


class StaticPricingProvider:
    """Prices models from the built-in table.

    Attributes:
        version: The table version this provider was built from.
    """

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        """Build a provider.

        Args:
            prices: Price table. Defaults to the built-in one.
        """
        source = _PRICES if prices is None else prices
        self._prices = {name.lower(): price for name, price in source.items()}
        # Longest first so `gpt-4o-mini` wins over `gpt-4o` for a prefix match.
        self._by_length = sorted(self._prices, key=len, reverse=True)
        self.version = PRICING_TABLE_VERSION

    def price_for(self, model: str) -> ModelPrice | None:
        """Return the price for a model.

        Args:
            model: Model identifier as reported by the framework.

        Returns:
            The price, or ``None`` when the model is unknown.
        """
        if not model:
            return None

        normalised = model.strip().lower()
        exact = self._prices.get(normalised)
        if exact is not None:
            return exact

        # Vendor-prefixed ids ("openai/gpt-4o", "anthropic.claude-3-haiku-v1")
        # and dated snapshots ("gpt-4o-2024-11-20") resolve to their base model.
        for name in self._by_length:
            if name in normalised:
                return self._prices[name]

        return None

    def __repr__(self) -> str:
        """Return a debug representation with the model count."""
        return f"<StaticPricingProvider models={len(self._prices)} v={self.version}>"


#: The provider used unless a caller supplies its own.
DEFAULT_PROVIDER: Final = StaticPricingProvider()


def cost_of(
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: PricingProvider | None = None,
) -> float | None:
    """Compute the cost of one LLM call.

    Args:
        model: Model identifier as reported by the framework.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.
        provider: Pricing source. Defaults to the built-in table.

    Returns:
        Cost in USD, or ``None`` when the model is unpriced or the token counts
        are unusable. ``None`` means *unknown* and must stay distinguishable
        from a real zero all the way to the consumer.
    """
    resolved = provider if provider is not None else DEFAULT_PROVIDER
    price = resolved.price_for(model)

    if price is None:
        _log.debug("no price for model %r; cost signal omitted for this step", model)
        return None

    if input_tokens < 0 or output_tokens < 0:
        # Negative counts mean the framework reported something we do not
        # understand. Pricing it would invent a number.
        _log.debug("negative token counts for model %r; cost signal omitted", model)
        return None

    return price.cost(input_tokens, output_tokens)


def known_models() -> list[str]:
    """Return the models the built-in table prices.

    Returns:
        Sorted model identifiers.
    """
    return sorted(_PRICES)
