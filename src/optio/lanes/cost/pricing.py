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

import datetime as dt
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

_log: Final = logging.getLogger("optio")

#: Bumped whenever ``_PRICES`` changes. Emitted nowhere yet; it exists so a
#: support conversation can establish which table a given run was priced with.
PRICING_TABLE_VERSION: Final = "2026-07-31"

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
#:
#: A prefix match requires the *remainder* to name the same model -- a date or a
#: revision tag, never a version bump. ``claude-opus-4-5`` does not resolve to
#: ``claude-opus-4``; it has its own row or it has no price. See
#: :func:`_is_same_model`, and ADR-029 for the 3x overstatement that established
#: the rule.
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
    # Anthropic. Transcribed from the published price list of 2026-07-31
    # (ADR-031), which also confirmed the four rows ADR-029 had added and the
    # 3x Opus overstatement it removed: Opus 4.5 really is $5/$25 and really was
    # being answered by Opus 4's $15/$75 row.
    "claude-fable-5": ModelPrice(10.00, 50.00),
    # Limited availability, and absent from `models.list`. Carried anyway: a
    # price for a model the caller cannot reach costs nothing, while a missing
    # price for one they can costs the cost signal entirely.
    "claude-mythos-5": ModelPrice(10.00, 50.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-opus-4-5": ModelPrice(5.00, 25.00),
    "claude-opus-4-1": ModelPrice(15.00, 75.00),
    # Retired, and still reachable through Google Cloud, which is why it stays.
    "claude-opus-4": ModelPrice(15.00, 75.00),
    # The price in force *before* the first entry in `_SCHEDULED` below.
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-4": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    # Both spellings of the 3.5 generation. The vendor's page writes
    # `Claude Haiku 3.5` while the older ids read `claude-3-5-haiku`, and a
    # table carrying one of them returns None for the other.
    "claude-haiku-3-5": ModelPrice(0.80, 4.00),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00),
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-opus": ModelPrice(15.00, 75.00),
    "claude-3-sonnet": ModelPrice(3.00, 15.00),
    "claude-3-haiku": ModelPrice(0.25, 1.25),
    # Google
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00),
    "gemini-1.5-flash": ModelPrice(0.075, 0.30),
}


#: Published price changes with a known effective date, newest last.
#:
#: The vendor's page lists Sonnet 5 twice -- ``2 / 10`` "through Aug 31, 2026"
#: and ``3 / 15`` "from Sep 1, 2026" -- and a ``dict[str, ModelPrice]`` cannot
#: hold that. Whichever single number were written would be wrong on one side of
#: the boundary and wrong by 50%, which is larger than most savings this project
#: reports.
#:
#: Recording it is not the prediction this package has a standing rule against.
#: A prediction is a guess about what a vendor will do; this is a published,
#: dated commitment on the vendor's own page, exactly as auditable as the row
#: above it. Declining to record it would mean knowingly shipping a number that
#: becomes wrong on a date already known (ADR-031).
#:
#: :data:`_PRICES` holds the rate in force before the first date here; each
#: entry replaces it from its date onward.
_SCHEDULED: Final[dict[str, tuple[tuple[dt.date, ModelPrice], ...]]] = {
    "claude-sonnet-5": ((dt.date(2026, 9, 1), ModelPrice(3.00, 15.00)),),
}


#: Vendor namespaces that may precede a model id, as `openai/gpt-4o` or
#: `anthropic.claude-3-haiku-v1`. Matched as `[a-z_]+` rather than by a vendor
#: list so a new one needs no code change -- and requiring letters is what keeps
#: `gemini-1.5-pro` intact, whose first dot follows a digit.
_VENDOR = re.compile(r"^[a-z_]+[./]")

#: A suffix denoting the same model rather than a newer one. Either a release
#: date (`-2024-11-20`, `-20251101`) or a Bedrock revision tag (`-v1`, `-v1:0`).
#: Four digits is the discriminator that matters: `-20251101` is a snapshot of
#: one model and `-5` is the next one, and reading the second as the first is
#: what priced five Opus generations from a single row.
_SAME_MODEL_SUFFIX = re.compile(r"^-(?:\d{4,}|v\d+(?::\d+)?)(?:-|$)")


def _without_vendor(model: str) -> str:
    """Strip a leading vendor namespace, if there is one."""
    return _VENDOR.sub("", model, count=1)


def _is_same_model(model: str, key: str) -> bool:
    """Whether ``model`` is ``key``, or a dated snapshot or revision of it.

    Not "whether ``key`` appears in ``model``", which is what this used to ask.
    A version bump (`-5`, `-4-1`) and a descriptive suffix (`-mini`,
    `-distilled`) both name a *different* model, and neither may borrow a
    price.
    """
    if not model.startswith(key):
        return False
    remainder = model[len(key) :]
    return remainder == "" or bool(_SAME_MODEL_SUFFIX.match(remainder))


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

    def __init__(
        self,
        prices: dict[str, ModelPrice] | None = None,
        *,
        today: Callable[[], dt.date] | None = None,
    ) -> None:
        """Build a provider.

        Args:
            prices: Price table. Defaults to the built-in one.
            today: Source of the current date, for resolving a scheduled price
                change. Injectable because a table whose correctness depends on
                the calendar needs tests that can stand on both sides of a
                boundary without waiting five weeks for it.
        """
        self._today = today if today is not None else dt.date.today
        source = _PRICES if prices is None else prices
        self._prices = {name.lower(): price for name, price in source.items()}
        # Longest first so `gpt-4o-mini` wins over `gpt-4o` for a prefix match.
        self._by_length = sorted(self._prices, key=len, reverse=True)
        self._warned: set[str] = set()
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
        matched = self._lookup(normalised)
        if matched is None:
            self._warn_unknown(normalised)
            return None
        name, price = matched
        # Resolved here rather than at construction, so a process running across
        # an effective date does not keep serving the stale rate (ADR-031).
        for effective, scheduled in _SCHEDULED.get(name, ()):
            if self._today() >= effective:
                price = scheduled
        return price

    def _lookup(self, normalised: str) -> tuple[str, ModelPrice] | None:
        """Resolve an id to a row, or ``None`` rather than to something close.

        Exact first, then the same id with a vendor prefix removed, then a
        prefix match restricted to suffixes that denote *the same model*.

        Until 2026-07-31 the last step was ``if name in normalised`` --
        containment, over a table whose Anthropic rows were the three family
        keys ``claude-opus-4``, ``claude-sonnet-4`` and ``claude-haiku-4``, none
        of which is an id the API will serve. Five Opus generations resolved to
        one row, and ``not-really-gpt-4o-at-all-v2`` priced as ``gpt-4o``. The
        module docstring above has always said "not the price of a
        similar-looking model"; this is the code catching up to it (ADR-029).
        """
        exact = self._prices.get(normalised)
        if exact is not None:
            return normalised, exact

        bare = _without_vendor(normalised)
        if bare != normalised:
            exact = self._prices.get(bare)
            if exact is not None:
                return bare, exact

        for name in self._by_length:
            if _is_same_model(bare, name):
                return name, self._prices[name]

        return None

    def _warn_unknown(self, model: str) -> None:
        """Say once that a model is unpriced, and how to price it.

        Silence reads as a broken cost lane rather than a missing table row,
        and seven models Anthropic currently serves are deliberately unpriced
        (ADR-029 decision 3). Once per model: ADR-004's fail-open discipline
        means a pricing gap may never turn into a log flood on the hot path.
        """
        if model in self._warned:
            return
        self._warned.add(model)
        _log.warning(
            "optio: no price for model %r, so no cost signal will be emitted for it. "
            "Supply a PricingProvider to price models this table does not carry.",
            model,
        )

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
