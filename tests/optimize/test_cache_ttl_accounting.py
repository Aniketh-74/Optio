"""Pricing the one-hour cache band (ADR-021, decision 1).

Anthropic charges **2.0x** the base input rate to populate a one-hour cache
entry, against 1.25x for the five-minute one. ``_cost`` knows about a single
write rate, so the moment anything asks for ``ttl: "1h"`` the most expensive
band in the request is under-billed by 37.5% -- in the direction that inflates
this package's headline saving.

That asymmetry has already been paid for once here. Cache writes were left out
of the prompt total entirely: one turn reported 200 tokens against a true 4,805,
and a published 53.7% saving was really 50.1%. These tests exist so the second
occurrence is impossible rather than merely unlikely.

**This file is worth its weight with no stage attached.** A caller who sets
their own ``cache_control: {"ttl": "1h"}`` breakpoint today is already mis-priced
by this package, and the provider has been reporting the split all along in
``usage.cache_creation`` -- this package simply was not reading it.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import PRICING, ModelPricing
from optio_optimize.savings import _cost
from optio_optimize.types import LLMResponse

pytestmark = pytest.mark.optimize

#: Haiku 4.5's real rates: $1.00 input, $5.00 output, $0.10 read, $1.25 5m
#: write, $2.00 1h write.
_HAIKU = ModelPricing(1.00, 5.00, 0.10, 1.25, 2.00)


class TestTheOneHourBandIsPricedSeparately:
    """``written_1h`` is a **subset** of ``written``, not a sibling of it.

    That is how the provider reports it -- the total in
    ``cache_creation_input_tokens``, the breakdown in ``cache_creation`` -- so an
    all-one-hour write is ``written=N, written_1h=N``, and ``written=0,
    written_1h=N`` is incoherent input rather than a way to express one. The
    first draft of this file got that wrong in four tests, which is a decent
    argument for the clamping the implementation does.
    """

    def test_a_one_hour_write_costs_more_than_a_five_minute_one(self) -> None:
        five_minute = _cost(_HAIKU, 1_000_000, 0, 0, 1_000_000, 0)
        one_hour = _cost(_HAIKU, 1_000_000, 0, 0, 1_000_000, 1_000_000)

        assert one_hour > five_minute
        assert one_hour == pytest.approx(2.00)
        assert five_minute == pytest.approx(1.25)

    def test_the_bands_add_up_rather_than_overlapping(self) -> None:
        # 400k base + 300k read + 300k written, of which 100k went to an hour.
        cost = _cost(_HAIKU, 1_000_000, 0, 300_000, 300_000, 100_000)

        expected = 0.400 * 1.00 + 0.300 * 0.10 + 0.200 * 1.25 + 0.100 * 2.00
        assert cost == pytest.approx(expected)

    def test_writes_never_exceed_the_input_they_are_a_subset_of(self) -> None:
        """A response claiming more writes than prompt tokens is a provider or
        translation bug, and pricing it literally would report a cost above the
        real bill."""
        cost = _cost(_HAIKU, 1_000, 0, 0, 900_000, 900_000)

        # Clamped to the 1,000 prompt tokens that exist, all at the hour rate.
        assert cost == pytest.approx(1_000 * 2.00 / 1_000_000)

    def test_the_hour_count_cannot_exceed_the_total_write_count(self) -> None:
        # Incoherent input -- more hour-writes than writes -- must not price
        # tokens that were never written at the most expensive rate.
        assert _cost(_HAIKU, 1_000_000, 0, 0, 0, 1_000_000) == pytest.approx(1.00)

    def test_a_missing_one_hour_rate_falls_back_rather_than_crashing(self) -> None:
        # OpenAI populates its cache for free and offers no TTL control, so both
        # rates are absent there and 1h tokens should never appear -- but a
        # crash on a provider that reports something unexpected would be a
        # fail-closed accounting bug.
        no_rates = ModelPricing(1.00, 5.00)

        assert _cost(no_rates, 1_000_000, 0, 0, 1_000_000, 1_000_000) == pytest.approx(1.00)

    def test_the_five_minute_rate_is_used_when_only_it_is_known(self) -> None:
        only_5m = ModelPricing(1.00, 5.00, 0.10, 1.25)

        # A 1h write with no 1h rate falls back to the 5m rate rather than the
        # base rate: it is certainly not cheaper than a 5m write, and guessing
        # low here is the flattering direction.
        assert _cost(only_5m, 1_000_000, 0, 0, 1_000_000, 1_000_000) == pytest.approx(1.25)


class TestEveryAnthropicModelHasAOneHourRate:
    @pytest.mark.parametrize(
        "model", [name for name in PRICING if name.startswith(("claude", "anthropic"))]
    )
    def test_the_one_hour_rate_is_exactly_twice_the_base_input_rate(self, model: str) -> None:
        """Anthropic's published multiplier, checked as data rather than prose.

        A row that gains a 5-minute rate and not a one-hour one would silently
        price the expensive band at 1.25x, which is the whole defect this file
        guards.
        """
        pricing = PRICING[model]

        assert pricing.cache_write_1h_usd_per_m is not None, (
            f"{model} has no one-hour write rate, so a 1h write there is priced "
            f"at the 5-minute rate and understates the bill by 37.5%"
        )
        assert pricing.cache_write_1h_usd_per_m == pytest.approx(pricing.input_usd_per_m * 2.0), (
            f"{model}: Anthropic charges 2x base input to populate a one-hour entry"
        )

    @pytest.mark.parametrize(
        "model", [name for name in PRICING if name.startswith(("gpt", "o1", "o3", "o4"))]
    )
    def test_openai_rows_carry_no_write_premium_at_all(self, model: str) -> None:
        # OpenAI populates its cache for free. A write rate here would invent a
        # charge the provider does not make.
        pricing = PRICING[model]

        assert pricing.cache_write_usd_per_m is None
        assert pricing.cache_write_1h_usd_per_m is None


class TestTheResponseCarriesTheSplit:
    def test_a_one_hour_write_is_still_part_of_the_prompt_total(self) -> None:
        """``input_tokens`` means total prompt tokens, whatever band they sit in.

        The bug this mirrors: writes omitted from the total made a cached call
        report a fraction of its real cost, and every saving derived from it came
        out too high.
        """
        response = LLMResponse(
            content="x",
            input_tokens=5_000,
            output_tokens=10,
            cached_input_tokens=0,
            cache_write_tokens=4_000,
            cache_write_1h_tokens=4_000,
        )

        assert response.billable_input_tokens == 5_000

    def test_one_hour_writes_are_a_subset_of_all_writes_not_a_sibling(self) -> None:
        """``cache_write_1h_tokens`` refines ``cache_write_tokens``; it does not
        sit beside it.

        The provider reports ``cache_creation_input_tokens`` as the total and
        breaks it down in ``cache_creation``. Treating the 1h figure as
        additional would double-count the most expensive tokens in the request.
        """
        response = LLMResponse(
            content="x",
            input_tokens=5_000,
            output_tokens=10,
            cache_write_tokens=4_000,
            cache_write_1h_tokens=4_000,
        )

        assert response.cache_write_1h_tokens <= response.cache_write_tokens

    def test_the_default_is_zero_one_hour_writes(self) -> None:
        # Every existing caller and every provider that has no TTL control.
        assert LLMResponse(content="x").cache_write_1h_tokens == 0


class TestTheSplitIsReadOffTheProvidersOwnUsage:
    """Anthropic has reported this breakdown all along; nothing read it.

    ``usage.cache_creation`` carries ``ephemeral_1h_input_tokens`` and
    ``ephemeral_5m_input_tokens`` beside the combined
    ``cache_creation_input_tokens``. Verified live before ADR-021 was written: a
    request with ``ttl: "1h"`` and **no beta header** put 4,218 tokens in the
    one-hour field, so a caller who sets their own hour-long breakpoint has been
    mis-priced by this package rather than by the provider.
    """

    def test_a_one_hour_write_is_recognised(self) -> None:
        from optio_optimize.wire import response_from_anthropic_message

        message = _AnthropicMessage(_Usage(input_tokens=100, cache_creation_1h=4_218, total=4_218))

        response = response_from_anthropic_message(message)

        assert response.cache_write_tokens == 4_218
        assert response.cache_write_1h_tokens == 4_218
        assert response.input_tokens == 100 + 4_218

    def test_a_five_minute_write_reports_no_hour_tokens(self) -> None:
        from optio_optimize.wire import response_from_anthropic_message

        message = _AnthropicMessage(_Usage(input_tokens=100, cache_creation_1h=0, total=4_218))

        response = response_from_anthropic_message(message)

        assert response.cache_write_tokens == 4_218
        assert response.cache_write_1h_tokens == 0

    def test_a_usage_object_without_the_breakdown_is_not_an_error(self) -> None:
        # Older SDKs, other providers, and batch result shapes. Absent means
        # "no hour writes reported", not "crash".
        from optio_optimize.wire import response_from_anthropic_message

        message = _AnthropicMessage(_Usage(input_tokens=100, cache_creation_1h=None, total=4_218))

        response = response_from_anthropic_message(message)

        assert response.cache_write_tokens == 4_218
        assert response.cache_write_1h_tokens == 0


class _CacheCreation:
    def __init__(self, one_hour: int) -> None:
        self.ephemeral_1h_input_tokens = one_hour
        self.ephemeral_5m_input_tokens = 0


class _Usage:
    """The attribute shape ``response_from_anthropic_message`` reads.

    Attribute access rather than the real pydantic model, matching what that
    function documents: its two callers hand it structurally identical objects
    from different SDK code paths, and a batch result's usage is not the same
    class as a live one's.
    """

    def __init__(self, *, input_tokens: int, cache_creation_1h: int | None, total: int) -> None:
        self.input_tokens = input_tokens
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = total
        if cache_creation_1h is not None:
            self.cache_creation = _CacheCreation(cache_creation_1h)


class _AnthropicMessage:
    def __init__(self, usage: _Usage) -> None:
        self.usage = usage
        self.content: list[object] = []
        self.model = "claude-haiku-4-5"
        self.stop_reason = "end_turn"
