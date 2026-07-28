"""SimulatedProvider's automatic-cache model, calibrated against live traces.

Two live traces (2026-07-28 and 2026-07-29) confirmed OpenAI's automatic
prefix cache reports `cached_tokens` as an exact multiple of 128, never an
arbitrary value past the 1024-token floor -- see
``bench/providers.py``'s ``_AUTO_CACHE_QUANTUM_TOKENS`` docstring for the
raw numbers. These tests pin the simulator to that, not to whatever token
count a message boundary happens to land on.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from optio_optimize.bench.providers import SimulatedProvider
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _request(content: str, model: str = "gpt-4o") -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(Message(role="system", content=content),),
        temperature=0.0,
    )


class TestAutomaticCacheQuantization:
    def test_a_repeat_call_reports_cached_tokens_as_a_multiple_of_the_quantum(self) -> None:
        provider = SimulatedProvider(prefix_cache_style="automatic")
        # Long enough to comfortably clear the 1024-token floor.
        big_prompt = "You are a careful assistant. " * 300

        provider(_request(big_prompt))
        second = provider(_request(big_prompt))

        assert second.cached_input_tokens > 0
        assert second.cached_input_tokens % 128 == 0

    def test_a_prompt_under_the_floor_never_reports_a_cache_hit(self) -> None:
        provider = SimulatedProvider(prefix_cache_style="automatic")
        short_prompt = "Be terse."

        provider(_request(short_prompt))
        second = provider(_request(short_prompt))

        assert second.cached_input_tokens == 0

    def test_the_first_call_never_reports_a_hit(self) -> None:
        provider = SimulatedProvider(prefix_cache_style="automatic")
        big_prompt = "You are a careful assistant. " * 300

        first = provider(_request(big_prompt))

        assert first.cached_input_tokens == 0

    def test_a_growing_conversation_reports_a_non_decreasing_cache(self) -> None:
        """The cache only ever grows or plateaus as history accumulates,
        matching the live trace's 0 -> 1408 -> plateau -> 1536 -> plateau
        shape -- it must never drop back down while the prefix keeps growing.
        """
        provider = SimulatedProvider(prefix_cache_style="automatic")
        base = "You are a careful assistant. " * 300
        seen: list[int] = []

        text = base
        for turn in range(6):
            text += f" Turn {turn}: some more conversation content here."
            response = provider(_request(text))
            seen.append(response.cached_input_tokens)

        for earlier, later in pairwise(seen):
            assert later >= earlier, f"cache shrank: {seen}"
