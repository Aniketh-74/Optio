"""Token counting, where a wrong answer corrupts every number downstream.

Counts feed savings figures, cache keys and budget decisions. A memoizer that
returns a stale or wrong count does not fail loudly -- it produces a confident
report that is wrong, which is the failure mode this whole package has to avoid.

So the memoizing counter is tested against the counter it wraps rather than
against expected values: whatever the real tokenizer says, the cache must say
the same thing.
"""

from __future__ import annotations

import threading

import pytest

from optio_optimize.tokens import (
    HeuristicCounter,
    MemoizingCounter,
    count_request,
    default_counter,
    fits_in_window,
)
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


class CountingInner:
    """A counter that records how often it was actually consulted."""

    is_exact = True

    def __init__(self) -> None:
        self.calls = 0

    def count_text(self, text: str, model: str = "") -> int:
        self.calls += 1
        # Deterministic and content-dependent, so a cache returning the wrong
        # entry produces a visibly wrong number rather than a plausible one.
        return len(text) // 3 + sum(ord(c) for c in text[:4]) % 7


class TestMemoizingNeverChangesTheAnswer:
    @pytest.mark.parametrize(
        "text",
        [
            "x" * 250,
            "a longer piece of prose " * 40,
            '{"json": true, "nested": {"a": [1, 2, 3]}}' * 20,
            "unicode: éèê中文 " * 60,
        ],
    )
    def test_it_matches_the_wrapped_counter(self, text: str) -> None:
        inner = HeuristicCounter()
        memo = MemoizingCounter(HeuristicCounter())

        assert memo.count_text(text, "gpt-4o") == inner.count_text(text, "gpt-4o")
        # And again, now that it is cached.
        assert memo.count_text(text, "gpt-4o") == inner.count_text(text, "gpt-4o")

    def test_different_texts_get_different_counts(self) -> None:
        """A digest collision or key bug would show up here.

        The dangerous shape is not a crash -- it is the second text quietly
        receiving the first one's count.
        """
        memo = MemoizingCounter(CountingInner())
        first = "alpha " * 60
        second = "bravo " * 60

        assert memo.count_text(first, "m") != memo.count_text(second, "m")

    def test_the_same_text_under_different_models_is_counted_separately(self) -> None:
        # Models have different tokenizers; sharing an entry across them would
        # silently return o200k counts for a cl100k model.
        memo = MemoizingCounter(CountingInner())
        text = "shared content " * 40

        memo.count_text(text, "gpt-4o")
        before = memo.hits
        memo.count_text(text, "claude-sonnet-4")

        assert memo.hits == before, "a different model reused the cached count"


class TestItActuallyAvoidsWork:
    def test_a_repeated_prompt_is_counted_once(self) -> None:
        inner = CountingInner()
        memo = MemoizingCounter(inner)
        prompt = "system instructions " * 60

        for _ in range(20):
            memo.count_text(prompt, "gpt-4o")

        assert inner.calls == 1, f"tokenized {inner.calls} times instead of once"
        assert memo.hit_rate == pytest.approx(19 / 20)

    def test_short_strings_bypass_the_cache(self) -> None:
        """Below the threshold, hashing costs as much as counting.

        Memoizing them would fill the cache with entries that displace the large
        repeated prompts it exists to hold.
        """
        inner = CountingInner()
        memo = MemoizingCounter(inner)

        for _ in range(5):
            memo.count_text("short", "gpt-4o")

        assert inner.calls == 5
        assert memo.hits == 0

    def test_no_data_reports_none_rather_than_zero(self) -> None:
        assert MemoizingCounter(CountingInner()).hit_rate is None


class TestItStaysBounded:
    def test_entries_never_exceed_the_ceiling(self) -> None:
        # This lives for the process lifetime, so an unbounded cache here is the
        # same leak shape §11 exists to prevent.
        memo = MemoizingCounter(CountingInner())
        for n in range(MemoizingCounter.MAX_ENTRIES + 500):
            memo.count_text(f"unique text number {n} " * 15, "gpt-4o")
            assert len(memo._counts) <= MemoizingCounter.MAX_ENTRIES

    def test_no_prompt_content_is_retained(self) -> None:
        """Keys are digests, so a heap dump holds no prompts.

        The core forbids retaining prompt content (§10). This package must read
        it to do its job, but that is no reason to *keep* it.
        """
        memo = MemoizingCounter(CountingInner())
        secret = "patient SSN 123-45-6789 " * 20
        memo.count_text(secret, "gpt-4o")

        for digest, model in memo._counts:
            assert "SSN" not in digest
            assert "123-45-6789" not in digest
            assert secret not in digest
            assert model == "gpt-4o"


class TestItIsThreadSafe:
    def test_concurrent_counting_stays_correct(self) -> None:
        # The read-modify-write on the LRU is exactly the shape the core's
        # concurrency work showed the GIL does not protect.
        memo = MemoizingCounter(HeuristicCounter())
        reference = HeuristicCounter()
        texts = [f"prompt variant {n} " * 30 for n in range(16)]
        expected = [reference.count_text(t, "gpt-4o") for t in texts]
        wrong: list[str] = []

        def worker() -> None:
            for _ in range(50):
                for text, want in zip(texts, expected, strict=True):
                    if memo.count_text(text, "gpt-4o") != want:
                        wrong.append(text[:20])

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "counting deadlocked"

        assert not wrong, f"{len(wrong)} wrong counts under contention"


class TestRequestCounting:
    def test_tool_schemas_are_counted(self) -> None:
        """Tool definitions are billed and are easy to forget.

        A dozen schemas can outweigh the conversation, and they are resent on
        every step -- so omitting them understates the prompt badly on exactly
        the workloads that cost most.
        """
        counter = default_counter()
        bare = LLMRequest(model="gpt-4o", messages=(Message(role="user", content="hi"),))
        with_tools = LLMRequest(
            model="gpt-4o",
            messages=(Message(role="user", content="hi"),),
            tools=({"name": "search", "parameters": {"q": "string", "k": "integer"}},),
        )

        assert count_request(with_tools, counter) > count_request(bare, counter)

    def test_an_inexact_counter_is_treated_conservatively(self) -> None:
        """Guessing low means a provider-side rejection the user sees as a crash."""
        assert not fits_in_window(950, limit=1000, counter=HeuristicCounter())
        assert fits_in_window(950, limit=1000, counter=CountingInner())
