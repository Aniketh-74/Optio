"""The bench provider must read the usage fields it bills from (ADR-027 follow-on).

Found by a live run: `multi_turn_chat` on Sonnet 4.5 reported

    provider cache      reads 18,300   writes 0

which cannot happen. Nothing is read from a cache that was never written --
Anthropic bills the first call of a cached prefix as a write.

``AnthropicProvider.__call__`` read ``cache_read_input_tokens`` and never
``cache_creation_input_tokens``, so:

* written tokens were **dropped from ``input_tokens`` entirely**, not merely
  mispriced, and
* the cache-write premium landed in ``ABResult.cost_usd`` a commit earlier was
  **inert**, because nothing ever populated the field it prices.

``wire.response_from_anthropic_message`` has done this correctly since ADR-021,
and its docstring records the identical defect being fixed there: "it dropped
the most expensive band of prompt tokens from the total, so a cached call
reported a fraction of its real cost and every prefix-cache saving derived from
it came out too high." The streaming adapter uses it. The benchmark -- the one
component whose entire output is those savings figures -- had its own copy.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from optio_optimize.bench.providers import AnthropicProvider, SpendGuard
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


class _Usage:
    def __init__(self, *, plain: int, read: int, written: int, written_1h: int = 0) -> None:
        self.input_tokens = plain
        self.output_tokens = 20
        self.cache_read_input_tokens = read
        self.cache_creation_input_tokens = written
        self.cache_creation = type("C", (), {"ephemeral_1h_input_tokens": written_1h})()


class _Block:
    type = "text"
    text = "ok"


class _Reply:
    def __init__(self, usage: _Usage) -> None:
        self.usage = usage
        self.content = [_Block()]
        self.model = "claude-sonnet-4-5"
        self.stop_reason = "end_turn"


class _Messages:
    def __init__(self, usage: _Usage) -> None:
        self._usage = usage

    def create(self, **_: Any) -> _Reply:
        return _Reply(self._usage)


class _Client:
    def __init__(self, usage: _Usage) -> None:
        self.messages = _Messages(usage)


def _provider(usage: _Usage) -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = cast("Any", _Client(usage))
    provider.model = "claude-sonnet-4-5"
    provider.guard = SpendGuard(10.0)
    return provider


def _request() -> LLMRequest:
    return LLMRequest(
        model="claude-sonnet-4-5",
        messages=(
            Message(role="system", content="policy " * 400, cacheable=True),
            Message(role="user", content="q"),
        ),
        temperature=0.0,
    )


class TestWrittenTokensReachTheResult:
    def test_a_write_is_reported(self) -> None:
        """The live case: a first call that populates the cache."""
        response = _provider(_Usage(plain=40, read=0, written=1_406))(_request())

        assert response.cache_write_tokens == 1_406

    def test_a_write_is_counted_in_input_tokens(self) -> None:
        """Not merely mispriced -- they were missing from the total.

        ``wire`` sums plain + read + written. The bench provider summed plain +
        read, so a 1,406-token write vanished from the benchmark's own
        token-reduction figure as well as from its cost.
        """
        response = _provider(_Usage(plain=40, read=0, written=1_406))(_request())

        assert response.input_tokens == 40 + 1_406

    def test_a_read_is_still_reported(self) -> None:
        response = _provider(_Usage(plain=40, read=1_663, written=0))(_request())

        assert response.cached_input_tokens == 1_663
        assert response.input_tokens == 40 + 1_663

    def test_reads_and_writes_can_both_appear(self) -> None:
        response = _provider(_Usage(plain=40, read=1_663, written=200))(_request())

        assert (response.cached_input_tokens, response.cache_write_tokens) == (1_663, 200)
        assert response.input_tokens == 40 + 1_663 + 200

    def test_the_one_hour_band_is_reported(self) -> None:
        """A 1-hour write costs 2x base against 1.25x for five minutes.

        Folding the two together under-bills the dearer one by 37.5%, in the
        same flattering direction as omitting writes altogether (ADR-021).
        """
        response = _provider(_Usage(plain=40, read=0, written=900, written_1h=900))(_request())

        assert response.cache_write_1h_tokens == 900


class TestTheFlatteringDirection:
    def test_a_writing_call_is_not_cheaper_than_a_plain_one(self) -> None:
        """The regression this class exists for.

        Before the fix, a call that wrote 1,406 tokens into the cache reported
        the same 40 input tokens as one that wrote nothing -- so placing a
        breakpoint appeared free, and every saving measured against it was
        overstated.
        """
        wrote = _provider(_Usage(plain=40, read=0, written=1_406))(_request())
        did_not = _provider(_Usage(plain=40, read=0, written=0))(_request())

        assert wrote.input_tokens > did_not.input_tokens
