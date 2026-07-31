"""A stream that dies mid-generation must leave no trace (ADR-019's safety claim).

``anthropic_streaming``'s docstring makes the strongest claim in this package:

    Only the terminal event completes a request. Not exhaustion, not ``close()``.
    A transport that dies mid-generation, or a caller who stops reading, leaves a
    partial reply, and ``exact_cache`` storing a truncation would serve it
    confidently and permanently to everyone who later asks the same question. A
    missing report row costs a number; that costs correctness.

That is asserted about ``message_stop`` and had never been tested against the
ways a stream actually dies. It matters more after ADR-033, which found the
non-streaming version of this hazard *live*: a truncated answer was being cached
and served, because the truncation guard compared Anthropic's ``max_tokens``
against OpenAI's ``length``. There the guard existed and was inert; here the
guard is "the terminal event never arrived", and this pins it.

Four ways a stream ends without a ``message_stop``, all of them real: the
transport raises, the caller breaks out of the loop, the caller calls ``close()``
early, and the caller leaves a ``with`` block early.
"""

from __future__ import annotations

from typing import Any

import pytest

from optio_optimize.adapters.anthropic_streaming import StreamProxy
from optio_optimize.cache import MemoryCache, request_key
from optio_optimize.config import OptimizeConfig
from optio_optimize.pipeline import Pipeline
from optio_optimize.stages.caching import ExactCacheStage
from optio_optimize.types import LLMRequest, Message

pytestmark = [pytest.mark.failinject, pytest.mark.optimize]


class _Event:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        # `text_delta` specifically: the accumulator ignores `thinking_delta`,
        # because reasoning is billed but is not the answer.
        self.delta = type("D", (), {"type": "text_delta", "text": text})()


class _Usage:
    input_tokens = 100
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Start:
    type = "message_start"

    def __init__(self) -> None:
        self.message = type("M", (), {"usage": _Usage(), "model": "claude-sonnet-4-5"})()


def _events(*, terminal: bool, explode_at: int | None = None) -> list[Any]:
    body: list[Any] = [_Start(), _Event("content_block_delta", "partial ")]
    body.append(_Event("content_block_delta", "answer"))
    if terminal:
        body.append(_Event("message_stop"))
    if explode_at is not None:
        body.insert(explode_at, _Boom())
    return body


class _Boom:
    """A sentinel the fake stream turns into a raised transport error."""


class _FakeStream:
    """Yields events; raises where a real transport would die."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    def __iter__(self) -> Any:
        for event in self._events:
            if isinstance(event, _Boom):
                raise ConnectionError("transport died mid-generation")
            yield event

    def close(self) -> None:
        self.closed = True


def _request() -> LLMRequest:
    return LLMRequest(
        model="claude-sonnet-4-5",
        messages=(Message(role="user", content="explain at length"),),
        temperature=0.0,
    )


def _proxy(events: list[Any]) -> tuple[StreamProxy, MemoryCache, LLMRequest]:
    backend = MemoryCache()
    pipeline = Pipeline(OptimizeConfig(), stages=[ExactCacheStage(backend=backend)])
    request = _request()
    prepared = pipeline.prepare(request)
    return StreamProxy(_FakeStream(events), pipeline, prepared), backend, request


class TestAPartialStreamIsNeverCached:
    def test_a_transport_death_caches_nothing(self) -> None:
        proxy, backend, request = _proxy(_events(terminal=False, explode_at=2))

        with pytest.raises(ConnectionError):
            for _ in proxy:
                pass

        assert backend.get(request_key(request)) is None

    def test_a_caller_who_stops_reading_caches_nothing(self) -> None:
        """``break`` out of the loop -- the commonest way a stream is abandoned."""
        proxy, backend, request = _proxy(_events(terminal=True))

        for _ in proxy:
            break

        assert backend.get(request_key(request)) is None

    def test_an_early_close_caches_nothing(self) -> None:
        proxy, backend, request = _proxy(_events(terminal=True))

        next(iter(proxy))
        proxy.close()

        assert backend.get(request_key(request)) is None

    def test_leaving_a_with_block_early_caches_nothing(self) -> None:
        proxy, backend, request = _proxy(_events(terminal=True))

        with proxy as stream:
            next(iter(stream))

        assert backend.get(request_key(request)) is None

    def test_exhaustion_without_a_terminal_event_caches_nothing(self) -> None:
        """A stream that simply stops. Exhaustion is not completion."""
        proxy, backend, request = _proxy(_events(terminal=False))

        for _ in proxy:
            pass

        assert backend.get(request_key(request)) is None


class TestACompleteStreamIsCached:
    """The other half: the guard must not be so strict that nothing ever caches."""

    def test_a_terminal_event_completes_the_request(self) -> None:
        proxy, backend, request = _proxy(_events(terminal=True))

        for _ in proxy:
            pass

        assert backend.get(request_key(request)) is not None

    def test_the_cached_answer_is_the_whole_answer(self) -> None:
        proxy, backend, request = _proxy(_events(terminal=True))

        for _ in proxy:
            pass

        stored = backend.get(request_key(request))
        assert stored is not None
        assert stored.content == "partial answer"


class TestTheCallerIsUnaffected:
    def test_every_event_reaches_the_caller_before_the_death(self) -> None:
        proxy, _, _ = _proxy(_events(terminal=False, explode_at=3))
        seen: list[str] = []

        with pytest.raises(ConnectionError):
            for event in proxy:
                seen.append(event.type)

        assert seen == ["message_start", "content_block_delta", "content_block_delta"]

    def test_the_transport_error_is_not_swallowed(self) -> None:
        """A dead stream must look dead. Fail-open is about cost, not silence."""
        proxy, _, _ = _proxy(_events(terminal=False, explode_at=2))

        with pytest.raises(ConnectionError, match="transport died"):
            for _ in proxy:
                pass
