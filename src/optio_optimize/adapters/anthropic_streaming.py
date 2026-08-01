"""Streaming for the Anthropic adapter (ADR-019).

A ``stream=True`` call used to bypass this package entirely, on the grounds that
a pipeline built around one request producing one response can only buffer a
token stream. That is true of the few stages which read a reply and false of the
majority: ``prefix_cache`` writes a breakpoint onto the outgoing request and
never looks at the response, and on Anthropic it is the only reason the provider
caches anything. Streaming callers were getting zero of nineteen stages.

The shape here is the one ADR-017 already built for batch: ``Pipeline.prepare``
runs every ``before`` hook, and ``Pipeline.complete`` runs the ``after`` hooks
when the answer exists. Batch has hours between the halves; a stream has the
duration of one generation.

**Nothing is withheld.** The proxy forwards each event the instant it arrives and
accumulates alongside, so the caller sees the first token exactly as soon as they
would without this package. What is deferred is bookkeeping -- a cache write, an
observation, a report row -- which nobody waits on.

**Only the terminal event completes a request.** Not exhaustion, not ``close()``.
A transport that dies mid-generation, or a caller who stops reading, leaves a
partial reply, and ``exact_cache`` storing a truncation would serve it
confidently and permanently to everyone who later asks the same question. A
missing report row costs a number; that costs correctness.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from optio_optimize.types import LLMResponse

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from optio_optimize.pipeline import Pipeline, PreparedRequest

_log = logging.getLogger("optio_optimize")

#: The event that means the message is finished. The one trigger for running the
#: ``after`` hooks -- see this module's docstring for why exhaustion is not.
TERMINAL_EVENT = "message_stop"


@dataclass(slots=True)
class _Accumulator:
    """Folds stream events into the response the ``after`` hooks need.

    Deliberately mirrors :func:`~optio_optimize.wire.response_from_anthropic_message`
    field for field, including adding cache reads *and* writes back into
    ``input_tokens``. A second, subtly different reading of the same usage
    numbers is how the streaming and non-streaming totals would come to
    disagree, and the non-symmetric version of this exact arithmetic already
    reported 200 prompt tokens against a true 4,805 once.
    """

    parts: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached: int = 0
    written: int = 0
    written_1h: int = 0
    model: str = ""
    finish_reason: str | None = None

    def feed(self, event: Any) -> None:
        """Fold one event in. Unknown event types are ignored, not an error."""
        kind = getattr(event, "type", "")
        if kind == "message_start":
            self._start(getattr(event, "message", None))
        elif kind == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", "") == "text_delta":
                # Text only. A `thinking_delta` is reasoning, which
                # `response_from_anthropic_message` also excludes from content --
                # folding it in would put a thinking trace into what a stage may
                # rewrite and what `exact_cache` may later serve as the answer.
                self.parts.append(str(getattr(delta, "text", "")))
        elif kind == "message_delta":
            self._message_delta(event)

    def _start(self, message: Any) -> None:
        usage = getattr(message, "usage", None)
        self.model = str(getattr(message, "model", "") or "")
        if usage is None:
            return
        self.cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        self.written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        # The one-hour band, read exactly as `response_from_anthropic_message`
        # reads it. It was missing here, so every *streamed* reply reported
        # `cache_write_1h_tokens=0` and priced 2x tokens at 1.25x -- two paths to
        # one provider disagreeing about the same usage object, which is the
        # failure the docstring above warns about, in the field it did not name.
        # Clamped to `written` because the hour figure is a subset of the total
        # (ADR-021), not a sibling.
        creation = getattr(usage, "cache_creation", None)
        self.written_1h = min(
            self.written, int(getattr(creation, "ephemeral_1h_input_tokens", 0) or 0)
        )
        self.input_tokens = int(getattr(usage, "input_tokens", 0) or 0) + self.cached + self.written
        self.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    def _message_delta(self, event: Any) -> None:
        delta = getattr(event, "delta", None)
        reason = getattr(delta, "stop_reason", None)
        if reason is not None:
            self.finish_reason = str(reason)
        usage = getattr(event, "usage", None)
        total = getattr(usage, "output_tokens", None)
        if isinstance(total, int):
            # Cumulative for the whole message, not a per-event increment, so it
            # replaces rather than adds. Summing these would multiply the output
            # bill by the number of delta events on a long reply.
            self.output_tokens = total

    def response(self) -> LLMResponse:
        """The completed reply, in this package's model."""
        return LLMResponse(
            content="".join(self.parts),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached,
            cache_write_tokens=self.written,
            cache_write_1h_tokens=self.written_1h,
            model=self.model,
            finish_reason=self.finish_reason,
        )


class _Completion:
    """Runs ``Pipeline.complete`` at most once, whatever the caller does.

    A stream offers more than one route to "finished" -- the terminal event, and
    a caller who also calls ``close()`` -- and ``complete`` must fire exactly
    once per ``prepare`` or the savings report double-counts the request and the
    read-through caches get two write halves.
    """

    __slots__ = ("_done", "_pipeline", "_prepared")

    def __init__(self, pipeline: Pipeline, prepared: PreparedRequest) -> None:
        self._pipeline = pipeline
        self._prepared = prepared
        self._done = False

    def fire(self, response: LLMResponse) -> None:
        """Complete the request, or do nothing if it already happened."""
        if self._done:
            return
        self._done = True
        try:
            self._pipeline.complete(self._prepared, response)
        except Exception as exc:  # noqa: BLE001 - ADR-013 rule 1
            # The caller already has every token. A failed cache write or a
            # failed accounting row must not surface as an exception from the
            # last `next()` of an otherwise perfect stream.
            _log.warning(
                "optio_optimize: completing a streamed request failed (%s); "
                "the response is unaffected",
                type(exc).__name__,
            )


class StreamProxy:
    """A synchronous ``Stream``, passed through and accounted for.

    Not a ``Stream`` subclass: the SDK's constructor wants a cast type, an httpx
    response and a client, and faking those to satisfy ``isinstance`` would be
    worse than not satisfying it (ADR-019 records the difference). Iteration,
    ``with``, ``close()`` and attribute forwarding all work.
    """

    __slots__ = ("_acc", "_completion", "_events", "_stream")

    def __init__(self, stream: Any, pipeline: Pipeline, prepared: PreparedRequest) -> None:
        self._stream = stream
        self._events: Iterator[Any] = iter(stream)
        self._acc = _Accumulator()
        self._completion = _Completion(pipeline, prepared)

    def __iter__(self) -> StreamProxy:
        """Return self, so a second pass does not restart the underlying stream."""
        return self

    def __next__(self) -> Any:
        """Yield the next provider event, unchanged, recording it on the way."""
        event = next(self._events)
        self._acc.feed(event)
        if getattr(event, "type", "") == TERMINAL_EVENT:
            self._completion.fire(self._acc.response())
        return event

    def close(self) -> None:
        """Close the underlying stream. Completes nothing on its own."""
        self._stream.close()

    def __enter__(self) -> StreamProxy:
        """Enter the ``with`` block, as the SDK's own stream does."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on the way out. Completing is the terminal event's job."""
        self.close()

    def __getattr__(self, name: str) -> Any:
        """Forward anything unmodelled to the real stream.

        Private names raise instead of forwarding, which is what keeps this from
        recursing while ``__init__`` is still assigning ``_stream``.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)


class AsyncStreamProxy:
    """The ``AsyncStream`` twin of :class:`StreamProxy`."""

    __slots__ = ("_acc", "_completion", "_events", "_stream")

    def __init__(self, stream: Any, pipeline: Pipeline, prepared: PreparedRequest) -> None:
        self._stream = stream
        self._events = stream.__aiter__()
        self._acc = _Accumulator()
        self._completion = _Completion(pipeline, prepared)

    def __aiter__(self) -> AsyncStreamProxy:
        """Return self, so a second pass does not restart the stream."""
        return self

    async def __anext__(self) -> Any:
        """Yield the next provider event, unchanged, recording it on the way."""
        event = await self._events.__anext__()
        self._acc.feed(event)
        if getattr(event, "type", "") == TERMINAL_EVENT:
            self._completion.fire(self._acc.response())
        return event

    async def close(self) -> None:
        """Close the underlying stream. Completes nothing on its own."""
        await self._stream.close()

    async def __aenter__(self) -> AsyncStreamProxy:
        """Enter the ``async with`` block, as the SDK's own stream does."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on the way out. Completing is the terminal event's job."""
        await self.close()

    def __getattr__(self, name: str) -> Any:
        """Forward anything unmodelled to the real stream."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)


def replay_events(response: LLMResponse, kwargs: dict[str, Any]) -> list[Any]:
    """Build the event sequence a cache hit is delivered as.

    One delta carrying the whole answer, where a live call sent hundreds. The
    reassembled message is byte-identical; the chunking is not, and that is not a
    defect -- a replayed hit *did* arrive all at once, because nothing was
    generated for it.

    Built through the SDK's own pydantic models, so a shape this package gets
    wrong fails here rather than in the caller's event handler.
    """
    from anthropic.types import (
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
    )

    model = response.model or str(kwargs.get("model") or "")
    stop_reason = response.finish_reason or "end_turn"
    return [
        RawMessageStartEvent.model_validate(
            {
                "type": "message_start",
                "message": {
                    "id": f"optio-optimize-{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    # Zeroes, and they are the truthful numbers: this request was
                    # never sent. Echoing the stored call's usage would re-bill
                    # the call the hit avoided, on every hit -- the defect the
                    # non-streaming `_unwrap` already documents.
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        ),
        RawContentBlockStartEvent.model_validate(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        RawContentBlockDeltaEvent.model_validate(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": response.content},
            }
        ),
        RawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": 0}),
        RawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }
        ),
        RawMessageStopEvent.model_validate({"type": "message_stop"}),
    ]


class ReplayStream:
    """A synchronous stream over an answer that is already in hand.

    Same surface a caller uses on the real thing -- iteration, ``with``,
    ``close()`` -- so a cache hit needs no special case at the call site. That is
    the whole value: a streaming caller gets cache hits without knowing they can
    happen.
    """

    __slots__ = ("_events",)

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)

    def __iter__(self) -> ReplayStream:
        """Return self, matching the SDK's single-pass stream."""
        return self

    def __next__(self) -> Any:
        """Yield the next synthesized event."""
        return next(self._events)

    def close(self) -> None:
        """Nothing to release; there was never a connection."""

    def __enter__(self) -> ReplayStream:
        """Enter the ``with`` block, so a hit needs no special case."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on the way out. Completing is the terminal event's job."""
        self.close()


class AsyncReplayStream:
    """The async twin of :class:`ReplayStream`."""

    __slots__ = ("_events",)

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> AsyncReplayStream:
        """Return self, matching the SDK's single-pass stream."""
        return self

    async def __anext__(self) -> Any:
        """Yield the next synthesized event."""
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        """Nothing to release; there was never a connection."""

    async def __aenter__(self) -> AsyncReplayStream:
        """Enter the ``async with`` block, so a hit needs no special case."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on the way out. Completing is the terminal event's job."""
        await self.close()
