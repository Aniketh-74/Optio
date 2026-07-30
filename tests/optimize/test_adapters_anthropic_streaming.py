"""Streaming: the mode most production callers ship, getting zero stages.

ADR-019's premise, restated because these tests are its enforcement. Both
adapters used to say "streaming is not optimized" on the grounds that a pipeline
built around one request producing one response "can only buffer a token
stream". True of the handful of stages that read a reply, and false of the
majority: ``prefix_cache`` places a breakpoint on the outgoing request and never
looks at the response, and on Anthropic it is the only mechanism by which the
provider caches anything at all.

So the tests here are about two claims:

* every ``before`` hook reaches a streamed call, and the marker lands on the
  wire -- the same assertion the non-streaming suite makes, which is the point;
* the ``after`` hooks run when the stream *finishes*, and never when it is
  abandoned. A ``exact_cache`` that stored half an answer would serve that
  truncation confidently and permanently, which is worse than any missing
  report row.

Real SDK types against a mocked transport, same construction as
``test_adapters_anthropic.py``: SSE bytes go in, the SDK's own ``Stream`` parses
them, and the events this package hands back are validated by the SDK's pydantic
models rather than by a stand-in.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("anthropic")

import httpx
from anthropic import Anthropic, AsyncAnthropic

from optio_optimize import Optimizer
from optio_optimize.adapters.anthropic import wrap_anthropic_client

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.optimize


def _message_events(
    text: str,
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> list[dict[str, Any]]:
    """The event sequence Anthropic sends for one text-only reply."""
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        # Two deltas, not one: the accumulator has to join them, and a single
        # delta would let a bug that keeps only the last chunk pass.
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text[:2]},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text[2:]},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class _FakeAnthropic:
    """Answers streaming and non-streaming requests, recording every body."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = "hello there"

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse(_message_events(self.reply)),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": self.reply}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )


@pytest.fixture
def fake() -> _FakeAnthropic:
    return _FakeAnthropic()


@pytest.fixture
def sync_client(fake: _FakeAnthropic) -> Iterator[Anthropic]:
    transport = httpx.MockTransport(fake.handler)
    yield Anthropic(api_key="test", http_client=httpx.Client(transport=transport))


@pytest.fixture
def async_client(fake: _FakeAnthropic) -> Iterator[AsyncAnthropic]:
    transport = httpx.MockTransport(fake.handler)
    yield AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=transport))


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "temperature": 0.0,  # exact_cache only serves deterministic requests
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    base.update(overrides)
    return base


def _long_chat(turns: int = 40) -> list[dict[str, Any]]:
    """Long enough to clear PrefixCacheStage's minimum cacheable prefix."""
    filler = "policy detail " * 2400
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"q0 {filler}"}]
    for turn in range(turns):
        messages.append({"role": "assistant", "content": f"a{turn} {filler}"})
        messages.append({"role": "user", "content": f"q{turn + 1} {filler}"})
    return messages


def _text_of(events: list[Any]) -> str:
    """Reassemble the reply from whatever events came back."""
    return "".join(
        e.delta.text
        for e in events
        if e.type == "content_block_delta" and getattr(e.delta, "type", "") == "text_delta"
    )


class TestTheEventsReachTheCallerUnchanged:
    def test_a_streamed_call_yields_the_providers_own_events(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(sync_client)

        events = list(sync_client.messages.create(**_kwargs()))

        assert [e.type for e in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert _text_of(events) == "hello there"

    def test_the_stream_still_works_as_a_context_manager(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        # The SDK's own Stream is one, and callers write `with ... as stream:`.
        wrap_anthropic_client(sync_client)

        with sync_client.messages.create(**_kwargs()) as stream:
            events = list(stream)

        assert _text_of(events) == "hello there"

    def test_stream_stays_true_on_the_wire(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(sync_client)

        list(sync_client.messages.create(**_kwargs()))

        assert fake.requests[0]["stream"] is True

    def test_an_async_streamed_call_yields_the_same_events(
        self, async_client: AsyncAnthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(async_client)

        async def go() -> list[Any]:
            stream = await async_client.messages.create(**_kwargs())
            return [event async for event in stream]

        events = asyncio.run(go())

        assert _text_of(events) == "hello there"
        assert [e.type for e in events][-1] == "message_stop"


class TestTheRequestSideStagesReachAStreamedCall:
    def test_the_prefix_marker_lands_on_a_streamed_request(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        """ADR-019's headline. The largest lossless saving in the package was
        reaching exactly zero streaming callers.

        Same prompt shape as the non-streaming suite's equivalent, deliberately:
        a big system prompt above ``MIN_PREFIX_TOKENS`` and a few short turns.
        The breakpoint may land on the system block or on a turn -- either caches
        everything above it -- so both are searched rather than asserting which.
        """
        wrap_anthropic_client(sync_client, prefix_cache=True, exact_cache=False)

        list(
            sync_client.messages.create(
                **_kwargs(
                    system="You are a careful assistant. " * 900,
                    messages=[
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "q2"},
                    ],
                )
            )
        )

        body = fake.requests[0]
        blocks = [
            block
            for block in (
                *body.get("system", ()),
                *(
                    block
                    for message in body["messages"]
                    if isinstance(message.get("content"), list)
                    for block in message["content"]
                ),
            )
            if isinstance(block, dict)
        ]
        marked = [block for block in blocks if "cache_control" in block]
        assert marked, (
            "a streamed request reached the provider with no cache breakpoint, so "
            "prefix_cache contributed nothing -- which is the state ADR-019 exists "
            "to end"
        )
        assert marked[0]["cache_control"] == {"type": "ephemeral"}

    def test_a_transformed_request_is_what_goes_on_the_wire(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        # trim_history is on by default and must shorten a long conversation
        # here exactly as it does on the non-streaming path.
        wrap_anthropic_client(sync_client)
        sent_messages = _long_chat()

        list(sync_client.messages.create(**_kwargs(messages=sent_messages)))

        assert len(fake.requests[0]["messages"]) < len(sent_messages)

    def test_the_disabled_optimizer_passes_a_stream_through(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(sync_client, optimizer=Optimizer(enabled=False))
        sent_messages = _long_chat()

        events = list(sync_client.messages.create(**_kwargs(messages=sent_messages)))

        assert len(fake.requests[0]["messages"]) == len(sent_messages)
        assert _text_of(events) == "hello there"


class TestTheAfterHooksRunWhenTheStreamFinishes:
    def test_a_fully_consumed_stream_is_stored_and_served_to_a_later_call(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(sync_client)

        list(sync_client.messages.create(**_kwargs()))
        reply = sync_client.messages.create(**_kwargs(stream=False))

        assert len(fake.requests) == 1, (
            "the second call reached the provider despite a stored reply"
        )
        assert reply.content[0].text == "hello there"

    def test_an_abandoned_stream_stores_nothing(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        """A half-read reply must never become a cache entry.

        ``exact_cache`` is on by default. A stored truncation is served
        confidently, permanently, to every later caller asking the same
        question -- strictly worse than the report undercounting a stream
        nobody finished reading.
        """
        wrap_anthropic_client(sync_client)

        stream = sync_client.messages.create(**_kwargs())
        next(iter(stream))  # one event, then walk away
        stream.close()

        sync_client.messages.create(**_kwargs(stream=False))

        assert len(fake.requests) == 2, "a partially consumed stream was cached"

    def test_the_streamed_request_reaches_the_savings_report(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        optimizer = Optimizer()
        wrap_anthropic_client(sync_client, optimizer=optimizer)

        list(sync_client.messages.create(**_kwargs(messages=_long_chat())))

        assert optimizer.report.requests == 1
        assert optimizer.report.actual_output_tokens == 20

    def test_completion_happens_once_even_when_the_caller_also_closes(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        """Two paths to completion, one accounting.

        Iterating to the terminal event completes the request; so would treating
        exhaustion or ``close()`` as completion. A caller who does both must not
        be billed twice in the report, and the guard for that is a flag rather
        than a careful reading of the control flow.
        """
        optimizer = Optimizer()
        wrap_anthropic_client(sync_client, optimizer=optimizer)

        stream = sync_client.messages.create(**_kwargs())
        list(stream)
        stream.close()
        list(stream)  # exhausted; yields nothing and must not re-complete

        assert optimizer.report.requests == 1

    def test_an_async_stream_completes_too(
        self, async_client: AsyncAnthropic, fake: _FakeAnthropic
    ) -> None:
        optimizer = Optimizer()
        wrap_anthropic_client(async_client, optimizer=optimizer)

        async def go() -> Any:
            stream = await async_client.messages.create(**_kwargs())
            async for _ in stream:
                pass
            return await async_client.messages.create(**_kwargs(stream=False))

        reply = asyncio.run(go())

        assert len(fake.requests) == 1, "the async streamed reply was never stored"
        assert reply.content[0].text == "hello there"
        assert optimizer.report.requests == 2


class TestACacheHitIsReplayedAsAStream:
    def test_a_stored_reply_is_replayed_without_calling_the_provider(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(sync_client)

        sync_client.messages.create(**_kwargs(stream=False))
        events = list(sync_client.messages.create(**_kwargs()))

        assert len(fake.requests) == 1
        assert _text_of(events) == "hello there"

    def test_the_replay_carries_the_terminal_events_a_caller_waits_for(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        # A caller's handler usually keys off message_stop; a replay missing it
        # hangs a consumer that is otherwise correct.
        wrap_anthropic_client(sync_client)
        sync_client.messages.create(**_kwargs(stream=False))

        types = [e.type for e in sync_client.messages.create(**_kwargs())]

        assert len(fake.requests) == 1, "this asserted the shape of a live stream, not a replay"
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        assert "content_block_stop" in types
        assert "message_delta" in types

    def test_the_replay_works_as_a_context_manager_too(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(sync_client)
        sync_client.messages.create(**_kwargs(stream=False))

        with sync_client.messages.create(**_kwargs()) as stream:
            events = list(stream)

        assert len(fake.requests) == 1, "this asserted the shape of a live stream, not a replay"
        assert _text_of(events) == "hello there"

    def test_a_replayed_hit_does_not_rebill_the_original_calls_usage(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        """The defect the non-streaming path already documents, one path over.

        A cache hit carries the stored response, usage included. Reporting it as
        billed would charge every hit for the call it avoided.
        """
        optimizer = Optimizer()
        wrap_anthropic_client(sync_client, optimizer=optimizer)

        sync_client.messages.create(**_kwargs(stream=False))
        before = optimizer.report.actual_output_tokens
        list(sync_client.messages.create(**_kwargs()))

        assert optimizer.report.actual_output_tokens == before
        assert optimizer.report.short_circuits == 1

    def test_an_async_replay_is_iterable_too(
        self, async_client: AsyncAnthropic, fake: _FakeAnthropic
    ) -> None:
        wrap_anthropic_client(async_client)

        async def go() -> list[Any]:
            await async_client.messages.create(**_kwargs(stream=False))
            stream = await async_client.messages.create(**_kwargs())
            return [event async for event in stream]

        events = asyncio.run(go())

        assert len(fake.requests) == 1
        assert _text_of(events) == "hello there"


class TestFailOpen:
    def test_a_raising_stage_does_not_break_a_streamed_call(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        from optio_optimize.stages.base import Stage, StageResult

        class Exploding(Stage):
            @property
            def name(self) -> str:
                return "exploding"

            def before(self, request: Any, ctx: Any) -> StageResult:
                raise RuntimeError("boom")

        wrap_anthropic_client(sync_client, optimizer=Optimizer(stages=[Exploding()]))

        events = list(sync_client.messages.create(**_kwargs()))

        assert _text_of(events) == "hello there"

    def test_a_raising_after_hook_does_not_break_a_finished_stream(
        self, sync_client: Anthropic, fake: _FakeAnthropic
    ) -> None:
        from optio_optimize.stages.base import Stage, StageResult

        class ExplodingAfter(Stage):
            @property
            def name(self) -> str:
                return "exploding_after"

            def before(self, request: Any, ctx: Any) -> StageResult:
                return self.declines(request)

            def after(self, request: Any, response: Any, ctx: Any) -> None:
                raise RuntimeError("boom")

        wrap_anthropic_client(sync_client, optimizer=Optimizer(stages=[ExplodingAfter()]))

        events = list(sync_client.messages.create(**_kwargs()))

        assert _text_of(events) == "hello there"
