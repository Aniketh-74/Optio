"""wrap_openai_client against the real openai SDK types, not stand-ins.

Every test here builds a genuine ``openai.AsyncOpenAI`` client with its HTTP
transport mocked (``httpx.MockTransport``) rather than any part of the client
itself -- so the request this package builds and the response it parses are
real ``ChatCompletionMessageParam``/``ChatCompletion`` shapes, validated by
the SDK's own pydantic models, not by an assumption about what they look
like. No network, no API key, no real spend.

Two real bugs surfaced building this adapter, the same way the live
benchmark surfaced defects in earlier work -- neither would have been found
by hand-written test kwargs, only by driving real SDK code paths:

* ``TestFailOpen``'s cache-honesty test: the first version returned a cached
  response's *original* ``ChatCompletion``, usage numbers included, so a
  cache hit that cost nothing looked -- to anyone reading ``response.usage``
  -- exactly as expensive as the call that filled the cache. Fixed by gating
  on ``LLMResponse.served_from`` rather than merely "is there a native
  object available."
* ``TestOmitSentinelNormalization``: the Agents SDK's ``OpenAIChatCompletionsModel``
  represents "the caller never set this" as a falsy sentinel object
  (``openai.Omit``), not ``None`` -- found only by calling the real
  ``get_response()`` with its default ``ModelSettings()``, which never sets
  temperature or max_tokens explicitly. A bare ``kwargs.get(...)`` handed
  this package a live sentinel instead of ``None``, and ``is not None``
  checks downstream (``AdaptiveMaxTokensStage``'s "caller already capped
  this") read it as *present*.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("openai")

import httpx
from openai import NOT_GIVEN, AsyncOpenAI

from optio_optimize.adapters.openai_agents import wrap_openai_client

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.optimize

#: Turns the size of real ones. A two-token turn is not a conversation worth
#: trimming, and since ADR-026 the stage correctly declines to trim one.
_TURN_PADDING = " ".join(f"context{n}" for n in range(120))


class _FakeOpenAI:
    """Records every request body it receives and answers deterministically."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = "hello there"

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-real",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": body.get("model", "gpt-4o"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 5,
                    "total_tokens": 55,
                },
            },
        )


@pytest.fixture
def fake_openai() -> _FakeOpenAI:
    return _FakeOpenAI()


@pytest.fixture
def client(fake_openai: _FakeOpenAI) -> Iterator[AsyncOpenAI]:
    transport = httpx.MockTransport(fake_openai.handler)
    async_client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
    yield async_client


def _basic_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "you are terse"},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.0,
    }
    kwargs.update(overrides)
    return kwargs


class TestBasicRoundTrip:
    async def _create(self, client: AsyncOpenAI, **kwargs: Any) -> Any:
        return await client.chat.completions.create(**kwargs)

    def test_a_real_call_returns_the_providers_own_content(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        import asyncio

        wrap_openai_client(client, exact_cache=False, prefix_cache=False)

        response = asyncio.run(self._create(client, **_basic_kwargs()))

        assert response.choices[0].message.content == "hello there"
        assert len(fake_openai.requests) == 1

    def test_the_real_request_body_carries_the_right_messages(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        import asyncio

        wrap_openai_client(client, exact_cache=False, prefix_cache=False)

        asyncio.run(self._create(client, **_basic_kwargs()))

        sent = fake_openai.requests[0]
        assert sent["messages"][0] == {"role": "system", "content": "you are terse"}
        assert sent["messages"][1] == {"role": "user", "content": "hello"}

    def test_extra_kwargs_survive_untouched(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        """seed, top_p, etc. -- anything this package does not model."""
        import asyncio

        wrap_openai_client(client, exact_cache=False, prefix_cache=False)

        asyncio.run(self._create(client, **_basic_kwargs(seed=42, top_p=0.9)))

        sent = fake_openai.requests[0]
        assert sent["seed"] == 42
        assert sent["top_p"] == 0.9


class TestCacheHitsReportTheirRealCost:
    def test_a_second_identical_call_makes_no_real_request(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        import asyncio

        wrap_openai_client(client, exact_cache=True, prefix_cache=False)
        kwargs = _basic_kwargs()

        asyncio.run(client.chat.completions.create(**kwargs))
        asyncio.run(client.chat.completions.create(**kwargs))

        assert len(fake_openai.requests) == 1

    def test_the_cache_hits_usage_is_zeroed_not_the_original_calls(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        """The bug this adapter shipped with once: a cache hit must never
        report the tokens the *first* call actually billed."""
        import asyncio

        wrap_openai_client(client, exact_cache=True, prefix_cache=False)
        kwargs = _basic_kwargs()

        first = asyncio.run(client.chat.completions.create(**kwargs))
        second = asyncio.run(client.chat.completions.create(**kwargs))

        assert first.usage.prompt_tokens == 50
        assert second.usage.prompt_tokens == 0
        assert second.usage.completion_tokens == 0
        assert second.id != first.id

    def test_the_cache_hits_content_still_matches(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        import asyncio

        wrap_openai_client(client, exact_cache=True, prefix_cache=False)
        kwargs = _basic_kwargs()

        first = asyncio.run(client.chat.completions.create(**kwargs))
        second = asyncio.run(client.chat.completions.create(**kwargs))

        assert second.choices[0].message.content == first.choices[0].message.content


class TestStreamingBypassesOptimizationEntirely:
    def test_a_streaming_call_reaches_the_real_client_unmodified(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        import asyncio

        # A malformed "messages" that would break request translation --
        # proving the stream path never enters this package's code at all,
        # not just that it happens to tolerate this input.
        wrap_openai_client(client, exact_cache=True)
        fake_openai.handler = lambda request: httpx.Response(  # type: ignore[method-assign]
            200,
            content=(
                b'data: {"id":"x","object":"chat.completion.chunk","created":1,'
                b'"model":"gpt-4o","choices":[]}\n\ndata: [DONE]\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

        async def run() -> list[object]:
            stream = await client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
            )
            return [chunk async for chunk in stream]

        # Must not raise -- proves the wrapper detected stream=True and
        # handed off before touching request translation at all.
        asyncio.run(run())


class TestFailOpen:
    def test_untranslatable_kwargs_fall_back_to_the_real_client(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        """A request-translation bug must degrade to the unoptimized call,
        not raise -- the same guarantee every stage in this package makes,
        extended to this wrapper's own kwargs-to-LLMRequest step, which runs
        outside Pipeline's guard.
        """
        import asyncio

        wrap_openai_client(client, exact_cache=False)

        async def run() -> Any:
            # "messages" containing something with no .get -- forces
            # _message_from_param to raise inside _request_from_kwargs.
            return await client.chat.completions.create(
                model="gpt-4o",
                messages=["not-a-dict"],  # type: ignore[list-item]
                temperature=0.0,
            )

        response = asyncio.run(run())

        assert response.choices[0].message.content == "hello there"
        assert len(fake_openai.requests) == 1


class TestToolCallsRoundTripUntouched:
    def test_tool_calls_and_tool_call_id_reach_the_real_request(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        """Fields this package does not model must survive verbatim when no
        stage changed the message's text -- the ``_raw`` passthrough.
        """
        import asyncio

        wrap_openai_client(client, exact_cache=False, prefix_cache=False)
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fetch", "arguments": "{}"},
            }
        ]
        kwargs = _basic_kwargs(
            messages=[
                {"role": "user", "content": "look it up"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {"role": "tool", "content": "result", "tool_call_id": "call_1"},
            ]
        )

        asyncio.run(client.chat.completions.create(**kwargs))

        sent_messages = fake_openai.requests[0]["messages"]
        assert sent_messages[1]["tool_calls"] == tool_calls
        assert sent_messages[2]["tool_call_id"] == "call_1"


class TestTrimHistoryShrinksTheRealRequest:
    def test_a_long_conversation_sends_fewer_messages_over_the_wire(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        import asyncio

        wrap_openai_client(
            client,
            exact_cache=False,
            prefix_cache=False,
            trim_history=True,
            recent_turns=4,
        )
        messages = [{"role": "system", "content": "sys"}]
        for turn in range(10):
            messages.append({"role": "user", "content": f"q{turn}"})
            messages.append({"role": "assistant", "content": f"a{turn} {_TURN_PADDING}"})
        messages.append({"role": "user", "content": "final question"})

        asyncio.run(client.chat.completions.create(**_basic_kwargs(messages=messages)))

        sent = fake_openai.requests[0]["messages"]
        assert len(sent) < len(messages)
        # system + the anchored opening question + an elision marker + window.
        assert len(sent) == 1 + 1 + 1 + 4
        assert sent[1]["content"] == "q0", "the opening question is the task; it is never history"
        assert sent[-1]["content"] == "final question"


class TestOmitSentinelNormalization:
    """``openai.Omit``/``NOT_GIVEN`` must normalize to ``None``, not survive
    as a live sentinel object in ``LLMRequest`` fields typed ``X | None``.
    """

    def test_temperature_and_bare_dict_get_normalizes_correctly(self) -> None:
        from optio_optimize.adapters.openai_agents import _dict_or_none, _numeric_or_none

        assert _numeric_or_none(NOT_GIVEN) is None
        assert _numeric_or_none(None) is None
        assert _numeric_or_none(True) is None  # bool is an int subclass; never a real value
        assert _numeric_or_none(0.0) == 0.0  # a real, meaningful value -- must survive
        assert _numeric_or_none(256) == 256

        assert _dict_or_none(NOT_GIVEN) is None
        assert _dict_or_none({"type": "json_object"}) == {"type": "json_object"}

    def test_an_omitted_temperature_is_not_mistaken_for_deterministic(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        """temperature=NOT_GIVEN must not equal 0.0's caching behaviour."""
        import asyncio

        wrap_openai_client(client, exact_cache=True)
        kwargs = _basic_kwargs(temperature=NOT_GIVEN)

        asyncio.run(client.chat.completions.create(**kwargs))
        asyncio.run(client.chat.completions.create(**kwargs))

        # Not deterministic (temperature was never really set to 0), so the
        # exact cache must not treat the two calls as safely interchangeable.
        assert len(fake_openai.requests) == 2

    def test_an_omitted_max_tokens_still_lets_the_adaptive_stage_learn(
        self, client: AsyncOpenAI, fake_openai: _FakeOpenAI
    ) -> None:
        """The regression this bug actually caused: AdaptiveMaxTokensStage's
        "the caller already capped this" check must not fire on a sentinel.
        """
        from optio_optimize.adapters.openai_agents import _request_from_kwargs

        kwargs = _basic_kwargs(max_tokens=NOT_GIVEN)
        request = _request_from_kwargs(kwargs)

        assert request.max_tokens is None, (
            "a NOT_GIVEN max_tokens must read as unset, or AdaptiveMaxTokensStage "
            "will decline forever, believing the caller already set a ceiling"
        )
