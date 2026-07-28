"""SimulatedProvider's automatic-cache model, calibrated against live traces.

Two live traces (2026-07-28 and 2026-07-29) confirmed OpenAI's automatic
prefix cache reports `cached_tokens` as an exact multiple of 128, never an
arbitrary value past the 1024-token floor -- see
``bench/providers.py``'s ``_AUTO_CACHE_QUANTUM_TOKENS`` docstring for the
raw numbers. These tests pin the simulator to that, not to whatever token
count a message boundary happens to land on.

Also covers ``OpenAIProvider``/``AnthropicProvider`` themselves, against the
real SDK types with the HTTP transport mocked (``httpx.MockTransport``) --
same pattern as ``test_adapters_openai_agents.py``. No network, no key, no
spend, but a genuine ``ChatCompletion``/``Message`` object parsed by the
SDK's own pydantic models, not a hand-built stand-in that could drift from
what the real SDK actually returns.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import httpx
import pytest

from optio_optimize.bench.providers import (
    AnthropicProvider,
    OpenAIProvider,
    SimulatedProvider,
    SpendGuard,
    _actual_cost,
    _estimate_cost,
    available_live_provider,
)
from optio_optimize.types import LLMRequest, LLMResponse, Message

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


def _chat_request(**overrides: Any) -> LLMRequest:
    kwargs: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": (Message(role="user", content="hello"),),
        "temperature": 0.0,
    }
    kwargs.update(overrides)
    return LLMRequest(**kwargs)


pytest.importorskip("openai")


class _FakeOpenAIBackend:
    """Records every request body and answers with a fixed usage/content."""

    def __init__(self, usage: dict[str, Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = "the answer"
        self.usage = (
            usage
            if usage is not None
            else {"prompt_tokens": 60, "completion_tokens": 10, "total_tokens": 70}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1_700_000_000,
                "model": body.get("model", "gpt-4o-mini"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": self.usage,
            },
        )


def _openai_provider(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeOpenAIBackend, **kwargs: Any
) -> OpenAIProvider:
    """A real OpenAIProvider with its client's transport swapped for a mock.

    __init__ needs OPENAI_API_KEY set to construct at all, and reads it via
    the SDK's own mechanism -- a fake value is enough since no real request
    ever leaves the mocked transport.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider(**kwargs)
    from openai import OpenAI

    provider._client = OpenAI(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(backend.handler))
    )
    return provider


class TestOpenAIProviderMockedSDK:
    """Real ``openai.OpenAI`` client, HTTP transport mocked -- no network, no key."""

    def test_a_call_returns_the_providers_content_and_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeOpenAIBackend()
        provider = _openai_provider(monkeypatch, backend)

        response = provider(_chat_request())

        assert response.content == "the answer"
        assert response.input_tokens == 60
        assert response.output_tokens == 10
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4o-mini"

    def test_cached_tokens_are_read_from_prompt_tokens_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeOpenAIBackend(
            usage={
                "prompt_tokens": 1500,
                "completion_tokens": 10,
                "total_tokens": 1510,
                "prompt_tokens_details": {"cached_tokens": 1408},
            }
        )
        provider = _openai_provider(monkeypatch, backend)

        response = provider(_chat_request())

        assert response.cached_input_tokens == 1408

    def test_usage_with_no_prompt_tokens_details_reports_zero_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real API omits this key entirely on a cold prompt -- confirmed
        # against the real SDK's parsed type, which sets it to None rather
        # than an empty object.
        backend = _FakeOpenAIBackend(
            usage={"prompt_tokens": 60, "completion_tokens": 10, "total_tokens": 70}
        )
        provider = _openai_provider(monkeypatch, backend)

        response = provider(_chat_request())

        assert response.cached_input_tokens == 0

    def test_name_field_is_included_only_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeOpenAIBackend()
        provider = _openai_provider(monkeypatch, backend)
        named = Message(role="user", content="hi", name="bob")
        unnamed = Message(role="user", content="hi")

        provider(_chat_request(messages=(named,)))
        provider(_chat_request(messages=(unnamed,)))

        assert backend.requests[0]["messages"][0]["name"] == "bob"
        assert "name" not in backend.requests[1]["messages"][0]

    def test_tool_calls_and_tool_call_id_are_forwarded_from_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Dropping these makes OpenAI reject any "tool" message with no
        # preceding tool_calls -- found live running tool_calling_chat.
        backend = _FakeOpenAIBackend()
        provider = _openai_provider(monkeypatch, backend)
        assistant_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]
        assistant_msg = Message(
            role="assistant", content="", extra={"tool_calls": assistant_calls}
        )
        tool_msg = Message(role="tool", content="result", extra={"tool_call_id": "call_1"})

        provider(_chat_request(messages=(assistant_msg, tool_msg)))

        sent = backend.requests[0]["messages"]
        assert sent[0]["tool_calls"] == assistant_calls
        assert sent[1]["tool_call_id"] == "call_1"

    def test_max_tokens_temperature_and_response_format_are_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeOpenAIBackend()
        provider = _openai_provider(monkeypatch, backend)

        provider(
            _chat_request(
                max_tokens=256, temperature=0.7, response_format={"type": "json_object"}
            )
        )

        sent = backend.requests[0]
        assert sent["max_completion_tokens"] == 256
        assert sent["temperature"] == 0.7
        assert sent["response_format"] == {"type": "json_object"}

    def test_the_spend_guard_records_actual_cost_after_a_successful_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeOpenAIBackend()
        guard = SpendGuard(cap_usd=1.0)
        provider = _openai_provider(monkeypatch, backend, guard=guard)

        provider(_chat_request())

        assert guard.calls == 1
        assert guard.spent_usd > 0

    def test_the_spend_guard_blocks_the_call_when_the_estimate_exceeds_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeOpenAIBackend()
        guard = SpendGuard(cap_usd=0.000001)
        provider = _openai_provider(monkeypatch, backend, guard=guard)

        with pytest.raises(RuntimeError, match="spend cap reached"):
            provider(_chat_request())

        assert not backend.requests, "the guard must stop the call before it is sent"

    def test_missing_api_key_raises_a_clear_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIProvider()

    def test_declares_itself_live_with_modelled_latency_and_a_reportable_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIProvider(model="gpt-4o-mini")

        assert provider.is_live is True
        assert provider.models_latency is True
        assert provider.label == "openai(gpt-4o-mini)"
        provider.reset()  # No-op; must not raise.


pytest.importorskip("anthropic")


class _FakeAnthropicBackend:
    """Records every request body and answers with a fixed usage/content."""

    def __init__(self, cache_read_input_tokens: int = 0) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = "the answer"
        self.cache_read_input_tokens = cache_read_input_tokens

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "claude-haiku-4"),
                "content": [{"type": "text", "text": self.reply}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 8,
                    "cache_read_input_tokens": self.cache_read_input_tokens,
                },
            },
        )


def _anthropic_provider(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeAnthropicBackend, **kwargs: Any
) -> AnthropicProvider:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider(**kwargs)
    from anthropic import Anthropic

    provider._client = Anthropic(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(backend.handler))
    )
    return provider


class TestAnthropicProviderMockedSDK:
    """Real ``anthropic.Anthropic`` client, HTTP transport mocked -- no network, no key."""

    def test_a_call_returns_the_providers_content_and_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeAnthropicBackend()
        provider = _anthropic_provider(monkeypatch, backend)

        response = provider(_chat_request(model="claude-haiku-4"))

        assert response.content == "the answer"
        assert response.output_tokens == 8
        assert response.finish_reason == "end_turn"
        assert response.model == "claude-haiku-4"

    def test_cache_read_tokens_are_added_into_billable_input_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # input_tokens on the wire excludes what was served from cache; this
        # library's LLMResponse.input_tokens is the full prompt, with
        # cached_input_tokens tracked separately -- unlike OpenAI, where
        # prompt_tokens already includes the cached count.
        backend = _FakeAnthropicBackend(cache_read_input_tokens=15)
        provider = _anthropic_provider(monkeypatch, backend)

        response = provider(_chat_request(model="claude-haiku-4"))

        assert response.cached_input_tokens == 15
        assert response.input_tokens == 40 + 15
        assert response.billable_input_tokens == 40

    def test_a_cacheable_system_message_gets_the_cache_control_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeAnthropicBackend()
        provider = _anthropic_provider(monkeypatch, backend)
        system = Message(role="system", content="be terse", cacheable=True)
        user = Message(role="user", content="hi")

        provider(_chat_request(model="claude-haiku-4", messages=(system, user)))

        sent_system = backend.requests[0]["system"]
        assert sent_system[0]["cache_control"] == {"type": "ephemeral"}

    def test_a_non_cacheable_system_message_carries_no_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeAnthropicBackend()
        provider = _anthropic_provider(monkeypatch, backend)
        system = Message(role="system", content="be terse", cacheable=False)
        user = Message(role="user", content="hi")

        provider(_chat_request(model="claude-haiku-4", messages=(system, user)))

        sent_system = backend.requests[0]["system"]
        assert "cache_control" not in sent_system[0]

    def test_non_system_messages_become_turns_not_system_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeAnthropicBackend()
        provider = _anthropic_provider(monkeypatch, backend)
        messages = (
            Message(role="system", content="be terse"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        )

        provider(_chat_request(model="claude-haiku-4", messages=messages))

        sent = backend.requests[0]
        assert [m["role"] for m in sent["messages"]] == ["user", "assistant"]

    def test_temperature_none_defaults_to_one_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # None means "the caller never set it" -- defaulting it to 0 would
        # silently make every unset-temperature request deterministic.
        backend = _FakeAnthropicBackend()
        provider = _anthropic_provider(monkeypatch, backend)

        provider(_chat_request(model="claude-haiku-4", temperature=None))

        assert backend.requests[0]["temperature"] == 1.0

    def test_max_tokens_none_defaults_to_1024(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeAnthropicBackend()
        provider = _anthropic_provider(monkeypatch, backend)

        provider(_chat_request(model="claude-haiku-4", max_tokens=None))

        assert backend.requests[0]["max_tokens"] == 1024

    def test_the_spend_guard_records_actual_cost_after_a_successful_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _FakeAnthropicBackend()
        guard = SpendGuard(cap_usd=1.0)
        provider = _anthropic_provider(monkeypatch, backend, guard=guard)

        provider(_chat_request(model="claude-haiku-4"))

        assert guard.calls == 1
        assert guard.spent_usd > 0

    def test_missing_api_key_raises_a_clear_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider()

    def test_declares_itself_live_with_modelled_latency_and_a_reportable_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        provider = AnthropicProvider(model="claude-haiku-4")

        assert provider.is_live is True
        assert provider.models_latency is True
        assert provider.label == "anthropic(claude-haiku-4)"
        provider.reset()  # No-op; must not raise.


class TestCostHelpers:
    """`_estimate_cost`/`_actual_cost`: the two functions the spend guard trusts."""

    def test_estimate_cost_is_positive_for_a_priced_model(self) -> None:
        estimate = _estimate_cost(_chat_request(model="gpt-4o-mini"), "gpt-4o-mini")

        assert estimate > 0

    def test_estimate_cost_is_zero_for_an_unpriced_model(self) -> None:
        estimate = _estimate_cost(_chat_request(model="some-future-model"), "some-future-model")

        assert estimate == 0.0

    def test_actual_cost_is_zero_for_an_unpriced_model(self) -> None:
        response = LLMResponse(content="x", input_tokens=100, output_tokens=10)

        assert _actual_cost(response, "some-future-model") == 0.0

    def test_actual_cost_reflects_cached_tokens_at_a_discount(self) -> None:
        # A response with some of its input served from cache must cost less
        # than the same token counts with nothing cached -- otherwise the
        # cache discount this whole benchmark suite exists to measure would
        # not show up in the one number the spend guard actually tracks.
        full_price = LLMResponse(content="x", input_tokens=2000, output_tokens=100)
        with_cache = LLMResponse(
            content="x", input_tokens=2000, output_tokens=100, cached_input_tokens=1500
        )

        assert _actual_cost(with_cache, "gpt-4o-mini") < _actual_cost(full_price, "gpt-4o-mini")


class TestAvailableLiveProvider:
    def test_returns_none_when_neither_key_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        assert available_live_provider() is None

    def test_prefers_anthropic_when_both_keys_are_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Anthropic is the only provider whose prefix caching this library
        # controls explicitly, so it measures strictly more than the others.
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        provider = available_live_provider()

        assert isinstance(provider, AnthropicProvider)

    def test_falls_back_to_openai_when_only_its_key_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        provider = available_live_provider()

        assert isinstance(provider, OpenAIProvider)
