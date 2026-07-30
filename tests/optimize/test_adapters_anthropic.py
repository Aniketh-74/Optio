"""wrap_anthropic_client against the real anthropic SDK types, not stand-ins.

Same construction as ``test_adapters_openai_agents.py``: a genuine client with
its HTTP transport mocked, so the request this package builds and the response
it parses are real SDK shapes validated by the SDK's own pydantic models. No
network, no API key, no spend.

``asyncio.run`` rather than a pytest-asyncio plugin, matching
``test_pipeline_async.py`` -- nothing here needs an event-loop fixture, and a
plugin would be a dev dependency bought for syntax.

This adapter matters more than its size suggests. ``PrefixCacheStage`` is
described in its own source as the largest lossless saving in the package, and
on OpenAI it contributes exactly zero -- automatic caching lands on both A/B
arms, which is why a simulated 36.3% corrected to -1.8% live. Anthropic caches
nothing without an explicit ``cache_control`` breakpoint, and this adapter is
the only path by which the marker our pipeline places becomes that field.
"""

from __future__ import annotations

import asyncio
import contextlib
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


class _FakeAnthropic:
    """Records every request body and answers deterministically."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = "hello there"
        self.cache_read = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
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
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": self.cache_read,
                },
            },
        )


@pytest.fixture
def fake() -> _FakeAnthropic:
    return _FakeAnthropic()


@pytest.fixture
def async_client(fake: _FakeAnthropic) -> Iterator[AsyncAnthropic]:
    transport = httpx.MockTransport(fake.handler)
    yield AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=transport))


@pytest.fixture
def sync_client(fake: _FakeAnthropic) -> Iterator[Anthropic]:
    transport = httpx.MockTransport(fake.handler)
    yield Anthropic(api_key="test", http_client=httpx.Client(transport=transport))


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return base


def _content_blocks(messages: list[dict[str, Any]]) -> list[Any]:
    """Every content block across a turn list, skipping plain-string contents."""
    blocks: list[Any] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            blocks.extend(content)
    return blocks


def _growing_chat(turns: int = 10) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "q0"}]
    for turn in range(turns):
        messages.append({"role": "assistant", "content": f"a{turn}"})
        messages.append({"role": "user", "content": f"q{turn + 1}"})
    return messages


class TestItStillWorks:
    def test_a_real_call_returns_the_providers_own_content(self, async_client, fake):
        wrap_anthropic_client(async_client)
        reply = asyncio.run(async_client.messages.create(**_kwargs()))
        assert reply.content[0].text == "hello there"
        assert reply.usage.input_tokens == 100

    def test_the_system_prompt_reaches_the_wire_as_a_block(self, async_client, fake):
        wrap_anthropic_client(async_client)
        asyncio.run(async_client.messages.create(**_kwargs(system="You are terse.")))
        assert fake.requests[0]["system"] == [{"type": "text", "text": "You are terse."}]

    def test_unmodelled_kwargs_survive_untouched(self, async_client, fake):
        wrap_anthropic_client(async_client)
        asyncio.run(async_client.messages.create(**_kwargs(top_p=0.5, metadata={"user_id": "u1"})))
        body = fake.requests[0]
        assert body["top_p"] == 0.5
        assert body["metadata"] == {"user_id": "u1"}


class TestThePrefixMarkerBecomesCacheControl:
    def test_a_long_conversation_gets_a_cache_control_breakpoint(self, async_client, fake):
        # The entire reason this adapter exists. PrefixCacheStage only places a
        # marker above MIN_PREFIX_TOKENS (1024), so the prompt must be big.
        #
        # The breakpoint lands on the last message of the stable prefix, which
        # here is a *turn* rather than the system block -- and a breakpoint on a
        # turn caches everything above it, system prompt included. Asserting
        # "the system block carries it" would have been asserting my assumption
        # rather than the stage's contract.
        wrap_anthropic_client(async_client, prefix_cache=True, exact_cache=False)
        asyncio.run(
            async_client.messages.create(
                **_kwargs(
                    system="You are a careful assistant. " * 400,
                    messages=[
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "q2"},
                    ],
                )
            )
        )
        body = fake.requests[0]
        marked = [
            block
            for block in (*body["system"], *_content_blocks(body["messages"]))
            if isinstance(block, dict) and "cache_control" in block
        ]
        assert marked, "the marker our pipeline placed never reached the wire"
        assert marked[0]["cache_control"] == {"type": "ephemeral"}

    def test_the_breakpoint_holds_back_the_newest_turn(self, async_client, fake):
        # Marking right up to the newest message would invalidate the cached
        # prefix on the very next call -- the classic way to get zero benefit
        # while believing the feature is on.
        wrap_anthropic_client(async_client, prefix_cache=True, exact_cache=False)
        asyncio.run(
            async_client.messages.create(
                **_kwargs(
                    system="You are a careful assistant. " * 400,
                    messages=[
                        {"role": "user", "content": "q1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "q2"},
                    ],
                )
            )
        )
        last = fake.requests[0]["messages"][-1]
        assert last["content"] == "q2", "the newest turn was marked and will invalidate itself"

    def test_a_short_prompt_gets_no_breakpoint(self, async_client, fake):
        # Below the provider's floor a marker is ignored, so placing one would
        # show up in reports as work done for no effect.
        wrap_anthropic_client(async_client, prefix_cache=True, exact_cache=False)
        asyncio.run(async_client.messages.create(**_kwargs(system="short")))
        assert "cache_control" not in fake.requests[0]["system"][0]


class TestStagesReachTheWire:
    def test_a_long_conversation_sends_fewer_messages(self, async_client, fake):
        wrap_anthropic_client(
            async_client,
            exact_cache=False,
            prefix_cache=False,
            trim_history=True,
            recent_turns=4,
        )
        messages = _growing_chat()
        asyncio.run(async_client.messages.create(**_kwargs(messages=messages)))

        sent = fake.requests[0]["messages"]
        assert len(sent) < len(messages)
        assert sent[0]["content"] == "q0", "the opening question is the task; never history"

    def test_tools_are_translated_into_anthropic_shape(self, async_client, fake):
        wrap_anthropic_client(async_client, exact_cache=False)
        asyncio.run(
            async_client.messages.create(
                **_kwargs(
                    tools=[
                        {"name": "search", "description": "d", "input_schema": {"type": "object"}}
                    ]
                )
            )
        )
        assert fake.requests[0]["tools"][0]["name"] == "search"


class TestCacheHonesty:
    def test_a_second_identical_call_makes_no_real_request(self, async_client, fake):
        wrap_anthropic_client(async_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        asyncio.run(async_client.messages.create(**kwargs))
        asyncio.run(async_client.messages.create(**kwargs))
        assert len(fake.requests) == 1

    def test_the_cache_hits_usage_is_zeroed_not_the_originals(self, async_client, fake):
        # Returning the stored object would re-bill the original call's usage on
        # every hit, making a cache that saves money look like one that spends
        # it repeatedly. The OpenAI adapter documents the same defect.
        wrap_anthropic_client(async_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        first = asyncio.run(async_client.messages.create(**kwargs))
        second = asyncio.run(async_client.messages.create(**kwargs))
        assert first.usage.input_tokens == 100
        assert second.usage.input_tokens == 0
        assert second.content[0].text == first.content[0].text


class TestFailOpen:
    def test_an_unmodellable_request_still_reaches_the_provider(self, async_client, fake):
        # Fail-open (ADR-013 rule 1) promises the call still happens -- not that
        # *some* exception occurs, which `pytest.raises(Exception)` would accept
        # from any bug anywhere. Assert the promise: content this package does
        # not model (a block list rather than a string) goes out and the caller
        # gets the provider's own answer.
        optimizer = Optimizer()
        wrap_anthropic_client(async_client, optimizer=optimizer)

        reply = asyncio.run(
            async_client.messages.create(
                **_kwargs(messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
            )
        )

        assert reply.content[0].text == "hello there"
        assert len(fake.requests) == 1

    def test_a_streaming_call_bypasses_the_wrapper(self, async_client, fake):
        # Wrapped exactly once. Wrapping twice would satisfy the assertion
        # below because the outer wrapper never saw the call -- a different
        # fact from the one under test.
        optimizer = Optimizer()
        wrap_anthropic_client(async_client, optimizer=optimizer)

        # The mock transport returns JSON rather than SSE, so the SDK's stream
        # parser will object. Beside the point: what matters is that the
        # request reached the real client and that no stage ran.
        with contextlib.suppress(Exception):
            asyncio.run(async_client.messages.create(**_kwargs(stream=True)))

        assert len(fake.requests) == 1, "the streaming call never reached the client"
        assert fake.requests[0]["stream"] is True
        assert optimizer.report.requests == 0, "a streaming call went through the pipeline"


class TestTheSyncClient:
    def test_a_real_call_returns_the_providers_own_content(self, sync_client, fake):
        wrap_anthropic_client(sync_client)
        reply = sync_client.messages.create(**_kwargs())
        assert reply.content[0].text == "hello there"

    def test_stages_reach_the_wire(self, sync_client, fake):
        wrap_anthropic_client(
            sync_client,
            exact_cache=False,
            prefix_cache=False,
            trim_history=True,
            recent_turns=4,
        )
        messages = _growing_chat()
        sync_client.messages.create(**_kwargs(messages=messages))

        sent = fake.requests[0]["messages"]
        assert len(sent) < len(messages)
        assert sent[0]["content"] == "q0"

    def test_a_second_identical_call_makes_no_real_request(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        sync_client.messages.create(**kwargs)
        sync_client.messages.create(**kwargs)
        assert len(fake.requests) == 1

    def test_the_cache_hit_is_a_valid_sdk_object(self, sync_client, fake):
        # The reconstructed object must satisfy the SDK's own model, or a
        # caller reading .usage or .stop_reason on a cache hit gets an
        # AttributeError instead of an answer.
        wrap_anthropic_client(sync_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        sync_client.messages.create(**kwargs)
        hit = sync_client.messages.create(**kwargs)
        assert hit.stop_reason == "end_turn"
        assert hit.usage.output_tokens == 0
        assert hit.content[0].text == "hello there"

    def test_a_streaming_call_bypasses_the_wrapper(self, sync_client, fake):
        optimizer = Optimizer()
        wrap_anthropic_client(sync_client, optimizer=optimizer)
        with contextlib.suppress(Exception):
            sync_client.messages.create(**_kwargs(stream=True))
        assert len(fake.requests) == 1
        assert optimizer.report.requests == 0


class TestBothClientsAgree:
    def test_the_same_request_produces_the_same_wire_body(self):
        # Two branches in one wrapper is two chances to diverge, and a
        # divergence would show up as sync and async callers being optimized
        # differently for reasons nobody could see. The async branch was
        # entirely dead until the coroutine detection was fixed, and every
        # test still passed -- so "both branches work" needs asserting, not
        # assuming.
        sync_fake, async_fake = _FakeAnthropic(), _FakeAnthropic()
        sync = Anthropic(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(sync_fake.handler)),
        )
        asynchronous = AsyncAnthropic(
            api_key="test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(async_fake.handler)),
        )
        wrap_anthropic_client(sync, exact_cache=False)
        wrap_anthropic_client(asynchronous, exact_cache=False)

        kwargs = _kwargs(system="You are terse.", stop_sequences=["<END>"])
        sync.messages.create(**kwargs)
        asyncio.run(asynchronous.messages.create(**kwargs))

        assert sync_fake.requests[0] == async_fake.requests[0]

    def test_both_branches_are_actually_taken(self):
        # Guards the defect directly: if _is_async regressed, both clients would
        # take the sync branch and the test above would still pass, because both
        # bodies would be built by the same code.
        from optio_optimize.adapters.anthropic import _is_async

        sync = Anthropic(api_key="test", http_client=httpx.Client())
        asynchronous = AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient())
        assert _is_async(asynchronous.messages.create) is True
        assert _is_async(sync.messages.create) is False


class TestBlockContentSurvivesTheRoundTrip:
    """Every one of these passed before the defect they cover was fixed.

    The adapter modelled block content as the empty string, so
    ``_param_from_message``'s "did a stage change the text?" check compared a
    list against ``""`` -- never equal -- and rebuilt every block-shaped message
    as ``{"role": ..., "content": ""}``. That is every tool-using Anthropic
    conversation, and the full suite of 1,366 tests did not have one.
    """

    def test_a_tool_result_turn_reaches_the_wire_intact(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=False)
        blocks = [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42 orders"}]
        sync_client.messages.create(
            **_kwargs(
                messages=[
                    {"role": "user", "content": "how many orders?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "toolu_1", "name": "count", "input": {}}
                        ],
                    },
                    {"role": "user", "content": blocks},
                ]
            )
        )
        sent = fake.requests[0]["messages"]
        assert sent[1]["content"] == [
            {"type": "tool_use", "id": "toolu_1", "name": "count", "input": {}}
        ]
        assert sent[2]["content"] == blocks

    def test_an_image_block_is_not_flattened(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=False)
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        }
        blocks = [image, {"type": "text", "text": "?"}]
        sync_client.messages.create(**_kwargs(messages=[{"role": "user", "content": blocks}]))
        assert fake.requests[0]["messages"][0]["content"] == blocks

    def test_text_is_still_readable_by_stages(self):
        """A stage must see the prose, or every text-rewriting stage no-ops."""
        from optio_optimize.adapters.anthropic import _message_from_param

        message = _message_from_param(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look. "},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                    {"type": "text", "text": "One moment."},
                ],
            }
        )
        assert message.content == "Let me look. One moment."

    def test_a_rewritten_single_text_block_keeps_its_siblings(self):
        """The common assistant shape: one text block beside a tool_use."""
        from optio_optimize.adapters.anthropic import _message_from_param, _param_from_message

        tool_use = {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}
        message = _message_from_param(
            {"role": "assistant", "content": [{"type": "text", "text": "long"}, tool_use]}
        )
        rewritten = _param_from_message(message.with_content("short"))
        assert rewritten["content"] == [{"type": "text", "text": "short"}, tool_use]

    def test_ambiguous_content_passes_through_rather_than_guessing(self):
        """Two text blocks: no way to know which a stage meant. Fail open."""
        from optio_optimize.adapters.anthropic import _message_from_param, _param_from_message

        original = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        message = _message_from_param({"role": "user", "content": original})
        assert _param_from_message(message.with_content("z"))["content"] == original

    def test_an_untouched_turn_is_the_very_same_object(self):
        """Identity, not equality: nothing was rebuilt, so nothing can be lost."""
        from optio_optimize.adapters.anthropic import _message_from_param, _param_from_message

        raw = {"role": "user", "content": [{"type": "text", "text": "hi"}], "custom": "kept"}
        assert _param_from_message(_message_from_param(raw)) is raw


class TestTheCallersOwnCacheControlIsNotClobbered:
    """A cost-reduction library must never cause a cost regression."""

    def test_a_one_hour_ttl_on_a_system_block_survives(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=False)
        system = [
            {
                "type": "text",
                "text": "You are terse." * 400,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]
        sync_client.messages.create(**_kwargs(system=system, messages=_growing_chat(12)))
        # The caller paid Anthropic's 2x write premium for an hour of cache.
        # Replacing it with the 5-minute default, or with a bare text block,
        # spends their money and then discards what it bought.
        assert fake.requests[0]["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_a_plain_system_string_still_gets_a_marker(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=False)
        # One turn on purpose. With a longer conversation the stable prefix ends
        # on a *turn* and the system block correctly carries no marker -- the
        # first version of this test used 12 turns and failed for that reason,
        # asserting the stage's behaviour rather than this function's.
        sync_client.messages.create(**_kwargs(system="You are terse. " * 400))
        assert fake.requests[0]["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_an_unmodelled_system_block_field_is_preserved(self):
        from optio_optimize.adapters.anthropic import _system_block_from_message
        from optio_optimize.types import Message

        block = {"type": "text", "text": "sys", "citations": [{"type": "char_location"}]}
        rebuilt = _system_block_from_message(
            Message(role="system", content="sys", extra={"_raw": block})
        )
        assert rebuilt["citations"] == [{"type": "char_location"}]


class TestTheMarkerReachesNonDictBlocks:
    """Appending a response's own content to the history is what the SDK's
    examples do, and it yields pydantic blocks rather than dicts. The marker
    used to fall through silently for exactly that shape -- the third time this
    package placed a field it never sent."""

    def test_a_pydantic_text_block_can_carry_a_breakpoint(self):
        from anthropic.types import TextBlock

        from optio_optimize.adapters.anthropic import _with_cache_control

        blocks = _with_cache_control([TextBlock(type="text", text="hello")], "hello")
        assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
        assert blocks[-1]["text"] == "hello"

    def test_an_uncoercible_block_warns_and_keeps_the_content(self, caplog):
        from optio_optimize.adapters.anthropic import _with_cache_control

        opaque = object()
        with caplog.at_level("WARNING", logger="optio_optimize"):
            blocks = _with_cache_control([opaque], "text")
        assert blocks == [opaque]  # content intact; only the discount is lost
        assert "cache breakpoint" in caplog.text


class TestReasoningBudgetReachesTheProvider:
    """`wire` being correct does not prove the adapter sends it.

    The adapter builds `create(**kwargs)` itself rather than going through
    `anthropic_body`, which is exactly how `cacheable` came to be emitted by
    `wire` and dropped by the adapter. Two translation sites, one field.
    """

    def test_a_caller_set_budget_is_forwarded(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=False)
        sync_client.messages.create(**_kwargs(thinking={"type": "enabled", "budget_tokens": 2048}))
        assert fake.requests[0]["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_no_thinking_key_when_the_caller_set_none(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=False)
        sync_client.messages.create(**_kwargs())
        assert "thinking" not in fake.requests[0]

    def test_a_budget_a_stage_set_on_the_request_reaches_the_wire(self):
        """The case that actually matters, and the one that passes for the
        wrong reason if you only test the caller's own kwarg.

        A caller-supplied `thinking` survives because `_kwargs_from_request`
        starts from `dict(original)` and preserves everything unmodelled -- so
        that test passes without the adapter understanding the field at all.
        A budget set by a *stage* lives on the LLMRequest, and nothing carries
        it over unless this function does. That asymmetry is precisely how
        `cacheable` reached the wire from `wire` and not from here.
        """
        from optio_optimize.adapters.anthropic import _kwargs_from_request
        from optio_optimize.types import LLMRequest, Message

        sent = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="hi"),),
            max_tokens=256,
            thinking_budget=1_024,
        )
        kwargs = _kwargs_from_request(sent, _kwargs())
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1_024}

    def test_a_stage_that_lowers_a_budget_overrides_the_callers(self):
        from optio_optimize.adapters.anthropic import _kwargs_from_request
        from optio_optimize.types import LLMRequest, Message

        sent = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="hi"),),
            max_tokens=256,
            thinking_budget=512,
        )
        original = _kwargs(thinking={"type": "enabled", "budget_tokens": 8_192})
        assert _kwargs_from_request(sent, original)["thinking"]["budget_tokens"] == 512


class TestTheCacheLifetimeReachesTheWire:
    """ADR-021: a chosen TTL is worth nothing if the adapter drops it.

    This is the fourth field in this module to need such a test. `tools` went
    unsent, then `cacheable` from the turn path, then `cacheable` on a pydantic
    block, then `thinking_budget`. Each was reported as done and never sent, and
    each was found by a test that checked the wire rather than the request.
    """

    def test_a_one_hour_ttl_appears_on_the_breakpoint(self):
        from optio_optimize.adapters.anthropic import _kwargs_from_request
        from optio_optimize.types import LLMRequest, Message

        sent = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(role="system", content="brief", cacheable=True, cache_ttl="1h"),
                Message(role="user", content="hi"),
            ),
            max_tokens=256,
        )

        kwargs = _kwargs_from_request(sent, _kwargs())

        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_no_ttl_means_no_ttl_key_rather_than_an_explicit_five_minutes(self):
        # Absent and "5m" are different instructions: absent is this package
        # expressing no preference, which is what it does by default.
        from optio_optimize.adapters.anthropic import _kwargs_from_request
        from optio_optimize.types import LLMRequest, Message

        sent = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(role="system", content="brief", cacheable=True),
                Message(role="user", content="hi"),
            ),
            max_tokens=256,
        )

        kwargs = _kwargs_from_request(sent, _kwargs())

        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_a_ttl_reaches_a_turn_breakpoint_too(self):
        from optio_optimize.adapters.anthropic import _kwargs_from_request
        from optio_optimize.types import LLMRequest, Message

        sent = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1", cacheable=True, cache_ttl="1h"),
                Message(role="user", content="q2"),
            ),
            max_tokens=256,
        )

        kwargs = _kwargs_from_request(sent, _kwargs())
        blocks = _content_blocks(kwargs["messages"])
        marked = [b for b in blocks if isinstance(b, dict) and "cache_control" in b]

        assert marked, "the breakpoint never reached the wire at all"
        assert marked[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_the_callers_own_cache_control_is_still_never_touched(self):
        """A caller who chose their own TTL has better information than we do.

        Overwriting a caller's `ttl: "1h"` with our own choice would be the cost
        *regression* the system-block path already had once: they paid 2x for an
        hour and this package silently downgraded them.
        """
        from optio_optimize.adapters.anthropic import _RAW, _kwargs_from_request
        from optio_optimize.types import LLMRequest, Message

        caller_block = {
            "type": "text",
            "text": "brief",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        sent = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(
                    role="system",
                    content="brief",
                    cacheable=True,
                    cache_ttl=None,
                    extra={_RAW: caller_block},
                ),
                Message(role="user", content="hi"),
            ),
            max_tokens=256,
        )

        kwargs = _kwargs_from_request(sent, _kwargs())

        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
