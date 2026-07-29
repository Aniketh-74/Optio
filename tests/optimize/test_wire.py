"""What actually reaches the provider.

This file exists because of one incident. The live OpenAI adapter did not
forward ``request.tools``. Nothing raised, nothing failed, and the whole
``mcp_agent`` benchmark run reported ``minify_tools`` saving 3,240 tokens while
the provider billed byte-identical totals in both arms -- because the field the
stage had rewritten was never sent. A confident wrong number, produced by an
omission no test could see.

ADR-017 adds a second translation site (batch submission builds JSON rather than
SDK keyword arguments), which doubles the chance of that omission. So the rule
here is: **every field on LLMRequest is either demonstrably on the wire or
listed in wire.UNSENT_FIELDS with a reason.** Adding a field to the request type
fails this file until someone decides which it is.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import pytest

from optio_optimize import LLMRequest, Message
from optio_optimize.wire import (
    UNSENT_FIELDS,
    anthropic_body,
    anthropic_system_and_turns,
    anthropic_tools,
    as_anthropic_tool,
    openai_body,
    openai_messages,
    openai_tools,
)

pytestmark = pytest.mark.optimize

_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the corpus.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


def _full_request() -> LLMRequest:
    """A request exercising every field that can reach a provider."""
    return LLMRequest(
        model="gpt-4o-mini",
        messages=(
            Message(role="system", content="You are terse.", cacheable=True),
            Message(role="user", content="find the invoice"),
            Message(
                role="assistant",
                content="",
                extra={"tool_calls": [{"id": "call_1", "type": "function"}]},
            ),
            Message(role="tool", content="found it", extra={"tool_call_id": "call_1"}),
        ),
        max_tokens=256,
        tools=(_TOOL,),
        temperature=0.0,
        response_format={"type": "json_object"},
        stop=("<END>",),
    )


# --------------------------------------------------------------------------
# The guard the tools incident earned.
# --------------------------------------------------------------------------


def test_every_request_field_is_sent_or_explicitly_excused():
    request = _full_request()
    body = openai_body(request, request.model)
    # `model` and `messages` are structural; the rest must each show up under
    # some provider key or be excused by name.
    sent_somewhere = {
        "model": "model" in body,
        "messages": "messages" in body,
        "max_tokens": "max_completion_tokens" in body,
        "tools": "tools" in body,
        "temperature": "temperature" in body,
        "response_format": "response_format" in body,
        "stop": "stop" in body,
    }
    for spec in fields(LLMRequest):
        if spec.name in UNSENT_FIELDS:
            continue
        assert spec.name in sent_somewhere, (
            f"LLMRequest.{spec.name} is new: put it on the wire in optio_optimize.wire "
            f"or name it in UNSENT_FIELDS with a reason. Silently not sending a field "
            f"is what made a whole live benchmark measure nothing."
        )
        assert sent_somewhere[spec.name], f"LLMRequest.{spec.name} did not reach the body"


def test_unsent_fields_are_real_fields():
    names = {spec.name for spec in fields(LLMRequest)}
    assert set(UNSENT_FIELDS) <= names, (
        "UNSENT_FIELDS names something LLMRequest does not have; a stale excuse "
        "silently excuses nothing"
    )


def test_anthropic_body_carries_the_same_request():
    request = _full_request()
    body = anthropic_body(request, "claude-haiku-4")
    assert body["model"] == "claude-haiku-4"
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.0
    assert body["stop_sequences"] == ["<END>"]
    assert body["tools"][0]["name"] == "search"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


# --------------------------------------------------------------------------
# Message translation.
# --------------------------------------------------------------------------


def test_tool_call_fields_are_lifted_out_of_extra():
    # OpenAI rejects a "tool" message with no preceding tool_calls, so dropping
    # these makes every tool-calling workload 400 regardless of any stage.
    messages = openai_messages(_full_request())
    assert messages[2]["tool_calls"] == [{"id": "call_1", "type": "function"}]
    assert messages[3]["tool_call_id"] == "call_1"


def test_unknown_extra_keys_are_not_forwarded():
    request = LLMRequest(
        model="gpt-4o",
        messages=(Message(role="user", content="hi", extra={"internal_marker": 1}),),
    )
    assert openai_messages(request) == [{"role": "user", "content": "hi"}]


def test_name_is_included_only_when_set():
    request = LLMRequest(
        model="gpt-4o",
        messages=(
            Message(role="user", content="a", name="alice"),
            Message(role="user", content="b"),
        ),
    )
    messages = openai_messages(request)
    assert messages[0]["name"] == "alice"
    assert "name" not in messages[1]


# --------------------------------------------------------------------------
# Tools: absent is not empty.
# --------------------------------------------------------------------------


def test_no_tools_yields_none_not_empty_list():
    request = LLMRequest(model="gpt-4o", messages=(Message(role="user", content="hi"),))
    assert openai_tools(request) is None
    assert anthropic_tools(request) is None


def test_no_tools_omits_the_key_entirely():
    request = LLMRequest(model="gpt-4o", messages=(Message(role="user", content="hi"),))
    assert "tools" not in openai_body(request, "gpt-4o")
    assert "tools" not in anthropic_body(request, "claude-haiku-4")


def test_openai_tool_schema_is_translated_for_anthropic():
    translated = as_anthropic_tool(_TOOL)
    assert translated == {
        "name": "search",
        "description": "Search the corpus.",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }


def test_anthropic_shaped_schema_passes_through():
    native = {"name": "search", "description": "d", "input_schema": {"type": "object"}}
    assert as_anthropic_tool(native) == native


def test_tool_with_empty_function_gets_a_usable_input_schema():
    # Anthropic rejects a tool with no input_schema, so an OpenAI schema that
    # omitted `parameters` must not translate into a rejected request.
    translated = as_anthropic_tool({"type": "function", "function": {"name": "noop"}})
    assert translated["input_schema"] == {"type": "object", "properties": {}}


# --------------------------------------------------------------------------
# Anthropic's system/turn split.
# --------------------------------------------------------------------------


def test_system_messages_become_blocks_and_leave_the_turn_list():
    system, turns = anthropic_system_and_turns(_full_request())
    assert len(system) == 1
    assert all(t["role"] != "system" for t in turns)
    assert len(turns) == 3


def test_cacheable_marker_becomes_cache_control():
    # The entire value of the prefix stage on Anthropic: without this the
    # marker our pipeline places is inert and the discount is zero.
    system, _ = anthropic_system_and_turns(_full_request())
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_unmarked_system_block_carries_no_cache_control():
    request = LLMRequest(
        model="claude-haiku-4",
        messages=(Message(role="system", content="terse"), Message(role="user", content="hi")),
    )
    system, _ = anthropic_system_and_turns(request)
    assert "cache_control" not in system[0]


def test_no_system_message_omits_the_key():
    request = LLMRequest(model="claude-haiku-4", messages=(Message(role="user", content="hi"),))
    assert "system" not in anthropic_body(request, "claude-haiku-4")


# --------------------------------------------------------------------------
# Null handling: the batch endpoint validates envelopes, so an explicit null
# is a rejection rather than a default.
# --------------------------------------------------------------------------


def test_unset_optional_fields_are_omitted_not_nulled():
    request = LLMRequest(model="gpt-4o", messages=(Message(role="user", content="hi"),))
    body = openai_body(request, "gpt-4o")
    assert set(body) == {"model", "messages"}


def test_anthropic_invents_a_max_tokens_because_the_field_is_required():
    request = LLMRequest(model="claude-haiku-4", messages=(Message(role="user", content="hi"),))
    assert anthropic_body(request, "claude-haiku-4")["max_tokens"] == 1024
    assert anthropic_body(request, "claude-haiku-4", default_max_tokens=64)["max_tokens"] == 64


def test_body_is_json_serializable():
    # Batch submission writes these to a JSONL file; a body that cannot
    # serialize fails at upload, after the caller believed work was queued.
    import json

    json.dumps(openai_body(_full_request(), "gpt-4o-mini"))
    json.dumps(anthropic_body(_full_request(), "claude-haiku-4"))


def test_routing_model_overrides_the_request_model():
    # The routing stage may already have changed request.model; the body must
    # bill against what the backend actually submits to.
    body = openai_body(_full_request(), "gpt-4o")
    assert body["model"] == "gpt-4o"


# --------------------------------------------------------------------------
# Anthropic response translation.
# --------------------------------------------------------------------------


# Dataclasses rather than hand-written __init__s: the translator reads these by
# attribute, so a structural stand-in is enough, and a dataclass gives mypy a
# typed constructor for free. The same shape test_batch_backends.py uses.
@dataclass
class _Usage:
    input_tokens: int = 300
    output_tokens: int = 40
    cache_read_input_tokens: int = 100


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Msg:
    content: list[Any]
    usage: _Usage | None
    model: str = "claude-haiku-4-5"
    stop_reason: str = "end_turn"


def test_anthropic_response_adds_cache_reads_into_input_tokens():
    # Anthropic reports input_tokens EXCLUDING cache reads. Everywhere else in
    # this package input_tokens means "total prompt tokens, some discounted",
    # so getting this wrong in one place makes batched, synchronous and
    # adapter totals silently incomparable.
    from optio_optimize.wire import response_from_anthropic_message

    response = response_from_anthropic_message(_Msg([_Block("hi")], _Usage()))

    assert response.input_tokens == 400
    assert response.cached_input_tokens == 100
    assert response.billable_input_tokens == 300
    assert response.content == "hi"
    assert response.finish_reason == "end_turn"


def test_anthropic_response_joins_only_text_blocks():
    from optio_optimize.wire import response_from_anthropic_message

    class _ToolBlock:
        type = "tool_use"
        text = "SHOULD NOT APPEAR"

    message = _Msg([_Block("part one "), _ToolBlock(), _Block("part two")], _Usage())
    assert response_from_anthropic_message(message).content == "part one part two"


def test_anthropic_response_survives_missing_usage():
    from optio_optimize.wire import response_from_anthropic_message

    class _NoUsage:
        content: ClassVar[list[_Block]] = [_Block("hi")]
        usage = None
        model = "claude-haiku-4-5"
        stop_reason = "end_turn"

    response = response_from_anthropic_message(_NoUsage())
    assert response.input_tokens == 0
    assert response.output_tokens == 0


class TestCacheWriteTokensAreAccountedAndPriced:
    """Anthropic reports ``input_tokens`` excluding reads *and* writes.

    Writes were left out of the total, which dropped the single most expensive
    band of prompt tokens -- they carry a 1.25x premium, not a discount -- from
    every Anthropic report. The error only ever ran in the direction that makes
    this package's saving look larger, which is why no report ever looked wrong.
    """

    def test_writes_are_added_into_the_prompt_total(self):
        from optio_optimize.wire import response_from_anthropic_message

        class _Usage:
            input_tokens = 200
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 4_605
            output_tokens = 50

        class _Message:
            usage = _Usage()
            content: ClassVar[list[object]] = []
            model = "claude-haiku-4-5"
            stop_reason = "end_turn"

        response = response_from_anthropic_message(_Message())
        # 200 was what this reported for a turn that really billed 4,805.
        assert response.input_tokens == 4_805
        assert response.cache_write_tokens == 4_605

    def test_a_write_costs_more_than_a_plain_input_token(self):
        from optio_optimize.config import PRICING
        from optio_optimize.savings import _cost

        pricing = PRICING["claude-haiku-4-5"]
        as_writes = _cost(pricing, 10_000, 0, 0, 10_000)
        as_plain = _cost(pricing, 10_000, 0, 0, 0)
        assert as_writes > as_plain
        assert as_writes == pytest.approx(as_plain * 1.25)

    def test_openai_writes_are_free_because_openai_does_not_charge_them(self):
        from optio_optimize.config import PRICING
        from optio_optimize.savings import _cost

        pricing = PRICING["gpt-4o"]
        assert pricing.cache_write_usd_per_m is None
        assert _cost(pricing, 1_000, 0, 0, 1_000) == _cost(pricing, 1_000, 0, 0, 0)

    def test_the_measured_anthropic_run_prices_out_at_fifty_percent(self):
        """The published figure, recomputed from the run's own token counts.

        Locks the arithmetic behind the number in ``caching.py`` and
        ``docs/optimize-benchmarks.md``. It was first published as 53.7% with
        the 5,487 writes priced at the base rate; correctly priced it is 50.1%.
        The token counts never changed -- only what they cost.
        """
        from optio_optimize.config import PRICING
        from optio_optimize.savings import _cost

        pricing = PRICING["claude-haiku-4-5"]
        off = _cost(pricing, 30_111, 1_623, 0, 0)
        on = _cost(pricing, 30_113, 1_662, 23_023, 5_487)
        assert (off - on) / off == pytest.approx(0.501, abs=0.001)

    def test_writes_cannot_exceed_the_uncached_remainder(self):
        """A provider reporting nonsense must not produce a negative band."""
        from optio_optimize.config import PRICING
        from optio_optimize.savings import _cost

        pricing = PRICING["claude-haiku-4-5"]
        assert _cost(pricing, 100, 0, 90, 500) >= 0
