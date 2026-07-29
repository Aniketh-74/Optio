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

from dataclasses import fields

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
