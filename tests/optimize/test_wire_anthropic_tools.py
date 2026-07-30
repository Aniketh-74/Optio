"""Translating tool turns into Anthropic's shape (ADR-025).

The first full live Anthropic benchmark failed two of twelve workloads outright::

    mcp_agent          9 errors,  1 of 10 calls succeeded
    tool_calling_chat 19 errors,  1 of 20 calls succeeded

    400 invalid_request_error: messages: Unexpected role "tool".
    Allowed roles are "user" or "assistant".

``anthropic_system_and_turns`` copied ``message.role`` onto the wire, and this
package's neutral request models a tool result as ``role="tool"``.

**The adapter never hit this** because a real Anthropic caller's original params
ride through in ``extra["_raw"]`` and are restored verbatim. The translation is
reached only when a request is *built* neutrally -- the benchmark, ADR-017's
batch submission, and any caller constructing an ``LLMRequest`` directly. The
defect sat behind the one path nothing had run.

``openai_messages`` records the identical bug being paid for once on the other
vendor: "OpenAI rejects a ``tool`` message with no preceding ``tool_calls``, so
dropping them silently -- which the first version of the live adapter did --
makes every tool-calling workload fail with a 400."
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import pytest

from optio_optimize.types import LLMRequest, Message
from optio_optimize.wire import anthropic_system_and_turns

pytestmark = pytest.mark.optimize


def _tool_call(call_id: str = "call_1", name: str = "search", **arguments: Any) -> dict[str, Any]:
    """One OpenAI-shaped proposed call, the neutral representation."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _conversation(*messages: Message) -> LLMRequest:
    return LLMRequest(model="claude-haiku-4-5", messages=messages, temperature=0.0)


class TestAToolResultBecomesAUserTurn:
    def test_a_tool_role_never_reaches_the_wire(self) -> None:
        """The defect, as the assertion that used to fail.

        Anthropic allows only ``user`` and ``assistant``; anything else is a 400
        that names the role.
        """
        request = _conversation(
            Message(role="user", content="what is the weather?"),
            Message(role="assistant", content="", extra={"tool_calls": [_tool_call(city="Paris")]}),
            Message(role="tool", content="18C", name="search", extra={"tool_call_id": "call_1"}),
        )

        _, turns = anthropic_system_and_turns(request)

        assert all(t["role"] in {"user", "assistant"} for t in turns)

    def test_the_result_is_carried_as_a_tool_result_block(self) -> None:
        request = _conversation(
            Message(role="assistant", content="", extra={"tool_calls": [_tool_call()]}),
            Message(role="tool", content="18C", extra={"tool_call_id": "call_1"}),
        )

        _, turns = anthropic_system_and_turns(request)

        result_turn = turns[-1]
        assert result_turn["role"] == "user"
        block = result_turn["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_1"
        assert block["content"] == "18C"

    def test_consecutive_tool_results_merge_into_one_turn(self) -> None:
        """Anthropic requires alternating roles; parallel calls produce two in a row.

        Two ``user`` turns back to back is a 400. Real agents make parallel tool
        calls constantly -- the benchmark's own workloads happen not to, which is
        exactly why this is worth pinning rather than discovering in production.
        """
        request = _conversation(
            Message(
                role="assistant",
                content="",
                extra={"tool_calls": [_tool_call("call_1"), _tool_call("call_2")]},
            ),
            Message(role="tool", content="18C", extra={"tool_call_id": "call_1"}),
            Message(role="tool", content="cloudy", extra={"tool_call_id": "call_2"}),
        )

        _, turns = anthropic_system_and_turns(request)

        assert [t["role"] for t in turns] == ["assistant", "user"]
        blocks = turns[-1]["content"]
        assert [b["tool_use_id"] for b in blocks] == ["call_1", "call_2"]

    def test_roles_alternate_across_a_realistic_agent_loop(self) -> None:
        # The shape the failing workloads actually send: propose, result, answer.
        request = _conversation(
            Message(role="system", content="be terse"),
            Message(role="user", content="q"),
            Message(role="assistant", content="", extra={"tool_calls": [_tool_call()]}),
            Message(role="tool", content="payload", extra={"tool_call_id": "call_1"}),
            Message(role="assistant", content="the answer"),
        )

        _, turns = anthropic_system_and_turns(request)

        roles = [t["role"] for t in turns]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert all(a != b for a, b in pairwise(roles))


class TestAProposedCallBecomesAToolUseBlock:
    def test_the_call_is_carried_as_a_tool_use_block(self) -> None:
        request = _conversation(
            Message(role="assistant", content="", extra={"tool_calls": [_tool_call(city="Paris")]}),
        )

        _, turns = anthropic_system_and_turns(request)

        block = turns[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_1"
        assert block["name"] == "search"

    def test_arguments_become_a_dict_not_a_json_string(self) -> None:
        """Anthropic's ``input`` is an object; the neutral shape is a string.

        Forwarding the string would be accepted as a *string-valued* input by
        some models and rejected by others -- the kind of divergence that shows
        up as a mysterious tool failure rather than a translation bug.
        """
        request = _conversation(
            Message(role="assistant", content="", extra={"tool_calls": [_tool_call(city="Paris")]}),
        )

        _, turns = anthropic_system_and_turns(request)

        assert turns[0]["content"][0]["input"] == {"city": "Paris"}

    def test_assistant_text_alongside_a_call_is_preserved(self) -> None:
        # A model may narrate before calling. Dropping that text loses content
        # the caller was billed for and the model may refer back to.
        request = _conversation(
            Message(
                role="assistant",
                content="Let me look that up.",
                extra={"tool_calls": [_tool_call()]},
            ),
        )

        _, turns = anthropic_system_and_turns(request)

        kinds = [b["type"] for b in turns[0]["content"]]
        assert kinds == ["text", "tool_use"]
        assert turns[0]["content"][0]["text"] == "Let me look that up."

    def test_unparseable_arguments_raise_rather_than_inventing_a_call(self) -> None:
        """Neither way of being wrong is safe, so this refuses to guess.

        Dropping the block orphans the ``tool_result`` that answers it and
        produces a 400 naming the wrong problem; sending ``{}`` sends a call the
        model never made. Raising reaches the pipeline's per-stage fail-open,
        which sends the request unoptimized -- a lost optimization rather than a
        corrupted conversation.
        """
        broken = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{not json"},
        }
        request = _conversation(
            Message(role="assistant", content="", extra={"tool_calls": [broken]}),
        )

        with pytest.raises(ValueError, match="tool_calls"):
            anthropic_system_and_turns(request)


class TestTheCacheMarkerSurvives:
    def test_a_marked_tool_result_keeps_its_cache_control(self) -> None:
        """``PrefixCacheStage`` marks the last stable message, often a tool result.

        This function has already lost a marker once -- it emitted
        ``cache_control`` for system blocks alone, so on any real conversation
        the marker was computed, placed, reported in the savings ledger, and
        silently dropped on the way to the wire. Losing it on tool turns would
        be the same bug on the workloads that need caching most.
        """
        request = _conversation(
            Message(role="assistant", content="", extra={"tool_calls": [_tool_call()]}),
            Message(
                role="tool", content="payload", cacheable=True, extra={"tool_call_id": "call_1"}
            ),
        )

        _, turns = anthropic_system_and_turns(request)

        assert turns[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_a_marked_tool_call_keeps_its_cache_control(self) -> None:
        request = _conversation(
            Message(
                role="assistant", content="", cacheable=True, extra={"tool_calls": [_tool_call()]}
            ),
        )

        _, turns = anthropic_system_and_turns(request)

        assert turns[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}


class TestOrdinaryTurnsAreUnchanged:
    def test_a_plain_conversation_still_uses_string_content(self) -> None:
        # The translation must not promote every turn to blocks: that would
        # change the bytes of every request this package has ever sent.
        request = _conversation(
            Message(role="system", content="be terse"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        )

        system, turns = anthropic_system_and_turns(request)

        assert system == [{"type": "text", "text": "be terse"}]
        assert turns == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_an_assistant_without_tool_calls_is_untouched(self) -> None:
        request = _conversation(Message(role="assistant", content="just text"))

        _, turns = anthropic_system_and_turns(request)

        assert turns == [{"role": "assistant", "content": "just text"}]
