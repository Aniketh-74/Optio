"""Claude Agent SDK adapter (M4-3).

This SDK has a different shape from the other three. There is no long-lived
"agent object" to hand us: the surface is ``query()`` for one-shot calls and
``ClaudeSDKClient`` for a conversation. The client is the only durable object,
so it is what we recognise.

The SDK also runs the agent loop in the Claude Code process rather than in the
caller's, which means the spans we tap are the ones the user's own OTel
instrumentation produces around the SDK calls. Cost signals therefore reflect
what the SDK reports back through those spans, not a token count we observe
directly -- the same reconciliation path as every other adapter, but worth
stating because the out-of-process execution makes it easy to assume otherwise.

``ClaudeAgentOptions`` is deliberately *not* matched. It is a configuration
dataclass, and accepting it would let ``instrument(options)`` look like it
worked while instrumenting nothing.
"""

from __future__ import annotations

from typing import Final

from agentmeter.adapters._common import ModuleMatchedAdapter

#: The SDK publishes as ``claude-agent-sdk`` and imports as ``claude_agent_sdk``.
#: ``claude_code_sdk`` is the pre-rename name, still installed in the wild.
_MODULES: Final[tuple[str, ...]] = ("claude_agent_sdk", "claude_code_sdk")

#: ``ClaudeSDKClient`` drives a conversation: connect, send, receive.
_DUCK_ATTRS: Final[tuple[str, ...]] = ("query", "receive_response")


class ClaudeAgentAdapter(ModuleMatchedAdapter):
    """Wires agentmeter to a Claude Agent SDK client."""

    name = "claude_agent"
    modules = _MODULES
    duck_attrs = _DUCK_ATTRS
    instrumentation_hint = (
        "OTel instrumentation around your Claude Agent SDK calls "
        "(the SDK runs the agent loop out of process, so spans come from your side)"
    )
