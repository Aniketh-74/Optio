"""OpenAI Agents SDK adapter (M4-1).

The OpenAI Agents SDK has its own tracing system, and it is *not* OpenTelemetry:
it exports to OpenAI's own trace backend by default. GenAI spans reach us only
when the user has bridged that tracing to OTel -- via the OTel
``opentelemetry-instrumentation-openai-agents`` package, Logfire's
``logfire.instrument_openai_agents()``, or an OpenInference processor.

So this adapter cannot verify that spans will arrive; it can only install the
tap and say clearly what else is required. Being explicit about that boundary is
the point (R-TECH-3): emitting the spans is the ecosystem's job, and taking it
on would put us on the hook for every SDK release.

An ``Agent`` in this SDK is a passive configuration object -- name, instructions,
tools, handoffs -- executed by a separate ``Runner``. Instrumenting it is
therefore process-wide rather than object-scoped, which is why the target is
returned untouched and the tap goes on the tracer provider.
"""

from __future__ import annotations

from typing import Final

from optio.adapters._common import ModuleMatchedAdapter

#: Module prefixes for the SDK. It publishes as ``openai-agents`` but imports as
#: ``agents``; ``openai.agents`` is matched too in case that namespace is used.
_MODULES: Final[tuple[str, ...]] = ("agents.", "agents", "openai.agents")

#: An ``Agent`` carries instructions and a tool list. ``name`` alone would match
#: half the objects in Python, so both are required.
_DUCK_ATTRS: Final[tuple[str, ...]] = ("name", "tools")


class OpenAIAgentsAdapter(ModuleMatchedAdapter):
    """Wires optio to an OpenAI Agents SDK agent."""

    name = "openai_agents"
    modules = _MODULES
    duck_attrs = _DUCK_ATTRS
    instrumentation_hint = (
        "an OTel bridge for the Agents SDK "
        "(opentelemetry-instrumentation-openai-agents, logfire, or OpenInference) "
        "-- the SDK's built-in tracing does not emit OTel spans on its own"
    )

    def matches(self, target: object) -> bool:
        """Whether the target is an Agents SDK ``Agent``.

        Args:
            target: The object passed to ``instrument()``.

        Returns:
            ``True`` for an SDK agent.
        """
        if not super().matches(target):
            return False
        # ``agents`` is a short, generic top-level name and a user package could
        # plausibly claim it. Requiring one SDK-specific field as well keeps a
        # coincidental match from silently instrumenting the wrong thing.
        return any(hasattr(target, attr) for attr in ("instructions", "handoffs", "model"))
