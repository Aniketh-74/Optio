"""LangGraph adapter -- the first one-line integration (M1-4).

LangGraph builds on LangChain, whose OTel instrumentation is supplied by
third-party packages rather than by us. That division is deliberate: emitting
GenAI spans is the ecosystem's job and re-implementing it would put us on the
hook for every LangChain release (R-TECH-3). Our job starts at the span.

So this adapter is thin by design:

1. Confirm the target really is a LangGraph object, so a typo does not silently
   instrument nothing.
2. Install the span tap on the user's tracer provider.

If the user has no LangChain OTel instrumentation configured, no GenAI spans
exist to tap. That is reported at setup rather than discovered later as missing
dashboards -- but it is a *warning*, not an error: the user may be wiring
instrumentation after ``instrument()``, and refusing to proceed would break a
legitimate ordering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from agentmeter.errors import UnsupportedFrameworkError
from agentmeter.runtime.adapter_base import Adapter
from agentmeter.runtime.installer import install_tap

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from agentmeter.config import Config

_log: Final = logging.getLogger("agentmeter")

#: Module prefixes that identify a LangGraph/LangChain object. Checked against
#: the target's own class rather than by importing LangGraph, so resolution
#: costs nothing on machines where it is not installed.
_LANGGRAPH_MODULES: Final[tuple[str, ...]] = ("langgraph.", "langchain")

#: Attributes a compiled LangGraph exposes. Used as a secondary signal so a
#: subclass defined in the user's own module is still recognised.
_LANGGRAPH_DUCK_ATTRS: Final[tuple[str, ...]] = ("invoke", "stream")


class LangGraphAdapter(Adapter):
    """Wires agentmeter to a compiled LangGraph graph."""

    name = "langgraph"

    def matches(self, target: object) -> bool:
        """Whether the target looks like a compiled LangGraph graph.

        Args:
            target: The object passed to ``instrument()``.

        Returns:
            ``True`` for a LangGraph/LangChain object exposing the graph
            interface.
        """
        module = type(target).__module__ or ""
        if not module.startswith(_LANGGRAPH_MODULES):
            return False
        return all(hasattr(target, attr) for attr in _LANGGRAPH_DUCK_ATTRS)

    def instrument(
        self, target: Any, config: Config, provider: TracerProvider | None = None
    ) -> Any:
        """Install the span tap for a LangGraph agent.

        Args:
            target: The compiled graph.
            config: Active configuration.
            provider: Tracer provider to install on. Defaults to the global one.

        Returns:
            The same graph object.

        Raises:
            UnsupportedFrameworkError: If the target is not a LangGraph object.
                Setup-time and loud, so a mistyped target cannot masquerade as
                working instrumentation.
        """
        if not self.matches(target):
            raise UnsupportedFrameworkError(
                f"adapter 'langgraph' cannot instrument "
                f"{type(target).__module__}.{type(target).__qualname__}; "
                f"pass a compiled LangGraph graph, or name the right adapter"
            )

        tap = install_tap(config, provider)
        if tap is None:
            _log.warning(
                "agentmeter: no OTel SDK tracer provider is configured, so no "
                "signals will be emitted. Configure a TracerProvider and "
                "LangChain OTel instrumentation. See docs/runbooks.md."
            )
        return target
