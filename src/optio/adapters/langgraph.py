"""LangGraph adapter -- the first one-line integration (M1-4).

LangGraph builds on LangChain, whose OTel instrumentation is supplied by
third-party packages rather than by us. That division is deliberate: emitting
GenAI spans is the ecosystem's job and re-implementing it would put us on the
hook for every LangChain release (R-TECH-3). Our job starts at the span.

So this adapter is thin by design: confirm the target really is a LangGraph
object (so a typo does not silently instrument nothing), then install the span
tap. Both steps live in :class:`~optio.adapters._common.ModuleMatchedAdapter`,
which is shared with the M4 adapters; all that remains here is the recognition
rule.

If the user has no LangChain OTel instrumentation configured, no GenAI spans
exist to tap. That is reported at setup rather than discovered later as missing
dashboards -- but it is a *warning*, not an error: the user may be wiring
instrumentation after ``instrument()``, and refusing to proceed would break a
legitimate ordering.
"""

from __future__ import annotations

from typing import Final

from optio.adapters._common import ModuleMatchedAdapter

#: Module prefixes that identify a LangGraph/LangChain object. Checked against
#: the target's own class rather than by importing LangGraph, so resolution
#: costs nothing on machines where it is not installed.
_LANGGRAPH_MODULES: Final[tuple[str, ...]] = ("langgraph.", "langchain")

#: Attributes a compiled LangGraph exposes. Used as a secondary signal so a
#: subclass defined in the user's own module is still recognised.
_LANGGRAPH_DUCK_ATTRS: Final[tuple[str, ...]] = ("invoke", "stream")


class LangGraphAdapter(ModuleMatchedAdapter):
    """Wires optio to a compiled LangGraph graph."""

    name = "langgraph"
    modules = _LANGGRAPH_MODULES
    duck_attrs = _LANGGRAPH_DUCK_ATTRS
    instrumentation_hint = "LangChain OTel instrumentation"
