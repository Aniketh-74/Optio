"""Adapter lookup and duck-typed resolution (Section 6.1).

Lives in ``runtime`` rather than ``adapters`` because of the dependency
direction. Section 3.1 has ``adapters`` importing ``api``, and ``api``
importing ``runtime`` -- so a registry inside ``adapters`` would force ``api``
to import *upward* to reach it. Resolution is orchestration, which is what this
layer is for; the concrete adapters stay where they belong.

Adapters are loaded lazily by module path: importing this module must not pull
in LangGraph, CrewAI and the OpenAI SDK, because a user who installed one
framework should not pay the import cost -- or the failure -- of the other
three. Each entry is imported only when it is actually a candidate, and the
concrete module never appears as a static dependency of ``runtime``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Final

from agentmeter.errors import UnsupportedFrameworkError
from agentmeter.runtime.adapter_base import Adapter

_log: Final = logging.getLogger("agentmeter")

#: Adapter name to ``(module, class)``. LangGraph landed in M1; the other three
#: in M4. Names here are what users pass as ``adapter=``. The class is named
#: explicitly rather than discovered by a naming convention -- a scan would also
#: match the imported ``Adapter`` base class, and "find the class whose name
#: ends in Adapter" is the kind of cleverness that fails silently later.
#:
#: Order matters for duck-typed resolution: ``resolve_adapter`` returns the
#: first match, so the most specifically-matched frameworks come first. In
#: practice the match rules are disjoint, but relying on that silently would
#: make a future adapter's overlap a mystery rather than a merge conflict.
_ADAPTER_MODULES: Final[dict[str, tuple[str, str]]] = {
    "langgraph": ("agentmeter.adapters.langgraph", "LangGraphAdapter"),
    "openai_agents": ("agentmeter.adapters.openai_agents", "OpenAIAgentsAdapter"),
    "crewai": ("agentmeter.adapters.crewai", "CrewAIAdapter"),
    "claude_agent": ("agentmeter.adapters.claude_agent", "ClaudeAgentAdapter"),
}

#: Adapters designed but not yet implemented. Empty as of M4 -- kept, rather
#: than deleted, because it distinguishes "designed, not built yet" from
#: "unknown", which are different problems with different fixes for the user.
_PLANNED_ADAPTERS: Final[frozenset[str]] = frozenset()


def available_adapters() -> list[str]:
    """Return the adapter names that can be used today.

    Returns:
        Sorted adapter names.
    """
    return sorted(_ADAPTER_MODULES)


def load_adapter(name: str) -> Adapter:
    """Load an adapter by name.

    Args:
        name: The adapter name, e.g. ``"langgraph"``.

    Returns:
        An adapter instance.

    Raises:
        UnsupportedFrameworkError: If the name is unknown or not yet
            implemented. Setup-time and loud (Section 4.2).
    """
    entry = _ADAPTER_MODULES.get(name)
    if entry is None:
        if name in _PLANNED_ADAPTERS:
            raise UnsupportedFrameworkError(
                f"adapter {name!r} is planned but not implemented yet; "
                f"available now: {available_adapters()}"
            )
        raise UnsupportedFrameworkError(
            f"unknown adapter {name!r}; available: {available_adapters()}"
        )

    module_path, class_name = entry
    module = importlib.import_module(module_path)
    adapter_class: type[Adapter] = getattr(module, class_name)
    return adapter_class()


def resolve_adapter(target: object) -> Adapter:
    """Find the adapter that handles a target object.

    Args:
        target: The object passed to ``instrument()``.

    Returns:
        The matching adapter.

    Raises:
        UnsupportedFrameworkError: If no adapter matches. The message names what
            was passed and what is supported, because "unsupported framework"
            without either is a dead end for the user.
    """
    for name in _ADAPTER_MODULES:
        adapter = load_adapter(name)
        if adapter.matches(target):
            return adapter

    target_type = f"{type(target).__module__}.{type(target).__qualname__}"
    planned = f" (planned: {sorted(_PLANNED_ADAPTERS)})" if _PLANNED_ADAPTERS else ""
    raise UnsupportedFrameworkError(
        f"no adapter matches {target_type}; available: {available_adapters()}{planned}. "
        f"Pass adapter='...' to select one explicitly, or use RunContext "
        f"to govern an unsupported framework."
    )
