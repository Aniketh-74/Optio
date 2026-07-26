"""CrewAI adapter (M4-2).

CrewAI emits OTel spans through third-party instrumentation
(``openinference-instrumentation-crewai``, or CrewAI's own OTel exporter where
enabled). As with every adapter, producing the spans is the ecosystem's job and
ours starts at the span (R-TECH-3).

Both a ``Crew`` and a single ``Agent`` are accepted. A ``Crew`` is the usual
target -- it owns the ``kickoff()`` that runs the whole workflow -- but users
building a one-agent flow reasonably pass the agent, and refusing that would be
a papercut with no safety benefit: the tap is installed process-wide either way.
"""

from __future__ import annotations

from typing import Final

from agentmeter.adapters._common import ModuleMatchedAdapter

_MODULES: Final[tuple[str, ...]] = ("crewai.", "crewai")

#: A ``Crew`` exposes ``agents`` and ``tasks``; a bare ``Agent`` exposes neither,
#: so recognition is handled in ``matches`` rather than by a single attribute set.
_CREW_ATTRS: Final[tuple[str, ...]] = ("agents", "tasks")
_AGENT_ATTRS: Final[tuple[str, ...]] = ("role", "goal")


class CrewAIAdapter(ModuleMatchedAdapter):
    """Wires agentmeter to a CrewAI crew or agent."""

    name = "crewai"
    modules = _MODULES
    instrumentation_hint = "CrewAI OTel instrumentation (e.g. openinference-instrumentation-crewai)"

    def matches(self, target: object) -> bool:
        """Whether the target is a CrewAI crew or agent.

        Args:
            target: The object passed to ``instrument()``.

        Returns:
            ``True`` for a ``Crew`` (has ``agents`` and ``tasks``) or an
            ``Agent`` (has ``role`` and ``goal``).
        """
        module = type(target).__module__ or ""
        if not module.startswith(self.modules):
            return False
        is_crew = all(hasattr(target, attr) for attr in _CREW_ATTRS)
        is_agent = all(hasattr(target, attr) for attr in _AGENT_ATTRS)
        return is_crew or is_agent
