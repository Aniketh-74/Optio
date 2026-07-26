"""Shared adapter machinery (M4-1..3).

Every adapter we ship does the same three things: recognise the target, refuse
loudly if it does not match, install the tap. Only the *recognition* differs.
That shape emerged from the LangGraph adapter (M1-4) and repeated verbatim
three times in M4, so it lives here once.

This is deliberately a base class and not a framework. It holds no per-framework
knowledge -- each adapter supplies two class attributes and, where duck-typing
is not enough, overrides :meth:`matches`. Anything cleverer (entry points,
declarative match DSLs) would be over-engineering for four adapters in one
repository (Section 16 rule 10).

Recognition never imports the framework. A probe that imported every candidate
would make ``instrument(agent)`` cost the import time -- and the failure -- of
three frameworks the user did not install (Section 6.7).
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


class ModuleMatchedAdapter(Adapter):
    """An adapter that recognises its framework by module path and duck-typing.

    Subclasses set :attr:`modules` and :attr:`duck_attrs`, and may override
    :meth:`matches` when those two are not sufficient to separate the framework's
    agent objects from its other exports.

    Attributes:
        modules: Module-name prefixes that identify the framework. Checked
            against the target's own class, so resolution costs nothing on a
            machine where the framework is absent.
        duck_attrs: Attributes the agent object must expose. A secondary signal,
            so a subclass defined in the user's own module is still recognised
            when combined with an overridden :meth:`matches`.
        instrumentation_hint: What the user must have configured for GenAI spans
            to exist. Named in the warning when no tracer provider is present,
            because "no signals" is otherwise discovered weeks later as an empty
            dashboard.
    """

    modules: tuple[str, ...] = ()
    duck_attrs: tuple[str, ...] = ()
    instrumentation_hint: str = "OTel instrumentation"

    def matches(self, target: object) -> bool:
        """Whether the target belongs to this adapter's framework.

        Args:
            target: The object passed to ``instrument()``.

        Returns:
            ``True`` when the target's class comes from one of :attr:`modules`
            and exposes every attribute in :attr:`duck_attrs`.
        """
        module = type(target).__module__ or ""
        if not module.startswith(self.modules):
            return False
        return all(hasattr(target, attr) for attr in self.duck_attrs)

    def instrument(
        self, target: Any, config: Config, provider: TracerProvider | None = None
    ) -> Any:
        """Install the span tap for this framework's agent.

        Args:
            target: The agent object.
            config: Active configuration.
            provider: Tracer provider to install on. Defaults to the global one.

        Returns:
            The same object that was passed in (Section 6.7 identity contract).

        Raises:
            UnsupportedFrameworkError: If the target is not this framework's
                agent. Setup-time and loud, so a mistyped target cannot
                masquerade as working instrumentation (Section 4.2).
        """
        if not self.matches(target):
            raise UnsupportedFrameworkError(
                f"adapter {self.name!r} cannot instrument "
                f"{type(target).__module__}.{type(target).__qualname__}; "
                f"pass a {self.name} agent, or name the right adapter"
            )

        if install_tap(config, provider) is None:
            _log.warning(
                "agentmeter: no OTel SDK tracer provider is configured, so no "
                "signals will be emitted. Configure a TracerProvider and %s. "
                "See docs/runbooks.md.",
                self.instrumentation_hint,
            )
        return target
