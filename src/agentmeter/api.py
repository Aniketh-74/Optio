"""Public API surface -- the only module users touch.

This is the frozen contract described in Section 8.1. Breaking a signature here
after M1 requires an ADR (Section 16 rule 12).

**M0 status:** these entry points are typed no-ops. They validate configuration
and manage run identity, but wire no lanes -- the span tap arrives in M1, cost
signals in M2. A no-op ``instrument()`` is deliberate: it lets downstream code
and adapter tests compile against the final surface from commit one.

Failure discipline (Section 4.2): everything in this module raises *eagerly* on
bad input, because setup-time is exactly where a misconfiguration should be loud.
The opposite rule governs the runtime -- see ``runtime/failopen.py``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar, overload

from agentmeter.config import BudgetPolicy, Config, default_config
from agentmeter.errors import AgentMeterConfigError
from agentmeter.runtime.run_context import RunContext

__all__ = ["RunContext", "instrument", "meter"]

_log = logging.getLogger("agentmeter")

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

#: Adapters planned for M1 (langgraph) and M4 (the rest). Named here so an
#: unsupported target produces a useful error instead of a silent no-op.
_KNOWN_ADAPTERS: frozenset[str] = frozenset(
    {"langgraph", "openai_agents", "crewai", "claude_agent"}
)


def instrument(
    target: T,
    *,
    adapter: str | None = None,
    config: Config | None = None,
    **overrides: object,
) -> T:
    """Attach agentmeter to an agent.

    Wires the span tap so the agent's OTel GenAI spans flow through the enabled
    signal lanes. The agent's own behavior is never altered.

    Args:
        target: The agent object to instrument.
        adapter: Explicit adapter name. When omitted, the adapter is resolved by
            duck-typing the target.
        config: Full configuration. Defaults to environment-resolved values.
        **overrides: Individual config fields, taking precedence over both
            ``config`` and the environment (Section 4.3).

    Returns:
        The same object that was passed in. Instrumentation is a side effect;
        the identity guarantee lets ``agent = instrument(agent)`` and a bare
        ``instrument(agent)`` behave identically.

    Raises:
        AgentMeterConfigError: If ``adapter`` names an unknown framework or an
            override is invalid. Raised at call time -- never on the hot path.

    Example:
        >>> from agentmeter import instrument
        >>> agent = object()
        >>> instrument(agent) is agent
        True
    """
    resolved = (config or default_config()).merged_with(**overrides)

    if adapter is not None and adapter not in _KNOWN_ADAPTERS:
        raise AgentMeterConfigError(
            f"unknown adapter {adapter!r}; supported: {sorted(_KNOWN_ADAPTERS)}"
        )

    # M0: surface only. M1-4 replaces this with real adapter resolution + span-tap
    # registration. The debug log makes the no-op observable during bring-up.
    _log.debug(
        "instrument() called (M0 no-op): adapter=%s cost=%s behavior=%s quality=%s",
        adapter,
        resolved.cost_lane,
        resolved.behavior_lane,
        resolved.quality_lane,
    )
    return target


@overload
def meter(_fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def meter(
    _fn: None = None,
    *,
    budget: str | float | BudgetPolicy | None = ...,
    config: Config | None = ...,
    **overrides: object,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def meter(
    _fn: Callable[P, R] | None = None,
    *,
    budget: str | float | BudgetPolicy | None = None,
    config: Config | None = None,
    **overrides: object,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function so each call is metered as one run.

    Usable bare (``@meter``) or called (``@meter(budget="$0.50")``).

    Args:
        _fn: The decorated function, supplied positionally when used bare.
        budget: Optional per-run spend limit, e.g. ``"$0.50"``. Emitted as a
            signal; never enforced by agentmeter (ADR-001).
        config: Full configuration for runs started by this function.
        **overrides: Individual config fields.

    Returns:
        The decorated function, or a decorator when called with arguments.

    Raises:
        AgentMeterConfigError: If the budget or an override is invalid. Raised at
            decoration time.

    Example:
        >>> @meter(budget="$0.50")
        ... def run_agent(prompt: str) -> str:
        ...     return prompt.upper()
        >>> run_agent("hi")
        'HI'
    """
    resolved_config = (config or default_config()).merged_with(**overrides)
    resolved_budget = BudgetPolicy.parse(budget) if budget is not None else None

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with RunContext(budget=resolved_budget, config=resolved_config):
                return fn(*args, **kwargs)

        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate
