"""Public API surface -- the only module users touch.

This is the frozen contract described in Section 8.1. Breaking a signature here
after M1 requires an ADR (Section 16 rule 12).

**M1 status:** ``instrument()`` resolves an adapter and installs the span tap, so
GenAI spans now reach the lane dispatch path. No lanes are registered yet -- cost
lands in M2, behavior in M3, quality in M5 -- so the pipeline runs end to end
while emitting nothing.

Failure discipline (Section 4.2): everything in this module raises *eagerly* on
bad input, because setup-time is exactly where a misconfiguration should be loud.
The opposite rule governs the runtime -- see ``runtime/failopen.py``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar, overload

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from agentmeter.config import BudgetPolicy, Config, default_config
from agentmeter.errors import AgentMeterConfigError
from agentmeter.runtime.adapter_registry import load_adapter, resolve_adapter
from agentmeter.runtime.installer import install_tap
from agentmeter.runtime.run_context import RunContext

__all__ = ["RunContext", "instrument", "meter"]

_log = logging.getLogger("agentmeter")

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def instrument(
    target: T,
    *,
    adapter: str | None = None,
    config: Config | None = None,
    provider: TracerProvider | None = None,
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
        provider: Tracer provider to install the span tap on. Defaults to the
            global one, which is the right choice for essentially all users.
        **overrides: Individual config fields, taking precedence over both
            ``config`` and the environment (Section 4.3).

    Returns:
        The same object that was passed in. Instrumentation is a side effect;
        the identity guarantee lets ``agent = instrument(agent)`` and a bare
        ``instrument(agent)`` behave identically.

    Raises:
        AgentMeterConfigError: If ``adapter`` names an unknown framework, no
            adapter matches the target, or an override is invalid. Raised at call
            time -- never on the hot path.

    Example:
        >>> from agentmeter import instrument
        >>> instrument(object())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        agentmeter.errors.UnsupportedFrameworkError: ...
    """
    resolved = (config or default_config()).merged_with(**overrides)

    selected = load_adapter(adapter) if adapter is not None else resolve_adapter(target)

    _log.debug(
        "instrument(): adapter=%s cost=%s behavior=%s quality=%s",
        selected.name,
        resolved.cost_lane,
        resolved.behavior_lane,
        resolved.quality_lane,
    )
    instrumented = selected.instrument(target, resolved, provider)

    # Adapters are contractually identity-returning (Section 6.7). Checking
    # rather than casting keeps a buggy adapter from silently handing the user a
    # different object than the one they passed -- `agent = instrument(agent)`
    # would then quietly replace their agent.
    if instrumented is not target:
        raise AgentMeterConfigError(
            f"adapter {selected.name!r} returned a different object than it was "
            f"given; instrument() must return the same agent"
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
    provider: TracerProvider | None = ...,
    **overrides: object,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def meter(
    _fn: Callable[P, R] | None = None,
    *,
    budget: str | float | BudgetPolicy | None = None,
    config: Config | None = None,
    provider: TracerProvider | None = None,
    **overrides: object,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function so each call is metered as one run.

    Usable bare (``@meter``) or called (``@meter(budget="$0.50")``).

    Args:
        _fn: The decorated function, supplied positionally when used bare.
        budget: Optional per-run spend limit, e.g. ``"$0.50"``. Emitted as a
            signal; never enforced by agentmeter (ADR-001).
        config: Full configuration for runs started by this function.
        provider: Tracer provider for the run span. Defaults to the global one.
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

    # @meter is the framework-agnostic entry point, so no adapter has run to
    # install the tap. Installing here is idempotent per provider.
    install_tap(resolved_config, provider)

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        tracer = (
            provider.get_tracer("agentmeter")
            if provider is not None
            else trace.get_tracer("agentmeter")
        )

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # The run span is the home for every run-scoped signal: it stays open
            # while the steps inside it start and end, and a span processor
            # cannot write back to a span that has already ended (see span_tap).
            with (
                tracer.start_as_current_span(f"agentmeter.run.{fn.__name__}"),
                RunContext(budget=resolved_budget, config=resolved_config),
            ):
                return fn(*args, **kwargs)

        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate
