"""Public API surface -- the only module users touch.

This is the frozen contract described in Section 8.1. Breaking a signature here
after M1 requires an ADR (Section 16 rule 12).

**Status:** the cost (M2) and behavior (M3) lanes are live, so an instrumented
agent emits spend, projection, budget headroom, and loop state today. The quality
lane is M5 and is off by default even once it lands (ADR-003), so its signals are
absent rather than zero until then -- see ``docs/signals.md`` on why that
distinction is load-bearing for downstream policies.

Failure discipline (Section 4.2): everything in this module raises *eagerly* on
bad input, because setup-time is exactly where a misconfiguration should be loud.
The opposite rule governs the runtime -- see ``runtime/failopen.py``.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast, overload

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from optio.config import BudgetPolicy, Config, default_config
from optio.errors import OptioConfigError
from optio.runtime.adapter_registry import load_adapter, resolve_adapter
from optio.runtime.installer import install_tap
from optio.runtime.run_context import RunContext

__all__ = ["RunContext", "instrument", "meter"]

_log = logging.getLogger("optio")

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
    """Attach optio to an agent.

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
        OptioConfigError: If ``adapter`` names an unknown framework, no
            adapter matches the target, or an override is invalid. Raised at call
            time -- never on the hot path.

    Example:
        >>> from optio import instrument
        >>> instrument(object())  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        optio.errors.UnsupportedFrameworkError: ...
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
        raise OptioConfigError(
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
            signal; never enforced by optio (ADR-001).
        config: Full configuration for runs started by this function.
        provider: Tracer provider for the run span. Defaults to the global one.
        **overrides: Individual config fields.

    Returns:
        The decorated function, or a decorator when called with arguments.

    Raises:
        OptioConfigError: If the budget or an override is invalid. Raised at
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
        tracer = provider.get_tracer("optio") if provider is not None else trace.get_tracer("optio")
        span_name = f"optio.run.{fn.__name__}"

        # An `async def` has to be wrapped by an `async def`. Wrapping one with
        # the sync branch below measures the *construction of the coroutine* --
        # `fn(...)` returns immediately, the `with` closes the run before a
        # single step has run, and every signal is silently absent. The
        # decorator looks applied and the agent works, which is the worst
        # possible way to be wrong, and most agent code is async.
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                # `RunContext` is a sync context manager used inside a
                # coroutine, which is correct: it sets a `ContextVar`, and each
                # asyncio task runs with its own copy of the context, so
                # concurrent runs cannot see each other's.
                with (
                    tracer.start_as_current_span(span_name),
                    RunContext(budget=resolved_budget, config=resolved_config),
                ):
                    return await fn(*args, **kwargs)

            # `P` is preserved; only the return type is unrepresentable here,
            # because `R` is already the coroutine type that `await` unwraps.
            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # The run span is the home for every run-scoped signal: it stays open
            # while the steps inside it start and end, and a span processor
            # cannot write back to a span that has already ended (see span_tap).
            with (
                tracer.start_as_current_span(span_name),
                RunContext(budget=resolved_budget, config=resolved_config),
            ):
                return fn(*args, **kwargs)

        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate
