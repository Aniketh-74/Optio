"""The fail-open guard -- nothing in ``runtime/`` or ``lanes/`` raises past this.

This is the single most important safety component in the library (ADR-004,
SC-4, R-TECH-4). ``optio`` runs in-process, on the critical path of every
agent step, so the asymmetry is severe:

* A dropped signal costs the user one gap in a graph.
* A raised exception costs the user a failed agent run.

There is no observability value that justifies the second outcome, so every lane
and runtime entry point is wrapped by :func:`guard` or :func:`guarded`.

Three rules shape the implementation:

1. **Catch** :class:`Exception` **broadly, not just our own types.** A lane
   raising a plain ``KeyError`` must not break the agent either. The guarantee
   is about the agent's safety, not about our exception hygiene (ADR-004).
2. **Never catch** :class:`BaseException`. ``KeyboardInterrupt``,
   ``SystemExit``, and ``GeneratorExit`` are control flow belonging to the
   user's process. Swallowing a Ctrl-C to protect a cost calculation would make
   the library un-killable, which is its own kind of breaking the agent.
3. **Setup errors are exempt.** :class:`~optio.errors.OptioConfigError`
   propagates. Failing open at runtime is safety; failing open at setup would
   mean silently shipping a meter that measures nothing (Section 4.2).

The guard body itself is written so that it cannot raise even if the pieces it
depends on misbehave -- a fallback that explodes, or a logging handler that
does. A guard that could raise while handling a failure would defeat its own
purpose.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Final, TypeVar

from optio.errors import OptioConfigError
from optio.runtime import selfobs

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec

    P = ParamSpec("P")

T = TypeVar("T")
S = TypeVar("S")

_log: Final = logging.getLogger("optio")

#: Exception types that are *not* absorbed. Configuration is a setup-time
#: contract with the user, and must fail loudly rather than degrade silently.
_NEVER_ABSORB: Final[tuple[type[BaseException], ...]] = (OptioConfigError,)

#: Fail-open activations, keyed by ``(component, exception type name)``. Feeds
#: the ``optio.internal.lane_errors`` self-metric (Section 12) and is the
#: mitigation for the "bugs hide" cost accepted in ADR-004: a rising count is
#: how a silently-broken lane becomes visible.
_activations: Final[dict[tuple[str, str], int]] = {}

#: Components that have already logged. Each component logs its *first*
#: activation at WARN, then falls silent -- a lane failing once per span would
#: otherwise flood the user's logs at exactly the moment their agent is
#: struggling, turning our bug into their second incident.
_logged_components: Final[set[str]] = set()


def _record(component: str, exc: BaseException) -> None:
    """Count one fail-open activation and log the first per component.

    Deliberately total: any failure inside bookkeeping is discarded, because
    this runs while already handling a failure. A raise here would escape
    :func:`guard` and break the agent -- the exact outcome the guard exists to
    prevent.

    Args:
        component: Lane or runtime component name, used in logs and metrics.
        exc: The absorbed exception. Only its *type* is logged, never its
            payload -- exception messages can contain prompt or completion
            content, which this library never logs (Section 7.2, Section 10).
    """
    try:
        exc_type = type(exc).__name__
        key = (component, exc_type)
        _activations[key] = _activations.get(key, 0) + 1

        # Publish to the metric as well as the in-process counter. The counter
        # is only readable by a test; an operator needs this on their dashboard,
        # because a lane that fails on every span is invisible by construction
        # -- the agent keeps working, which is the whole point of the guard.
        selfobs.record_lane_error(component)

        if component not in _logged_components:
            _logged_components.add(component)
            _log.warning(
                "optio: %s failed with %s; signal dropped, agent unaffected. "
                "Further failures from this component are counted but not logged. "
                "See docs/runbooks.md.",
                component,
                exc_type,
                exc_info=exc,
            )
    except Exception:  # noqa: BLE001 - see docstring: bookkeeping must be total
        pass


def guard(
    fn: Callable[..., T],
    fallback: T,
    *args: Any,
    component: str = "runtime",
    **kwargs: Any,
) -> T:
    """Call ``fn``; on any internal failure return ``fallback`` instead.

    The one-shot form, for a call site that is not itself a reusable function.
    Use :func:`guarded` to wrap a function once at definition time.

    Args:
        fn: The callable to protect.
        fallback: Value returned when ``fn`` raises. Should represent "no
            signal" rather than a neutral-looking number: absence is a valid
            signal state, and a fabricated zero is indistinguishable from a
            real one downstream (ADR-004, ``docs/signals.md``).
        *args: Positional arguments for ``fn``.
        component: Name used in the WARN log and the activation counter.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        ``fn``'s result, or ``fallback`` if it raised.

    Raises:
        OptioConfigError: Setup errors are never absorbed.
        BaseException: ``KeyboardInterrupt``, ``SystemExit`` and other
            non-:class:`Exception` control flow propagate untouched.
    """
    try:
        return fn(*args, **kwargs)
    except _NEVER_ABSORB:
        raise
    except Exception as exc:  # noqa: BLE001 - absorbing everything is the point
        _record(component, exc)
        return fallback


def guarded(
    fallback: T,
    *,
    component: str = "runtime",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap a function so it can never raise past its own boundary.

    The decorator form, for functions on the hot path::

        @guarded(fallback=[], component="cost")
        def process_span(span, run) -> list[Signal]:
            ...

    Args:
        fallback: Value returned when the wrapped function raises.
        component: Name used in the WARN log and the activation counter.

    Returns:
        A decorator producing a guarded version of the function, with its
        signature and metadata preserved.
    """

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return fn(*args, **kwargs)
            except _NEVER_ABSORB:
                raise
            except Exception as exc:  # noqa: BLE001 - absorbing is the point
                _record(component, exc)
                return fallback

        return wrapper

    return decorate


def guard_signals(
    fn: Callable[..., list[S]],
    *args: Any,
    component: str = "runtime",
    **kwargs: Any,
) -> list[S]:
    """Call a signal-producing function, returning no signals if it fails.

    The overwhelmingly common guard call is "run this lane; on failure emit
    nothing". Spelling that with :func:`guard` requires an annotated empty-list
    variable at the call site, because a bare ``[]`` makes the type checker
    infer ``list[Never]`` for the return type. This wrapper carries the element
    type through instead.

    Args:
        fn: A callable returning a list of signals.
        *args: Positional arguments for ``fn``.
        component: Name used in the WARN log and the activation counter.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        ``fn``'s signals, or an empty list if it raised.
    """
    empty: list[S] = []
    return guard(fn, empty, *args, component=component, **kwargs)


def activation_count(component: str | None = None) -> int:
    """Return how many times the guard has absorbed a failure.

    Exposed for the fault-injection suite and for the
    ``optio.internal.lane_errors`` self-metric (Section 12). A rising count
    means a lane is broken and its signals are missing -- the user's agent is
    still safe, which is the trade ADR-004 makes explicit.

    Args:
        component: Restrict the count to one component. ``None`` totals all.

    Returns:
        Number of absorbed failures.
    """
    if component is None:
        return sum(_activations.values())
    return sum(count for (name, _), count in _activations.items() if name == component)


def reset_activations() -> None:
    """Clear activation counters and re-arm first-failure logging.

    Test-support only. Production code has no reason to erase its own error
    history.
    """
    _activations.clear()
    _logged_components.clear()
