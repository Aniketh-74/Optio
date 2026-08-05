"""Per-run identity and lifecycle anchor.

Every signal optio emits is scoped to a run, so ``run_id`` is the join key
across all three lanes and the state store.

Two invariants carry weight here (M1-2 acceptance criteria):

1. **Unique id per run.** Ids are UUID4 hex; concurrent and nested runs never
   collide.
2. **Idempotent end.** ``end()`` may be called more than once -- by the context
   manager on exit, by an adapter's own run-end hook, by a framework callback --
   but run-end work (reconcile sweep, quality scoring) must fire *exactly once*.
   A double-fire would double-count cost, which is the silent-wrong class of bug
   R-TECH-1 exists to prevent.

Nesting: runs are tracked in a :class:`~contextvars.ContextVar` stack, so an
inner run restores the outer one on exit and async tasks each see their own
current run.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from contextvars import ContextVar, Token
from types import TracebackType
from typing import TYPE_CHECKING, Final, Literal

from opentelemetry import trace

from optio.config import BudgetPolicy, Config, default_config
from optio.runtime.failopen import guard

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from opentelemetry.trace import Span, Tracer

    # typing.Self is 3.11+; the floor is 3.10. Type-checking-only, so this adds
    # no runtime dependency.
    from typing_extensions import Self

_log = logging.getLogger("optio")

#: Callables invoked when any run ends, in registration order.
#:
#: A registry rather than a direct call into the span tap: the tap imports this
#: module, so importing it back would be a cycle. Inverting the dependency keeps
#: run lifecycle ignorant of what observes it, which is also what lets the tap
#: be swapped or absent entirely.
_run_end_observers: list[Callable[[RunContext], None]] = []


def register_run_end_observer(observer: Callable[[RunContext], None]) -> None:
    """Register a callable to run when any run ends.

    Args:
        observer: Called with the ending run. Invoked behind the fail-open
            guard, so it may raise without reaching the agent.
    """
    if observer not in _run_end_observers:
        _run_end_observers.append(observer)


def unregister_run_end_observer(observer: Callable[[RunContext], None]) -> None:
    """Remove a previously registered observer.

    Args:
        observer: The observer to remove. Removing an unregistered observer is
            not an error.
    """
    if observer in _run_end_observers:
        _run_end_observers.remove(observer)


_current_run: Final[ContextVar[RunContext | None]] = ContextVar("optio_current_run", default=None)


#: Tracer used for run spans this module opens itself. Set by the installer.
#:
#: The same inversion as :data:`_run_end_observers`, and for the same reason: a
#: run has to reach something the installer configured, and importing the
#: installer from here would be a cycle. ``None`` falls back to the global
#: provider, which is right until a user passes one of their own -- at which
#: point ``trace.get_tracer`` would put their run spans somewhere nothing they
#: configured is listening.
_default_tracer: Tracer | None = None


def set_default_tracer(tracer: Tracer | None) -> None:
    """Set the tracer used for run spans opened by :class:`RunContext`.

    Args:
        tracer: Tracer from the provider the tap was installed on, or ``None``
            to fall back to the global provider.
    """
    global _default_tracer
    _default_tracer = tracer


def current_run() -> RunContext | None:
    """Return the run active in this context, if any.

    Returns:
        The innermost active :class:`RunContext`, or ``None`` outside a run.
    """
    return _current_run.get()


class RunContext:
    """A single metered agent run.

    Usable directly as a context manager for raw or unsupported frameworks::

        with RunContext(budget="$0.50") as run:
            ...  # spans emitted inside are governed

    Attributes:
        run_id: Stable unique identifier for this run.
        budget: Optional spend limit, emitted as a signal, never enforced.
        config: Configuration governing which lanes are active.
        sampled: Whether this run is selected for the expensive quality path.
        successes: Successful tasks recorded for this run, or ``None`` when the
            run was never scored. Written by the quality lane (M5) and read by
            the cost lane as the denominator of ``cost_per_successful_task``.
            ``None`` rather than ``0`` is the whole point: an unscored run has an
            *unknown* success count, and treating it as zero would make the
            headline unit-economics number infinite for every run of the default
            configuration.
        actual_cost: Final reconciled cost of this run in USD, or ``None`` when
            nothing could be priced. Written by the cost lane at run end and
            read by the quality lane's deferred emitter, which needs a cost and
            a judged outcome together and is the only place both are known --
            the judge answers after the run span has closed. The exact mirror of
            ``successes``, which travels the other way, and for the same reason:
            the two lanes may not import each other (Section 3.1).
    """

    __slots__ = (
        "_ended",
        "_span",
        "_token",
        "actual_cost",
        "budget",
        "config",
        "ended_at",
        "run_id",
        "sampled",
        "started_at",
        "successes",
    )

    def __init__(
        self,
        *,
        run_id: str | None = None,
        budget: str | float | BudgetPolicy | None = None,
        config: Config | None = None,
    ) -> None:
        """Create a run.

        Args:
            run_id: Explicit id. Generated when omitted; supply one only to
                correlate with an external system's identifier.
            budget: Per-run spend limit, e.g. ``"$0.50"``.
            config: Configuration. Defaults to environment-resolved values.

        Raises:
            OptioConfigError: If the budget is unparseable. Setup-time only.
        """
        self.run_id: str = run_id or uuid.uuid4().hex
        self.budget: BudgetPolicy | None = (
            BudgetPolicy.parse(budget) if budget is not None else None
        )
        self.config: Config = config or default_config()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.successes: int | None = None
        self.actual_cost: float | None = None
        self._ended: bool = False
        self._token: Token[RunContext | None] | None = None
        self._span: AbstractContextManager[Span] | None = None
        self.sampled: bool = self._decide_sampling()

    def _decide_sampling(self) -> bool:
        """Decide once, at construction, whether this run takes the costly path.

        Frozen for the run's lifetime rather than re-rolled per query. Re-rolling
        would let run start and run end disagree about whether a run is sampled,
        which produces quality signals attached to runs that were never scored
        -- the silent-wrong class of bug, not a crash.

        The decision is made even when the quality lane is off, so enabling the
        lane mid-flight cannot retroactively change a run's tier. Selecting the
        tier itself (heuristic-inline vs async LLM-judge) is M5-1's job; this
        only answers "is this run eligible".

        Returns:
            ``True`` if this run is selected for sampling.
        """
        if not self.config.quality_lane:
            return False
        rate = self.config.quality_sample_rate
        # Exact 0.0 and 1.0 are worth short-circuiting: random() returns a value
        # in [0, 1), so `random() < 1.0` is always true but `< 0.0` is never,
        # and being explicit keeps the boundary behaviour obvious.
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return random.random() < rate

    @property
    def is_active(self) -> bool:
        """Whether the run has started and not yet ended."""
        return self.started_at is not None and not self._ended

    def start(self) -> Self:
        """Mark the run started and make it the current run.

        Idempotent: a second call is a no-op that leaves the first start time and
        context token intact.

        Returns:
            This run, so ``RunContext(...).start()`` chains.
        """
        if self.started_at is not None:
            return self
        self.started_at = time.monotonic()
        self._token = _current_run.set(self)
        self._open_span()
        return self

    def _open_span(self) -> None:
        """Open a run span when the caller has not provided one.

        Signals are span attributes, so they need a live span to be written to.
        ``@meter`` opens one; a bare ``with RunContext(...)`` -- the documented
        path for a raw SDK loop or an unsupported framework -- did not, and
        every signal it produced was computed and then dropped for want of
        somewhere to put it. The run worked, the budget parsed, and nothing was
        emitted.

        Nothing is opened when a recording span is already current, so the
        ``@meter`` and adapter paths are untouched and a run never gets two.
        """
        current = trace.get_current_span()
        if current is not trace.INVALID_SPAN and current.is_recording():
            return

        tracer = _default_tracer if _default_tracer is not None else trace.get_tracer("optio")
        # Held as the context manager rather than the span, so exiting restores
        # whatever context was current before -- ending the span alone would
        # leave it attached as current for everything the caller does next.
        span = tracer.start_as_current_span(f"optio.run.{self.run_id[:8]}")
        span.__enter__()
        self._span = span

    def _close_span(self) -> None:
        """Close the span opened by :meth:`_open_span`, if there was one.

        Best-effort: a run whose span cannot be closed is a dropped signal, not
        a broken agent (ADR-004). Called after the run-end observers, so their
        signals still land on it.
        """
        span, self._span = self._span, None
        if span is None:
            return
        try:
            span.__exit__(None, None, None)
        except Exception as error:  # noqa: BLE001 - a dropped span never breaks the agent
            _log.debug("could not close the run span (%s)", type(error).__name__)

    def end(self) -> None:
        """Mark the run ended and run end-of-run work exactly once.

        Idempotent by contract (M1-2). Subsequent calls return immediately
        without re-firing reconcile or quality hooks.
        """
        if self._ended:
            return
        self._ended = True
        self.ended_at = time.monotonic()

        # Run-end work -- the cost lane's close/reconcile sweep (M2), quality
        # scoring (M5). Each observer is guarded individually so one failing
        # cannot stop the others, and none can surface to the agent (ADR-004).
        # The context is cleared afterwards in a `finally`, because a stale
        # current run would contaminate everything the agent does next.
        try:
            for observer in tuple(_run_end_observers):
                guard(observer, None, self, component="run_end")
        finally:
            # Observers first, then the span they wrote to. Closing it earlier
            # would drop exactly the end-of-run signals -- the final cost, the
            # loop verdict -- that this whole call exists to produce.
            self._close_span()
            self._clear_current()

    def _clear_current(self) -> None:
        """Restore the previously-current run, best-effort.

        ``ContextVar.reset`` raises ``ValueError`` when the token was created in a
        different context -- which happens whenever ``start()`` and ``end()`` land
        in different contexts, e.g. a run started in a callback and ended by the
        framework's own run-end hook, or a run spanning an ``asyncio`` task
        boundary.

        Letting that escape would be a live instance of the bug ADR-004 exists to
        prevent: run-end sits on the agent's path, so a failure here would surface
        as an agent-visible error. Worse, it would abort ``end()`` mid-way and
        leave a stale run marked current for everything that follows. So the reset
        is best-effort, with a hard clear as the fallback.
        """
        token, self._token = self._token, None
        if token is None:
            return
        try:
            _current_run.reset(token)
        except ValueError:
            # Token belongs to another context; clear rather than restore. The
            # outer run is lost in this context, which is strictly better than a
            # stale inner run leaking into subsequent work.
            _current_run.set(None)
            _log.debug("run %s ended in a different context than it started", self.run_id)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration, or ``None`` if the run has not ended."""
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def __enter__(self) -> Self:
        """Start the run on scope entry."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """End the run on scope exit, including when the agent raised.

        Returns:
            Always ``False``, so an exception raised by the agent propagates
            normally. optio observes runs; it never swallows the user's
            errors. This is the standard context-manager contract, stated
            explicitly because returning anything truthy here would silently
            discard agent failures.
        """
        self.end()
        return False

    def __repr__(self) -> str:
        """Return a debug representation with a short id and lifecycle state."""
        state = "active" if self.is_active else ("ended" if self._ended else "new")
        return f"<RunContext {self.run_id[:8]} {state}>"
