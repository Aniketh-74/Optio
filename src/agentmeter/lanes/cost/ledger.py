"""The reserve/reconcile cost ledger (M2-2) -- the moat.

This module holds the invariant the product's core number depends on
(**R-TECH-1**):

    *reserve always precedes the step; reconcile replaces the reservation
    exactly once.*

Two ways to break it, both of which produce a **wrong number rather than an
error**:

* **A reserve with no reconcile leaks budget.** The run looks more expensive
  than it was, and ``budget_remaining`` under-reports for the rest of the run.
* **A double reconcile under-counts.** The run looks cheaper than it was, which
  is the dangerous direction: a budget policy lets a run continue that should
  have been gated.

Neither failure raises on its own. That is why this file gets property tests
over random interleavings rather than review alone -- fail-open protects against
crashes, not against arithmetic that is confidently incorrect (ADR-004).

**Why reserve at all?** A ledger that only recorded actuals could not answer
"can this run afford the next step" until after the tokens burned. Reserving the
worst case *before* the step is what makes pre-spend gating possible, which is
the whole economic argument for this library (``docs/signals.md``).

Design notes that carry weight:

* **Reservations are keyed by step id.** Not a running counter -- a framework
  that retries a step reuses its id, and the retry must replace the original
  reservation rather than stack a second one on top.
* **Totals are derived, never accumulated separately.** ``reserved`` is the sum
  of open reservations, computed on read. A separately-maintained running total
  is one missed decrement away from silent drift, and the drift would look
  plausible.
* **Every mutation holds a lock.** Steps run concurrently in most frameworks.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Final

from agentmeter.errors import LedgerInvariantError

_log: Final = logging.getLogger("agentmeter")


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """A consistent view of one run's cost state.

    Attributes:
        reserved: Sum of open (unreconciled) reservations, in USD.
        actual: Sum of reconciled costs, in USD.
        committed: ``reserved + actual`` -- the worst case if every open step
            costs its full reservation.
        open_steps: How many reservations are awaiting reconciliation.
        reconciled_steps: How many steps have been reconciled.
        leaked_steps: Reservations abandoned without reconciliation, detected at
            run end.
    """

    reserved: float
    actual: float
    committed: float
    open_steps: int
    reconciled_steps: int
    leaked_steps: int


@dataclass(slots=True)
class _RunLedger:
    """Mutable per-run state. Guarded by the ledger's lock."""

    #: Open reservations by step id: step_id -> reserved USD.
    open: dict[str, float] = field(default_factory=dict)
    actual: float = 0.0
    reconciled_steps: int = 0
    leaked_steps: int = 0
    closed: bool = False


#: How many recently-closed run ids to remember after their state is evicted.
#:
#: Closing is final (ADR-010), but eviction has to release memory or a
#: long-lived agent process leaks a few hundred bytes per run forever. These two
#: requirements conflict: forget a run entirely and a straggling callback can
#: reserve against its id again, silently starting a *new* total under a run
#: that was already reported.
#:
#: A bounded FIFO of closed ids resolves it. Finality survives eviction for the
#: most recent N runs -- comfortably longer than any real straggler -- while
#: memory stays capped. Beyond N, a late arrival is treated as a fresh run,
#: which is the same behaviour as a process restart and is acceptable because
#: nothing that old can still be in flight.
_CLOSED_MEMORY: Final = 4096


class CostLedger:
    """Per-run reserve/reconcile accounting.

    Holds state for many concurrent runs, keyed by ``run_id``. Call
    :meth:`close_run` to finalise a run and :meth:`evict` to release its state;
    the cost lane does both at run end.
    """

    def __init__(self, closed_memory: int = _CLOSED_MEMORY) -> None:
        """Create an empty ledger.

        Args:
            closed_memory: How many recently-closed run ids to remember after
                eviction, so a late reconcile is still rejected.
        """
        self._lock: Final = threading.RLock()
        self._runs: Final[dict[str, _RunLedger]] = {}
        self._closed_memory: Final = closed_memory
        self._recently_closed: Final[OrderedDict[str, None]] = OrderedDict()

    def _remember_closed(self, run_id: str) -> None:
        """Record a run id as closed, evicting the oldest when full.

        Args:
            run_id: The run that was closed.
        """
        self._recently_closed[run_id] = None
        self._recently_closed.move_to_end(run_id)
        while len(self._recently_closed) > self._closed_memory:
            self._recently_closed.popitem(last=False)

    def _is_closed(self, run_id: str) -> bool:
        """Whether a run has been closed, including after eviction.

        Args:
            run_id: The run to check.

        Returns:
            ``True`` if the run is closed or was recently closed.
        """
        ledger = self._runs.get(run_id)
        if ledger is not None and ledger.closed:
            return True
        return run_id in self._recently_closed

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record the worst-case cost of a step *before* it runs.

        Re-reserving the same ``step_id`` **replaces** the previous reservation
        rather than adding to it. Frameworks retry steps and reuse their ids;
        stacking reservations would inflate ``reserved`` and under-report
        ``budget_remaining`` for the rest of the run.

        Args:
            run_id: The run's identifier.
            step_id: Identifier for this step, stable across a retry.
            projected: Worst-case cost of the step in USD. Must be finite and
                non-negative.

        Raises:
            LedgerInvariantError: If ``projected`` is negative. Absorbed by the
                fail-open guard at the call site; the signal is dropped and the
                agent proceeds.
        """
        if projected < 0:
            raise LedgerInvariantError(
                f"cannot reserve a negative cost ({projected}) for {run_id}/{step_id}"
            )

        with self._lock:
            if self._is_closed(run_id):
                raise LedgerInvariantError(
                    f"cannot reserve on closed run {run_id!r}; the run's cost "
                    f"has already been reported"
                )
            ledger = self._runs.setdefault(run_id, _RunLedger())
            previous = ledger.open.get(step_id)
            if previous is not None:
                _log.debug(
                    "re-reserving %s/%s: %s replaces %s",
                    run_id,
                    step_id,
                    projected,
                    previous,
                )
            ledger.open[step_id] = projected

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost.

        Exactly-once by construction: the reservation is removed as it is
        reconciled, so a second call for the same step finds nothing open and
        raises rather than adding the cost twice.

        Args:
            run_id: The run's identifier.
            step_id: The step being reconciled.
            actual: Actual cost of the step in USD.

        Raises:
            LedgerInvariantError: On a double reconcile, on a reconcile with no
                matching reservation, or on a negative cost. All three mean the
                caller violated the ordering rule, and continuing would produce
                a wrong total rather than a missing one.
        """
        if actual < 0:
            raise LedgerInvariantError(
                f"cannot reconcile a negative cost ({actual}) for {run_id}/{step_id}"
            )

        with self._lock:
            if self._is_closed(run_id):
                # A straggling callback after the run's cost was reported.
                # Folding it in now would silently change a number a policy may
                # already have acted on -- or, after eviction, start a brand new
                # total under a run id that has already been finalised.
                raise LedgerInvariantError(
                    f"cannot reconcile {run_id}/{step_id} on a closed run; "
                    f"the run's cost has already been reported"
                )

            ledger = self._runs.get(run_id)
            if ledger is None:
                raise LedgerInvariantError(
                    f"reconcile for unknown run {run_id!r}; reserve must come first"
                )

            if step_id not in ledger.open:
                # Either a second reconcile for this step, or a step that was
                # never reserved. Both are ordering violations; neither may be
                # folded into the total, because the total is what a budget
                # policy gates real money on.
                raise LedgerInvariantError(
                    f"no open reservation for {run_id}/{step_id}; "
                    f"either reconciled twice or never reserved"
                )

            del ledger.open[step_id]
            ledger.actual += actual
            ledger.reconciled_steps += 1

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        """Return a consistent view of a run's cost state.

        Totals are derived from the open reservations on every read rather than
        maintained alongside them. A separately-accumulated total is one missed
        decrement away from drifting, and the drift would look plausible.

        **Cost is O(open reservations)**, not O(steps). In normal operation
        reservations close immediately, so this is flat at any run length
        (~2 microseconds, measured). It degrades only when reservations stay
        open -- the unpriceable-model case, where every step leaks -- reaching
        roughly 96 microseconds at 10,000 open steps. Still two orders of
        magnitude inside the SC-5 budget, but the shape is quadratic in that
        pathological case, so a run with very many unpriced steps is the one to
        watch. Verified by ``tests/bench/test_overhead.py``.

        Args:
            run_id: The run's identifier.

        Returns:
            The snapshot. An unknown run yields an all-zero snapshot, since a
            run that has spent nothing and a run that does not exist are the
            same thing to a consumer.
        """
        with self._lock:
            ledger = self._runs.get(run_id)
            if ledger is None:
                return LedgerSnapshot(0.0, 0.0, 0.0, 0, 0, 0)

            reserved = sum(ledger.open.values())
            return LedgerSnapshot(
                reserved=reserved,
                actual=ledger.actual,
                committed=reserved + ledger.actual,
                open_steps=len(ledger.open),
                reconciled_steps=ledger.reconciled_steps,
                leaked_steps=ledger.leaked_steps,
            )

    def close_run(self, run_id: str) -> LedgerSnapshot:
        """Finalise a run, recording any leaked reservations.

        A reservation still open at run end never got its actual cost. The
        reservation is *kept* in the returned snapshot rather than discarded:
        dropping it would make the run look cheaper than the evidence supports,
        and under-reporting cost is the direction that lets an over-budget run
        through. Discarding is the guess; keeping the worst case is the honest
        answer.

        Closing is **final**. Once a run's cost has been reported, a late
        reconcile arriving from a straggling callback would silently change a
        number a policy may already have acted on, so it is rejected instead.
        Calling this method again is safe and returns the same snapshot -- run
        end can fire more than once (M1-2).

        Args:
            run_id: The run's identifier.

        Returns:
            The final snapshot, with ``leaked_steps`` set to the number of
            reservations that were never reconciled.
        """
        with self._lock:
            # Remembered separately from the run's state, so finality outlives
            # eviction: the state is released to bound memory, the id is not.
            self._remember_closed(run_id)

            # A run with no state is still closed by this call. Returning early
            # would leave it re-openable, so a late reserve could start
            # accumulating cost against a run that has already been reported --
            # and the run would look like it began after it ended.
            ledger = self._runs.setdefault(run_id, _RunLedger())

            if ledger.closed:
                return self.snapshot(run_id)

            # Assigned rather than accumulated: the leak count describes what
            # was open at close. Accumulating would let the number drift on a
            # repeated close.
            leaked = len(ledger.open)
            ledger.leaked_steps = leaked
            ledger.closed = True

            if leaked:
                _log.warning(
                    "agentmeter: run %s ended with %d unreconciled reservation(s); "
                    "cost is reported as the reserved worst case for those steps. "
                    "See docs/runbooks.md.",
                    run_id,
                    leaked,
                )

            return self.snapshot(run_id)

    def is_finalised(self, run_id: str) -> bool:
        """Whether this run has already been closed.

        Distinct from "unknown": a caller needs to tell *this run is over* from
        *I have never seen this run*, because an all-zero snapshot looks
        identical either way. Deriving signals from the zeros of a finalised
        run would report a full budget and no spend.

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run was closed, whether or not its state survives.
        """
        with self._lock:
            return self._is_closed(run_id)

    def evict(self, run_id: str) -> None:
        """Drop all state for a run.

        Separate from :meth:`close_run` so a caller can read the final snapshot
        before the state disappears. Idempotent.

        Args:
            run_id: The run's identifier.
        """
        with self._lock:
            self._runs.pop(run_id, None)

    def run_count(self) -> int:
        """Return how many runs currently hold state.

        Returns:
            Number of tracked runs. Growth without bound means runs are not
            being evicted.
        """
        with self._lock:
            return len(self._runs)

    def __repr__(self) -> str:
        """Return a debug representation with the tracked run count."""
        return f"<CostLedger runs={self.run_count()}>"
