"""What a ledger backend must do, and the view it returns.

Supersedes ADR-005's generic ``StateStore``. That interface offered
``get``/``set``/``incr``/``delete`` and could not express this one:
:meth:`LedgerStore.reconcile` has to check a reservation is open, remove it, and
fold the cost into the total **together**, and a caller holding primitives
cannot do those three things atomically across processes.

So the store speaks the domain instead. Each backend keeps the exactly-once
promise its own way -- a lock in memory, a Lua script in Redis -- and the
promise is enforced where the check and the mutation happen together, which is
the only place it can be.

:class:`LedgerSnapshot` lives here rather than beside ``CostLedger`` so both the
facade and every backend can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


class LedgerStore(Protocol):
    """Reserve/reconcile accounting for many concurrent runs.

    Implementations must be safe under concurrent use. In-process that means a
    lock; across processes it means the backend's own atomicity, because a lock
    held in one process says nothing to another.
    """

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record a step's worst-case cost before it runs.

        Re-reserving the same ``step_id`` replaces the previous reservation
        rather than adding to it: frameworks retry steps and reuse their ids,
        and stacking would inflate ``reserved`` for the rest of the run.

        Args:
            run_id: The run's identifier.
            step_id: Identifier for this step, stable across a retry.
            projected: Worst-case cost in USD. Finite and non-negative.

        Raises:
            LedgerInvariantError: If ``projected`` is negative, or the run is
                closed.
        """
        ...

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost.

        Exactly-once: the reservation is removed as it is reconciled, so a
        second call finds nothing open and raises rather than double-counting.

        Args:
            run_id: The run's identifier.
            step_id: The step being reconciled.
            actual: Actual cost in USD.

        Raises:
            LedgerInvariantError: On a double reconcile, a reconcile with no
                matching reservation, a negative cost, or a closed run.
        """
        ...

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        """Return a consistent view of a run's cost state.

        Args:
            run_id: The run's identifier.

        Returns:
            The snapshot. An unknown run yields an all-zero snapshot.
        """
        ...

    def close_run(self, run_id: str) -> LedgerSnapshot:
        """Finalise a run, recording any reservations left open.

        Idempotent: run end can fire more than once.

        Args:
            run_id: The run's identifier.

        Returns:
            The final snapshot, with ``leaked_steps`` set.
        """
        ...

    def is_finalised(self, run_id: str) -> bool:
        """Whether this run has been closed, whether or not its state survives.

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run was closed.
        """
        ...

    def knows(self, run_id: str) -> bool:
        """Whether this store has ever recorded anything for a run.

        Distinct from :meth:`is_finalised`: an all-zero snapshot for a run
        nobody metered is indistinguishable from one for a run that has not
        started, and only the first is a lie (ADR-044).

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run has state here.
        """
        ...

    def evict(self, run_id: str) -> None:
        """Drop a run's state. Finality outlives eviction. Idempotent.

        Args:
            run_id: The run's identifier.
        """
        ...

    def run_count(self) -> int:
        """How many runs currently hold state.

        Returns:
            Number of tracked runs. Unbounded growth means runs are not being
            evicted.
        """
        ...
