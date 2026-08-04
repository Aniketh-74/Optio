"""The reserve/reconcile cost ledger (M2-2) -- the moat.

This module is the lane's entry point to cost accounting. The invariant it
guards (**R-TECH-1**) is:

    *reserve always precedes the step; reconcile replaces the reservation
    exactly once.*

Two ways to break it, both of which produce a **wrong number rather than an
error**:

* **A reserve with no reconcile leaks budget.** The run looks more expensive
  than it was, and ``budget_remaining`` under-reports for the rest of the run.
* **A double reconcile under-counts.** The run looks cheaper than it was, which
  is the dangerous direction: a budget policy lets a run continue that should
  have been gated.

Neither failure raises on its own. That is why this area gets property tests
over random interleavings rather than review alone -- fail-open protects against
crashes, not against arithmetic that is confidently incorrect (ADR-004).

**Why reserve at all?** A ledger that only recorded actuals could not answer
"can this run afford the next step" until after the tokens burned. Reserving the
worst case *before* the step is what makes pre-spend gating possible, which is
the whole economic argument for this library (``docs/signals.md``).

**Where the state lives is a backend decision** (ADR-050). :class:`CostLedger`
delegates to a :class:`~optio.lanes.cost.ledger_store.LedgerStore`: a
process-local dictionary by default, Redis when a run is sharded across
processes. The rules above hold either way, because each backend enforces them
where its own check and mutation happen together -- a lock in memory, one Lua
script in Redis.
"""

from __future__ import annotations

from typing import Final

from optio.lanes.cost.ledger_memory import _CLOSED_MEMORY, InMemoryLedgerStore
from optio.lanes.cost.ledger_store import LedgerSnapshot, LedgerStore

#: Re-exported so ``from optio.lanes.cost.ledger import LedgerSnapshot`` keeps
#: working for ``lane.py`` and ``project.py``, and ``_CLOSED_MEMORY`` for the
#: soak test that asserts the closed-run memory stays bounded. The definitions
#: moved to break an import cycle; the import paths did not have to.
__all__ = ["_CLOSED_MEMORY", "CostLedger", "LedgerSnapshot", "LedgerStore"]


class CostLedger:
    """Per-run reserve/reconcile accounting, delegated to a backend.

    Kept as the lane's entry point so the ledger's callers are unaffected by
    where state actually lives. The backend decides that: in-process by default,
    Redis when a run spans processes.
    """

    def __init__(
        self,
        store: LedgerStore | None = None,
        closed_memory: int = _CLOSED_MEMORY,
    ) -> None:
        """Create a ledger over a backend.

        Args:
            store: Backend to delegate to. An in-memory store when omitted,
                which preserves the pre-0.4 constructor exactly.
            closed_memory: How many recently-closed run ids to remember after
                eviction. Applies to the default in-memory backend only, and is
                ignored when ``store`` is supplied -- a backend that was handed
                in has already chosen its own retention, and silently
                overriding it would make this argument look effective when it
                is not.
        """
        self._store: Final[LedgerStore] = (
            store if store is not None else InMemoryLedgerStore(closed_memory=closed_memory)
        )

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record the worst-case cost of a step *before* it runs.

        Args:
            run_id: The run's identifier.
            step_id: Identifier for this step, stable across a retry.
            projected: Worst-case cost of the step in USD.

        Raises:
            LedgerInvariantError: If ``projected`` is negative, or the run is
                closed.
        """
        self._store.reserve(run_id, step_id, projected)

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost.

        Args:
            run_id: The run's identifier.
            step_id: The step being reconciled.
            actual: Actual cost of the step in USD.

        Raises:
            LedgerInvariantError: On a double reconcile, a reconcile with no
                matching reservation, a negative cost, or a closed run.
        """
        self._store.reconcile(run_id, step_id, actual)

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        """Return a consistent view of a run's cost state.

        Args:
            run_id: The run's identifier.

        Returns:
            The snapshot. An unknown run yields an all-zero snapshot.
        """
        return self._store.snapshot(run_id)

    def close_run(self, run_id: str) -> LedgerSnapshot:
        """Finalise a run, recording any leaked reservations.

        Args:
            run_id: The run's identifier.

        Returns:
            The final snapshot, with ``leaked_steps`` set.
        """
        return self._store.close_run(run_id)

    def is_finalised(self, run_id: str) -> bool:
        """Whether this run has already been closed.

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run was closed, whether or not its state survives.
        """
        return self._store.is_finalised(run_id)

    def knows(self, run_id: str) -> bool:
        """Whether this ledger has ever recorded anything for a run.

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run has state in this ledger (ADR-044).
        """
        return self._store.knows(run_id)

    def evict(self, run_id: str) -> None:
        """Drop all state for a run. Idempotent.

        Args:
            run_id: The run's identifier.
        """
        self._store.evict(run_id)

    def run_count(self) -> int:
        """Return how many runs currently hold state.

        Returns:
            Number of tracked runs.
        """
        return self._store.run_count()

    def __repr__(self) -> str:
        """Return a debug representation naming the backend and run count."""
        return f"<CostLedger backend={type(self._store).__name__} runs={self.run_count()}>"
