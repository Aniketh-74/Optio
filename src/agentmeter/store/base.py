"""The StateStore contract.

Per-run state (ledger entries, behavior window, quality results) lives behind
this ABC so the in-memory default and the optional Redis backend are
interchangeable (ADR-005). In-memory is the default because zero new
infrastructure is what makes five-minute first value possible (SC-1); Redis
exists for multi-process and distributed runs.

Store failures are lane failures: a backend that is slow or unreachable must
degrade to a dropped signal, never to a blocked agent (ADR-004). Implementations
may raise :class:`~agentmeter.errors.StateStoreError`; the fail-open guard
absorbs it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateStore(ABC):
    """Per-run state persistence keyed by ``run_id``."""

    @abstractmethod
    def get(self, run_id: str, key: str) -> Any | None:
        """Return a stored value.

        Args:
            run_id: The run's identifier.
            key: State key within the run.

        Returns:
            The value, or ``None`` if absent.
        """

    @abstractmethod
    def set(self, run_id: str, key: str, value: Any) -> None:
        """Store a value.

        Args:
            run_id: The run's identifier.
            key: State key within the run.
            value: Value to store.
        """

    @abstractmethod
    def incr(self, run_id: str, key: str, delta: float) -> float:
        """Atomically add to a numeric value.

        Atomicity matters: the ledger's running totals are updated through this
        method, and a lost update is a wrong cost signal (R-TECH-1).

        Args:
            run_id: The run's identifier.
            key: State key within the run.
            delta: Amount to add. May be negative.

        Returns:
            The value after the increment.
        """

    @abstractmethod
    def delete(self, run_id: str) -> None:
        """Evict all state for a run.

        Called at run end. Idempotent -- deleting an unknown run is not an error.

        Args:
            run_id: The run's identifier.
        """
