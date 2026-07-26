"""In-process state store -- the default backend (ADR-005).

Zero new infrastructure is what makes five-minute first value possible (SC-1),
so this is what a user gets unless they ask for Redis.

**Thread safety is not optional here.** The ledger's running totals go through
:meth:`InMemoryStateStore.incr`, and agent frameworks routinely run steps on a
thread pool. A lost update would not raise; it would produce a cost total that
is quietly too low -- the silent-wrong class of bug R-TECH-1 exists to prevent.
So every mutation holds a lock, and ``incr`` is a genuine read-modify-write
under that lock rather than two separate operations.

Scope: state lives for the life of the process and is evicted per run by
:meth:`delete`. Runs spanning multiple processes need the Redis backend, since
each process would otherwise hold a partial view of the same run.
"""

from __future__ import annotations

import threading
from typing import Any, Final

from agentmeter.errors import StateStoreError


class InMemoryStateStore:
    """Per-run state held in a process-local dictionary.

    Structurally implements :class:`~agentmeter.store.base.StateStore` rather
    than inheriting it, so the store layer stays free of import-order concerns.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._lock: Final = threading.RLock()
        self._state: Final[dict[str, dict[str, Any]]] = {}

    def get(self, run_id: str, key: str) -> Any | None:
        """Return a stored value.

        Args:
            run_id: The run's identifier.
            key: State key within the run.

        Returns:
            The value, or ``None`` if absent.
        """
        with self._lock:
            return self._state.get(run_id, {}).get(key)

    def set(self, run_id: str, key: str, value: Any) -> None:
        """Store a value.

        Args:
            run_id: The run's identifier.
            key: State key within the run.
            value: Value to store.
        """
        with self._lock:
            self._state.setdefault(run_id, {})[key] = value

    def incr(self, run_id: str, key: str, delta: float) -> float:
        """Atomically add to a numeric value.

        The read, the addition and the write all happen under one lock. Doing
        them as separate operations would let two threads read the same total
        and each write back their own sum, losing one increment -- a cost signal
        that is wrong rather than missing.

        Args:
            run_id: The run's identifier.
            key: State key within the run.
            delta: Amount to add. May be negative.

        Returns:
            The value after the increment.

        Raises:
            StateStoreError: If the existing value is not numeric.
        """
        with self._lock:
            bucket = self._state.setdefault(run_id, {})
            current = bucket.get(key, 0.0)
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                raise StateStoreError(f"cannot increment non-numeric value at {run_id}/{key}")
            updated = float(current) + delta
            bucket[key] = updated
            return updated

    def delete(self, run_id: str) -> None:
        """Evict all state for a run.

        Idempotent: deleting an unknown run is not an error, because run end can
        fire more than once and must stay cheap and safe.

        Args:
            run_id: The run's identifier.
        """
        with self._lock:
            self._state.pop(run_id, None)

    def run_count(self) -> int:
        """Return how many runs currently hold state.

        Exposed for leak detection: a process whose run count grows without
        bound is failing to evict at run end.

        Returns:
            Number of runs with retained state.
        """
        with self._lock:
            return len(self._state)

    def __repr__(self) -> str:
        """Return a debug representation with the retained run count."""
        return f"<InMemoryStateStore runs={self.run_count()}>"
