"""The in-process quality backend -- the default (ADR-050).

Replaces the ``dict[str, list[ReadableSpan]]`` the lane held through 0.3.0. Not
a verbatim move, unlike the ledger's and the behaviour window's: the list of
spans is gone entirely, because deriving the projection showed that run-end
reads a count and the final step and nothing else. Sixty-four retained spans, of
which one was ever inspected.

So the state per run is a counter and one :class:`QualityStep`, and
:data:`~optio.lanes.quality.lane.MAX_RETAINED_SPANS` has nothing left to bound.
Bounded memory stops being a policy someone has to remember to enforce and
becomes a property of the shape -- a run costs the same whether it takes three
steps or thirty thousand.

**Single-process by construction.** A lock says nothing to another process. A
run sharded across workers scores from whichever worker happened to observe run
end, using only the steps that landed in that worker -- so the judge is told a
fraction of the run's length and the heuristic scores a step that was last in
one process rather than last overall.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

from optio.lanes.quality.store import QualitySummary

if TYPE_CHECKING:
    from optio.lanes.quality.store import QualityStep


class InMemoryQualityStore:
    """Per-run scoring state in a process-local dictionary.

    Structurally implements :class:`~optio.lanes.quality.store.QualityStore`
    rather than inheriting it, matching the other two lanes' backends.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._lock: Final = threading.RLock()
        self._runs: Final[dict[str, QualitySummary]] = {}

    def record(self, run_id: str, step: QualityStep) -> None:
        """Note that a step happened, and that it was the latest.

        Args:
            run_id: The run's identifier. Its state is created on first use.
            step: The step's projection. Never a span -- see
                :mod:`optio.lanes.quality.store`.
        """
        with self._lock:
            previous = self._runs.get(run_id)
            self._runs[run_id] = QualitySummary(
                step_count=previous.step_count + 1 if previous is not None else 1,
                last=step,
            )

    def close_run(self, run_id: str) -> QualitySummary | None:
        """Release a run's state and return what it held.

        Args:
            run_id: The run's identifier.

        Returns:
            The summary, or ``None`` if the run held no state -- either it
            never recorded a step, or run end already fired once (M1-2).
        """
        with self._lock:
            return self._runs.pop(run_id, None)

    def run_count(self) -> int:
        """Return how many runs currently hold state.

        Returns:
            Number of tracked runs. Growth without bound means runs are not
            being closed.
        """
        with self._lock:
            return len(self._runs)

    def __repr__(self) -> str:
        """Return a debug representation with the tracked run count.

        Reports only a count. A repr that dumped summaries would put run
        outcomes into a log line or a crash report.
        """
        return f"<InMemoryQualityStore runs={self.run_count()}>"
