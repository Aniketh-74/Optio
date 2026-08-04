"""What a behaviour backend must do, and the summary it returns.

Follows ADR-050: the store speaks the domain rather than offering primitives,
so each backend keeps its own promises where the check and the mutation happen
together -- a lock in memory, one Lua script in Redis.

:class:`WindowState` is deliberately three scalars plus a short tuple.
:func:`~optio.lanes.behavior.detectors.classify_state` reads nothing else --
not the step signatures, not the counter itself -- so nothing else has any
business crossing a process boundary. Returning the counter instead would put
up to ``behavior_window_size`` entries on the wire per step (1,000 at the
documented ceiling) and quietly convert a published O(1)-in-window-size
guarantee into O(window), without failing a single existing test.

``top_counts`` is bounded by the caller's ``k`` rather than by a constant here,
so ``LOOP_MAX_DISTINCT`` stays in the detector that owns it and the store never
learns a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WindowState:
    """Everything a verdict needs about one run's recent steps.

    Attributes:
        size: Signatures currently retained -- the window, not the run. The
            distinction matters: ``MIN_STEPS_FOR_VERDICT`` gates on how much
            recent evidence exists, not on how long the run has been going.
        errors: How many retained steps ended in error.
        distinct_calls: Distinct call identities retained. Counted rather than
            inferred from ``top_counts``, which is truncated.
        top_counts: The ``k`` largest per-call counts, descending. Shorter than
            ``k`` when fewer distinct calls exist; empty for an empty window.
    """

    size: int
    errors: int
    distinct_calls: int
    top_counts: tuple[int, ...]


class BehaviorStore(Protocol):
    """Per-run step windows, for many concurrent runs.

    Implementations must be safe under concurrent use. In-process that means a
    lock; across processes it means the backend's own atomicity, because a lock
    held in one process says nothing to another.
    """

    def record(
        self,
        run_id: str,
        signature_call: tuple[str, str],
        errored: bool,
        maxlen: int,
        k: int,
    ) -> WindowState:
        """Add one step and return the resulting summary.

        One call rather than write-then-read: a step costs one round trip, and
        the summary is read on every step anyway, so splitting them would
        double the network cost and open a window for another worker's step to
        land in between.

        Args:
            run_id: The run's identifier.
            signature_call: The step's call identity -- ``(tool, args_digest)``,
                never the arguments themselves (Section 10).
            errored: Whether the step ended in error.
            maxlen: Window bound, from ``Config.behavior_window_size``.
            k: How many top counts to return.

        Returns:
            The window's state after adding the step.
        """
        ...

    def close_run(self, run_id: str, k: int) -> WindowState | None:
        """Release a run's window and return the state it held.

        Reading then releasing would be two operations, and on a shared backend
        two round trips with a gap another worker's step can land in -- so the
        final verdict would describe a window that no longer existed. One call
        makes the read and the release the same event.

        Args:
            run_id: The run's identifier.
            k: How many top counts to return.

        Returns:
            The final state, or ``None`` if the run held no window. ``None`` is
            not an empty window: run end can fire more than once (M1-2), and a
            second close that reported an empty state would let the caller emit
            ``healthy`` with no repeats, overwriting a real ``looping`` verdict
            on the run span. Absence is reported as absence (ADR-044).
        """
        ...

    def run_count(self) -> int:
        """How many runs currently hold a window.

        Exposed for leak detection: unbounded growth means runs are not being
        released at run end.

        Returns:
            Number of tracked runs.
        """
        ...
