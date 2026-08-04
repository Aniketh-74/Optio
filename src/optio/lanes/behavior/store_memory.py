"""The in-process behaviour backend -- the default (ADR-050).

This is the window handling that shipped inside ``BehaviorLane`` through 0.3.0,
moved behind :class:`~optio.lanes.behavior.store.BehaviorStore` **unchanged**:
the same ``RLock``, the same create-on-first-step, the same pop at run end. The
move is verbatim on purpose, so the existing behaviour-lane tests act as the
regression net for the extraction and any difference is a bug in the move
rather than a design choice nobody wrote down.

The lock guards the dictionary, not the window. Agent frameworks are routinely
multi-threaded and the span tap runs on whichever thread ended the span, so two
threads starting the same run can each install a window and one set of steps is
silently lost -- a shorter window, a missed pathology, and nothing raised.

**Single-process by construction.** A lock says nothing to another process. Runs
sharded across workers need the Redis backend; each process here would classify
from its own slice of the steps, and a slice is not a smaller sample of the run
-- below :data:`~optio.lanes.behavior.detectors.MIN_STEPS_FOR_VERDICT` it is a
confident ``healthy`` for an agent that is visibly stuck.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

from optio.lanes.behavior.window import BehaviorWindow, StepSignature

if TYPE_CHECKING:
    from optio.lanes.behavior.store import WindowState


class InMemoryBehaviorStore:
    """Per-run step windows in a process-local dictionary.

    Holds windows for many concurrent runs, keyed by ``run_id``. Call
    :meth:`close_run` at run end to read the final state and release the window
    in one step; a window that is never released is an unbounded leak, which is
    the bug stress testing found in the ledger after every unit test missed it.

    Structurally implements :class:`~optio.lanes.behavior.store.BehaviorStore`
    rather than inheriting it, matching the ledger's backends.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._lock: Final = threading.RLock()
        self._windows: Final[dict[str, BehaviorWindow]] = {}

    def record(
        self,
        run_id: str,
        signature_call: tuple[str, str],
        errored: bool,
        maxlen: int,
        k: int,
    ) -> WindowState:
        """Add one step to a run's window and return the resulting summary.

        ``maxlen`` is read when the window is created and the window keeps it
        for the run's lifetime. ``Config`` is frozen for a process's lifetime,
        so every call passes the same bound; the argument exists because a
        shared backend has no config of its own to consult.

        Args:
            run_id: The run's identifier. Its window is created on first use.
            signature_call: The step's call identity -- ``(tool, args_digest)``,
                never the arguments themselves (Section 10).
            errored: Whether the step ended in error.
            maxlen: Window bound, from ``Config.behavior_window_size``.
            k: How many top counts to return.

        Returns:
            The window's state after adding the step.

        Raises:
            ValueError: If ``maxlen`` is not positive and the window is being
                created. Config validates this at setup (Section 4.2); a
                non-positive bound here would silently disable detection.
        """
        tool, args_digest = signature_call
        signature = StepSignature(tool=tool, args_digest=args_digest, errored=errored)

        with self._lock:
            window = self._windows.get(run_id)
            if window is None:
                window = BehaviorWindow(maxlen)
                self._windows[run_id] = window
            window.add(signature)
            # Summarised under the lock. Handing the window back and letting the
            # caller read it would expose live mutable state to another thread
            # mid-eviction, where the counts and the deque disagree.
            return window.state(k)

    def close_run(self, run_id: str, k: int) -> WindowState | None:
        """Release a run's window and return the state it held.

        Args:
            run_id: The run's identifier.
            k: How many top counts to return.

        Returns:
            The final state, or ``None`` if the run held no window -- either it
            never recorded a step, or run end already fired once (M1-2).
        """
        with self._lock:
            window = self._windows.pop(run_id, None)
            if window is None:
                return None
            return window.state(k)

    def run_count(self) -> int:
        """Return how many runs currently hold a window.

        Returns:
            Number of live windows. Growth without bound means runs are not
            being closed.
        """
        with self._lock:
            return len(self._windows)

    def __repr__(self) -> str:
        """Return a debug representation with the live window count.

        Reports only a count. A repr that dumped windows could put
        fingerprinted call shapes into a log line or a crash report.
        """
        return f"<InMemoryBehaviorStore runs={self.run_count()}>"
