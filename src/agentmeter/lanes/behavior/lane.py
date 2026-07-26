"""The behavior lane (M3-3) -- turns step signatures into health signals.

Holds one bounded :class:`~agentmeter.lanes.behavior.window.BehaviorWindow` per
in-flight run and emits ``gen_ai.run.loop_state`` and ``gen_ai.run.repeat_count``
as each step lands.

**Windows are evicted at run end.** This is the same discipline the cost
ledger needed: an agent process is long-lived, and per-run state that is never
released is an unbounded leak rather than a slow one. Stress testing found that
bug in the ledger after every unit test missed it, because a test that builds
one lane and ends one run cannot see state accumulating across runs.

Unlike the ledger, the behavior lane needs no closed-run memory. Re-adding steps
to an evicted run's window starts a fresh window, which at worst under-reports
(the window is short again, so no pathology is claimed) -- it cannot invent a
pathology or corrupt a published total the way a reopened cost run could
(ADR-010). Failing toward ``healthy`` is exactly the bias Section 6.4 requires.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from agentmeter import semconv
from agentmeter.lanes.base import Lane, Signal
from agentmeter.lanes.behavior.detectors import classify
from agentmeter.lanes.behavior.window import BehaviorWindow, signature_of

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from agentmeter.config import Config
    from agentmeter.lanes.base import RunLike


class BehaviorLane(Lane):
    """Detects loops, repetition, and retry storms from span signatures.

    Attributes:
        window_size: Maximum step signatures retained per run.
    """

    name = "behavior"

    def __init__(self, config: Config) -> None:
        """Build the lane.

        Args:
            config: Active configuration. ``behavior_window_size`` bounds each
                run's window and is validated at setup (Section 4.2).
        """
        super().__init__(config)
        self.window_size = config.behavior_window_size
        # Agent frameworks are frequently multi-threaded, and the span tap runs
        # on whichever thread ended the span. The dict itself needs guarding or
        # two threads starting the same run can each install a window and one
        # set of steps is silently lost.
        self._lock = threading.RLock()
        self._windows: dict[str, BehaviorWindow] = {}

    def process_span(self, span: ReadableSpan, run: RunLike) -> list[Signal]:
        """Record one step and emit the current verdict.

        Args:
            span: The finished GenAI span.
            run: The run the span belongs to.

        Returns:
            The loop state and repeat count for the run so far.
        """
        signature = signature_of(span)

        with self._lock:
            window = self._windows.get(run.run_id)
            if window is None:
                window = BehaviorWindow(self.window_size)
                self._windows[run.run_id] = window
            window.add(signature)
            verdict = classify(window)

        return [
            Signal(semconv.RUN_LOOP_STATE, verdict.state),
            Signal(semconv.RUN_REPEAT_COUNT, verdict.repeat_count),
        ]

    def on_run_end(self, run: RunLike) -> list[Signal]:
        """Emit the final verdict and release the run's window.

        Args:
            run: The run that just ended.

        Returns:
            The final loop state and repeat count, or nothing if the run had no
            steps -- or if its window was already released.
        """
        with self._lock:
            # Run end can fire more than once (M1-2). The window is gone after
            # the first call, and re-deriving a verdict from its absence would
            # emit `healthy` with repeat_count 0, overwriting a real `looping`
            # verdict on the run span. That is the same failure the cost lane
            # hit, in the direction that hides a pathology.
            window = self._windows.pop(run.run_id, None)
            if window is None or len(window) == 0:
                return []
            verdict = classify(window)

        return [
            Signal(semconv.RUN_LOOP_STATE, verdict.state),
            Signal(semconv.RUN_REPEAT_COUNT, verdict.repeat_count),
        ]

    def run_count(self) -> int:
        """Return the number of runs currently holding a window.

        Exists so a test can assert the lane releases state; without an
        observable count the leak that stress testing found in the ledger would
        be untestable here too.

        Returns:
            Number of live windows.
        """
        with self._lock:
            return len(self._windows)
