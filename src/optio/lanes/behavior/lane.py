"""The behavior lane (M3-3) -- turns step signatures into health signals.

Records each step against a :class:`~optio.lanes.behavior.store.BehaviorStore`
and emits ``gen_ai.run.loop_state`` and ``gen_ai.run.repeat_count`` from the
summary it returns. The window itself lives in the store: process-local by
default, in Redis when runs are sharded across workers (ADR-050).

**Windows are released at run end.** This is the same discipline the cost
ledger needed: an agent process is long-lived, and per-run state that is never
released is an unbounded leak rather than a slow one. Stress testing found that
bug in the ledger after every unit test missed it, because a test that builds
one lane and ends one run cannot see state accumulating across runs.

Unlike the ledger, the behavior lane needs no closed-run memory. Re-adding steps
to a released run's window starts a fresh window, which at worst under-reports
(the window is short again, so no pathology is claimed) -- it cannot invent a
pathology or corrupt a published total the way a reopened cost run could
(ADR-010). Failing toward ``healthy`` is exactly the bias Section 6.4 requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio import semconv
from optio.lanes.base import Lane, Signal
from optio.lanes.behavior.detectors import LOOP_MAX_DISTINCT, classify_state
from optio.lanes.behavior.store_memory import InMemoryBehaviorStore
from optio.lanes.behavior.window import signature_of

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from optio.config import Config
    from optio.lanes.base import RunLike
    from optio.lanes.behavior.store import BehaviorStore, WindowState


class BehaviorLane(Lane):
    """Detects loops, repetition, and retry storms from span signatures.

    Attributes:
        window_size: Maximum step signatures retained per run.
    """

    name = "behavior"

    def __init__(self, config: Config, store: BehaviorStore | None = None) -> None:
        """Build the lane.

        Args:
            config: Active configuration. ``behavior_window_size`` bounds each
                run's window and is validated at setup (Section 4.2).
            store: Where windows live. Defaults to the process-local backend,
                which is correct for the single-process case and wrong in a way
                nothing reports for a run sharded across workers -- so the
                registry injects the shared one when configured.
        """
        super().__init__(config)
        self.window_size = config.behavior_window_size
        # The lock that used to guard `self._windows` moved into the store with
        # the dictionary. It has to live with the state it protects: a lock here
        # would say nothing about a window another process is writing.
        self._store: BehaviorStore = store if store is not None else InMemoryBehaviorStore()

    def process_span(self, span: ReadableSpan, run: RunLike) -> list[Signal]:
        """Record one step and emit the current verdict.

        Args:
            span: The finished GenAI span.
            run: The run the span belongs to.

        Returns:
            The loop state and repeat count for the run so far.
        """
        signature = signature_of(span)
        state = self._store.record(
            run.run_id,
            signature.call,
            signature.errored,
            maxlen=self.window_size,
            k=LOOP_MAX_DISTINCT,
        )
        return _signals(state)

    def on_run_end(self, run: RunLike) -> list[Signal]:
        """Emit the final verdict and release the run's window.

        Args:
            run: The run that just ended.

        Returns:
            The final loop state and repeat count, or nothing if the run had no
            steps -- or if its window was already released.
        """
        # Run end can fire more than once (M1-2). The window is gone after the
        # first call, and re-deriving a verdict from its absence would emit
        # `healthy` with repeat_count 0, overwriting a real `looping` verdict on
        # the run span. That is the same failure the cost lane hit, in the
        # direction that hides a pathology -- so the store reports absence as
        # `None` rather than as an empty window, and this returns nothing.
        state = self._store.close_run(run.run_id, k=LOOP_MAX_DISTINCT)
        if state is None or state.size == 0:
            return []
        return _signals(state)

    def run_count(self) -> int:
        """Return the number of runs currently holding a window.

        Exists so a test can assert the lane releases state; without an
        observable count the leak that stress testing found in the ledger would
        be untestable here too.

        Returns:
            Number of live windows.
        """
        return self._store.run_count()


def _signals(state: WindowState) -> list[Signal]:
    """Turn a window summary into the lane's two signals.

    Args:
        state: The window's state.

    Returns:
        The loop state and repeat count.
    """
    verdict = classify_state(state)
    return [
        Signal(semconv.RUN_LOOP_STATE, verdict.state),
        Signal(semconv.RUN_REPEAT_COUNT, verdict.repeat_count),
    ]
