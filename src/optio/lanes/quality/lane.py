"""The quality lane (M5-4) -- outcome scoring, opt-in and off by default.

Ties the three M5 pieces together: :mod:`sampling` picks a tier,
:mod:`heuristic` scores cheaply inline, :mod:`judge` scores deeply on sampled
runs, and this module turns whatever came back into signals.

## Where the signals go

Two destinations, because the two tiers answer at different times.

The heuristic is synchronous, so its verdict goes on the **run span**, with
everything else optio emits. The judge is a model call dispatched at run end:
it answers long after that span has closed, and an ended span cannot be
modified. Its scores go on a **separate ``optio.quality`` span**, linked back to
the run's and carrying ``gen_ai.run.id`` to join on
(:mod:`optio.lanes.quality.deferred`).

Through 0.3.0 this lane dispatched the judge and then polled it with a
zero-second timeout on the very next line, so a judge that made a network call
never delivered a single score. See :mod:`optio.lanes.quality.judge`.

## What it emits, and when it stays silent

``gen_ai.run.success`` is the signal most consumers want, and it is the one this
lane is most careful about. It is emitted **only on evidence**:

* the heuristic found positive evidence of failure -> ``False`` on the run span
* the judge scored ``task_success`` -> thresholded, on the quality span
* neither -> **nothing at all**

That third case is the common one and it is deliberate. A run that produced
fluent output looks identical, from the outside, to a run that produced fluent
*wrong* output -- and "well-formed but wrong" is precisely the failure Section
1.3 says permission-based governance cannot see. Emitting ``success=true``
because nothing looked obviously broken would manufacture exactly the false
assurance the lane exists to replace.

The knock-on is that ``cost_per_successful_task`` stays absent for unscored
runs, which is correct: its denominator is unknown, and a headline unit-economics
number derived from a guess is worse than no number.

## Memory

The lane held every run's spans until run end, capped at 64, because scoring
needed them. Deriving what scoring actually reads showed it needs a step count
and the final step -- so the buffer is gone, and the state per run is a counter
and one :class:`~optio.lanes.quality.store.QualityStep` (ADR-050). Bounded
memory stops being a cap someone has to remember to enforce and becomes a
property of the shape: a run costs the same at three steps or thirty thousand.

State is still released at run end, which is the discipline all three lanes
share. A long-lived agent process that never releases per-run state leaks
whatever that state happens to be.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from opentelemetry import trace

from optio import semconv
from optio.lanes.base import Lane, Signal
from optio.lanes.quality import deferred, heuristic
from optio.lanes.quality.judge import (
    DEFAULT_DRAIN_TIMEOUT,
    JudgeRequest,
    JudgeRunner,
    JudgeScores,
)
from optio.lanes.quality.sampling import Tier, decide
from optio.lanes.quality.store_memory import InMemoryQualityStore

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import Tracer

    from optio.config import Config
    from optio.lanes.base import RunLike
    from optio.lanes.quality.judge import Judge
    from optio.lanes.quality.store import QualityStore

_log: Final = logging.getLogger("optio")

#: ``task_success`` at or above this counts as a success. A judge score is a
#: confidence, and turning it into a boolean needs a line somewhere; 0.5 is the
#: neutral one. Consumers wanting a stricter bar should gate on the raw
#: ``gen_ai.run.quality.task_success`` value instead, which is why it is emitted
#: separately rather than folded into the boolean.
SUCCESS_THRESHOLD: Final = 0.5


class QualityLane(Lane):
    """Scores run outcomes. Off unless ``quality_lane`` is enabled (ADR-003)."""

    name = "quality"

    def __init__(
        self,
        config: Config,
        judge: Judge | None = None,
        store: QualityStore | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Create the lane.

        Args:
            config: Active configuration.
            judge: The user's evaluator. ``None`` means heuristic-only --
                optio ships no default judge, because a default would
                spend the user's money the moment they enabled the lane
                (Section 10).
            store: Where per-run scoring state lives. Defaults to the
                process-local backend, which is correct for a single process
                and wrong in a way nothing reports for a run sharded across
                workers -- so the registry injects the shared one when
                configured.
            tracer: Tracer the deferred judge span is emitted with. Must belong
                to the provider the tap was installed on: a user who passes
                their own ``TracerProvider`` would otherwise have quality spans
                recorded by the global one, where nothing they configured is
                listening. ``None`` uses the global provider.
        """
        super().__init__(config)
        # The lock that guarded the span buffer moved into the store with the
        # state it protects. One here would say nothing about a run another
        # process is writing.
        self._store: QualityStore = store if store is not None else InMemoryQualityStore()
        self._runner = JudgeRunner(judge)
        self._tracer: Tracer = tracer if tracer is not None else trace.get_tracer("optio")

        if config.quality_lane and judge is None:
            # Said once, at setup, because the alternative is a user believing
            # they enabled deep scoring and quietly getting only the heuristic.
            _log.warning(
                "optio: quality lane is enabled but no judge was supplied; "
                "only the inline heuristic will run. Pass judge=... to score "
                "groundedness and task success. See docs/quality.md."
            )

    def process_span(self, span: ReadableSpan, run: RunLike) -> list[Signal]:
        """Retain a span for end-of-run scoring.

        Emits nothing: quality is a run-scoped property and cannot be judged
        from one step. Kept cheap -- this is on the hot path (SC-5).

        Args:
            span: The span that just finished.
            run: The run it belongs to.

        Returns:
            Always empty.
        """
        # Projected here rather than retained. A span is not serializable, and
        # holding one is what kept this lane process-local. The store overwrites
        # the step it holds, so which step counts as last is decided by arrival
        # order -- within a process today, across every worker on a shared
        # backend.
        self._store.record(run.run_id, heuristic.project(span))
        return []

    def on_run_end(self, run: RunLike) -> list[Signal]:
        """Score the finished run.

        Args:
            run: The run that just ended.

        Returns:
            Quality signals, or empty when the run was not scored or the
            evidence did not support a verdict.
        """
        # One call, so the read and the release are the same event. The count
        # cannot outlive the step it describes and be handed to the next run's
        # judge as a plausible number.
        summary = self._store.close_run(run.run_id)

        decision = decide(run, self.config)
        if not decision.tier.scores:
            self._runner.discard(run.run_id)
            return []

        if summary is None:
            # Run end can fire more than once (M1-2); the first call took the
            # state. Re-scoring would emit a second, weaker verdict over the
            # first -- the failure the behavior lane hit in M3, and here it
            # would overwrite a judge result the user paid for.
            return []

        signals: list[Signal] = []
        inline = heuristic.score(summary.last)

        # Dispatched, not awaited. Its scores land on their own span later
        # (`deferred`), because the run span this returns into is about to
        # close and cannot be written to afterwards.
        self._dispatch_judge(run, decision.tier, summary.step_count)

        # The heuristic is what the run span can carry, because it is the only
        # verdict available synchronously. It never claims success, so this is
        # `False` or nothing.
        success = inline.succeeded
        if success is not None:
            signals.append(Signal(semconv.RUN_SUCCESS, success))
            _record_successes(run, success)

        return signals

    def _dispatch_judge(self, run: RunLike, tier: Tier, step_count: int) -> None:
        """Send the run to the judge, to be scored on a worker thread.

        Args:
            run: The run being scored.
            tier: The assigned tier.
            step_count: How many steps the run took. The counted total, not the
                size of a buffer -- ``docs/quality.md`` shows users passing this
                number straight into their own evaluator, and it reported a
                capped buffer's length through 0.3.0.
        """
        if not tier.uses_judge or not self._runner.enabled:
            return

        # Captured here, on the run's thread, while its span is still current.
        # The worker thread that emits has no OTel context of its own.
        parent = deferred.parent_context()

        def deliver(scores: JudgeScores) -> None:
            """Emit the scores. Runs on a judge worker thread."""
            deferred.emit_scores(
                self._tracer, run, parent, scores, success_threshold=SUCCESS_THRESHOLD
            )

        # Content is the caller's to provide. optio passes no trace text of
        # its own (Section 10): a judge that needs prompts should be closed over
        # the user's own record of them.
        self._runner.submit(
            JudgeRequest(run_id=run.run_id, step_count=step_count, content={}),
            on_scores=deliver,
        )

    def run_count(self) -> int:
        """Return how many runs are holding scoring state.

        Exposed so a test can observe that state is released.
        """
        return self._store.run_count()

    def shutdown(self, drain_timeout: float = DEFAULT_DRAIN_TIMEOUT) -> None:
        """Let outstanding judgements land, then release the worker pool.

        Args:
            drain_timeout: Seconds to wait for in-flight judge calls to finish
                emitting their spans. Scores are dispatched asynchronously, so
                a process that exits the moment its last run ends would discard
                judgements it has already been billed for.
        """
        self._runner.shutdown(drain_timeout)

    def drain(self, timeout: float = DEFAULT_DRAIN_TIMEOUT) -> int:
        """Wait for outstanding judge scores to be emitted.

        Args:
            timeout: Seconds to wait.

        Returns:
            How many judgements were still undelivered when the wait ended.
        """
        return self._runner.drain(timeout)


def _record_successes(run: RunLike, success: bool) -> None:
    """Record the success count on the run.

    Written for anyone downstream who reads the run object, and consumed by the
    cost lane, which computes ``cost_per_successful_task`` from it when it is
    positive. In practice this lane can only ever write ``0`` here: the value
    comes from the heuristic, which reports failure or abstains and never claims
    success. The judge -- the only thing that can -- answers after the run span
    has closed, so its verdict and the ratio derived from it are emitted on the
    deferred quality span instead (ADR-051).

    Kept because the field is part of the run object's contract and a caller who
    determines success themselves may set it, in which case the cost lane's copy
    of the signal is correct and reachable.

    A run is one task, so the count is 1 or 0. It is an ``int`` rather than a
    ``bool`` because the field is a count -- a future batching run could record
    more -- and because the cost lane rejects ``bool`` explicitly to avoid
    ``True`` arriving as the integer 1 by accident.

    Args:
        run: The run that just ended.
        success: Whether it succeeded.
    """
    # `RunLike` (Section 3.1) deliberately exposes only `run_id` and `budget`;
    # widening it to carry a quality-specific field would push one lane's
    # concern into the contract every other lane is typed against. `setattr`
    # rather than a cast, so the AttributeError path below stays honest.
    try:
        setattr(run, "successes", 1 if success else 0)  # noqa: B010
    except AttributeError:
        # A run object without the field -- a minimal stub, or an older
        # RunContext. The cost signal is then omitted rather than wrong, which
        # is the correct degradation (docs/signals.md).
        _log.debug("run object does not accept a success count; skipping")
