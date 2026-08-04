"""The quality lane (M5-4) -- outcome scoring, opt-in and off by default.

Ties the three M5 pieces together: :mod:`sampling` picks a tier,
:mod:`heuristic` scores cheaply inline, :mod:`judge` scores deeply on sampled
runs, and this module turns whatever came back into signals.

## What it emits, and when it stays silent

``gen_ai.run.success`` is the signal most consumers want, and it is the one this
lane is most careful about. It is emitted **only on evidence**:

* the heuristic found positive evidence of failure -> ``False``
* the judge scored ``task_success`` -> thresholded
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

from optio import semconv
from optio.lanes.base import Lane, Signal
from optio.lanes.quality import heuristic
from optio.lanes.quality.judge import JudgeRequest, JudgeRunner, JudgeScores
from optio.lanes.quality.sampling import Tier, decide
from optio.lanes.quality.store_memory import InMemoryQualityStore

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

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
        self, config: Config, judge: Judge | None = None, store: QualityStore | None = None
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
        """
        super().__init__(config)
        # The lock that guarded the span buffer moved into the store with the
        # state it protects. One here would say nothing about a run another
        # process is writing.
        self._store: QualityStore = store if store is not None else InMemoryQualityStore()
        self._runner = JudgeRunner(judge)

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

        scores = self._judge_scores(run, decision.tier, summary.step_count)
        if scores is not None:
            if scores.groundedness is not None:
                signals.append(Signal(semconv.RUN_QUALITY_GROUNDEDNESS, scores.groundedness))
            if scores.task_success is not None:
                signals.append(Signal(semconv.RUN_QUALITY_TASK_SUCCESS, scores.task_success))

        success = _resolve_success(inline, scores)
        if success is not None:
            signals.append(Signal(semconv.RUN_SUCCESS, success))
            _record_successes(run, success)

        return signals

    def _judge_scores(self, run: RunLike, tier: Tier, step_count: int) -> JudgeScores | None:
        """Dispatch and collect the judge, without blocking the run.

        Args:
            run: The run being scored.
            tier: The assigned tier.
            step_count: How many steps the run took. The counted total, not
                ``len(retained spans)`` -- the buffer is capped at
                :data:`MAX_RETAINED_SPANS`, so measuring it reported the cap for
                every longer run, and ``docs/quality.md`` shows users passing
                this number straight into their own evaluator.

        Returns:
            Validated scores, or ``None``.
        """
        if not tier.uses_judge or not self._runner.enabled:
            return None

        # Content is the caller's to provide. optio passes no trace text of
        # its own (Section 10): a judge that needs prompts should be closed over
        # the user's own record of them.
        self._runner.submit(JudgeRequest(run_id=run.run_id, step_count=step_count, content={}))
        return self._runner.collect(run.run_id)

    def run_count(self) -> int:
        """Return how many runs are holding scoring state.

        Exposed so a test can observe that state is released.
        """
        return self._store.run_count()

    def shutdown(self) -> None:
        """Release the judge's worker pool."""
        self._runner.shutdown()


def _record_successes(run: RunLike, success: bool) -> None:
    """Record the success count on the run for the cost lane to read.

    This is the seam ``cost_per_successful_task`` needs: the cost lane owns the
    numerator and this lane owns the denominator, and neither imports the other
    (Section 3.1). The value is written onto the run object, which both already
    hold. The registry orders quality ahead of cost so the value is present when
    cost reads it.

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


def _resolve_success(inline: heuristic.HeuristicResult, scores: JudgeScores | None) -> bool | None:
    """Decide the run's success flag from the available evidence.

    Args:
        inline: The heuristic verdict.
        scores: Judge scores, when the judge answered.

    Returns:
        ``True``/``False`` on evidence, ``None`` when there is none.
    """
    # The judge wins when it spoke: it is the only component that looked at what
    # the run actually said, whereas the heuristic only saw that it said
    # something.
    if scores is not None and scores.task_success is not None:
        return scores.task_success >= SUCCESS_THRESHOLD

    # The heuristic reports failure or abstains; it never claims success.
    return inline.succeeded
