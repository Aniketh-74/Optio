"""The quality lane (M5-4).

The lane's most important behaviour is a refusal: it does not emit
``gen_ai.run.success`` unless something actually judged the outcome. A run that
produced fluent output is indistinguishable from one that produced fluent
*wrong* output without reading it, so claiming success on "nothing looked broken"
would manufacture the false assurance the lane exists to replace.

The knock-on tested here is that ``cost_per_successful_task`` stays absent for
unscored runs, because its denominator is genuinely unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from opentelemetry.trace import StatusCode

from optio import semconv
from optio.config import Config
from optio.lanes.quality.judge import Judge, JudgeRequest, JudgeRunner, JudgeScores
from optio.lanes.quality.lane import MAX_RETAINED_SPANS, QualityLane

if TYPE_CHECKING:
    from collections.abc import Mapping

    from optio.lanes.base import Signal


@dataclass
class FakeRun:
    """A run the lane can score and write a success count onto."""

    run_id: str = "run-1"
    budget: Any = None
    sampled: bool = False
    successes: int | None = None


def span(
    attributes: Mapping[str, object] | None = None,
    *,
    errored: bool = False,
) -> Mock:
    """Build a stand-in for a finished ReadableSpan."""
    mock = Mock()
    mock.name = "step"
    mock.attributes = attributes or {}
    mock.status = Mock(status_code=StatusCode.ERROR if errored else StatusCode.OK)
    return mock


def answered(tokens: int = 50) -> Mock:
    """A span that looks like a completed generation."""
    return span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: tokens})


def values(signals: list[Signal]) -> dict[str, object]:
    """Index emitted signals by name."""
    return {signal.name: signal.value for signal in signals}


def scoring_judge(groundedness: float = 0.9, task_success: float = 0.8) -> Judge:
    """A judge returning fixed scores."""

    def judge(_request: JudgeRequest) -> JudgeScores:
        return JudgeScores(groundedness=groundedness, task_success=task_success)

    return judge


def enabled(**overrides: object) -> Config:
    """Config with the quality lane on."""
    return Config(quality_lane=True, **overrides)  # type: ignore[arg-type]


class TestTheLaneIsSilentWhenOff:
    def test_a_disabled_lane_emits_nothing(self) -> None:
        lane = QualityLane(Config())
        lane.process_span(answered(), FakeRun())
        assert lane.on_run_end(FakeRun()) == []

    def test_a_disabled_lane_records_no_success_count(self) -> None:
        run = FakeRun()
        lane = QualityLane(Config())
        lane.process_span(answered(), run)
        lane.on_run_end(run)
        assert run.successes is None


class TestSuccessIsNeverAssumed:
    """The refusal that matters."""

    def test_a_normal_run_emits_no_success_signal(self) -> None:
        # No judge, nothing broken. The honest answer is "unknown", and the
        # signal is therefore absent rather than True.
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(answered(), run)

        assert semconv.RUN_SUCCESS not in values(lane.on_run_end(run))

    def test_a_normal_run_records_no_success_count(self) -> None:
        # Which is what keeps cost_per_successful_task absent: its denominator
        # is unknown, and 1 would be a guess published as a headline number.
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(answered(), run)
        lane.on_run_end(run)

        assert run.successes is None

    def test_a_failed_run_does_emit_success_false(self) -> None:
        # Failure is evidence-based, so it *is* claimed.
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(span({}, errored=True), run)

        assert values(lane.on_run_end(run))[semconv.RUN_SUCCESS] is False

    def test_a_failed_run_records_zero_successes(self) -> None:
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(answered(tokens=0), run)
        lane.on_run_end(run)

        assert run.successes == 0


class TestTheJudgeDecidesWhenItSpeaks:
    def test_judge_scores_are_emitted(self) -> None:
        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(quality_sample_rate=1.0), judge=scoring_judge())
        lane.process_span(answered(), run)
        emitted = values(lane.on_run_end(run))

        assert emitted[semconv.RUN_QUALITY_GROUNDEDNESS] == 0.9
        assert emitted[semconv.RUN_QUALITY_TASK_SUCCESS] == 0.8
        assert emitted[semconv.RUN_SUCCESS] is True

    def test_a_low_task_success_score_is_a_failure(self) -> None:
        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=scoring_judge(task_success=0.2))
        lane.process_span(answered(), run)

        assert values(lane.on_run_end(run))[semconv.RUN_SUCCESS] is False

    def test_the_judge_overrides_a_silent_heuristic(self) -> None:
        # The heuristic abstains on a normal run; the judge is the only thing
        # that looked at what the run actually said.
        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=scoring_judge(task_success=0.95))
        lane.process_span(answered(), run)

        assert values(lane.on_run_end(run))[semconv.RUN_SUCCESS] is True
        assert run.successes == 1

    def test_an_unsampled_run_never_reaches_the_judge(self) -> None:
        called: list[str] = []

        def judge(req: JudgeRequest) -> JudgeScores:
            called.append(req.run_id)
            return JudgeScores(task_success=1.0)

        run = FakeRun(sampled=False)
        lane = QualityLane(enabled(), judge=judge)
        lane.process_span(answered(), run)
        lane.on_run_end(run)

        assert called == [], "an unsampled run must not spend the user's money"

    def test_a_failing_judge_leaves_the_heuristic_verdict(self) -> None:
        def raises(_request: JudgeRequest) -> JudgeScores:
            raise RuntimeError("judge exploded")

        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=raises)
        lane.process_span(span({}, errored=True), run)
        emitted = values(lane.on_run_end(run))

        # No judge scores, but the heuristic's failure evidence still stands.
        assert semconv.RUN_QUALITY_TASK_SUCCESS not in emitted
        assert emitted[semconv.RUN_SUCCESS] is False


class TestNothingIsEmittedOnTheHotPath:
    def test_process_span_emits_no_signals(self) -> None:
        # Quality is run-scoped and cannot be judged from one step. Also keeps
        # the hot path cheap (SC-5).
        lane = QualityLane(enabled())
        assert lane.process_span(answered(), FakeRun()) == []


class RecordingRunner(JudgeRunner):
    """A runner that remembers what it was asked to judge.

    Reads what the lane *sends* rather than what the judge received. The judge
    runs in a worker pool and ``collect`` waits ``DEFAULT_COLLECT_TIMEOUT``,
    which is ``0.0`` -- so a judge that records what it saw is racing the pool,
    and a test built on one passes or fails on scheduling. ``submit`` is called
    synchronously on the run-end thread, and the request is the contract.
    """

    def __init__(self, judge: Judge | None) -> None:
        super().__init__(judge)
        self.requests: list[JudgeRequest] = []

    def submit(self, request: JudgeRequest) -> bool:
        """Record the request, then dispatch it normally."""
        self.requests.append(request)
        return super().submit(request)


def recording_lane(**overrides: object) -> tuple[QualityLane, RecordingRunner]:
    """A judging lane whose dispatches are observable."""
    lane = QualityLane(enabled(quality_sample_rate=1.0, **overrides), judge=scoring_judge())
    runner = RecordingRunner(scoring_judge())
    lane._runner = runner
    return lane, runner


class TestTheJudgeIsToldTheTruth:
    def test_the_step_count_is_the_run_length_not_the_buffer_size(self) -> None:
        """``step_count`` is documented as "how many steps the run took", and
        ``docs/quality.md`` shows a user passing it straight to their evaluator
        as ``steps=request.step_count``.

        It was ``len(spans)``, and the span buffer is capped at
        :data:`MAX_RETAINED_SPANS` -- so every run longer than that reported the
        cap. A 500-step run told the judge it took 64. Wrong rather than
        missing, and plausible enough that nobody would query it.
        """
        steps = MAX_RETAINED_SPANS + 40
        run = FakeRun(sampled=True)
        lane, runner = recording_lane()
        for _ in range(steps):
            lane.process_span(answered(), run)
        lane.on_run_end(run)

        assert [request.step_count for request in runner.requests] == [steps]

    def test_the_count_is_released_with_the_spans(self) -> None:
        """A second per-run dictionary is a second thing to leak.

        The lane's whole eviction discipline exists because per-run state in a
        long-lived agent process grows without bound -- the bug stress testing
        found in the ledger after every unit test missed it. ``run_count()``
        observes the span buffer, so a counter released separately, or not at
        all, would be invisible to it.

        Reaching in is deliberate, as it is for the span cap below: this is a
        memory-safety property with no public surface.
        """
        lane = QualityLane(enabled(quality_sample_rate=1.0), judge=scoring_judge())
        for index in range(20):
            run = FakeRun(f"run-{index}", sampled=True)
            for _ in range(3):
                lane.process_span(answered(), run)
            lane.on_run_end(run)

        assert lane._steps == {}, "the step counters outlived their runs"
        assert lane.run_count() == 0

    def test_a_straggling_step_does_not_resume_the_previous_count(self) -> None:
        """Run end can fire more than once (M1-2), and a late span can land
        after it. The count restarts rather than continuing, so a re-opened run
        reports what it can actually account for."""
        run = FakeRun(sampled=True)
        lane, runner = recording_lane()
        for _ in range(7):
            lane.process_span(answered(), run)
        lane.on_run_end(run)

        lane.process_span(answered(), run)
        lane.on_run_end(run)

        assert [request.step_count for request in runner.requests] == [7, 1]


class TestStateIsBounded:
    def test_spans_are_capped_per_run(self) -> None:
        run = FakeRun()
        lane = QualityLane(enabled())
        for _ in range(MAX_RETAINED_SPANS * 3):
            lane.process_span(answered(), run)

        # Reaching in is deliberate: the bound is a memory-safety property of a
        # long-lived agent process, and there is no public way to observe it.
        assert len(lane._spans[run.run_id]) == MAX_RETAINED_SPANS

    def test_only_the_final_step_decides_the_verdict(self) -> None:
        """An error mid-run that the agent recovered from is not a failed task.

        Lived in the heuristic suite until the projection split. The heuristic
        now scores one step, so *which* step it is handed is the lane's
        responsibility -- and after ADR-050 the store's. Asserted here so the
        behaviour keeps a test at the level where it now lives.
        """
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(span({}, errored=True), run)
        lane.process_span(answered(), run)

        assert semconv.RUN_SUCCESS not in values(lane.on_run_end(run))

    def test_an_errored_final_step_fails_the_run(self) -> None:
        """The mirror, so the pair pins the ordering rather than one direction
        of it."""
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(answered(), run)
        lane.process_span(span({}, errored=True), run)

        assert values(lane.on_run_end(run))[semconv.RUN_SUCCESS] is False

    def test_the_newest_spans_are_kept(self) -> None:
        # The heuristic judges the final answer, so the tail is what matters.
        run = FakeRun()
        lane = QualityLane(enabled())
        for _ in range(MAX_RETAINED_SPANS):
            lane.process_span(answered(), run)
        lane.process_span(span({}, errored=True), run)

        assert values(lane.on_run_end(run))[semconv.RUN_SUCCESS] is False

    def test_run_state_is_released_at_run_end(self) -> None:
        run = FakeRun()
        lane = QualityLane(enabled())
        lane.process_span(answered(), run)
        assert lane.run_count() == 1

        lane.on_run_end(run)
        assert lane.run_count() == 0

    def test_a_second_run_end_emits_nothing(self) -> None:
        # Run end can fire more than once (M1-2). Re-scoring would overwrite a
        # real verdict with a weaker one -- the bug the behavior lane hit in M3.
        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=scoring_judge())
        lane.process_span(answered(), run)

        first = lane.on_run_end(run)
        second = lane.on_run_end(run)

        assert first != []
        assert second == []

    def test_concurrent_runs_do_not_mix(self) -> None:
        lane = QualityLane(enabled())
        good, bad = FakeRun(run_id="a"), FakeRun(run_id="b")
        lane.process_span(answered(), good)
        lane.process_span(span({}, errored=True), bad)

        assert semconv.RUN_SUCCESS not in values(lane.on_run_end(good))
        assert values(lane.on_run_end(bad))[semconv.RUN_SUCCESS] is False


class TestSetupWarnsAboutAMissingJudge:
    def test_enabling_without_a_judge_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        # Otherwise a user believes they enabled deep scoring and quietly gets
        # only the heuristic.
        with caplog.at_level("WARNING", logger="optio"):
            QualityLane(enabled())

        assert "no judge was supplied" in caplog.text

    def test_supplying_a_judge_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="optio"):
            QualityLane(enabled(), judge=scoring_judge())

        assert "no judge was supplied" not in caplog.text

    def test_a_disabled_lane_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="optio"):
            QualityLane(Config())

        assert caplog.text == ""


class TestARunObjectWithoutTheFieldDegrades:
    def test_a_run_that_rejects_a_success_count_still_emits(self) -> None:
        # A minimal run stub, or an older RunContext. The quality signal is
        # still emitted; only the derived cost signal is lost.
        class Frozen:
            __slots__ = ("budget", "run_id", "sampled")

            def __init__(self) -> None:
                self.run_id = "run-1"
                self.budget = None
                self.sampled = False

        run = Frozen()
        lane = QualityLane(enabled())
        lane.process_span(span({}, errored=True), run)

        emitted = values(lane.on_run_end(run))
        assert emitted[semconv.RUN_SUCCESS] is False


class TestNoContentIsRetained:
    """Section 10, checked at the lane boundary."""

    def test_emitted_signals_are_numbers_and_booleans_only(self) -> None:
        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=scoring_judge())
        secret = "my private prompt"
        lane.process_span(
            span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10, "gen_ai.prompt": secret}), run
        )

        for signal in lane.on_run_end(run):
            assert isinstance(signal.value, (bool, int, float))
            assert secret not in str(signal.value)


class TestAPartialJudgement:
    """A judge may score one dimension and not the other."""

    def test_groundedness_alone_is_emitted(self) -> None:
        def partial(_request: JudgeRequest) -> JudgeScores:
            return JudgeScores(groundedness=0.8)

        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=partial)
        lane.process_span(answered(), run)
        emitted = values(lane.on_run_end(run))

        assert emitted[semconv.RUN_QUALITY_GROUNDEDNESS] == 0.8
        assert semconv.RUN_QUALITY_TASK_SUCCESS not in emitted
        # No task_success means no basis for a success flag, and the heuristic
        # does not supply one -- so the run stays unscored on that axis.
        assert semconv.RUN_SUCCESS not in emitted

    def test_task_success_alone_is_emitted(self) -> None:
        def partial(_request: JudgeRequest) -> JudgeScores:
            return JudgeScores(task_success=0.9)

        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=partial)
        lane.process_span(answered(), run)
        emitted = values(lane.on_run_end(run))

        assert semconv.RUN_QUALITY_GROUNDEDNESS not in emitted
        assert emitted[semconv.RUN_QUALITY_TASK_SUCCESS] == 0.9
        assert emitted[semconv.RUN_SUCCESS] is True

    def test_a_partial_judgement_over_a_failed_heuristic(self) -> None:
        # The judge scored groundedness but not success; the heuristic saw the
        # run error out. The failure evidence stands.
        def partial(_request: JudgeRequest) -> JudgeScores:
            return JudgeScores(groundedness=0.95)

        run = FakeRun(sampled=True)
        lane = QualityLane(enabled(), judge=partial)
        lane.process_span(span({}, errored=True), run)
        emitted = values(lane.on_run_end(run))

        assert emitted[semconv.RUN_QUALITY_GROUNDEDNESS] == 0.95
        assert emitted[semconv.RUN_SUCCESS] is False
