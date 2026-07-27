"""The sampled LLM-judge (M5-3).

Two guarantees carry the weight, and both are about what the judge must *not* do.

**It must never block the agent.** A judge is a model call against the SC-5
budget of 5 ms per step. It runs off the hot path, and a judge that hangs
produces a missing signal rather than a stalled run.

**It must never be a credential or spending risk.** agentmeter ships no default
judge and constructs no model client (Section 10), so enabling the quality lane
without supplying one costs nothing and calls nothing.

Score validation is third but not minor: a judge is itself a model, and a model
returning `7` on a `[0,1]` scale has misread the scale. Clamping to `1.0` would
publish a confident wrong number -- the "who evals the evaluator" trap (R-TECH-5).
"""

from __future__ import annotations

import threading
import time

import pytest

from agentmeter.lanes.quality.judge import (
    Judge,
    JudgeRequest,
    JudgeRunner,
    JudgeScores,
)


def request(run_id: str = "run-1") -> JudgeRequest:
    """Build a judge request."""
    return JudgeRequest(run_id=run_id, step_count=3, content={})


def _scoring(groundedness: float = 0.9, task_success: float = 0.8) -> Judge:
    """A judge returning fixed scores."""

    def judge(_request: JudgeRequest) -> JudgeScores:
        return JudgeScores(groundedness=groundedness, task_success=task_success)

    return judge


class TestNoJudgeMeansNoSpend:
    """agentmeter ships no default judge (Section 10)."""

    def test_a_runner_without_a_judge_is_disabled(self) -> None:
        assert JudgeRunner(None).enabled is False

    def test_submitting_without_a_judge_does_nothing(self) -> None:
        runner = JudgeRunner(None)
        assert runner.submit(request()) is False
        assert runner.pending_count() == 0

    def test_collecting_without_a_judge_returns_nothing(self) -> None:
        assert JudgeRunner(None).collect("run-1") is None

    def test_no_thread_pool_is_created_when_unused(self) -> None:
        # A user with the lane on but no judge should not pay for threads that
        # will never run anything.
        before = threading.active_count()
        runner = JudgeRunner(None)
        runner.submit(request())
        assert threading.active_count() == before
        runner.shutdown()


class TestTheJudgeNeverBlocksTheRun:
    def test_a_hanging_judge_does_not_stall_collection(self) -> None:
        release = threading.Event()

        def hangs(_request: JudgeRequest) -> JudgeScores:
            release.wait(timeout=10)
            return JudgeScores(task_success=1.0)

        runner = JudgeRunner(hangs)
        runner.submit(request())

        started = time.monotonic()
        result = runner.collect("run-1")
        elapsed = time.monotonic() - started

        release.set()
        runner.shutdown()

        assert result is None, "an unfinished judge must yield no signal"
        assert elapsed < 1.0, f"collection blocked for {elapsed:.2f}s"

    def test_a_slow_judge_can_be_waited_for_explicitly(self) -> None:
        def slow(_request: JudgeRequest) -> JudgeScores:
            time.sleep(0.05)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(slow)
        runner.submit(request())
        result = runner.collect("run-1", timeout=5.0)
        runner.shutdown()

        assert result is not None
        assert result.task_success == 0.9


class TestAFailingJudgeIsAMissingSignal:
    """ADR-004: never an error the agent sees."""

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("boom"), ValueError("bad"), KeyError("k"), TimeoutError()],
        ids=["runtime", "value", "key", "timeout"],
    )
    def test_any_exception_yields_no_score(self, error: Exception) -> None:
        def raises(_request: JudgeRequest) -> JudgeScores:
            raise error

        runner = JudgeRunner(raises)
        runner.submit(request())
        result = runner.collect("run-1", timeout=5.0)
        runner.shutdown()

        assert result is None

    def test_the_exception_message_is_never_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        # A model client's exception routinely carries the prompt in its
        # payload, so only the type may be logged (Section 10).
        secret = "patient SSN 123-45-6789"

        def raises(_request: JudgeRequest) -> JudgeScores:
            raise RuntimeError(secret)

        runner = JudgeRunner(raises)
        runner.submit(request())
        with caplog.at_level("WARNING", logger="agentmeter"):
            runner.collect("run-1", timeout=5.0)
        runner.shutdown()

        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text

    def test_a_judge_returning_the_wrong_type_yields_no_score(self) -> None:
        def wrong(_request: JudgeRequest) -> JudgeScores:
            return {"task_success": 0.9}  # type: ignore[return-value]

        runner = JudgeRunner(wrong)
        runner.submit(request())
        result = runner.collect("run-1", timeout=5.0)
        runner.shutdown()

        assert result is None


class TestScoresAreValidatedNotClamped:
    @pytest.mark.parametrize("value", [1.5, 7, -0.1, -1, 100])
    def test_an_out_of_range_score_is_dropped(self, value: float) -> None:
        # Dropped, not clamped: a judge returning 7 misread the scale, and
        # clamping to 1.0 publishes a confident wrong number where absence is
        # the honest answer.
        scores = JudgeScores(task_success=value).validated()
        assert scores.task_success is None

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_an_in_range_score_survives(self, value: float) -> None:
        assert JudgeScores(task_success=value).validated().task_success == value

    def test_the_boundaries_are_inclusive(self) -> None:
        scores = JudgeScores(groundedness=0.0, task_success=1.0).validated()
        assert scores.groundedness == 0.0
        assert scores.task_success == 1.0

    def test_nan_is_dropped(self) -> None:
        assert JudgeScores(task_success=float("nan")).validated().task_success is None

    def test_infinity_is_dropped(self) -> None:
        assert JudgeScores(task_success=float("inf")).validated().task_success is None

    @pytest.mark.parametrize("value", ["0.9", None, [0.9], True])
    def test_a_non_numeric_score_is_dropped(self, value: object) -> None:
        # `True` included deliberately: bool subclasses int, so it would
        # otherwise validate as the number 1.0.
        scores = JudgeScores(task_success=value).validated()  # type: ignore[arg-type]
        assert scores.task_success is None

    def test_one_bad_score_does_not_discard_the_other(self) -> None:
        scores = JudgeScores(groundedness=0.9, task_success=99).validated()
        assert scores.groundedness == 0.9
        assert scores.task_success is None

    def test_all_scores_invalid_collects_as_nothing(self) -> None:
        runner = JudgeRunner(_scoring(groundedness=5.0, task_success=9.0))
        runner.submit(request())
        result = runner.collect("run-1", timeout=5.0)
        runner.shutdown()

        assert result is None, "a fully invalid judgement is no judgement"


class TestStateIsReleased:
    """Agents are long-lived; nothing may accumulate per run."""

    def test_collecting_releases_the_run(self) -> None:
        runner = JudgeRunner(_scoring())
        runner.submit(request())
        runner.collect("run-1", timeout=5.0)
        assert runner.pending_count() == 0
        runner.shutdown()

    def test_discarding_releases_the_run(self) -> None:
        runner = JudgeRunner(_scoring())
        runner.submit(request())
        runner.discard("run-1")
        assert runner.pending_count() == 0
        runner.shutdown()

    def test_a_duplicate_submit_does_not_pay_twice(self) -> None:
        # Run end can fire more than once (M1-2). A second dispatch would buy
        # the same judgement again, on the user's money.
        calls = []

        def counting(req: JudgeRequest) -> JudgeScores:
            calls.append(req.run_id)
            return JudgeScores(task_success=1.0)

        runner = JudgeRunner(counting)
        assert runner.submit(request()) is True
        assert runner.submit(request()) is False
        runner.collect("run-1", timeout=5.0)
        runner.shutdown()

        assert len(calls) == 1

    def test_collecting_an_unknown_run_is_not_an_error(self) -> None:
        runner = JudgeRunner(_scoring())
        assert runner.collect("never-submitted") is None
        runner.shutdown()

    def test_shutdown_is_idempotent(self) -> None:
        runner = JudgeRunner(_scoring())
        runner.submit(request())
        runner.shutdown()
        runner.shutdown()
        assert runner.pending_count() == 0


class TestConcurrency:
    def test_many_runs_are_scored_independently(self) -> None:
        runner = JudgeRunner(_scoring(task_success=0.7))
        for n in range(20):
            runner.submit(request(f"run-{n}"))

        results = [runner.collect(f"run-{n}", timeout=5.0) for n in range(20)]
        runner.shutdown()

        assert all(r is not None and r.task_success == 0.7 for r in results)

    def test_concurrent_submits_do_not_lose_runs(self) -> None:
        runner = JudgeRunner(_scoring())
        errors: list[BaseException] = []

        def submit(n: int) -> None:
            try:
                runner.submit(request(f"run-{n}"))
            except BaseException as error:  # noqa: BLE001 - recorded, then asserted
                errors.append(error)

        threads = [threading.Thread(target=submit, args=(n,)) for n in range(30)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        pending = runner.pending_count()
        runner.shutdown()

        assert not errors
        assert pending == 30


class TestRequestCarriesNoContentByDefault:
    def test_the_request_is_frozen(self) -> None:
        req = request()
        with pytest.raises(AttributeError):
            req.run_id = "other"  # type: ignore[misc]

    def test_content_is_the_callers_to_supply(self) -> None:
        # agentmeter passes none of its own (Section 10); the lane builds an
        # empty mapping and the user closes over their own record if they want
        # the judge to read text.
        assert request().content == {}
