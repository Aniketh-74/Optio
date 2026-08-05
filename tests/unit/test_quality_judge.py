"""The sampled LLM-judge (M5-3).

Two guarantees carry the weight, and both are about what the judge must *not* do.

**It must never block the agent.** A judge is a model call against the SC-5
budget of 5 ms per step. It runs off the hot path, and a judge that hangs
produces a missing signal rather than a stalled run.

**It must never be a credential or spending risk.** optio ships no default
judge and constructs no model client (Section 10), so enabling the quality lane
without supplying one costs nothing and calls nothing.

Score validation is third but not minor: a judge is itself a model, and a model
returning `7` on a `[0,1]` scale has misread the scale. Clamping to `1.0` would
publish a confident wrong number -- the "who evals the evaluator" trap (R-TECH-5).
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from optio.lanes.quality.judge import (
    MAX_PENDING,
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
    """optio ships no default judge (Section 10)."""

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
        with caplog.at_level("WARNING", logger="optio"):
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
        # optio passes none of its own (Section 10); the lane builds an
        # empty mapping and the user closes over their own record if they want
        # the judge to read text.
        assert request().content == {}


class TestScoresArePushedRatherThanPolled:
    """The delivery mode that can actually deliver a model call's result.

    Polling with a zero-second timeout -- what the lane did through 0.4.0 --
    could only succeed if the judge finished between two adjacent statements on
    the calling thread. Against a warm pool that is 2 successes in 200 with an
    *instant* judge, and none at all with a realistic one.
    """

    def test_a_callback_receives_the_scores(self) -> None:
        runner = JudgeRunner(_scoring())
        delivered: list[JudgeScores] = []

        runner.submit(request(), on_scores=delivered.append)
        assert runner.drain(5.0) == 0
        runner.shutdown()

        assert delivered == [JudgeScores(groundedness=0.9, task_success=0.8)]

    def test_a_slow_judge_still_delivers(self) -> None:
        """The case the polling model could never serve."""

        def slow(_request: JudgeRequest) -> JudgeScores:
            time.sleep(0.2)
            return JudgeScores(task_success=0.7)

        runner = JudgeRunner(slow)
        delivered: list[JudgeScores] = []

        started = time.perf_counter()
        runner.submit(request(), on_scores=delivered.append)
        dispatch_took = time.perf_counter() - started

        assert runner.drain(5.0) == 0
        runner.shutdown()

        assert dispatch_took < 0.1, "submit blocked on the judge"
        assert delivered == [JudgeScores(task_success=0.7)]

    def test_a_failing_judge_delivers_nothing(self) -> None:
        """Absence, not a zero score (ADR-044)."""

        def raises(_request: JudgeRequest) -> JudgeScores:
            raise RuntimeError("judge exploded")

        runner = JudgeRunner(raises)
        delivered: list[JudgeScores] = []

        runner.submit(request(), on_scores=delivered.append)
        assert runner.drain(5.0) == 0
        runner.shutdown()

        assert delivered == []

    def test_an_out_of_range_score_is_dropped_on_this_path_too(self) -> None:
        """Validation cannot live on one delivery path only.

        A rule enforced by ``collect`` alone would have stopped running the day
        the callback became the mode the lane actually uses.
        """

        def wild(_request: JudgeRequest) -> JudgeScores:
            return JudgeScores(groundedness=7.0, task_success=0.8)

        runner = JudgeRunner(wild)
        delivered: list[JudgeScores] = []

        runner.submit(request(), on_scores=delivered.append)
        assert runner.drain(5.0) == 0
        runner.shutdown()

        assert delivered == [JudgeScores(groundedness=None, task_success=0.8)]

    def test_a_judge_returning_the_wrong_type_delivers_nothing(self) -> None:
        runner = JudgeRunner(lambda _request: "not scores")  # type: ignore[arg-type,return-value]
        delivered: list[JudgeScores] = []

        runner.submit(request(), on_scores=delivered.append)
        assert runner.drain(5.0) == 0
        runner.shutdown()

        assert delivered == []

    def test_a_callback_that_raises_does_not_escape(self) -> None:
        """It runs on a pool thread, where nothing is left to catch it.

        The fail-open guard wraps the agent's call stack; this is no longer on
        it. An escaping exception would surface as an unhandled callback error
        from ``concurrent.futures``, which is a confusing way to learn anything.
        """

        def explode(_scores: JudgeScores) -> None:
            raise RuntimeError("emitter exploded")

        runner = JudgeRunner(_scoring())
        runner.submit(request(), on_scores=explode)

        # Delivery completed -- the run was released -- despite the failure.
        assert runner.drain(5.0) == 0
        assert runner.pending_count() == 0
        runner.shutdown()

    def test_delivery_releases_the_run(self) -> None:
        """State released on completion, not left for a later collect.

        A long-lived process that judged and never released would accumulate one
        future per sampled run -- the leak class stress testing found in the
        ledger.
        """
        runner = JudgeRunner(_scoring())

        runner.submit(request("a"), on_scores=lambda _scores: None)
        runner.submit(request("b"), on_scores=lambda _scores: None)
        assert runner.drain(5.0) == 0

        assert runner.pending_count() == 0
        runner.shutdown()


class TestDrainingBeforeShutdown:
    def test_shutdown_lets_an_outstanding_judgement_land(self) -> None:
        """It was already paid for.

        ``shutdown`` cancelled in-flight futures, so a process that stopped just
        after its last run discarded a model call the user had been billed for.
        Cancelling does not refund it -- it only throws away what it bought.
        """

        def slow(_request: JudgeRequest) -> JudgeScores:
            time.sleep(0.15)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(slow)
        delivered: list[JudgeScores] = []

        runner.submit(request(), on_scores=delivered.append)
        runner.shutdown()

        assert delivered == [JudgeScores(task_success=0.9)]

    def test_a_zero_timeout_discards_immediately(self) -> None:
        """The escape hatch for a caller that must not wait at all."""
        gate = threading.Event()

        def blocked(_request: JudgeRequest) -> JudgeScores:
            gate.wait(timeout=5.0)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(blocked)
        delivered: list[JudgeScores] = []

        runner.submit(request(), on_scores=delivered.append)
        started = time.perf_counter()
        runner.shutdown(drain_timeout=0.0)
        elapsed = time.perf_counter() - started
        gate.set()

        assert elapsed < 0.5
        assert delivered == []

    def test_a_hung_judge_does_not_hang_shutdown(self) -> None:
        """Bounded, because the alternative is a process that will not exit."""
        gate = threading.Event()

        def hangs(_request: JudgeRequest) -> JudgeScores:
            gate.wait(timeout=10.0)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(hangs)
        runner.submit(request(), on_scores=lambda _scores: None)

        started = time.perf_counter()
        outstanding = runner.drain(0.2)
        elapsed = time.perf_counter() - started
        gate.set()
        runner.shutdown(drain_timeout=0.0)

        assert elapsed < 1.0
        assert outstanding == 1, "a hung judgement should be reported as outstanding"

    def test_draining_nothing_returns_immediately(self) -> None:
        runner = JudgeRunner(_scoring())

        started = time.perf_counter()
        assert runner.drain(5.0) == 0

        assert time.perf_counter() - started < 0.5


class TestTheQueueIsBounded:
    """Delivery holds a run until its scores land, so the pending set is now
    state that can grow -- and the pool is two threads wide, so a judge slower
    than the run rate is not a hypothetical."""

    def test_submissions_are_refused_at_the_cap(self) -> None:
        gate = threading.Event()

        def blocked(_request: JudgeRequest) -> JudgeScores:
            gate.wait(timeout=10.0)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(blocked)
        try:
            accepted = [runner.submit(request(f"run-{n}")) for n in range(MAX_PENDING + 10)]

            assert accepted[:MAX_PENDING] == [True] * MAX_PENDING
            assert accepted[MAX_PENDING:] == [False] * 10
            assert runner.pending_count() == MAX_PENDING
        finally:
            gate.set()
            runner.shutdown(drain_timeout=0.0)

    def test_capacity_returns_as_judgements_land(self) -> None:
        """A burst must not disable the judge permanently."""
        gate = threading.Event()

        def blocked(_request: JudgeRequest) -> JudgeScores:
            gate.wait(timeout=10.0)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(blocked)
        try:
            for n in range(MAX_PENDING):
                runner.submit(request(f"run-{n}"), on_scores=lambda _scores: None)
            assert runner.submit(request("over")) is False

            gate.set()
            assert runner.drain(5.0) == 0

            assert runner.submit(request("after"), on_scores=lambda _scores: None) is True
        finally:
            gate.set()
            runner.shutdown(drain_timeout=1.0)

    def test_the_refusal_is_logged_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """A saturated judge saturates on every run; a line each would flood the
        user's logs exactly when their system is already struggling."""
        gate = threading.Event()

        def blocked(_request: JudgeRequest) -> JudgeScores:
            gate.wait(timeout=10.0)
            return JudgeScores(task_success=0.9)

        runner = JudgeRunner(blocked)
        try:
            with caplog.at_level(logging.WARNING, logger="optio"):
                for n in range(MAX_PENDING + 20):
                    runner.submit(request(f"run-{n}"))

            saturation = [r for r in caplog.records if "in flight" in r.getMessage()]
            assert len(saturation) == 1
        finally:
            gate.set()
            runner.shutdown(drain_timeout=0.0)
