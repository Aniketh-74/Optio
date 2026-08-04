"""Behavior lane unit tests (M3-3).

Beyond signal shape, two of these encode lessons the cost lane paid for:
per-run state must be released at run end (the unbounded leak stress testing
found), and a second run-end call must not emit a re-derived verdict (the
duplicate-emission bug that overwrote a correct value with a wrong one).
"""

from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest
from opentelemetry.trace import StatusCode

from optio import semconv
from optio.config import Config, default_config
from optio.lanes.base import Signal
from optio.lanes.behavior.lane import BehaviorLane


class Run:
    """Minimal RunLike stub."""

    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.budget = None


def span(tool: str = "search", *, errored: bool = False) -> Mock:
    mock = Mock()
    mock.name = "step"
    mock.attributes = {semconv.GEN_AI_TOOL_NAME: tool}
    mock.status = Mock(status_code=StatusCode.ERROR if errored else StatusCode.OK)
    return mock


def values(signals: list[Signal]) -> dict[str, bool | int | float | str]:
    return {s.name: s.value for s in signals}


class TestSignalShape:
    def test_every_step_emits_both_signals(self) -> None:
        lane = BehaviorLane(default_config())
        emitted = values(lane.process_span(span(), Run()))

        assert set(emitted) == {semconv.RUN_LOOP_STATE, semconv.RUN_REPEAT_COUNT}

    def test_the_state_is_a_contract_value(self) -> None:
        lane = BehaviorLane(default_config())
        for _ in range(20):
            emitted = values(lane.process_span(span(), Run()))
            assert emitted[semconv.RUN_LOOP_STATE] in semconv.LOOP_STATES

    def test_repeat_count_is_an_int(self) -> None:
        lane = BehaviorLane(default_config())
        emitted = values(lane.process_span(span(), Run()))

        count = emitted[semconv.RUN_REPEAT_COUNT]
        assert isinstance(count, int) and not isinstance(count, bool)

    def test_a_loop_is_reported_through_the_lane(self) -> None:
        lane = BehaviorLane(default_config())
        run = Run()
        for _ in range(10):
            emitted = values(lane.process_span(span("stuck"), run))

        assert emitted[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_LOOPING
        assert emitted[semconv.RUN_REPEAT_COUNT] == 10

    def test_a_healthy_run_stays_healthy(self) -> None:
        lane = BehaviorLane(default_config())
        run = Run()
        for n in range(20):
            emitted = values(lane.process_span(span(f"tool{n}"), run))

        assert emitted[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_HEALTHY


class TestRunIsolation:
    def test_two_runs_do_not_share_a_window(self) -> None:
        # Cross-run contamination would let one agent's loop be reported
        # against another agent's run.
        lane = BehaviorLane(default_config())
        looping, healthy = Run("looping"), Run("healthy")

        for _ in range(10):
            lane.process_span(span("stuck"), looping)
        emitted = values(lane.process_span(span("fresh"), healthy))

        assert emitted[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_HEALTHY
        assert emitted[semconv.RUN_REPEAT_COUNT] == 1

    def test_concurrent_runs_do_not_lose_steps(self) -> None:
        # Agent frameworks are frequently multi-threaded and the tap runs on
        # whichever thread ended the span.
        lane = BehaviorLane(Config(behavior_window_size=500))
        runs = [Run(f"run-{n}") for n in range(8)]
        errors: list[BaseException] = []

        def hammer(run: Run) -> None:
            try:
                for _ in range(200):
                    lane.process_span(span("t"), run)
            except BaseException as exc:  # noqa: BLE001 - surfaced via `errors`
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(run,)) for run in runs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert lane.run_count() == 8
        for run in runs:
            # Reaches through to the store since ADR-050 moved the windows and
            # their lock there. The assertion is unchanged -- what it protects
            # is that no step was lost, not where the dictionary lives.
            assert lane._store._windows[run.run_id].total_steps == 200  # type: ignore[attr-defined]


class TestWindowIsBounded:
    def test_a_long_run_retains_only_the_window(self) -> None:
        lane = BehaviorLane(Config(behavior_window_size=25))
        run = Run()
        for _ in range(5000):
            lane.process_span(span(), run)

        assert len(lane._store._windows[run.run_id]) == 25  # type: ignore[attr-defined]

    def test_repeat_count_is_bounded_by_the_window(self) -> None:
        # Not by the run length: a policy reading a count of 5000 from a
        # 25-step window would be reading a fabricated number.
        lane = BehaviorLane(Config(behavior_window_size=25))
        run = Run()
        for _ in range(5000):
            emitted = values(lane.process_span(span("same"), run))

        assert emitted[semconv.RUN_REPEAT_COUNT] == 25


class TestStateIsReleased:
    def test_the_window_is_evicted_at_run_end(self) -> None:
        # The ledger's leak, which every unit test missed because a single-run
        # test cannot see state accumulate across runs.
        lane = BehaviorLane(default_config())
        for n in range(2000):
            run = Run(f"run-{n}")
            lane.process_span(span(), run)
            lane.on_run_end(run)

        assert lane.run_count() == 0

    def test_run_end_emits_the_final_verdict(self) -> None:
        lane = BehaviorLane(default_config())
        run = Run()
        for _ in range(10):
            lane.process_span(span("stuck"), run)

        emitted = values(lane.on_run_end(run))
        assert emitted[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_LOOPING

    def test_a_repeat_run_end_emits_nothing(self) -> None:
        # Run end can fire more than once (M1-2). Re-deriving from the released
        # window would emit `healthy` with count 0, overwriting a real
        # `looping` verdict -- the cost lane's duplicate-emission bug, in the
        # direction that hides a pathology.
        lane = BehaviorLane(default_config())
        run = Run()
        for _ in range(10):
            lane.process_span(span("stuck"), run)

        assert lane.on_run_end(run) != []
        assert lane.on_run_end(run) == []
        assert lane.on_run_end(run) == []

    def test_run_end_on_a_run_with_no_steps_emits_nothing(self) -> None:
        # Absence, not `healthy`: we observed nothing, so we know nothing.
        assert BehaviorLane(default_config()).on_run_end(Run()) == []


class TestConfiguration:
    def test_the_window_size_comes_from_config(self) -> None:
        assert BehaviorLane(Config(behavior_window_size=7)).window_size == 7

    def test_the_lane_is_absent_when_disabled(self) -> None:
        from optio.lanes.registry import enabled_lanes

        names = [lane.name for lane in enabled_lanes(Config(behavior_lane=False))]
        assert "behavior" not in names

    def test_the_lane_is_wired_by_default(self) -> None:
        from optio.lanes.registry import enabled_lanes

        names = [lane.name for lane in enabled_lanes(default_config())]
        assert "behavior" in names

    def test_an_invalid_window_size_fails_at_setup(self) -> None:
        from optio.errors import OptioConfigError

        with pytest.raises(OptioConfigError):
            Config(behavior_window_size=0)
