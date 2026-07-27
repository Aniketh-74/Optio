"""SC-4 for the quality lane: no internal failure reaches the agent.

The generic proof lives in ``test_broken_lane_never_breaks_agent.py``. This file
covers what is specific to *this* lane, which has a failure surface none of the
others do: **it calls user-supplied code**.

The judge is written by the user against their own SDK. It can raise anything,
hang forever, return garbage, or -- since it runs on a worker thread -- fail in a
way that would otherwise surface far from its cause. A monitoring layer that let
a broken evaluator take down an agent would be strictly worse than not scoring
at all (ADR-004).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import pytest
from opentelemetry.trace import StatusCode

from optio import semconv
from optio.config import Config
from optio.lanes.base import Signal
from optio.lanes.quality.judge import JudgeRequest, JudgeScores
from optio.lanes.quality.lane import QualityLane
from optio.runtime import failopen
from optio.runtime.failopen import guard_signals

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.failinject


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    failopen.reset_activations()
    yield
    failopen.reset_activations()


class Run:
    """A minimal run object."""

    run_id = "run-1"
    budget = None
    sampled = True
    successes: int | None = None


def enabled(judge: Any = None) -> Config:
    """Config with the quality lane on."""
    return Config(quality_lane=True, judge=judge)


def span(attributes: dict[str, object] | None = None, *, errored: bool = False) -> Mock:
    """Build a stand-in span."""
    mock = Mock()
    mock.name = "step"
    mock.attributes = attributes if attributes is not None else {}
    mock.status = Mock(status_code=StatusCode.ERROR if errored else StatusCode.OK)
    return mock


def _run_lane(lane: QualityLane, spans: list[Mock], run: Any) -> list[Signal]:
    """Drive a lane through a run behind the guard, as the tap does."""
    for one in spans:
        guard_signals(lane.process_span, one, run, component=lane.name)
    return cast("list[Signal]", guard_signals(lane.on_run_end, run, component=lane.name))


class TestAHostileJudgeCannotBreakTheAgent:
    """The judge is user code. Every way it can misbehave is absorbed."""

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("boom"),
            ValueError("bad"),
            KeyError("missing"),
            TypeError("wrong"),
            AttributeError("nope"),
            MemoryError(),
            RecursionError(),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_a_raising_judge_yields_no_signal_and_no_error(self, error: Exception) -> None:
        def raises(_request: JudgeRequest) -> JudgeScores:
            raise error

        lane = QualityLane(enabled(judge=raises))
        signals = _run_lane(lane, [span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10})], Run())

        # The agent survived; the quality signal is simply missing.
        assert not any(
            s.name in {semconv.RUN_QUALITY_GROUNDEDNESS, semconv.RUN_QUALITY_TASK_SUCCESS}
            for s in signals
        )
        lane.shutdown()

    def test_a_judge_returning_garbage_yields_no_signal(self) -> None:
        for junk in (None, "excellent", 0.9, {"score": 1}, [1, 2], object()):

            def returns_junk(_request: JudgeRequest, value: object = junk) -> JudgeScores:
                return value  # type: ignore[return-value]

            lane = QualityLane(enabled(judge=returns_junk))
            signals = _run_lane(lane, [span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10})], Run())

            assert not any(s.name == semconv.RUN_QUALITY_TASK_SUCCESS for s in signals)
            lane.shutdown()

    def test_a_judge_that_mutates_its_request_cannot_corrupt_the_run(self) -> None:
        def hostile(request: JudgeRequest) -> JudgeScores:
            with contextlib.suppress(AttributeError):
                request.run_id = "hijacked"  # type: ignore[misc]
            return JudgeScores(task_success=1.0)

        run = Run()
        lane = QualityLane(enabled(judge=hostile))
        _run_lane(lane, [span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10})], run)

        assert run.run_id == "run-1"
        lane.shutdown()

    def test_a_judge_that_never_returns_does_not_hang_the_run(self) -> None:
        import threading

        release = threading.Event()

        def hangs(_request: JudgeRequest) -> JudgeScores:
            release.wait(timeout=30)
            return JudgeScores(task_success=1.0)

        lane = QualityLane(enabled(judge=hangs))
        try:
            signals = _run_lane(lane, [span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10})], Run())
            assert not any(s.name == semconv.RUN_QUALITY_TASK_SUCCESS for s in signals)
        finally:
            release.set()
            lane.shutdown()


class TestHostileSpansCannotBreakTheLane:
    """Spans are framework output; the lane must survive whatever arrives."""

    def test_a_span_with_no_attributes_is_survivable(self) -> None:
        lane = QualityLane(enabled())
        bare = Mock()
        bare.name = "step"
        bare.attributes = None
        bare.status = Mock(status_code=StatusCode.OK)

        _run_lane(lane, [bare], Run())
        lane.shutdown()

    def test_an_exploding_attribute_mapping_is_survivable(self) -> None:
        class HostileMappingError(dict):  # type: ignore[type-arg]
            def get(self, *args: object, **kwargs: object) -> object:
                raise RuntimeError("attribute access exploded")

        lane = QualityLane(enabled())
        _run_lane(lane, [span(HostileMappingError())], Run())
        lane.shutdown()

    def test_a_span_with_no_status_is_survivable(self) -> None:
        lane = QualityLane(enabled())
        odd = Mock(spec=["name", "attributes"])
        odd.name = "step"
        odd.attributes = {semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 5}

        _run_lane(lane, [odd], Run())
        lane.shutdown()

    def test_a_run_without_a_run_id_is_survivable(self) -> None:
        class NoId:
            budget = None
            sampled = False

        lane = QualityLane(enabled())
        _run_lane(lane, [span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 5})], NoId())
        lane.shutdown()


class TestTheGuardRecordsWhatItAbsorbed:
    def test_a_lane_that_raises_outright_is_counted(self) -> None:
        lane = QualityLane(enabled())

        def explode(_run: object) -> list[object]:
            raise RuntimeError("lane is broken")

        lane.on_run_end = explode  # type: ignore[method-assign,assignment]
        result = guard_signals(lane.on_run_end, Run(), component=lane.name)

        assert result == []
        assert failopen.activation_count() >= 1
        lane.shutdown()
