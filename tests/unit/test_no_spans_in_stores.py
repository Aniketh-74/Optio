"""Nothing a lane hands its store may be a span, or hold one.

The design spec names this test. It is written once for all three lanes rather
than three times, because the property is about the boundary, not about any
lane's logic: everything crossing into a store must be a value, so that the same
call works whether the store is a dictionary in this process or a Redis on
another machine.

A ``ReadableSpan`` is the counter-example that motivated it. It is not
serializable, it holds a reference to the tracer's resource and instrumentation
scope, and the quality lane kept sixty-four of them per run -- which is why that
lane could not be shared across processes at all.

The failure this guards against is quiet in the worst way. A lane that passed a
span would work perfectly in every single-process test, in CI, and in
development, and would fail only for the users who configured the shared backend
-- who are, by definition, the ones running the largest deployments.

So each lane is driven through its real code path with a store that inspects
what it is given, rather than asserting on the store implementations, which
could agree with each other about the wrong thing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from optio import semconv
from optio.config import Config

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Types a value may be built from and still cross a process boundary. Checked
#: recursively, so a tuple of spans fails as surely as a bare one.
_SERIALIZABLE = (bool, int, float, str, type(None))


class SpanRefusingStore:
    """Records what it is handed, and rejects anything that cannot travel.

    Not a mock. The point is to run the *lane's* real call, with real arguments,
    and inspect them -- a mock would record the call and prove only that one was
    made.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _check(self, value: object, path: str) -> None:
        """Fail if a value could not survive leaving this process.

        Args:
            value: The value to inspect.
            path: Where it sits, for the failure message.

        Raises:
            AssertionError: If the value is a span, or contains one.
        """
        assert not isinstance(value, ReadableSpan), f"{path} is a ReadableSpan"
        assert not isinstance(value, Mock), f"{path} is a Mock -- a span stand-in"

        if isinstance(value, _SERIALIZABLE):
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._check(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._check(item, f"{path}[{key!r}]")
            return
        if hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                self._check(getattr(value, name), f"{path}.{name}")
            return

        raise AssertionError(f"{path} is a {type(value).__name__}, which cannot cross a process")

    def _record(self, method: str, *args: object, **kwargs: object) -> None:
        """Check and remember one call's arguments."""
        for index, arg in enumerate(args):
            self._check(arg, f"{method}(arg {index})")
        for name, value in kwargs.items():
            self._check(value, f"{method}({name}=)")
        self.seen.append((method, args, kwargs))


class QualityRecorder(SpanRefusingStore):
    """A ``QualityStore`` that refuses spans."""

    def record(self, run_id: str, step: object) -> None:
        self._record("record", run_id, step)

    def close_run(self, run_id: str) -> None:
        self._record("close_run", run_id)
        return None

    def run_count(self) -> int:
        return 0


class BehaviorRecorder(SpanRefusingStore):
    """A ``BehaviorStore`` that refuses spans."""

    def record(
        self,
        run_id: str,
        signature_call: tuple[str, str],
        errored: bool,
        maxlen: int,
        k: int,
    ) -> object:
        self._record("record", run_id, signature_call, errored, maxlen=maxlen, k=k)
        from optio.lanes.behavior.store import WindowState

        return WindowState(size=1, errors=0, distinct_calls=1, top_counts=(1,))

    def close_run(self, run_id: str, k: int) -> None:
        self._record("close_run", run_id, k=k)
        return None

    def run_count(self) -> int:
        return 0


def span(attributes: Mapping[str, object] | None = None) -> Any:
    """A finished-span stand-in carrying what the lanes read.

    Given a real span id, because the cost lane derives its step id from one --
    a mock there raises inside ``format()`` rather than reaching the store, and
    a guard that never reaches the store proves nothing.
    """
    mock = Mock()
    mock.name = "gen_ai.chat"
    mock.attributes = {
        semconv.GEN_AI_SYSTEM: "openai",
        semconv.GEN_AI_TOOL_NAME: "search",
        semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
        semconv.GEN_AI_USAGE_INPUT_TOKENS: 100,
        semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 12,
        **(attributes or {}),
    }
    mock.status = Mock(status_code=StatusCode.OK)
    mock.get_span_context.return_value = Mock(span_id=0x1234ABCD, trace_id=0x99)
    return mock


class Run:
    """A run object the lanes accept."""

    run_id = "run-1"
    budget = None
    sampled = True
    successes: int | None = None


class TestNoLaneHandsASpanToItsStore:
    def test_the_quality_lane_projects_before_recording(self) -> None:
        from optio.lanes.quality.lane import QualityLane

        store = QualityRecorder()
        lane = QualityLane(Config(quality_lane=True), store=store)
        run = Run()

        lane.process_span(span(), run)
        lane.on_run_end(run)

        assert [call[0] for call in store.seen] == ["record", "close_run"]

    def test_the_behavior_lane_signs_before_recording(self) -> None:
        from optio.lanes.behavior.lane import BehaviorLane

        store = BehaviorRecorder()
        lane = BehaviorLane(Config(), store=store)  # type: ignore[arg-type]
        run = Run()

        lane.process_span(span(), run)
        lane.on_run_end(run)

        assert [call[0] for call in store.seen] == ["record", "close_run"]

    def test_the_cost_lane_passes_only_numbers(self) -> None:
        """The cost lane never held spans, so this is a regression guard rather
        than a fix -- its ledger takes a run id, a step id and a float, and the
        cheapest way for that to change is someone passing the span along "for
        context"."""
        from optio.lanes.cost.ledger import CostLedger
        from optio.lanes.cost.ledger_memory import InMemoryLedgerStore

        checker = SpanRefusingStore()
        backend = InMemoryLedgerStore()

        class Checked(InMemoryLedgerStore):
            def reserve(self, run_id: str, step_id: str, projected: float) -> None:
                checker._record("reserve", run_id, step_id, projected)
                backend.reserve(run_id, step_id, projected)

            def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
                checker._record("reconcile", run_id, step_id, actual)
                backend.reconcile(run_id, step_id, actual)

        from optio.lanes.cost.lane import CostLane

        lane = CostLane(Config(), ledger=CostLedger(store=Checked()))
        run = Run()

        lane.process_span(span(), run)

        assert checker.seen, "the cost lane recorded nothing, so nothing was checked"


class TestTheGuardItselfWorks:
    """A test that cannot fail is worse than no test. These drive the checker
    with the values it exists to reject."""

    def test_a_span_is_rejected(self) -> None:
        store = SpanRefusingStore()

        with pytest.raises(AssertionError, match="Mock"):
            store._record("record", "run", span())

    def test_a_span_nested_in_a_tuple_is_rejected(self) -> None:
        store = SpanRefusingStore()

        with pytest.raises(AssertionError, match=r"\[1\]"):
            store._record("record", "run", ("fine", span()))

    def test_a_span_inside_a_dataclass_field_is_rejected(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class Sneaky:
            tokens: int
            source: object

        store = SpanRefusingStore()

        with pytest.raises(AssertionError, match="source"):
            store._record("record", "run", Sneaky(tokens=1, source=span()))

    def test_an_arbitrary_object_is_rejected(self) -> None:
        store = SpanRefusingStore()

        with pytest.raises(AssertionError, match="cannot cross a process"):
            store._record("record", "run", object())

    def test_the_real_projections_pass(self) -> None:
        """The other direction: the values the lanes actually send are accepted,
        so the tests above are green because the code is right rather than
        because the checker is lenient."""
        from optio.lanes.behavior.window import signature_of
        from optio.lanes.quality.heuristic import project

        store = SpanRefusingStore()

        store._record("record", "run", project(span()))
        store._record("record", "run", signature_of(span()).call)

        assert len(store.seen) == 2

    def test_accepted_values_really_are_serializable(self) -> None:
        """``_SERIALIZABLE`` is a list of types someone has to keep honest, so
        it is checked against an actual encoder rather than trusted."""
        from dataclasses import asdict

        from optio.lanes.quality.heuristic import project

        json.dumps(asdict(project(span())))
