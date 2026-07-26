"""Architecture boundaries (Section 3.1, rule 11).

``lint-imports`` enforces the layering statically in CI. These tests cover the
part it cannot: that the abstractions the layering *forces* actually fit the
concrete types on the other side of the boundary.

The risk being managed is a decoupling that succeeds structurally and fails in
practice -- a ``RunLike`` protocol that ``RunContext`` does not satisfy would pass
the import check and then break every lane at wiring time.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agentmeter import semconv
from agentmeter.lanes.base import Lane, RunLike, Signal, enabled_lanes
from agentmeter.runtime.run_context import RunContext
from agentmeter.store.base import StateStore


class TestRunLikeProtocol:
    def test_run_context_satisfies_the_protocol(self):
        # If this fails, lanes cannot accept a real run despite type-checking.
        assert isinstance(RunContext(), RunLike)

    def test_protocol_exposes_only_what_lanes_need(self):
        run = RunContext(budget="$1.00")
        checked: RunLike = run
        assert checked.run_id == run.run_id
        assert checked.budget is run.budget

    def test_a_plain_stub_satisfies_the_protocol(self):
        # Lanes must be testable without constructing runtime objects.
        class StubRun:
            run_id = "stub"
            budget = None

        assert isinstance(StubRun(), RunLike)


class TestAbstractContracts:
    def test_lane_cannot_be_instantiated_directly(self):
        try:
            Lane(config=None)  # type: ignore[abstract,arg-type]
        except TypeError:
            return
        raise AssertionError("Lane must be abstract; process_span is required")

    def test_state_store_cannot_be_instantiated_directly(self):
        try:
            StateStore()  # type: ignore[abstract]
        except TypeError:
            return
        raise AssertionError("StateStore must be abstract")

    def test_no_lanes_are_enabled_in_m0(self):
        # Lanes land in M2/M3/M5. Asserting the current state makes their arrival
        # a deliberate, visible change rather than a silent one.
        from agentmeter.config import Config

        assert enabled_lanes(Config()) == []


class TestSignal:
    def test_signal_carries_a_semconv_name_and_value(self):
        signal = Signal(name=semconv.RUN_ACTUAL_COST, value=1.0)
        assert signal.name == semconv.RUN_ACTUAL_COST
        assert signal.value == 1.0

    def test_signal_is_immutable(self):
        signal = Signal(name=semconv.RUN_ACTUAL_COST, value=1.0)
        with pytest.raises(FrozenInstanceError):
            signal.value = 2.0  # type: ignore[misc]
