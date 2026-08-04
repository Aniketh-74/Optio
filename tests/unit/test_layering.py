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

from optio import semconv
from optio.lanes.base import Lane, RunLike, Signal
from optio.lanes.registry import enabled_lanes
from optio.runtime.run_context import RunContext


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

    def test_the_ledger_store_protocol_is_not_instantiable_state(self):
        """ADR-050 replaced the generic ``StateStore`` ABC with per-lane
        Protocols, so there is no abstract base to instantiate any more.

        What the old test protected -- that the storage contract is a contract
        and not a usable object -- now holds structurally: a ``Protocol`` is a
        typing construct, and the backends satisfy it without inheriting it.
        The property worth asserting instead is that both backends really do
        satisfy the same set of operations, and the parametrised contract suite
        in ``test_ledger_store_contract.py`` does that against live objects
        rather than against a declaration.
        """
        from optio.lanes.cost.ledger_memory import InMemoryLedgerStore
        from optio.lanes.cost.ledger_redis import RedisLedgerStore
        from optio.lanes.cost.ledger_store import LedgerStore

        required = [name for name in vars(LedgerStore) if not name.startswith("_")]
        assert required, "the Protocol declares no operations"
        for backend in (InMemoryLedgerStore, RedisLedgerStore):
            missing = [name for name in required if not hasattr(backend, name)]
            assert not missing, f"{backend.__name__} is missing {missing}"

    def test_both_behavior_backends_satisfy_their_protocol(self):
        """The same check for the behaviour lane's own Protocol.

        Per-lane rather than shared, which is the point of ADR-050: a store
        that spoke both domains would have to offer primitives, and primitives
        cannot express either lane's atomicity requirement.
        """
        from optio.lanes.behavior.store import BehaviorStore
        from optio.lanes.behavior.store_memory import InMemoryBehaviorStore
        from optio.lanes.behavior.store_redis import RedisBehaviorStore

        required = [name for name in vars(BehaviorStore) if not name.startswith("_")]
        assert required, "the Protocol declares no operations"
        for backend in (InMemoryBehaviorStore, RedisBehaviorStore):
            missing = [name for name in required if not hasattr(backend, name)]
            assert not missing, f"{backend.__name__} is missing {missing}"

    def test_cost_and_behavior_are_enabled_by_default(self):
        # Cost landed in M2, behavior in M3. Quality (M5) is off by default
        # regardless (ADR-003), so it must not appear even once it exists.
        from optio.config import Config

        names = [lane.name for lane in enabled_lanes(Config())]
        assert names == ["cost", "behavior"]
        assert "quality" not in names

    def test_each_lane_flag_is_honoured_independently(self):
        # Asserts the rule rather than a snapshot of today's lane set: each
        # flag controls exactly its own lane and nothing else. A snapshot
        # assertion would have to be rewritten every milestone, which is how a
        # test stops being read and starts being updated reflexively.
        from optio.config import Config

        assert [lane.name for lane in enabled_lanes(Config(cost_lane=False))] == ["behavior"]
        assert [lane.name for lane in enabled_lanes(Config(behavior_lane=False))] == ["cost"]
        assert enabled_lanes(Config(cost_lane=False, behavior_lane=False)) == []


class TestSignal:
    def test_signal_carries_a_semconv_name_and_value(self):
        signal = Signal(name=semconv.RUN_ACTUAL_COST, value=1.0)
        assert signal.name == semconv.RUN_ACTUAL_COST
        assert signal.value == 1.0

    def test_signal_is_immutable(self):
        signal = Signal(name=semconv.RUN_ACTUAL_COST, value=1.0)
        with pytest.raises(FrozenInstanceError):
            signal.value = 2.0  # type: ignore[misc]
