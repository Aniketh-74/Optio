"""Property tests for the R-TECH-1 ledger invariant (M2-2 acceptance criterion).

    *reserve always precedes the step; reconcile replaces the reservation
    exactly once.*

These are the acceptance criterion, not a supplement to it. Both ways of
breaking the invariant produce a **wrong number rather than an error** -- a leak
over-reports cost, a double reconcile under-reports it -- so no amount of
fail-open protection helps and review alone is not evidence. The spec asks for
random interleavings; that is what these generate.

The model-based test is the load-bearing one: it runs a random operation
sequence against both the ledger and a naive reference implementation, and
asserts they agree. A bug that survives that has to be present in both, which is
a much harder mistake to make than an arithmetic slip in one.
"""

from __future__ import annotations

import threading

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

from optio.errors import LedgerInvariantError
from optio.lanes.cost.ledger import CostLedger

pytestmark = pytest.mark.property

# Costs are money: bounded, finite, and not absurdly precise. Denormals and
# 1e300 are not realistic token costs and would only test float behaviour.
costs = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
step_ids = st.text(min_size=1, max_size=8)
run_ids = st.text(min_size=1, max_size=8)


class TestReserveReconcilePairing:
    @given(run_id=run_ids, step_id=step_ids, cost=costs)
    def test_a_reserved_step_shows_as_reserved(
        self, run_id: str, step_id: str, cost: float
    ) -> None:
        ledger = CostLedger()
        ledger.reserve(run_id, step_id, cost)

        snapshot = ledger.snapshot(run_id)
        assert snapshot.reserved == pytest.approx(cost)
        assert snapshot.actual == 0.0
        assert snapshot.open_steps == 1

    @given(run_id=run_ids, step_id=step_ids, reserved=costs, actual=costs)
    def test_reconcile_replaces_the_reservation_exactly(
        self, run_id: str, step_id: str, reserved: float, actual: float
    ) -> None:
        # The core of R-TECH-1: after reconciling, the reservation is gone and
        # only the actual remains. Not both, not neither.
        ledger = CostLedger()
        ledger.reserve(run_id, step_id, reserved)
        ledger.reconcile(run_id, step_id, actual)

        snapshot = ledger.snapshot(run_id)
        assert snapshot.reserved == 0.0
        assert snapshot.actual == pytest.approx(actual)
        assert snapshot.open_steps == 0
        assert snapshot.reconciled_steps == 1

    @given(run_id=run_ids, step_id=step_ids, reserved=costs, actual=costs)
    def test_double_reconcile_is_rejected(
        self, run_id: str, step_id: str, reserved: float, actual: float
    ) -> None:
        # The dangerous direction: silently adding the cost twice would make a
        # run look cheaper per step and let a budget policy pass a run it should
        # have gated.
        ledger = CostLedger()
        ledger.reserve(run_id, step_id, reserved)
        ledger.reconcile(run_id, step_id, actual)

        with pytest.raises(LedgerInvariantError):
            ledger.reconcile(run_id, step_id, actual)

        assert ledger.snapshot(run_id).actual == pytest.approx(actual)

    @given(run_id=run_ids, step_id=step_ids, actual=costs)
    def test_reconcile_without_reserve_is_rejected(
        self, run_id: str, step_id: str, actual: float
    ) -> None:
        ledger = CostLedger()
        with pytest.raises(LedgerInvariantError):
            ledger.reconcile(run_id, step_id, actual)

    @given(run_id=run_ids, step_id=step_ids, first=costs, second=costs)
    def test_re_reserving_replaces_rather_than_stacks(
        self, run_id: str, step_id: str, first: float, second: float
    ) -> None:
        # A retried step reuses its id. Stacking would inflate `reserved` and
        # under-report budget_remaining for the rest of the run.
        ledger = CostLedger()
        ledger.reserve(run_id, step_id, first)
        ledger.reserve(run_id, step_id, second)

        snapshot = ledger.snapshot(run_id)
        assert snapshot.reserved == pytest.approx(second)
        assert snapshot.open_steps == 1


class TestInterleavings:
    @given(
        run_id=run_ids,
        steps=st.lists(
            st.tuples(step_ids, costs, costs), min_size=1, max_size=20, unique_by=lambda t: t[0]
        ),
    )
    def test_all_reserved_then_all_reconciled(
        self, run_id: str, steps: list[tuple[str, float, float]]
    ) -> None:
        # Fully-pipelined shape: every step reserved before any completes.
        ledger = CostLedger()
        for step_id, reserved, _ in steps:
            ledger.reserve(run_id, step_id, reserved)

        assert ledger.snapshot(run_id).open_steps == len(steps)

        for step_id, _, actual in steps:
            ledger.reconcile(run_id, step_id, actual)

        snapshot = ledger.snapshot(run_id)
        assert snapshot.reserved == 0.0
        assert snapshot.open_steps == 0
        assert snapshot.actual == pytest.approx(sum(a for _, _, a in steps))

    @given(
        run_id=run_ids,
        steps=st.lists(
            st.tuples(step_ids, costs, costs), min_size=1, max_size=20, unique_by=lambda t: t[0]
        ),
    )
    def test_sequential_reserve_reconcile_pairs(
        self, run_id: str, steps: list[tuple[str, float, float]]
    ) -> None:
        # Serial shape: each step completes before the next begins.
        ledger = CostLedger()
        for step_id, reserved, actual in steps:
            ledger.reserve(run_id, step_id, reserved)
            ledger.reconcile(run_id, step_id, actual)

        snapshot = ledger.snapshot(run_id)
        assert snapshot.reserved == 0.0
        assert snapshot.actual == pytest.approx(sum(a for _, _, a in steps))
        assert snapshot.reconciled_steps == len(steps)

    @given(
        run_id=run_ids,
        steps=st.lists(
            st.tuples(step_ids, costs, costs), min_size=2, max_size=20, unique_by=lambda t: t[0]
        ),
        split=st.integers(min_value=1, max_value=19),
    )
    def test_partial_reconciliation_leaves_the_rest_reserved(
        self, run_id: str, steps: list[tuple[str, float, float]], split: int
    ) -> None:
        assume(split < len(steps))
        ledger = CostLedger()
        for step_id, reserved, _ in steps:
            ledger.reserve(run_id, step_id, reserved)
        for step_id, _, actual in steps[:split]:
            ledger.reconcile(run_id, step_id, actual)

        snapshot = ledger.snapshot(run_id)
        assert snapshot.open_steps == len(steps) - split
        assert snapshot.reserved == pytest.approx(sum(r for _, r, _ in steps[split:]))
        assert snapshot.actual == pytest.approx(sum(a for _, _, a in steps[:split]))


class TestRunIsolation:
    @given(
        runs=st.lists(
            st.tuples(run_ids, step_ids, costs), min_size=2, max_size=10, unique_by=lambda t: t[0]
        )
    )
    def test_runs_do_not_bleed_into_each_other(self, runs: list[tuple[str, str, float]]) -> None:
        # run_id is the join key across lanes; a leak here would merge two
        # customers' costs.
        ledger = CostLedger()
        for run_id, step_id, cost in runs:
            ledger.reserve(run_id, step_id, cost)

        for run_id, _, cost in runs:
            assert ledger.snapshot(run_id).reserved == pytest.approx(cost)

    @given(run_id=run_ids, step_id=step_ids, cost=costs)
    def test_unknown_run_reads_as_zero(self, run_id: str, step_id: str, cost: float) -> None:
        ledger = CostLedger()
        ledger.reserve(run_id, step_id, cost)

        other = ledger.snapshot(run_id + "-absent")
        assert other.reserved == 0.0
        assert other.actual == 0.0
        assert other.open_steps == 0


class TestLeakDetection:
    @given(
        run_id=run_ids,
        steps=st.lists(
            st.tuples(step_ids, costs), min_size=1, max_size=10, unique_by=lambda t: t[0]
        ),
    )
    def test_unreconciled_reservations_are_reported_as_leaks(
        self, run_id: str, steps: list[tuple[str, float]]
    ) -> None:
        ledger = CostLedger()
        for step_id, cost in steps:
            ledger.reserve(run_id, step_id, cost)

        final = ledger.close_run(run_id)
        assert final.leaked_steps == len(steps)
        # The reservation is kept, not discarded: dropping it would make the run
        # look cheaper than the evidence supports.
        assert final.reserved == pytest.approx(sum(c for _, c in steps))

    @given(
        run_id=run_ids,
        steps=st.lists(
            st.tuples(step_ids, costs, costs), min_size=1, max_size=10, unique_by=lambda t: t[0]
        ),
    )
    def test_a_clean_run_leaks_nothing(
        self, run_id: str, steps: list[tuple[str, float, float]]
    ) -> None:
        ledger = CostLedger()
        for step_id, reserved, actual in steps:
            ledger.reserve(run_id, step_id, reserved)
            ledger.reconcile(run_id, step_id, actual)

        final = ledger.close_run(run_id)
        assert final.leaked_steps == 0
        assert final.reserved == 0.0


class TestNegativeCosts:
    @given(
        run_id=run_ids,
        step_id=step_ids,
        cost=st.floats(max_value=-0.000001, allow_nan=False, allow_infinity=False),
    )
    def test_negative_reserve_is_rejected(self, run_id: str, step_id: str, cost: float) -> None:
        ledger = CostLedger()
        with pytest.raises(LedgerInvariantError):
            ledger.reserve(run_id, step_id, cost)

    @given(
        run_id=run_ids,
        step_id=step_ids,
        cost=st.floats(max_value=-0.000001, allow_nan=False, allow_infinity=False),
    )
    def test_negative_reconcile_is_rejected(self, run_id: str, step_id: str, cost: float) -> None:
        ledger = CostLedger()
        ledger.reserve(run_id, step_id, 1.0)
        with pytest.raises(LedgerInvariantError):
            ledger.reconcile(run_id, step_id, cost)


class LedgerStateMachine(RuleBasedStateMachine):
    """Random operation sequences checked against a reference implementation.

    The ledger has to agree with a naive model that tracks the same facts in the
    most obvious possible way. A bug that survives this has to exist in both,
    which is a substantially harder mistake to make than an arithmetic slip.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ledger = CostLedger()
        # The model: open reservations, accumulated actual, and closed-ness.
        self.model_open: dict[str, dict[str, float]] = {}
        self.model_actual: dict[str, float] = {}
        self.model_closed: set[str] = set()
        self.model_leaked: dict[str, int] = {}

    @initialize()
    def setup(self) -> None:
        self.ledger = CostLedger()
        self.model_open = {}
        self.model_actual = {}
        self.model_closed = set()
        self.model_leaked = {}

    @rule(
        run_id=st.sampled_from(["r1", "r2"]),
        step_id=st.sampled_from(["s1", "s2", "s3"]),
        cost=costs,
    )
    def reserve(self, run_id: str, step_id: str, cost: float) -> None:
        if run_id in self.model_closed:
            # Closing is final: a run's reported cost must not change after.
            with pytest.raises(LedgerInvariantError):
                self.ledger.reserve(run_id, step_id, cost)
            return

        self.ledger.reserve(run_id, step_id, cost)
        self.model_open.setdefault(run_id, {})[step_id] = cost
        self.model_actual.setdefault(run_id, 0.0)

    @rule(
        run_id=st.sampled_from(["r1", "r2"]),
        step_id=st.sampled_from(["s1", "s2", "s3"]),
        cost=costs,
    )
    def reconcile(self, run_id: str, step_id: str, cost: float) -> None:
        open_steps = self.model_open.get(run_id, {})
        if run_id not in self.model_closed and step_id in open_steps:
            self.ledger.reconcile(run_id, step_id, cost)
            del open_steps[step_id]
            self.model_actual[run_id] = self.model_actual.get(run_id, 0.0) + cost
        else:
            # Either nothing is open for this step, or the run is closed. Both
            # are ordering violations the ledger must refuse rather than absorb.
            with pytest.raises(LedgerInvariantError):
                self.ledger.reconcile(run_id, step_id, cost)

    @precondition(lambda self: bool(self.model_open))
    @rule(run_id=st.sampled_from(["r1", "r2"]))
    def close(self, run_id: str) -> None:
        if run_id not in self.model_closed:
            self.model_leaked[run_id] = len(self.model_open.get(run_id, {}))
            self.model_closed.add(run_id)

        final = self.ledger.close_run(run_id)
        assert final.leaked_steps == self.model_leaked.get(run_id, 0)

    @invariant()
    def totals_match_the_model(self) -> None:
        for run_id in set(self.model_open) | set(self.model_actual):
            snapshot = self.ledger.snapshot(run_id)
            expected_reserved = sum(self.model_open.get(run_id, {}).values())
            expected_actual = self.model_actual.get(run_id, 0.0)

            assert snapshot.reserved == pytest.approx(expected_reserved)
            assert snapshot.actual == pytest.approx(expected_actual)
            assert snapshot.committed == pytest.approx(expected_reserved + expected_actual)
            assert snapshot.open_steps == len(self.model_open.get(run_id, {}))


TestLedgerStateMachine = LedgerStateMachine.TestCase


class TestConcurrency:
    """Steps run on thread pools in most frameworks (M2-2: 'concurrent runs')."""

    @settings(max_examples=20)  # deadline and too_slow come from the property profile
    @given(
        threads=st.integers(min_value=2, max_value=8),
        per_thread=st.integers(min_value=1, max_value=25),
    )
    def test_concurrent_reserve_reconcile_loses_no_cost(
        self, threads: int, per_thread: int
    ) -> None:
        # A lost update here would not raise; it would produce a total that is
        # quietly too low. That is the whole reason the ledger locks.
        ledger = CostLedger()
        run_id = "concurrent-run"

        def worker(worker_id: int) -> None:
            for i in range(per_thread):
                step_id = f"w{worker_id}-s{i}"
                ledger.reserve(run_id, step_id, 1.0)
                ledger.reconcile(run_id, step_id, 0.5)

        workers = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        snapshot = ledger.snapshot(run_id)
        assert snapshot.reconciled_steps == threads * per_thread
        assert snapshot.actual == pytest.approx(threads * per_thread * 0.5)
        assert snapshot.reserved == 0.0

    @settings(max_examples=20)  # deadline and too_slow come from the property profile
    @given(threads=st.integers(min_value=2, max_value=8))
    def test_concurrent_runs_stay_isolated(self, threads: int) -> None:
        ledger = CostLedger()

        def worker(worker_id: int) -> None:
            run_id = f"run-{worker_id}"
            for i in range(20):
                step_id = f"s{i}"
                ledger.reserve(run_id, step_id, 2.0)
                ledger.reconcile(run_id, step_id, 1.0)

        workers = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        for n in range(threads):
            snapshot = ledger.snapshot(f"run-{n}")
            assert snapshot.actual == pytest.approx(20.0)
            assert snapshot.reserved == 0.0
