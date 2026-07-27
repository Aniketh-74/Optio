"""Ledger memory lifecycle: eviction, and finality that outlives it.

These exist because stress testing found an unbounded leak that every unit test
missed. Each test here builds a fresh ledger, so no test could see state
accumulating *across* runs -- the leak only appears when one ledger handles many
runs, which is exactly how a long-lived agent process uses it.

Measured before the fix: ~368 bytes retained per run, forever. At one run per
second that is ~30 MB/day, growing until the process dies.

The fix creates a tension worth understanding. Eviction releases the state that
records "this run is closed", which would let a straggling callback reserve
against the id again and silently start a *new* total under a run whose cost was
already reported (ADR-010). A bounded FIFO of closed ids resolves it: finality
survives eviction, memory stays capped.
"""

from __future__ import annotations

import pytest

from optio import semconv
from optio.errors import LedgerInvariantError
from optio.lanes.cost.ledger import CostLedger


class TestEvictionReleasesState:
    def test_many_closed_and_evicted_runs_retain_nothing(self) -> None:
        ledger = CostLedger()
        for n in range(2000):
            run_id = f"run-{n}"
            ledger.reserve(run_id, "s0", 1.0)
            ledger.reconcile(run_id, "s0", 1.0)
            ledger.close_run(run_id)
            ledger.evict(run_id)

        assert ledger.run_count() == 0

    def test_the_closed_id_window_is_bounded(self) -> None:
        # The other half of the leak fix: remembering closed ids forever would
        # just move the unbounded growth somewhere else.
        ledger = CostLedger(closed_memory=64)
        for n in range(5000):
            ledger.close_run(f"run-{n}")
            ledger.evict(f"run-{n}")

        assert len(ledger._recently_closed) == 64

    def test_an_active_run_is_not_evicted(self) -> None:
        ledger = CostLedger()
        ledger.reserve("live", "s0", 1.0)

        assert ledger.run_count() == 1


class TestFinalityOutlivesEviction:
    """ADR-010 has to survive the memory fix, not be traded away for it."""

    def test_reserve_on_an_evicted_closed_run_is_rejected(self) -> None:
        ledger = CostLedger()
        ledger.reserve("run", "s0", 1.0)
        ledger.reconcile("run", "s0", 1.0)
        ledger.close_run("run")
        ledger.evict("run")

        with pytest.raises(LedgerInvariantError, match="closed run"):
            ledger.reserve("run", "s1", 5.0)

    def test_reconcile_on_an_evicted_closed_run_is_rejected(self) -> None:
        ledger = CostLedger()
        ledger.reserve("run", "s0", 1.0)
        ledger.close_run("run")
        ledger.evict("run")

        with pytest.raises(LedgerInvariantError, match="closed run"):
            ledger.reconcile("run", "s0", 1.0)

    def test_an_evicted_run_cannot_accumulate_a_second_total(self) -> None:
        # The specific regression: without the closed-id window this silently
        # succeeded and reported 5.0 as the run's cost, replacing the 1.0 that
        # had already been published.
        ledger = CostLedger()
        ledger.reserve("run", "s0", 1.0)
        ledger.reconcile("run", "s0", 1.0)
        ledger.close_run("run")
        ledger.evict("run")

        with pytest.raises(LedgerInvariantError):
            ledger.reserve("run", "s1", 5.0)
        assert ledger.snapshot("run").actual == 0.0

    def test_a_fresh_run_id_is_unaffected(self) -> None:
        ledger = CostLedger()
        ledger.close_run("old")
        ledger.evict("old")

        ledger.reserve("brand-new", "s0", 1.0)
        ledger.reconcile("brand-new", "s0", 1.0)
        assert ledger.snapshot("brand-new").actual == pytest.approx(1.0)

    def test_beyond_the_window_a_run_id_becomes_reusable(self) -> None:
        # Documented consequence, not an accident: past the window a late
        # arrival is treated as a fresh run, the same as after a process
        # restart. Acceptable because nothing that old is still in flight.
        ledger = CostLedger(closed_memory=2)
        for n in range(5):
            ledger.close_run(f"run-{n}")
            ledger.evict(f"run-{n}")

        ledger.reserve("run-0", "s0", 1.0)  # forgotten; treated as new
        with pytest.raises(LedgerInvariantError):
            ledger.reserve("run-4", "s0", 1.0)  # still remembered


class TestCostLaneReleasesState:
    def test_the_lane_evicts_at_run_end(self) -> None:
        from optio.config import default_config
        from optio.lanes.cost.lane import CostLane

        class _Run:
            run_id = "lane-run"
            budget = None

        lane = CostLane(default_config())
        run = _Run()
        lane.ledger.reserve(run.run_id, "s0", 1.0)
        lane.ledger.reconcile(run.run_id, "s0", 1.0)

        lane.on_run_end(run)

        assert lane.ledger.run_count() == 0

    def test_a_repeat_run_end_emits_nothing(self) -> None:
        # A real bug found by stress testing, and the nastier half of the leak
        # fix. Run end can fire more than once (M1-2); after eviction the second
        # call saw an all-zero snapshot and emitted budget_remaining = the FULL
        # budget, overwriting the correct value on the run span. A policy
        # reading it would conclude the run spent nothing.
        from optio.config import BudgetPolicy, default_config
        from optio.lanes.cost.lane import CostLane

        class _Run:
            run_id = "lane-run"
            budget = BudgetPolicy(limit_usd=10.0)

        lane = CostLane(default_config())
        run = _Run()
        lane.ledger.reserve(run.run_id, "s0", 1.0)
        lane.ledger.reconcile(run.run_id, "s0", 1.0)

        first = {s.name: s.value for s in lane.on_run_end(run)}
        assert first[semconv.RUN_ACTUAL_COST] == pytest.approx(1.0)
        assert first[semconv.RUN_BUDGET_REMAINING] == pytest.approx(9.0)

        # The evidence is gone; re-deriving signals from its absence would
        # invent them.
        assert lane.on_run_end(run) == []
        assert lane.on_run_end(run) == []

    def test_a_repeat_run_end_on_an_unpriced_run_also_emits_nothing(self) -> None:
        from optio.config import BudgetPolicy, default_config
        from optio.lanes.cost.lane import CostLane

        class _Run:
            run_id = "unpriced"
            budget = BudgetPolicy(limit_usd=10.0)

        lane = CostLane(default_config())
        run = _Run()
        lane.ledger.reserve(run.run_id, "s0", 0.0)  # unpriceable step

        lane.on_run_end(run)
        assert lane.on_run_end(run) == []


class TestTheLeakWarningReachesTheOperator:
    """The warning is the only thing distinguishing an estimate from a measurement.

    ``leaked_steps`` as a number is covered by the property suite. The WARNING
    is separate and was asserted nowhere -- mutation testing found it by
    inverting ``if leaked:`` and watching every test still pass.

    It matters more than a log line usually would. When a run ends with open
    reservations, the reported cost is the *reserved worst case* for those
    steps rather than measured spend, and this warning is the only place that
    is said. Silently reporting an estimate as though it were a measurement is
    the silent-wrongness failure R-TECH-1 treats as the worst possible bug.
    """

    def test_an_unreconciled_reservation_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        ledger = CostLedger()
        ledger.reserve("run-leak", "step-0", 0.75)  # never reconciled

        with caplog.at_level("WARNING", logger="optio"):
            snapshot = ledger.close_run("run-leak")

        assert snapshot.leaked_steps == 1
        assert "unreconciled" in caplog.text
        assert "run-leak" in caplog.text

    def test_a_clean_run_stays_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # The other half: a warning on every healthy run would be noise, and
        # noise is how a real warning gets filtered out.
        ledger = CostLedger()
        ledger.reserve("run-clean", "step-0", 0.75)
        ledger.reconcile("run-clean", "step-0", 0.50)

        with caplog.at_level("WARNING", logger="optio"):
            snapshot = ledger.close_run("run-clean")

        assert snapshot.leaked_steps == 0
        assert "unreconciled" not in caplog.text

    def test_the_warning_counts_every_leaked_step(self, caplog: pytest.LogCaptureFixture) -> None:
        ledger = CostLedger()
        for i in range(3):
            ledger.reserve("run-many", f"step-{i}", 1.0)
        ledger.reconcile("run-many", "step-0", 1.0)

        with caplog.at_level("WARNING", logger="optio"):
            snapshot = ledger.close_run("run-many")

        assert snapshot.leaked_steps == 2
        assert "2 unreconciled" in caplog.text
