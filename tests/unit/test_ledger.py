"""Cost ledger semantics (M2-2).

The property tests in ``tests/property/test_ledger_invariant.py`` cover the
R-TECH-1 invariant over random interleavings. These cover the specific
behaviours a reader needs stated explicitly -- particularly the closed-run
rules, which the state machine forced into existence by finding two bugs:

* ``close_run`` left a stale ``leaked_steps`` after the step was reconciled.
* Closing a run with no state left it re-openable, so a late reserve could
  accumulate cost against an already-reported run.
"""

from __future__ import annotations

import pytest

from agentmeter.errors import LedgerInvariantError
from agentmeter.lanes.cost.ledger import CostLedger


@pytest.fixture
def ledger() -> CostLedger:
    """A fresh ledger."""
    return CostLedger()


class TestBasicAccounting:
    def test_empty_run_is_all_zero(self, ledger: CostLedger) -> None:
        # An unknown run and a run that spent nothing are the same to a
        # consumer, so both read as zero rather than raising.
        snapshot = ledger.snapshot("nobody")
        assert snapshot.reserved == 0.0
        assert snapshot.actual == 0.0
        assert snapshot.committed == 0.0

    def test_committed_is_reserved_plus_actual(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        ledger.reserve("run", "s2", 2.00)
        ledger.reconcile("run", "s1", 0.75)

        snapshot = ledger.snapshot("run")
        assert snapshot.reserved == pytest.approx(2.00)
        assert snapshot.actual == pytest.approx(0.75)
        assert snapshot.committed == pytest.approx(2.75)

    def test_reserving_zero_is_allowed(self, ledger: CostLedger) -> None:
        # A free step (cached completion, a tool call with no tokens) is real.
        ledger.reserve("run", "s1", 0.0)
        assert ledger.snapshot("run").open_steps == 1


class TestClosedRunIsFinal:
    """Once a run's cost is reported, it must not change (found by the model)."""

    def test_closing_reports_leaks_once(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.50)

        first = ledger.close_run("run")
        assert first.leaked_steps == 1
        # The reservation is kept, not discarded: dropping it would make the run
        # look cheaper than the evidence supports.
        assert first.reserved == pytest.approx(1.50)

    def test_closing_twice_returns_the_same_snapshot(self, ledger: CostLedger) -> None:
        # Run end can fire more than once (M1-2), so this must be safe.
        ledger.reserve("run", "s1", 1.50)
        first = ledger.close_run("run")
        second = ledger.close_run("run")

        assert first == second

    def test_a_clean_run_reports_no_leak(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.50)
        ledger.reconcile("run", "s1", 1.20)

        final = ledger.close_run("run")
        assert final.leaked_steps == 0
        assert final.actual == pytest.approx(1.20)

    def test_reconcile_after_close_is_rejected(self, ledger: CostLedger) -> None:
        # A straggling callback must not silently change a number a policy may
        # already have acted on.
        ledger.reserve("run", "s1", 1.50)
        ledger.close_run("run")

        with pytest.raises(LedgerInvariantError, match="closed run"):
            ledger.reconcile("run", "s1", 1.20)

    def test_reserve_after_close_is_rejected(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.50)
        ledger.close_run("run")

        with pytest.raises(LedgerInvariantError, match="closed run"):
            ledger.reserve("run", "s2", 1.00)

    def test_closing_an_unknown_run_still_closes_it(self, ledger: CostLedger) -> None:
        # Returning early here left the run re-openable, so cost could
        # accumulate against a run that had already been reported.
        ledger.close_run("never-seen")

        with pytest.raises(LedgerInvariantError, match="closed run"):
            ledger.reserve("never-seen", "s1", 1.00)


class TestRetries:
    def test_re_reserving_a_step_replaces_the_reservation(self, ledger: CostLedger) -> None:
        # Frameworks retry steps and reuse their ids.
        ledger.reserve("run", "s1", 1.00)
        ledger.reserve("run", "s1", 3.00)

        snapshot = ledger.snapshot("run")
        assert snapshot.reserved == pytest.approx(3.00)
        assert snapshot.open_steps == 1

    def test_a_retried_step_reconciles_once(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        ledger.reserve("run", "s1", 3.00)
        ledger.reconcile("run", "s1", 2.50)

        snapshot = ledger.snapshot("run")
        assert snapshot.actual == pytest.approx(2.50)
        assert snapshot.reconciled_steps == 1
        assert snapshot.reserved == 0.0


class TestEviction:
    def test_evict_removes_run_state(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        assert ledger.run_count() == 1

        ledger.evict("run")
        assert ledger.run_count() == 0

    def test_evict_is_idempotent(self, ledger: CostLedger) -> None:
        ledger.evict("never-seen")
        ledger.evict("never-seen")

    def test_evicted_run_reads_as_zero(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        ledger.reconcile("run", "s1", 0.90)
        ledger.evict("run")

        assert ledger.snapshot("run").actual == 0.0

    def test_repr_reports_tracked_runs(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        assert "runs=1" in repr(ledger)


class TestRejections:
    def test_negative_reserve_is_rejected(self, ledger: CostLedger) -> None:
        with pytest.raises(LedgerInvariantError, match="negative"):
            ledger.reserve("run", "s1", -1.00)

    def test_negative_reconcile_is_rejected(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        with pytest.raises(LedgerInvariantError, match="negative"):
            ledger.reconcile("run", "s1", -1.00)

    def test_reconcile_for_unknown_run_is_rejected(self, ledger: CostLedger) -> None:
        with pytest.raises(LedgerInvariantError, match="unknown run"):
            ledger.reconcile("nobody", "s1", 1.00)

    def test_reconcile_for_unreserved_step_is_rejected(self, ledger: CostLedger) -> None:
        ledger.reserve("run", "s1", 1.00)
        with pytest.raises(LedgerInvariantError, match="no open reservation"):
            ledger.reconcile("run", "s2", 1.00)
