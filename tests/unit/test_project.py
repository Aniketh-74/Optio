"""Cost projection (M2-3).

The acceptance criterion is *projection monotonic w.r.t. remaining steps, with a
documented method*. Monotonicity is what makes the signal safe to gate on: a
policy that sees the projection fall as a run progresses cannot reason about it.

The omission rules matter equally. A projection that cannot be computed is
absent, never zero -- `docs/signals.md` is explicit that defaulting a missing
``projected_cost`` to 0 would silently allow the exact runs this signal exists
to catch.
"""

from __future__ import annotations

import pytest

from optio.config import BudgetPolicy
from optio.lanes.cost.ledger import LedgerSnapshot
from optio.lanes.cost.project import (
    budget_remaining,
    cost_per_successful_task,
    per_step_estimate,
    project_cost,
    steps_taken,
)


def _snapshot(
    reserved: float = 0.0,
    actual: float = 0.0,
    open_steps: int = 0,
    reconciled_steps: int = 0,
) -> LedgerSnapshot:
    """Build a snapshot with a consistent ``committed``."""
    return LedgerSnapshot(
        reserved=reserved,
        actual=actual,
        committed=reserved + actual,
        open_steps=open_steps,
        reconciled_steps=reconciled_steps,
        leaked_steps=0,
    )


class TestPerStepEstimate:
    def test_prefers_reconciled_actuals(self) -> None:
        # Real observed spend beats a worst-case reservation.
        snapshot = _snapshot(actual=6.0, reconciled_steps=3)
        assert per_step_estimate(snapshot) == pytest.approx(2.0)

    def test_falls_back_to_open_reservations(self) -> None:
        # Early in a run nothing has reconciled, so the estimate comes from the
        # reservations -- themselves worst-case, making the projection
        # conservative exactly when it is least certain.
        snapshot = _snapshot(reserved=9.0, open_steps=3)
        assert per_step_estimate(snapshot) == pytest.approx(3.0)

    def test_reconciled_wins_over_open(self) -> None:
        snapshot = _snapshot(reserved=100.0, actual=2.0, open_steps=1, reconciled_steps=2)
        assert per_step_estimate(snapshot) == pytest.approx(1.0)

    def test_no_evidence_yields_none(self) -> None:
        assert per_step_estimate(_snapshot()) is None

    def test_open_steps_reserved_at_zero_are_not_evidence(self) -> None:
        # Regression. The cost lane marks a step it could not price by
        # reserving 0.0 and leaving it open. Dividing that gives 0.0, which
        # would be published as "this run costs nothing per step" for a run
        # making real billable calls. Absence of a price is not a price of zero.
        snapshot = _snapshot(reserved=0.0, open_steps=10)
        assert per_step_estimate(snapshot) is None


class TestUnpricedRunsFabricateNothing:
    """Regression: unknown models must produce absent signals, not zeroes.

    The pricing table is static and hand-maintained, so "a model we have never
    heard of" is the ordinary state of any deployed version of this library --
    not a rare edge case. Every signal derived from an unpriceable run has to be
    absent, because the alternative is a policy engine confidently reading
    "nothing spent" while the agent burns money (R-TECH-1).
    """

    #: A run whose every step hit an unknown model, as the lane records it.
    UNPRICED = _snapshot(reserved=0.0, actual=0.0, open_steps=10, reconciled_steps=0)

    #: A run that has genuinely not started.
    IDLE = _snapshot()

    def test_budget_remaining_is_absent_when_nothing_could_be_priced(self) -> None:
        assert budget_remaining(self.UNPRICED, BudgetPolicy(limit_usd=5.0)) is None

    def test_projected_cost_is_absent_when_nothing_could_be_priced(self) -> None:
        budget = BudgetPolicy(limit_usd=5.0, max_steps=100)
        assert project_cost(self.UNPRICED, budget) is None

    def test_an_idle_run_still_reports_its_full_budget(self) -> None:
        # The distinction that makes the guard non-trivial: committed == 0 is
        # honest here and dishonest above, and the two must not be conflated.
        assert budget_remaining(self.IDLE, BudgetPolicy(limit_usd=5.0)) == pytest.approx(5.0)

    def test_steps_in_flight_on_a_priced_model_still_report(self) -> None:
        # Nothing reconciled yet, same as the unpriced case -- but the
        # reservations are real money already claimed. A guard keyed on
        # reconciled_steps alone would wrongly suppress this.
        in_flight = _snapshot(reserved=1.20, open_steps=3)
        assert budget_remaining(in_flight, BudgetPolicy(limit_usd=5.0)) == pytest.approx(3.80)

    def test_a_genuinely_free_model_still_reports(self) -> None:
        # 0.0 reconciled is a price we observed, not a price we are missing.
        # Suppressing it would make free models indistinguishable from
        # unpriceable ones, losing the distinction this guard exists to draw.
        free = _snapshot(actual=0.0, reconciled_steps=3)
        assert budget_remaining(free, BudgetPolicy(limit_usd=5.0)) == pytest.approx(5.0)

    def test_one_priced_step_is_enough_to_report(self) -> None:
        # Partial evidence is still evidence; the undercount is surfaced
        # separately as a leak warning at run end.
        mixed = _snapshot(actual=1.25, open_steps=5, reconciled_steps=5)
        assert budget_remaining(mixed, BudgetPolicy(limit_usd=5.0)) == pytest.approx(3.75)

    def test_an_unpriced_run_is_indistinguishable_from_idle_only_in_the_ledger(self) -> None:
        # Both have committed == 0. The whole fix rests on open_steps being the
        # thing that tells them apart, so assert that premise directly rather
        # than leaving it implicit in the cases above.
        assert self.UNPRICED.committed == self.IDLE.committed == 0.0
        assert self.UNPRICED.open_steps != self.IDLE.open_steps


class TestStepsTaken:
    def test_counts_both_open_and_reconciled(self) -> None:
        # An open step has begun, so its budget is already committed.
        assert steps_taken(_snapshot(open_steps=2, reconciled_steps=3)) == 5

    def test_empty_run_has_taken_no_steps(self) -> None:
        assert steps_taken(_snapshot()) == 0


class TestProjection:
    def test_extrapolates_over_remaining_steps(self) -> None:
        snapshot = _snapshot(actual=2.0, reconciled_steps=1)
        budget = BudgetPolicy(limit_usd=100.0, max_steps=5)

        # 2.0 committed + 4 remaining steps at 2.0 each.
        assert project_cost(snapshot, budget) == pytest.approx(10.0)

    def test_is_monotonic_as_the_run_progresses(self) -> None:
        # The acceptance criterion. At a steady per-step cost the projection
        # must not drift downward as steps complete -- a policy cannot gate on
        # a number that moves against the spend.
        budget = BudgetPolicy(limit_usd=100.0, max_steps=10)
        projections = [
            project_cost(_snapshot(actual=2.0 * n, reconciled_steps=n), budget)
            for n in range(1, 11)
        ]

        assert all(p is not None for p in projections)
        assert projections == sorted(projections)  # type: ignore[type-var]

    def test_converges_on_actual_at_the_step_ceiling(self) -> None:
        # With no steps left the projection is just what was spent.
        snapshot = _snapshot(actual=20.0, reconciled_steps=10)
        budget = BudgetPolicy(limit_usd=100.0, max_steps=10)

        assert project_cost(snapshot, budget) == pytest.approx(20.0)

    def test_overrunning_the_step_ceiling_does_not_go_backwards(self) -> None:
        # remaining is floored at zero; a negative would subtract spend that
        # already happened.
        snapshot = _snapshot(actual=30.0, reconciled_steps=15)
        budget = BudgetPolicy(limit_usd=100.0, max_steps=10)

        assert project_cost(snapshot, budget) == pytest.approx(30.0)

    def test_is_omitted_without_a_budget(self) -> None:
        assert project_cost(_snapshot(actual=2.0, reconciled_steps=1), None) is None

    def test_is_omitted_without_a_step_ceiling(self) -> None:
        # No horizon means no finite worst case; inventing one would make the
        # number arbitrary.
        budget = BudgetPolicy(limit_usd=100.0)
        assert project_cost(_snapshot(actual=2.0, reconciled_steps=1), budget) is None

    def test_is_omitted_without_cost_evidence(self) -> None:
        budget = BudgetPolicy(limit_usd=100.0, max_steps=5)
        assert project_cost(_snapshot(), budget) is None


class TestBudgetRemaining:
    def test_subtracts_committed_not_just_actual(self) -> None:
        # A step in flight has already claimed its budget. Reporting that money
        # as available would let a policy authorise spending it twice.
        snapshot = _snapshot(reserved=3.0, actual=2.0, open_steps=1, reconciled_steps=1)
        budget = BudgetPolicy(limit_usd=10.0)

        assert budget_remaining(snapshot, budget) == pytest.approx(5.0)

    def test_goes_negative_when_overspent(self) -> None:
        # Clamping would erase the most actionable fact: by how much.
        snapshot = _snapshot(actual=15.0, reconciled_steps=3)
        budget = BudgetPolicy(limit_usd=10.0)

        assert budget_remaining(snapshot, budget) == pytest.approx(-5.0)

    def test_is_omitted_without_a_budget(self) -> None:
        # Absent, not infinite: "no budget" means nobody told us the limit.
        assert budget_remaining(_snapshot(actual=5.0), None) is None

    def test_full_budget_remains_before_any_spend(self) -> None:
        assert budget_remaining(_snapshot(), BudgetPolicy(limit_usd=10.0)) == (pytest.approx(10.0))


class TestCostPerSuccessfulTask:
    def test_divides_actual_by_successes(self) -> None:
        snapshot = _snapshot(actual=10.0, reconciled_steps=5)
        assert cost_per_successful_task(snapshot, 4) == pytest.approx(2.5)

    def test_zero_successes_yields_none(self) -> None:
        # A run that succeeded at nothing has a cost and a failure, both already
        # reported separately. Dividing would emit infinity.
        assert cost_per_successful_task(_snapshot(actual=10.0), 0) is None

    def test_negative_successes_yield_none(self) -> None:
        assert cost_per_successful_task(_snapshot(actual=10.0), -1) is None

    def test_a_free_successful_run_reports_zero(self) -> None:
        # A genuine zero, distinct from unknown.
        assert cost_per_successful_task(_snapshot(actual=0.0), 2) == 0.0
