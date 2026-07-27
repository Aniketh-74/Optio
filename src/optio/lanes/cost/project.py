"""Worst-case run cost projection (M2-3).

``projected_cost`` is what makes **pre-spend** gating possible. A policy reading
only `actual_cost` learns a run was too expensive after the money is gone;
reading a projection, it can act while the run still has steps left. That is the
economic argument for this library, so the number has to be defensible.

**Method** (documented because a projection nobody can reason about is a
projection nobody should gate on):

    projected = committed + remaining_steps * per_step_estimate

where:

* ``committed`` = reconciled actual + open reservations. Money already spent or
  already promised.
* ``remaining_steps`` = ``max_steps - steps_taken``, floored at zero. Absent a
  step ceiling there is no bound, so no projection is emitted.
* ``per_step_estimate`` = the mean reconciled cost so far, falling back to the
  mean open reservation before anything has reconciled.

**Deliberately worst-case, not expected-case.** A projection that
under-estimates lets exactly the runs that should have been gated slip through,
which is the failure that costs the user money. Over-estimating merely gates a
run early -- recoverable, and visible.

**No projection is emitted when it cannot be computed.** No step ceiling, or no
evidence of what a step costs, means the attribute is omitted rather than
defaulted. `docs/signals.md` is explicit that a missing value means *unknown*:
defaulting to zero would silently permit the exact runs this signal exists to
catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optio.config import BudgetPolicy
    from optio.lanes.cost.ledger import LedgerSnapshot


def per_step_estimate(snapshot: LedgerSnapshot) -> float | None:
    """Estimate what one step costs, from evidence in the ledger.

    Prefers reconciled actuals -- real observed spend. Falls back to open
    reservations, which are themselves worst-case estimates, so early in a run
    the projection is conservative. That bias is intentional: the first steps
    are exactly when a projection is least certain and most useful.

    Args:
        snapshot: Current ledger state for the run.

    Returns:
        Mean cost per step in USD, or ``None`` when the run has no evidence yet.
    """
    if snapshot.reconciled_steps > 0:
        return snapshot.actual / snapshot.reconciled_steps

    if snapshot.open_steps > 0:
        if snapshot.reserved <= 0.0:
            # Open steps reserved at zero are the cost lane's marker for a step
            # it could not price (see CostLane.process_span). Dividing gives
            # 0.0, which would be published as a confident "this run costs
            # nothing per step" for a run making real, billable calls. The
            # arithmetic is right and the meaning is wrong: absence of a price
            # is not a price of zero (docs/signals.md).
            return None
        return snapshot.reserved / snapshot.open_steps

    return None


def steps_taken(snapshot: LedgerSnapshot) -> int:
    """Return how many steps the run has started.

    Open reservations count: a step that has begun is a step the budget has
    already been committed against.

    Args:
        snapshot: Current ledger state for the run.

    Returns:
        Number of steps started.
    """
    return snapshot.reconciled_steps + snapshot.open_steps


def project_cost(snapshot: LedgerSnapshot, budget: BudgetPolicy | None) -> float | None:
    """Project the worst-case total cost of a run.

    Args:
        snapshot: Current ledger state for the run.
        budget: The run's budget policy. ``max_steps`` supplies the horizon.

    Returns:
        Projected total cost in USD, or ``None`` when it cannot be computed --
        no budget policy, no ``max_steps``, or no evidence of per-step cost.
        ``None`` means *unknown* and must reach the consumer as an absent
        attribute, never as a zero.
    """
    if budget is None or budget.max_steps is None:
        # Without a step ceiling the run could take any number of further
        # steps, so no finite worst case exists. Inventing a horizon would make
        # the number arbitrary.
        return None

    estimate = per_step_estimate(snapshot)
    if estimate is None:
        return None

    remaining = max(0, budget.max_steps - steps_taken(snapshot))
    return snapshot.committed + remaining * estimate


def budget_remaining(snapshot: LedgerSnapshot, budget: BudgetPolicy | None) -> float | None:
    """Return how much of the budget is left.

    Computed against **committed** cost -- reconciled actual plus open
    reservations -- rather than actual alone. A step that is in flight has
    already claimed its budget; reporting that money as still available would
    let a policy authorise spending it twice.

    Can go negative, and is reported that way. Clamping at zero would erase the
    single most actionable fact: by how much the run went over.

    Absent, too, when the run has taken steps but none of them could be priced.
    ``budget - 0`` is arithmetically correct there and semantically a lie: it
    reports the full budget as available for a run that has been spending money
    the whole time, and a rule like ``budget_remaining < 0.50 -> deny`` would
    never fire. Since the pricing table is static and hand-maintained, "a model
    we have never heard of" is the *ordinary* state for any deployed version of
    this library rather than an edge case, which is what makes the wrong answer
    dangerous rather than merely imprecise (R-TECH-1).

    Args:
        snapshot: Current ledger state for the run.
        budget: The run's budget policy.

    Returns:
        Remaining budget in USD, or ``None`` when no budget was supplied or the
        run's spend is unknown. Absent rather than infinite, because "no budget"
        is not "unlimited money" -- it means nobody told us the limit.
    """
    if budget is None:
        return None
    if not _has_cost_evidence(snapshot):
        return None
    return budget.limit_usd - snapshot.committed


def _has_cost_evidence(snapshot: LedgerSnapshot) -> bool:
    """Whether the ledger knows enough to talk about this run's spend.

    Four states have to be told apart, and only the third is a problem:

    * **Nothing happened yet.** No steps at all. ``committed == 0`` is the
      truth and the full budget really is available.
    * **Steps in flight on a known model.** Nothing reconciled, but the
      reservations are real money already claimed. Reportable.
    * **Steps happened, none could be priced.** ``committed == 0`` because the
      cost is *unknown*, not because it is zero. Reporting a number here
      fabricates evidence.
    * **A genuinely free model.** Reconciled at 0.0. That is a measured price,
      not a missing one, and must stay reportable -- suppressing it would make
      free models indistinguishable from unpriceable ones and lose the very
      distinction this function exists to preserve.

    The test is therefore "did anything get priced", not "did anything get
    reconciled": reconciliation count alone would suppress a run with real
    reservations still in flight.

    Args:
        snapshot: Current ledger state for the run.

    Returns:
        ``True`` when a cost figure would mean something.
    """
    if snapshot.reconciled_steps > 0:
        # Includes the free-model case: 0.0 reconciled is a price we observed.
        return True
    if snapshot.reserved > 0.0:
        # In flight on a priced model: worst case is known and already claimed.
        return True
    # Nothing reconciled and nothing reserved. Honest only if nothing was
    # attempted; otherwise every step so far was unpriceable.
    return snapshot.open_steps == 0


def cost_per_successful_task(snapshot: LedgerSnapshot, successes: int) -> float | None:
    """Return cost divided by successful outcomes.

    The headline signal: what this agent costs to actually get work done, as
    opposed to what it costs to run.

    Args:
        snapshot: Current ledger state for the run.
        successes: Number of successful tasks in the run.

    Returns:
        Cost per success in USD, or ``None`` when there were no successes.
        Division by zero would yield infinity, and a run that succeeded at
        nothing has no meaningful cost-per-success -- it has a cost and a
        failure, which are already reported separately.
    """
    if successes <= 0:
        return None
    return snapshot.actual / successes
