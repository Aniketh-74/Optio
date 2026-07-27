"""The policy the demo runs against optio's signals.

This stands in for OPA, Cedar, or AGT. Those are the real targets -- the shipped
packs in ``policies/`` are the same rules in each engine's own language -- but
the demo evaluates them in Python so it runs with one command and no engine
install (ADR-006: the demo has to work on a fresh machine).

The rules are deliberately the same shape as the shipped packs, including the
part that is easy to get wrong:

**A missing signal means unknown, never zero.** optio omits a signal it
cannot compute rather than emitting a wrong number, so every rule checks
presence before comparing. Reading absence as zero would make a broken cost lane
look like a free run -- and would deny every run for anyone with no budget set.

Editing these thresholds and re-running the demo is the intended way to play
with it, which is why they are constants at the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from optio import semconv

#: Stop a run projected to cost more than this. Generous on purpose: the demo's
#: point is that the *behavior* signal catches the loop long before spend does,
#: which is the case a cost-only tool misses entirely.
MAX_PROJECTED_COST: Final = 0.95

#: Stop when headroom runs out.
MIN_BUDGET_REMAINING: Final = 0.02

#: Loop states worth stopping a run over. `repeating` is deliberately absent:
#: healthy agents repeat calls constantly (polling, paging, bounded retries), so
#: gating on it produces false positives -- our error becoming the user's
#: outage. See docs/behavior.md for the measured false-positive rate.
BLOCKING_LOOP_STATES: Final[frozenset[str]] = frozenset(
    {semconv.LOOP_STATE_LOOPING, semconv.LOOP_STATE_RETRY_STORM}
)


@dataclass(frozen=True, slots=True)
class Decision:
    """A policy decision.

    Attributes:
        denied: Whether the run should stop.
        reason: Human-readable rule that fired, when denied.
    """

    denied: bool
    reason: str | None = None

    @staticmethod
    def allow() -> Decision:
        """Return an allowing decision."""
        return Decision(denied=False)

    @staticmethod
    def deny(reason: str) -> Decision:
        """Return a denying decision.

        Args:
            reason: The rule that fired.
        """
        return Decision(denied=True, reason=reason)


def evaluate(signals: dict[str, object]) -> Decision:
    """Decide whether a run may continue, given its current signals.

    Args:
        signals: optio attributes read off the latest span. Signals that
            were not emitted are simply absent.

    Returns:
        The decision. Absent signals never produce a denial.
    """
    state = signals.get(semconv.RUN_LOOP_STATE)
    if isinstance(state, str) and state in BLOCKING_LOOP_STATES:
        repeats = signals.get(semconv.RUN_REPEAT_COUNT)
        detail = f" (repeat_count {repeats})" if isinstance(repeats, int) else ""
        return Decision.deny(f"loop_state == {state}{detail}")

    projected = signals.get(semconv.RUN_PROJECTED_COST)
    if isinstance(projected, (int, float)) and projected > MAX_PROJECTED_COST:
        return Decision.deny(f"projected_cost ${projected:.4f} > ${MAX_PROJECTED_COST:.2f}")

    remaining = signals.get(semconv.RUN_BUDGET_REMAINING)
    if isinstance(remaining, (int, float)) and remaining < MIN_BUDGET_REMAINING:
        return Decision.deny(f"budget_remaining ${remaining:.4f} exhausted")

    return Decision.allow()
