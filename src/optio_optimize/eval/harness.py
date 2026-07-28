"""Runs eval cases against a stage instance and reports what failed.

See the package docstring for why these checks are model-free by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optio_optimize.eval.cases import (
        CacheBehaviorCase,
        DecisionBoundaryCase,
        FactPreservationCase,
    )
    from optio_optimize.stages.base import Stage, StageContext, StageResult


@dataclass(frozen=True, slots=True)
class EvalFailure:
    """One case that did not pass.

    Attributes:
        case: The failing case's name.
        reason: Human-readable explanation, safe to print -- never includes
            full prompt content, only the specific missing fact or the
            nature of the mismatch (§10's content-privacy rule extends here).
    """

    case: str
    reason: str


@dataclass(slots=True)
class EvalReport:
    """Accumulated results across every case run against one stage.

    Attributes:
        passed: Count of cases that passed.
        failures: Every case that did not, in the order it ran.
    """

    passed: int = 0
    failures: list[EvalFailure] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Cases run so far."""
        return self.passed + len(self.failures)

    @property
    def ok(self) -> bool:
        """Whether every case run so far passed.

        ``True`` on zero cases run is deliberate: an empty suite is a setup
        bug for the *caller* to catch (nothing was tested), not a reason for
        this property to lie about what it does know, which is "no failure
        has been observed".
        """
        return not self.failures

    def record(self, failure: EvalFailure | None) -> None:
        """Fold a single-check case's outcome (fact-preservation, decision-boundary)."""
        if failure is None:
            self.passed += 1
        else:
            self.failures.append(failure)

    def record_batch(self, failures: list[EvalFailure]) -> None:
        """Fold one :func:`check_cache_behavior` case's outcome.

        That check can return zero, one, or two failures for a single case,
        so "did this case pass" is "was the list empty", not "how many
        failures came back".
        """
        if failures:
            self.failures.extend(failures)
        else:
            self.passed += 1


def check_fact_preservation(
    stage: Stage, case: FactPreservationCase, ctx: StageContext
) -> EvalFailure | None:
    """Verify every required fact survives the stage's transformation.

    Args:
        stage: The stage under test.
        case: What to check.
        ctx: Context to run the stage with.

    Returns:
        A failure describing what went missing, or ``None`` if every fact
        survived.
    """
    result = stage.before(case.request, ctx)
    text = _visible_text(result).lower()
    missing = [fact for fact in case.required_facts if fact.lower() not in text]
    if missing:
        return EvalFailure(case.name, f"missing after transform: {missing}")
    return None


def check_cache_behavior(
    stage: Stage, case: CacheBehaviorCase, ctx: StageContext
) -> list[EvalFailure]:
    """Verify a cache-style stage hits near matches and refuses strangers.

    Args:
        stage: The stage under test. Populated by this call -- pass a fresh
            instance per case unless testing accumulated state deliberately.
        case: What to check.
        ctx: Context to run the stage with.

    Returns:
        Zero, one, or two failures (a false miss, a false hit, or both).
    """
    failures: list[EvalFailure] = []
    stage.before(case.stored_request, ctx)
    stage.after(case.stored_request, case.stored_response, ctx)

    near = stage.before(case.near_request, ctx)
    if not near.short_circuited:
        failures.append(EvalFailure(case.name, "a near-duplicate request did not hit the cache"))

    far = stage.before(case.far_request, ctx)
    if far.short_circuited:
        failures.append(EvalFailure(case.name, "an unrelated request incorrectly hit the cache"))

    return failures


def check_decision_boundary(
    stage: Stage, case: DecisionBoundaryCase, ctx: StageContext
) -> EvalFailure | None:
    """Verify a stage acts exactly when its own documented rule says it should.

    Args:
        stage: The stage under test.
        case: What to check.
        ctx: Context to run the stage with.

    Returns:
        A failure if the stage's decision didn't match ``case.should_act``,
        else ``None``.
    """
    result = stage.before(case.request, ctx)
    acted = bool(result.note)
    if acted != case.should_act:
        verb = "acted on" if acted else "declined"
        expected = "act on" if case.should_act else "decline"
        return EvalFailure(case.name, f"stage {verb} a request expected to {expected}")
    return None


def _visible_text(result: StageResult) -> str:
    """Everything a stage's transformation actually sends onward.

    A short-circuited result never reaches the provider, so what "the model
    would see" means, for it, is the response text served in its place.
    """
    if result.short_circuited and result.response is not None:
        return result.response.content
    return "\n".join(m.content for m in result.request.messages)
