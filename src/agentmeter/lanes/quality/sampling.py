"""Quality tier selection (M5-1).

ADR-003 makes the quality lane tiered, sampled, opt-in, and off by default, and
every one of those words is a latency or cost control:

* **opt-in / off by default** -- cost and behavior deliver value alone. A user
  who never enables this pays nothing for it.
* **tiered** -- the cheap inline heuristic runs on every run of an enabled lane;
  the expensive LLM-judge runs only on sampled ones.
* **sampled** -- the judge costs real money on the user's own credentials and
  takes model latency to answer. Scoring every run would make agentmeter's
  quality signal more expensive than the agent it observes, which is the
  "who evals the evaluator" trap (R-TECH-5).

## Why the decision is frozen at run start

:class:`~agentmeter.runtime.run_context.RunContext` decides ``sampled`` once, at
construction, and this module reads that decision rather than re-rolling. A
re-roll at run end would let run start and run end disagree, producing quality
signals attached to runs that were never scored -- wrong data rather than
missing data, which is the failure class this project treats as worst.

This module answers the narrower question: *given* that a run is eligible, which
tier does it get?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentmeter.config import Config
    from agentmeter.lanes.base import RunLike


class Tier(Enum):
    """How much evaluation effort a run receives.

    Attributes:
        NONE: No quality scoring. The lane is off, or the run is not eligible.
        HEURISTIC: Inline checks only -- deterministic, no network, near-zero
            cost. Every run of an enabled lane gets at least this.
        JUDGE: Heuristic *plus* the async LLM-judge. Sampled runs only.
    """

    NONE = "none"
    HEURISTIC = "heuristic"
    JUDGE = "judge"

    @property
    def scores(self) -> bool:
        """Whether this tier produces any quality signal at all."""
        return self is not Tier.NONE

    @property
    def uses_judge(self) -> bool:
        """Whether this tier invokes the LLM-judge."""
        return self is Tier.JUDGE


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    """The tier a run was assigned, and why.

    Attributes:
        tier: The assigned tier.
        reason: Short explanation, for logs and for the self-observability
            metrics. Never contains run content.
    """

    tier: Tier
    reason: str


def decide(run: RunLike, config: Config) -> SamplingDecision:
    """Choose the evaluation tier for a run.

    Args:
        run: The run being scored. Only its ``sampled`` flag is read, and only
            when present -- a run object without one is treated as unsampled
            rather than as an error, so a caller using a minimal run stub
            degrades to the heuristic tier instead of failing.
        config: Active configuration.

    Returns:
        The decision. Never raises: an unexpected value falls back to the
        cheaper tier, because the expensive path is the one that can hurt.
    """
    if not config.quality_lane:
        return SamplingDecision(Tier.NONE, "quality lane disabled")

    # Read through `getattr`: `RunLike` (Section 3.1) deliberately exposes only
    # `run_id` and `budget`, and widening that protocol to carry a
    # quality-specific field would push a lane's concern into the shared
    # contract every other lane is typed against.
    sampled = getattr(run, "sampled", False)
    if sampled is not True:
        # Explicitly `is not True` rather than falsy: a stub returning a Mock,
        # a string, or anything else truthy must not buy a run into the paid
        # LLM-judge path by accident.
        return SamplingDecision(Tier.HEURISTIC, "run not sampled for judge")

    return SamplingDecision(Tier.JUDGE, "run sampled for judge")
