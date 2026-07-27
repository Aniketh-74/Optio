"""Quality tier selection (M5-1).

ADR-003 makes this the latency and cost guard for the whole lane, so the tests
that matter are the ones proving the *expensive* path stays shut: off by
default, judge only when explicitly sampled, and no truthy value able to buy its
way into a paid model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest

from agentmeter.config import Config
from agentmeter.lanes.quality.sampling import SamplingDecision, Tier, decide


@dataclass
class FakeRun:
    """A run with a controllable sampling flag."""

    run_id: str = "run-1"
    budget: Any = None
    sampled: bool = False


class TestTheLaneIsOffByDefault:
    def test_default_config_scores_nothing(self) -> None:
        # ADR-003. A user who never opts in must pay nothing for this lane.
        assert decide(FakeRun(), Config()).tier is Tier.NONE

    def test_an_unsampled_run_still_gets_no_tier_when_disabled(self) -> None:
        assert decide(FakeRun(sampled=True), Config()).tier is Tier.NONE

    def test_disabling_beats_sampling(self) -> None:
        # `quality_lane=False` with a run somehow marked sampled must not score.
        decision = decide(FakeRun(sampled=True), Config(quality_lane=False))
        assert decision.tier is Tier.NONE
        assert "disabled" in decision.reason


class TestTiers:
    def test_enabled_but_unsampled_gets_the_heuristic(self) -> None:
        decision = decide(FakeRun(sampled=False), Config(quality_lane=True))
        assert decision.tier is Tier.HEURISTIC

    def test_enabled_and_sampled_gets_the_judge(self) -> None:
        decision = decide(FakeRun(sampled=True), Config(quality_lane=True))
        assert decision.tier is Tier.JUDGE

    def test_every_decision_carries_a_reason(self) -> None:
        # Goes to logs and self-metrics, so it must never be empty.
        for run, config in (
            (FakeRun(), Config()),
            (FakeRun(sampled=False), Config(quality_lane=True)),
            (FakeRun(sampled=True), Config(quality_lane=True)),
        ):
            assert decide(run, config).reason


class TestNothingBuysItsWayIntoThePaidPath:
    """The judge costs the user real money. Only an explicit True opens it."""

    @pytest.mark.parametrize(
        "sampled",
        [1, "yes", [1], object(), Mock(), 0.5],
        ids=["int", "str", "list", "object", "mock", "float"],
    )
    def test_a_merely_truthy_flag_does_not_reach_the_judge(self, sampled: object) -> None:
        # A Mock is truthy, and a test stub or a framework wrapper reaching this
        # code with one would otherwise silently start billing the user.
        run = FakeRun()
        run.sampled = sampled  # type: ignore[assignment]

        assert decide(run, Config(quality_lane=True)).tier is Tier.HEURISTIC

    def test_a_run_without_the_flag_degrades_to_the_heuristic(self) -> None:
        # RunLike (Section 3.1) does not carry `sampled`, so a minimal run
        # object is legitimate. It must degrade, not raise.
        class MinimalRun:
            run_id = "run-1"
            budget = None

        assert decide(MinimalRun(), Config(quality_lane=True)).tier is Tier.HEURISTIC


class TestTierProperties:
    def test_none_scores_nothing(self) -> None:
        assert Tier.NONE.scores is False
        assert Tier.NONE.uses_judge is False

    def test_heuristic_scores_without_the_judge(self) -> None:
        assert Tier.HEURISTIC.scores is True
        assert Tier.HEURISTIC.uses_judge is False

    def test_judge_does_both(self) -> None:
        assert Tier.JUDGE.scores is True
        assert Tier.JUDGE.uses_judge is True


class TestDecisionIsInert:
    def test_the_decision_is_frozen(self) -> None:
        decision = SamplingDecision(Tier.JUDGE, "because")
        with pytest.raises(AttributeError):
            decision.tier = Tier.NONE  # type: ignore[misc]

    def test_deciding_does_not_mutate_the_run(self) -> None:
        # The sampling decision is frozen at run construction (see RunContext);
        # this module reads it and must never re-roll or overwrite it.
        run = FakeRun(sampled=True)
        decide(run, Config(quality_lane=True))
        assert run.sampled is True
