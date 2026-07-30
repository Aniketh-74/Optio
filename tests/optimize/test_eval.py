"""The ADR-013 rule 3 quality gate: an ordinary, CI-blocking pytest module.

No separate runner needed -- this project's CI gate for property tests,
fail-inject and contract tests is already "pytest failed", and this suite
works the same way. See ``src/optio_optimize/eval/__init__.py`` for what
these checks can and cannot validate.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.eval.cases import CacheBehaviorCase, DecisionBoundaryCase, FactPreservationCase
from optio_optimize.eval.harness import (
    EvalReport,
    check_cache_behavior,
    check_decision_boundary,
    check_fact_preservation,
)
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.compress import CompressPromptStage
from optio_optimize.stages.output import (
    MIN_OBSERVATIONS,
    MIN_THINKING_BUDGET,
    ReasoningBudgetStage,
)
from optio_optimize.stages.routing import RouteModelsStage
from optio_optimize.stages.semantic_cache import SemanticCacheStage
from optio_optimize.stages.summarize import SummarizeHistoryStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx(**config_overrides: object) -> StageContext:
    return StageContext(config=OptimizeConfig(**config_overrides), counter=HeuristicCounter())  # type: ignore[arg-type]


def _request(content: str, **overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": "gpt-4o",
        "messages": (Message(role="user", content=content),),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compress_prompt: fact preservation
# ---------------------------------------------------------------------------

COMPRESS_CASES = (
    FactPreservationCase(
        name="distinct_facts_in_adjacent_sentences_survive",
        request=_request(
            "The Q3 budget was $4.2 million. Headcount grew by 12 engineers. "
            "Customer churn dropped to 3 percent. What changed this quarter?"
        ),
        required_facts=("4.2 million", "12 engineers", "3 percent"),
    ),
    FactPreservationCase(
        name="a_fact_repeated_only_as_paraphrase_still_survives_once",
        request=_request(
            "The meeting is scheduled for 3pm on Tuesday. "
            "Bring the quarterly report to the meeting. "
            "When is the meeting?"
        ),
        required_facts=("3pm",),
    ),
)


class TestCompressPromptPreservesFacts:
    @pytest.mark.parametrize("case", COMPRESS_CASES, ids=lambda c: c.name)
    def test_case(self, case: FactPreservationCase) -> None:
        stage = CompressPromptStage()
        failure = check_fact_preservation(stage, case, _ctx())
        assert failure is None, failure.reason

    def test_the_full_suite_reports_a_clean_pass(self) -> None:
        stage = CompressPromptStage()
        report = EvalReport()
        for case in COMPRESS_CASES:
            report.record(check_fact_preservation(stage, case, _ctx()))

        assert report.ok, [f.reason for f in report.failures]
        assert report.total == len(COMPRESS_CASES)


# ---------------------------------------------------------------------------
# semantic_cache: hits the near case, refuses the far one
# ---------------------------------------------------------------------------

CACHE_CASES = (
    CacheBehaviorCase(
        name="byte_identical_request_hits",
        stored_request=_request("what is the capital of france"),
        stored_response=LLMResponse(
            content="Paris", input_tokens=10, output_tokens=2, model="gpt-4o", finish_reason="stop"
        ),
        near_request=_request("what is the capital of france"),
        far_request=_request("explain the theory of relativity"),
    ),
)


class TestSemanticCacheBehavior:
    @pytest.mark.parametrize("case", CACHE_CASES, ids=lambda c: c.name)
    def test_case(self, case: CacheBehaviorCase) -> None:
        stage = SemanticCacheStage()
        ctx = _ctx(semantic_threshold=0.97)
        failures = check_cache_behavior(stage, case, ctx)
        assert not failures, [f.reason for f in failures]

    def test_the_full_suite_reports_a_clean_pass(self) -> None:
        report = EvalReport()
        for case in CACHE_CASES:
            stage = SemanticCacheStage()  # fresh store per case
            ctx = _ctx(semantic_threshold=0.97)
            report.record_batch(check_cache_behavior(stage, case, ctx))

        assert report.ok, [f.reason for f in report.failures]


# ---------------------------------------------------------------------------
# route_models: acts and declines exactly where its own rule says it should
# ---------------------------------------------------------------------------

ROUTING_CASES = (
    DecisionBoundaryCase(
        name="short_simple_request_is_routed",
        request=_request("what is the capital of france"),
        should_act=True,
    ),
    DecisionBoundaryCase(
        name="a_request_with_tools_is_not_routed",
        request=_request("look this up", tools=({"name": "search"},)),
        should_act=False,
    ),
    DecisionBoundaryCase(
        name="a_request_with_a_response_format_is_not_routed",
        request=_request("extract fields", response_format={"type": "json_object"}),
        should_act=False,
    ),
    DecisionBoundaryCase(
        name="a_long_prompt_is_not_routed",
        request=_request("word " * 2000),
        should_act=False,
    ),
)


class TestRouteModelsDecisionBoundary:
    @pytest.mark.parametrize("case", ROUTING_CASES, ids=lambda c: c.name)
    def test_case(self, case: DecisionBoundaryCase) -> None:
        stage = RouteModelsStage()
        ctx = _ctx(route_models=True, cheap_model="gpt-4o-mini")
        failure = check_decision_boundary(stage, case, ctx)
        assert failure is None, failure.reason

    def test_the_full_suite_reports_a_clean_pass(self) -> None:
        stage = RouteModelsStage()
        ctx = _ctx(route_models=True, cheap_model="gpt-4o-mini")
        report = EvalReport()
        for case in ROUTING_CASES:
            report.record(check_decision_boundary(stage, case, ctx))

        assert report.ok, [f.reason for f in report.failures]
        assert report.total == len(ROUTING_CASES)


# ---------------------------------------------------------------------------
# summarize_history: with a deterministic stub summarizer, the summary text
# itself must carry forward whatever it was given -- proving the stage's own
# plumbing (not a real summarizer's quality, which no model-free check can
# validate; see the eval package docstring).
# ---------------------------------------------------------------------------


def _stub_summarizer(text: str) -> str:
    """A deterministic stand-in: echoes every line's first word.

    Real summarizers are free text generators and out of scope for a
    model-free gate (see the eval package docstring) -- this exists only to
    prove the stage correctly threads whatever a summarizer returns into the
    outgoing request.
    """
    first_words = [
        line.split(":", 1)[-1].strip().split(" ")[0] for line in text.splitlines() if line
    ]
    return "covered: " + ", ".join(first_words)


class TestSummarizeHistoryPlumbing:
    def test_the_summarizers_output_reaches_the_outgoing_request(self) -> None:
        stage = SummarizeHistoryStage(summarizer=_stub_summarizer)
        ctx = _ctx(recent_turns=2)
        messages = [Message(role="system", content="sys")]
        for turn in range(6):
            messages.append(Message(role="user", content=f"question{turn} details"))
            messages.append(Message(role="assistant", content=f"answer{turn} details"))
        request = LLMRequest(model="gpt-4o", messages=tuple(messages), temperature=0.0)

        case = FactPreservationCase(
            name="the_stub_summarys_own_output_reaches_the_request",
            request=request,
            required_facts=("covered:",),
        )
        failure = check_fact_preservation(stage, case, ctx)

        assert failure is None, failure.reason if failure else ""


# ---------------------------------------------------------------------------
# reasoning_budget: acts only where ADR-018 says it may. A decision-boundary
# suite is the whole of what a model-free gate can check here -- whether a
# reduced budget still reaches the right answer is a question only a live run
# with graded tasks can answer, which is what ADR-018 requires before the flag
# is ever recommended. What these cases do cover is the half that is checkable
# and is also where the money-losing failures live: raising a budget, or
# inventing one, would make a cost-reduction library increase the bill.
# ---------------------------------------------------------------------------

REASONING_CASES = (
    DecisionBoundaryCase(
        name="a_budget_far_above_observed_use_is_lowered",
        request=_request("think about this", thinking_budget=32_000),
        should_act=True,
    ),
    DecisionBoundaryCase(
        name="a_request_with_no_budget_is_left_alone",
        request=_request("think about this"),
        should_act=False,
    ),
    DecisionBoundaryCase(
        name="a_budget_already_at_the_provider_minimum_is_never_raised",
        request=_request("think about this", thinking_budget=MIN_THINKING_BUDGET),
        should_act=False,
    ),
    DecisionBoundaryCase(
        name="a_budget_below_the_derived_ceiling_is_never_raised",
        request=_request("think about this", thinking_budget=2_000),
        should_act=False,
    ),
)


def _warmed_reasoning_stage(observed_output_tokens: int = 2_000) -> ReasoningBudgetStage:
    """A stage with a real observation history, which is what unlocks it.

    Below :data:`MIN_OBSERVATIONS` the stage declines everything, so an
    unwarmed instance would pass three of the four cases above for the wrong
    reason and fail the fourth.
    """
    stage = ReasoningBudgetStage()
    sent = _request("think about this", thinking_budget=32_000)
    reply = LLMResponse(
        content="x",
        input_tokens=10,
        output_tokens=observed_output_tokens,
        model="gpt-4o",
        finish_reason="stop",
    )
    for _ in range(MIN_OBSERVATIONS):
        stage.after(sent, reply, _ctx(reasoning_budget=True))
    return stage


class TestReasoningBudgetDecisionBoundary:
    @pytest.mark.parametrize("case", REASONING_CASES, ids=lambda c: c.name)
    def test_case(self, case: DecisionBoundaryCase) -> None:
        failure = check_decision_boundary(
            _warmed_reasoning_stage(), case, _ctx(reasoning_budget=True)
        )
        assert failure is None, failure.reason

    def test_the_full_suite_reports_a_clean_pass(self) -> None:
        report = EvalReport()
        for case in REASONING_CASES:
            report.record(
                check_decision_boundary(
                    _warmed_reasoning_stage(), case, _ctx(reasoning_budget=True)
                )
            )

        assert report.ok, [f.reason for f in report.failures]
        assert report.total == len(REASONING_CASES)

    def test_an_unwarmed_stage_declines_the_case_it_would_otherwise_act_on(self) -> None:
        """No sample, no ceiling. The gate would otherwise pass vacuously."""
        acting_case = REASONING_CASES[0]
        failure = check_decision_boundary(
            ReasoningBudgetStage(), acting_case, _ctx(reasoning_budget=True)
        )

        assert failure is not None
        assert "declined" in failure.reason
