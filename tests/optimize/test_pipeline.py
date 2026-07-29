"""The pipeline's two non-negotiable guarantees.

**A stage failure never reaches the caller** (ADR-013 rule 1). This package sits
in the request path, so unlike the core -- where a bug drops a signal -- a bug
here can break someone's agent. Every test that removes a stage's ability to
work checks that the call still succeeds and still returns the real answer.

**The savings numbers are coherent.** The whole premise is a number, so a
report that can show baseline below actual, or negative dollars saved, is worse
than no report: it is a confident wrong answer. The first smoke test of this
pipeline did exactly that, by measuring baseline with our tokenizer and actual
with the provider's count. These tests pin the invariants that mistake violated.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from optio_optimize import LLMRequest, LLMResponse, Message, Optimizer
from optio_optimize.stages.base import Stage, StageResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from optio_optimize.stages.base import StageContext

pytestmark = pytest.mark.optimize


def _request(text: str = "hello", *, temperature: float | None = 0.0) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=(
            Message(role="system", content="You are terse."),
            Message(role="user", content=text),
        ),
        temperature=temperature,
    )


def _provider(calls: list[LLMRequest]) -> Callable[[LLMRequest], LLMResponse]:
    def call(request: LLMRequest) -> LLMResponse:
        calls.append(request)
        return LLMResponse(
            content="the real answer",
            input_tokens=500,
            output_tokens=20,
            model=request.model,
            finish_reason="stop",
        )

    return call


class ExplodingStage(Stage):
    """A stage that fails the way a real bug would: after partial work."""

    @property
    def name(self) -> str:
        return "exploding"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        raise RuntimeError("stage is broken")

    def after(self, request: LLMRequest, response: LLMResponse, ctx: StageContext) -> None:
        raise RuntimeError("after hook is broken too")


class Mangl1ngStage(Stage):
    """A stage that corrupts the request, then fails.

    The dangerous shape: if the pipeline passed a mutated request onward after a
    failure, the provider would receive the corrupted version. Frozen requests
    plus last-known-good tracking are what prevent it, and this proves it.
    """

    @property
    def name(self) -> str:
        return "mangling"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        request.with_messages((Message(role="user", content="CORRUPTED"),))
        raise ValueError("failed after building a bad request")


class TestAStageFailureNeverReachesTheCaller:
    def test_the_call_still_succeeds(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer(stages=[ExplodingStage()])

        response = optimizer.call(_request(), _provider(calls))

        assert response.content == "the real answer"
        assert len(calls) == 1

    def test_the_provider_receives_the_unmodified_request(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer(stages=[Mangl1ngStage()])
        original = _request("keep me")

        optimizer.call(original, _provider(calls))

        assert calls[0].messages == original.messages, (
            "a failed stage leaked a partially transformed request to the provider"
        )

    def test_a_failing_after_hook_does_not_lose_the_response(self) -> None:
        # The response already exists at this point. Turning a successful call
        # into an exception because a cache write failed would be the worst
        # possible trade.
        calls: list[LLMRequest] = []
        optimizer = Optimizer(stages=[ExplodingStage()])

        assert optimizer.call(_request(), _provider(calls)).content == "the real answer"

    def test_a_good_stage_still_runs_after_a_bad_one(self) -> None:
        from optio_optimize.stages.caching import ExactCacheStage

        calls: list[LLMRequest] = []
        optimizer = Optimizer(stages=[ExplodingStage(), ExactCacheStage()])
        request = _request()

        optimizer.call(request, _provider(calls))
        optimizer.call(request, _provider(calls))

        assert len(calls) == 1, "the working cache stage was skipped because a sibling failed"


class TestTheProviderErrorIsTheCallersToHandle:
    def test_a_provider_exception_propagates(self) -> None:
        """Only the provider's own failures reach the caller.

        Swallowing these would be far worse than a stage failure: the caller
        would receive a fabricated success for a call that never happened.
        """

        def broken(_request: LLMRequest) -> LLMResponse:
            raise ConnectionError("provider is down")

        with pytest.raises(ConnectionError, match="provider is down"):
            Optimizer().call(_request(), broken)


class TestSavingsAccountingIsCoherent:
    """Invariants that the derived-baseline model must satisfy."""

    def test_baseline_is_never_below_actual(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        request = _request()
        for _ in range(5):
            optimizer.call(request, _provider(calls))

        report = optimizer.report
        assert report.baseline_input_tokens >= report.actual_input_tokens
        assert report.baseline_output_tokens >= report.actual_output_tokens

    def test_savings_are_never_negative(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        for i in range(5):
            optimizer.call(_request(f"q{i}"), _provider(calls))

        assert (optimizer.report.estimated_saved_usd("gpt-4o") or 0.0) >= 0.0
        assert optimizer.report.total_saved_tokens >= 0

    def test_the_reduction_ratio_stays_within_zero_and_one(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        request = _request()
        for _ in range(10):
            optimizer.call(request, _provider(calls))

        ratio = optimizer.report.reduction_ratio
        assert ratio is not None
        assert 0.0 <= ratio <= 1.0, f"nonsensical reduction ratio {ratio}"

    def test_no_baseline_reports_none_rather_than_zero(self) -> None:
        # Absence is not zero -- the same rule the core applies to signals. A
        # 0.0 here reads as "measured, saved nothing", which is a much stronger
        # and more damning claim than "nothing has run yet".
        assert Optimizer().report.reduction_ratio is None

    def test_an_unpriced_model_reports_none_rather_than_zero_dollars(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        optimizer.call(_request(), _provider(calls))

        assert optimizer.report.estimated_saved_usd("some-model-we-never-priced") is None

    def test_every_avoided_call_is_fully_credited(self) -> None:
        """Three cache hits out of four calls is a 75% reduction."""
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        request = _request()
        for _ in range(4):
            optimizer.call(request, _provider(calls))

        assert len(calls) == 1
        ratio = optimizer.report.reduction_ratio
        assert ratio is not None
        assert ratio == pytest.approx(0.75, abs=0.02)


class TestTheLatencyBudgetIsEnforced:
    def test_stages_past_the_budget_are_skipped(self) -> None:
        class SlowStage(Stage):
            def __init__(self, name: str) -> None:
                self._name = name
                self.ran = False

            @property
            def name(self) -> str:
                return self._name

            def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
                self.ran = True
                time.sleep(0.02)
                return self.declines(request)

        first, second, third = SlowStage("s1"), SlowStage("s2"), SlowStage("s3")
        calls: list[LLMRequest] = []
        optimizer = Optimizer(
            latency_budget_ms=25.0,
            stages=[first, second, third],
        )

        optimizer.call(_request(), _provider(calls))

        assert first.ran, "the first stage should always get a chance"
        assert not third.ran, "the budget did not stop the pipeline"
        assert len(calls) == 1, "the provider call must happen regardless of the budget"


class TestACachingStageKeysConsistentlyAcrossBeforeAndAfter:
    """A cache must write under the same text it reads under.

    ``after`` receives the request **as sent** -- every later stage's rewrites
    included -- while ``before`` saw it as the earlier stages left it. A stage
    that recomputes its key in ``after`` therefore stores entries under text no
    lookup will ever produce, and its hit rate silently drops to zero. Not
    hypothetically: ``semantic_cache`` did exactly this, and the defect stayed
    invisible for as long as every message-rewriting stage happened to decline
    on the workloads that exercised it. Adding ``concision``, which fires on
    any plain chat request, took the audit's collision count from several to
    none in one commit.

    ``exact_cache`` was always correct here -- it stashes its key in
    ``ctx.scratch`` during ``before`` -- so these tests pin the contract for
    both rather than for the one that broke.
    """

    @staticmethod
    def _optimizer(stage_name: str) -> Optimizer:
        """An optimizer with one cache stage and one later message rewriter."""
        return Optimizer(
            exact_cache=stage_name == "exact_cache",
            semantic_cache=stage_name == "semantic_cache",
            prefix_cache=False,
            structured_output=False,
            adaptive_max_tokens=False,
            trim_history=False,
            deduplicate=False,
            prune_retrieval=False,
            cap_tool_results=False,
            minify_tools=False,
            concision=True,  # the later stage that rewrites messages
        )

    @pytest.mark.parametrize("stage_name", ["exact_cache", "semantic_cache"])
    def test_a_repeat_still_hits_when_a_later_stage_rewrites_the_prompt(
        self, stage_name: str
    ) -> None:
        calls: list[LLMRequest] = []
        optimizer = self._optimizer(stage_name)
        request = _request("what is the capital of France")

        optimizer.call(request, _provider(calls))
        optimizer.call(request, _provider(calls))

        assert len(calls) == 1, (
            f"{stage_name} missed on an identical repeat: it stored the entry under the "
            "rewritten prompt and looked it up under the original"
        )

    @pytest.mark.parametrize("stage_name", ["exact_cache", "semantic_cache"])
    def test_the_rewriting_stage_really_did_change_the_prompt(self, stage_name: str) -> None:
        # Without this the test above passes trivially if `concision` stops
        # firing -- it would be checking that a cache hits when nothing
        # interfered, which is not the property at issue.
        calls: list[LLMRequest] = []
        self._optimizer(stage_name).call(
            _request("what is the capital of France"), _provider(calls)
        )

        assert calls[0].messages != _request("what is the capital of France").messages


class TestDisablingWorks:
    def test_a_disabled_pipeline_is_a_pass_through(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer(enabled=False)
        request = _request()

        optimizer.call(request, _provider(calls))
        optimizer.call(request, _provider(calls))

        # This is the control arm for A/B measurement: it must call the
        # provider every single time, or the comparison is meaningless.
        assert len(calls) == 2
        assert calls[0].messages == request.messages

    def test_a_named_stage_can_be_disabled(self) -> None:
        optimizer = Optimizer(disabled_stages=frozenset({"exact_cache"}))
        assert "exact_cache" not in optimizer.stage_names
        assert "prefix_cache" in optimizer.stage_names
