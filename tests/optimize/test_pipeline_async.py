"""Optimizer.acall / Pipeline.aexecute: the async twin, exercised the same way.

No new test dependency: every test drives the coroutine with a bare
``asyncio.run`` rather than a pytest-asyncio plugin, since nothing here needs
more than "run one coroutine to completion."
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from optio_optimize import LLMRequest, LLMResponse, Message, Optimizer
from optio_optimize.stages.base import Stage, StageResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

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


def _async_provider(calls: list[LLMRequest]) -> Callable[[LLMRequest], Awaitable[LLMResponse]]:
    async def call(request: LLMRequest) -> LLMResponse:
        calls.append(request)
        return LLMResponse(
            content="the real answer",
            input_tokens=500,
            output_tokens=20,
            model=request.model,
            finish_reason="stop",
        )

    return call


def _run(coro: Coroutine[object, object, LLMResponse]) -> LLMResponse:
    return asyncio.run(coro)


class TestAcallMatchesCallsBehaviour:
    def test_a_normal_request_is_optimized_and_the_provider_is_awaited(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()

        response = _run(optimizer.acall(_request(), _async_provider(calls)))

        assert response.content == "the real answer"
        assert len(calls) == 1

    def test_a_cache_hit_never_awaits_the_provider_a_second_time(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        request = _request()

        _run(optimizer.acall(request, _async_provider(calls)))
        _run(optimizer.acall(request, _async_provider(calls)))

        assert len(calls) == 1, "exact_cache should have served the second call"

    def test_disabled_still_awaits_the_provider_directly(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer(enabled=False)

        _run(optimizer.acall(_request(), _async_provider(calls)))
        _run(optimizer.acall(_request(), _async_provider(calls)))

        assert len(calls) == 2

    def test_savings_accumulate_into_the_same_report_as_call(self) -> None:
        calls: list[LLMRequest] = []
        optimizer = Optimizer()
        request = _request()

        for _ in range(4):
            _run(optimizer.acall(request, _async_provider(calls)))

        assert len(calls) == 1
        ratio = optimizer.report.reduction_ratio
        assert ratio is not None
        assert ratio == pytest.approx(0.75, abs=0.02)


class TestAcallFailsOpenLikeCall:
    def test_a_stage_failure_never_reaches_the_caller(self) -> None:
        class ExplodingStage(Stage):
            @property
            def name(self) -> str:
                return "exploding"

            def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
                raise RuntimeError("stage is broken")

        calls: list[LLMRequest] = []
        optimizer = Optimizer(stages=[ExplodingStage()])

        response = _run(optimizer.acall(_request(), _async_provider(calls)))

        assert response.content == "the real answer"
        assert len(calls) == 1

    def test_a_provider_exception_still_propagates(self) -> None:
        """Only the provider's own failures reach the caller, same as call()."""

        async def broken(_request: LLMRequest) -> LLMResponse:
            raise ConnectionError("provider is down")

        with pytest.raises(ConnectionError, match="provider is down"):
            _run(Optimizer().acall(_request(), broken))
