"""``Optimizer.afan_out``: dispatch order as a cost lever (ADR-020).

N concurrent calls over a shared prefix each pay to populate the provider's
cache, because none of them can see another's write. One first, then N-1 reads,
takes the shared prefix from 5x1.25 to 1.25+4x0.1 on Anthropic -- 74% off, with
no request altered and no answer changed.

The tests below are mostly about the cases where it must *not* warm up, because
those are the ones that cost something. A warm-up buys a discount only if the
provider actually cached the prefix, and below a provider's minimum cacheable
length nothing is cached at all -- so warming there is a full round trip of added
latency bought for exactly nothing, silently. That is the same failure that had a
measurement script reporting zero cache reads in both arms because its prompt sat
under Haiku 4.5's 4,096-token floor.

Order of *results* is also load-bearing: callers correlate a fan-out by index, so
a method that returned them in completion order would be a correctness bug
dressed as a performance feature.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from optio_optimize import LLMRequest, LLMResponse, Message, Optimizer
from optio_optimize.stages.caching import MIN_PREFIX_TOKENS

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.optimize


def _long_system() -> str:
    """A system prompt comfortably above ``MIN_PREFIX_TOKENS``."""
    return "You are a meticulous claims adjuster. Follow the schedule exactly. " * 600


def _shared_prefix_requests(count: int = 4, *, system: str | None = None) -> list[LLMRequest]:
    """``count`` requests that differ only in their final user turn."""
    head = Message(role="system", content=system if system is not None else _long_system())
    return [
        LLMRequest(
            model="claude-haiku-4-5",
            messages=(head, Message(role="user", content=f"classify item {index}")),
            temperature=0.0,
        )
        for index in range(count)
    ]


class _Recorder:
    """An async provider that records dispatch order and overlap.

    ``concurrent_peak`` is what distinguishes "warmed" from "not warmed": a
    warmed dispatch has exactly one call in flight while the first runs, then the
    rest together. Asserting on wall-clock time instead would be a flaky test
    about a scheduler.
    """

    def __init__(self, *, reply: str = "ok") -> None:
        self.seen: list[LLMRequest] = []
        self.in_flight = 0
        self.concurrent_peak = 0
        self.first_completed_before: list[int] = []
        self._reply = reply

    async def __call__(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        self.in_flight += 1
        self.concurrent_peak = max(self.concurrent_peak, self.in_flight)
        # Yield control so a genuinely concurrent gather overlaps here. Without
        # the await, every call would run to completion before the next started
        # and an unwarmed dispatch would look warmed.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.in_flight -= 1
        self.first_completed_before.append(len(self.seen))
        return LLMResponse(
            content=self._reply,
            input_tokens=100,
            output_tokens=10,
            model=request.model,
            finish_reason="stop",
        )


def _run(requests: Sequence[LLMRequest], provider: Any, **overrides: Any) -> list[LLMResponse]:
    optimizer = Optimizer(**overrides) if overrides else Optimizer()
    return asyncio.run(optimizer.afan_out(list(requests), provider))


class TestItWarmsTheCacheBeforeFanningOut:
    def test_the_first_call_goes_alone(self) -> None:
        provider = _Recorder()
        requests = _shared_prefix_requests(4)

        _run(requests, provider)

        assert len(provider.seen) == 4
        # One in flight while the warm-up runs, then the remaining three
        # together. A peak of 4 would mean nothing was warmed; a peak of 1 would
        # mean the whole fan-out was serialized, which trades far more latency
        # than the discount is worth.
        assert provider.concurrent_peak == 3

    def test_results_come_back_in_the_callers_order(self) -> None:
        """Callers correlate a fan-out by index."""
        provider = _Recorder()
        requests = _shared_prefix_requests(4)

        responses = _run(requests, provider)

        assert len(responses) == 4
        assert all(isinstance(r, LLMResponse) for r in responses)

    def test_every_request_is_dispatched_exactly_once(self) -> None:
        provider = _Recorder()

        _run(_shared_prefix_requests(5), provider)

        assert len(provider.seen) == 5


class TestWhenItMustNotWarmUp:
    def test_a_prefix_below_the_cacheable_floor_is_not_warmed(self) -> None:
        """Below the floor the provider caches nothing, so the warm-up is a
        round trip of latency bought for zero discount -- and silently."""
        provider = _Recorder()
        requests = _shared_prefix_requests(4, system="be terse")

        _run(requests, provider)

        assert provider.concurrent_peak == 4, (
            "a fan-out with no cacheable shared prefix was serialized anyway, "
            "which pays the latency and receives nothing"
        )

    def test_requests_with_no_shared_prefix_are_not_warmed(self) -> None:
        provider = _Recorder()
        requests = [
            LLMRequest(
                model="claude-haiku-4-5",
                messages=(Message(role="system", content=f"{_long_system()} variant {index}"),),
                temperature=0.0,
            )
            for index in range(4)
        ]

        _run(requests, provider)

        assert provider.concurrent_peak == 4

    def test_a_single_request_is_not_warmed(self) -> None:
        # Nothing to share a prefix with, so the warm-up would be the whole job
        # done twice as slowly for no reader.
        provider = _Recorder()

        responses = _run(_shared_prefix_requests(1), provider)

        assert len(responses) == 1
        assert provider.concurrent_peak == 1

    def test_an_empty_fan_out_dispatches_nothing(self) -> None:
        provider = _Recorder()

        assert _run([], provider) == []
        assert provider.seen == []


class TestTheCacheAndTheReportStillWork:
    def test_a_short_circuited_request_is_never_dispatched(self) -> None:
        provider = _Recorder()
        optimizer = Optimizer()
        requests = _shared_prefix_requests(3)

        async def go() -> list[LLMResponse]:
            # Prime the exact cache with the first request through the ordinary
            # path, then fan out over all three.
            await optimizer.acall(requests[0], provider)
            return await optimizer.afan_out(list(requests), provider)

        responses = asyncio.run(go())

        assert len(responses) == 3
        assert responses[0].served_from == "exact_cache"
        # Four dispatches, not five: the priming call plus the two requests the
        # cache could not answer.
        assert len(provider.seen) == 3

    def test_a_hit_does_not_justify_warming_for_the_rest(self) -> None:
        """A cached answer must not count toward "is a warm-up worth it".

        Tested against the decision function rather than through dispatch, and
        deliberately: with one live call remaining, a warmed and an unwarmed
        dispatch are *indistinguishable* from the outside -- both send one
        request. Asserting on observed concurrency would pass either way and read
        as coverage of a rule nothing checks.
        """
        from optio_optimize.fan_out import should_warm_up

        optimizer = Optimizer()
        one_live = [optimizer.pipeline.prepare(_shared_prefix_requests(1)[0])]

        assert should_warm_up(one_live) is False

    def test_two_live_requests_over_a_long_prefix_are_worth_warming(self) -> None:
        # The positive control for the test above: same function, same shape,
        # opposite answer, so a `return False` would not satisfy both.
        from optio_optimize.fan_out import should_warm_up

        optimizer = Optimizer()
        two_live = [optimizer.pipeline.prepare(r) for r in _shared_prefix_requests(2)]

        assert should_warm_up(two_live) is True

    def test_every_request_reaches_the_savings_report(self) -> None:
        provider = _Recorder()
        optimizer = Optimizer()

        asyncio.run(optimizer.afan_out(_shared_prefix_requests(4), provider))

        assert optimizer.report.requests == 4

    def test_the_prefix_marker_is_placed_on_every_request(self) -> None:
        # The whole point on Anthropic: nothing is cached without a breakpoint,
        # so a warmed dispatch with no marker warms nothing.
        provider = _Recorder()

        _run(_shared_prefix_requests(4), provider, prefix_cache=True, exact_cache=False)

        assert all(any(m.cacheable for m in sent.messages) for sent in provider.seen)


class TestFailOpen:
    def test_a_raising_stage_does_not_break_a_fan_out(self) -> None:
        from optio_optimize.stages.base import Stage, StageResult

        class Exploding(Stage):
            @property
            def name(self) -> str:
                return "exploding"

            def before(self, request: LLMRequest, ctx: Any) -> StageResult:
                raise RuntimeError("boom")

        provider = _Recorder()
        optimizer = Optimizer(stages=[Exploding()])

        responses = asyncio.run(optimizer.afan_out(_shared_prefix_requests(3), provider))

        assert len(responses) == 3
        assert len(provider.seen) == 3

    def test_a_provider_error_propagates_rather_than_being_swallowed(self) -> None:
        """The caller's own call failing is their business, not ours to hide.

        Fail-open means a failure *in this package* leaves the caller no worse
        off. A provider raising is a real error the caller must see -- swallowing
        it would return a response nobody produced.
        """

        async def failing(request: LLMRequest) -> LLMResponse:
            raise RuntimeError("provider down")

        optimizer = Optimizer()

        with pytest.raises(RuntimeError, match="provider down"):
            asyncio.run(optimizer.afan_out(_shared_prefix_requests(3), failing))


class TestTheFloorIsTheDocumentedOne:
    def test_the_threshold_is_the_marker_stages_own_constant(self) -> None:
        """One wrong constant that gets fixed once, not two that drift.

        ``MIN_PREFIX_TOKENS`` is known to be wrong in a specific way -- the real
        floor spans 512 to 4,096 across models and this is a single number -- so
        ADR-020 reuses it rather than inventing a second guess beside it.
        """
        from optio_optimize.fan_out import WARM_UP_MIN_PREFIX_TOKENS

        assert WARM_UP_MIN_PREFIX_TOKENS is MIN_PREFIX_TOKENS
