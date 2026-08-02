"""The first request must not be optimized less than the second (ADR-038).

``tiktoken`` loads its BPE vocabulary lazily, on first use, and that costs
**395 ms** against a 100 ms ``latency_budget_ms``. Whichever stage counted
first paid it, and every stage after that one was skipped.

Measured through the real pipeline on a long conversation, default config,
first request::

    unstable_prefix            0.02 ms
    exact_cache                1.71 ms
    adaptive_max_tokens        0.00 ms
    trim_history             388.03 ms   <- pays the vocabulary load
    budget 100.0 ms; stages that ran: 4 of 9

Five of nine never ran, including ``prefix_cache`` -- the largest lossless
saving in the package and Anthropic's only cache mechanism. The second request
through the same process ran all nine, which is why no benchmark caught it: the
defect is invisible to anything that makes more than one call and looks at
aggregates.

These tests use a deliberately slow counter rather than ``tiktoken``, so the
invariant holds for **any** counter with a lazy initializer rather than for one
library's startup behaviour.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.pipeline import Pipeline
from optio_optimize.stages import build_stages
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


class _SlowFirstCounter:
    """A counter whose first call is ruinously expensive, like a real one.

    Models the shape of the defect without depending on ``tiktoken`` being
    installed or on how long its vocabulary takes to load on a given machine.
    """

    def __init__(self, first_call_ms: float = 400.0) -> None:
        self._inner = HeuristicCounter()
        self._first_call_ms = first_call_ms
        self.calls = 0

    @property
    def is_exact(self) -> bool:
        return False

    def count_text(self, text: str, model: str = "") -> int:
        self.calls += 1
        if self.calls == 1:
            import time

            time.sleep(self._first_call_ms / 1000.0)
        return self._inner.count_text(text, model)


def _long_chat() -> tuple[Message, ...]:
    return tuple(
        Message(
            role="user" if i % 2 == 0 else "assistant",
            content=f"q{i} " + "policy detail " * 200,
        )
        for i in range(81)
    )


def _request() -> LLMRequest:
    return LLMRequest(model="claude-haiku-4-5", messages=_long_chat(), temperature=0.0)


class TestTheWarmUpHappensAtConstruction:
    def test_building_a_pipeline_counts_something(self) -> None:
        """The vocabulary load belongs to *having* an optimizer, not using one."""
        counter = _SlowFirstCounter(first_call_ms=0.0)
        config = OptimizeConfig()

        Pipeline(config=config, stages=build_stages(config), counter=counter)

        assert counter.calls >= 1

    def test_a_counter_that_raises_while_warming_does_not_break_construction(self) -> None:
        """Warming changes *when* a cost is paid; it is never a new way to fail."""

        class _Exploding:
            is_exact = False

            def count_text(self, text: str, model: str = "") -> int:
                raise RuntimeError("no vocabulary here")

        config = OptimizeConfig()

        pipeline = Pipeline(config=config, stages=build_stages(config), counter=_Exploding())

        assert pipeline is not None


class _RecordsWhoPaidTheInitializer:
    """A counter whose expensive first call is *recorded* rather than slept.

    The earlier version of these tests slept 400 ms on call one and then asserted
    on which stages had run. That was flaky **and** weak, which is an unusual
    combination and worth stating:

    *Flaky*, because the assertion raced a 100 ms wall-clock budget on a shared
    CI runner. It failed on ``py3.12/ubuntu`` and the Windows wheel job while
    passing everywhere else, and 0 times in 15 local runs.

    *Weak*, because the budget is checked **between** stages, so the stage that
    pays the initializer still completes and still fires. Measured both ways:
    warmed and unwarmed both trimmed the conversation to 8 messages. The
    assertion could not distinguish the bug from the fix -- it was reporting on
    the weather.

    The invariant does not need a clock. It is simply: *no stage pays the first
    call*. That is a fact about call ordering, and it is exact.
    """

    is_exact = False

    def __init__(self) -> None:
        self._inner = HeuristicCounter()
        self.calls = 0
        self.first_call_paid_by: str | None = None
        self.phase = "construction"

    def count_text(self, text: str, model: str = "") -> int:
        self.calls += 1
        if self.calls == 1:
            self.first_call_paid_by = self.phase
        return self._inner.count_text(text, model)


class TestTheFirstRequestGetsTheWholePipeline:
    def test_construction_pays_the_initializer_not_the_first_request(self) -> None:
        """The invariant, without a clock in it.

        Whoever makes call one pays the vocabulary load. If that is a stage, the
        request it belongs to loses whatever budget the load consumes, and the
        stages behind it are skipped -- silently, and at full price.
        """
        counter = _RecordsWhoPaidTheInitializer()
        config = OptimizeConfig()
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=counter)

        counter.phase = "first request"
        pipeline.prepare(_request())

        assert counter.first_call_paid_by == "construction"

    def test_the_first_request_runs_the_same_stages_as_the_second(self) -> None:
        """Stated as the general invariant rather than as one stage's outcome.

        Compares the stage lists rather than a message count: two requests can
        agree on length while disagreeing about which stages produced it, and
        the defect is about *which stages ran*.

        Two deliberate choices here, both learned from this test failing on CI
        after the first attempt to fix it:

        *A stated budget.* Request one is legitimately slower than request two
        even with a warm tokenizer -- ``MemoizingCounter`` has nothing cached,
        the regexes are uncompiled, the bytecode is cold. Left at the 100 ms
        default this compared "did the same stages run" against a clock that
        request one loses on its own merits, and the Windows sdist runner found
        that: ``[] != ['trim_history']``. The gap being measured must be the
        initializer, not the cache warming up behind it.

        *A counter that actually pays one.* With a recording counter there is no
        initializer to skip, so removing the warm-up changed nothing and this
        test could not fail. It needs the slow one to have anything to detect.
        """
        counter = _SlowFirstCounter()
        # 300 ms against ~8 ms of real work: forty times the headroom a warm
        # first request needs, and still less than the 400 ms an unwarmed one
        # would spend on the initializer alone.
        config = OptimizeConfig(latency_budget_ms=300.0)
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=counter)

        first = pipeline.prepare(_request())
        second = pipeline.prepare(_request())

        assert first.fired == second.fired

    def test_prefix_cache_reaches_the_first_request(self) -> None:
        """The stage that was being lost, named.

        ``prefix_cache`` is last in the pipeline, so it is the first casualty of
        a blown budget and the most expensive one -- 90% off the tokens it
        covers, and Anthropic's only cache mechanism.

        The system prompt is deliberately large: Haiku's minimum cacheable
        prefix is 4,096 tokens, and below it the stage declines for its own good
        reasons rather than because the budget ran out. Testing the wrong
        decline would pass whether or not the bug were fixed.
        """
        counter = _SlowFirstCounter()
        # The budget is set here rather than left at the 100 ms default so the
        # margin is a stated quantity instead of an accident. Real work on this
        # request measures ~8 ms, so 300 ms tolerates a runner nearly forty
        # times slower than this machine; the unwarmed 400 ms first call still
        # exceeds it, so the test keeps failing when the bug returns. At the
        # default the margin was 16x, and CI found the edge of it.
        config = OptimizeConfig(latency_budget_ms=300.0)
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=counter)
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="system", content="policy detail " * 3000), *_long_chat()),
            temperature=0.0,
        )

        first = pipeline.prepare(request)

        assert any(m.cacheable for m in first.request.messages)
