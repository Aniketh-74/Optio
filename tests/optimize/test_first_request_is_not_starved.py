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


class TestTheFirstRequestGetsTheWholePipeline:
    def test_a_slow_first_count_does_not_starve_the_first_request(self) -> None:
        """The invariant, stated against the behaviour a user sees.

        ``trim_history`` shortens this conversation. If the first request pays a
        lazy initializer out of its own budget, it is skipped and the prompt
        goes to the provider at full length -- silently, and at full price.
        """
        counter = _SlowFirstCounter()
        config = OptimizeConfig()
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=counter)

        first = pipeline.prepare(_request())

        assert len(first.request.messages) < 81

    def test_the_first_request_is_optimized_as_much_as_the_second(self) -> None:
        """Stated as the general invariant rather than as one stage's outcome.

        Any future counter with a lazy initializer fails here, which is the
        point -- the defect was never about ``tiktoken`` specifically.
        """
        counter = _SlowFirstCounter()
        config = OptimizeConfig()
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=counter)

        first = pipeline.prepare(_request())
        second = pipeline.prepare(_request())

        assert len(first.request.messages) == len(second.request.messages)

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
        config = OptimizeConfig()
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=counter)
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="system", content="policy detail " * 3000), *_long_chat()),
            temperature=0.0,
        )

        first = pipeline.prepare(request)

        assert any(m.cacheable for m in first.request.messages)
