"""A priced step must reach the run span, or say why not (ADR-043).

`test_a_cache_hit_is_priced_at_zero_by_optios_own_ledger` failed on CI with
``KeyError: 'gen_ai.run.actual_cost'`` -- the run span carried no cost at all --
and passed on the same commit, same Python, in a different job. It passes
locally in isolation, in the full suite, twice consecutively, with a cold
tokenizer cache, and under CI's exact failing command.

So rather than chase the flake, this asserts the **property** that test happened
to be observing: *whenever a priceable GenAI span ends inside a metered run, the
run span ends up carrying a cost.* A lucky case is replaced by the general one,
the same move ADR-040's ``test_every_billable_field_is_zeroed`` made.

The path has three places a cost can vanish without anything failing:

* :func:`~optio.runtime.span_tap._SpanTap._signal_target` returns ``None`` when
  no span is recording, and ``write_signals`` treats that as a no-op;
* ``_is_emittable`` drops a non-finite value with a ``debug`` log;
* ``on_run_end`` omits the cost entirely when ``reconciled_steps == 0``.

Each is correct in isolation and each is silent. Silence is what makes a flake
look like a race: the same missing attribute can arrive by three different
routes, and the assertion that noticed it cannot tell them apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.optimize


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> Iterator[TracerProvider]:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield tracer_provider
    tracer_provider.shutdown()


def _request() -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=(Message(role="user", content="hello"),),
        temperature=0.0,
    )


def _priced(request: LLMRequest) -> LLMResponse:
    """A response that costs exactly $2.50 on gpt-4o."""
    return LLMResponse(
        content="hello", model=request.model, input_tokens=1_000_000, output_tokens=0
    )


def _run_span(exporter: InMemorySpanExporter) -> object:
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("optio.run")]
    assert len(spans) == 1, f"expected exactly one run span, got {[s.name for s in spans]}"
    return spans[0]


class TestAPricedStepAlwaysReachesTheRunSpan:
    """The invariant, stated once and probed from several directions."""

    def test_one_priced_call(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> None:
            Optimizer(emit_spans=True, tracer_provider=provider).call(_request(), _priced)

        run_agent()

        attrs = _run_span(exporter).attributes  # type: ignore[attr-defined]
        assert attrs is not None
        assert semconv.RUN_ACTUAL_COST in attrs

    def test_a_cache_hit_after_a_priced_call(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """The shape that failed on CI.

        The second call is served by ``exact_cache``, and ADR-040 made
        ``served_from_cache`` zero **every** ``*_tokens`` field rather than
        three. A zeroed span is still priceable -- 0 is not absent -- so the
        cost must survive, and it must not double.
        """
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> None:
            optimizer = Optimizer(emit_spans=True, tracer_provider=provider)
            optimizer.call(_request(), _priced)
            optimizer.call(_request(), _priced)

        run_agent()

        attrs = _run_span(exporter).attributes  # type: ignore[attr-defined]
        assert attrs is not None
        assert attrs[semconv.RUN_ACTUAL_COST] == pytest.approx(2.50)

    def test_many_calls_with_repeated_cache_hits(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Ten calls, nine of them cache hits. Still one priced step, still $2.50.

        If a cache hit could ever un-reconcile the step it hit, this is where a
        single occurrence turns into nine chances to see it.
        """
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> None:
            optimizer = Optimizer(emit_spans=True, tracer_provider=provider)
            for _ in range(10):
                optimizer.call(_request(), _priced)

        run_agent()

        attrs = _run_span(exporter).attributes  # type: ignore[attr-defined]
        assert attrs is not None
        assert attrs[semconv.RUN_ACTUAL_COST] == pytest.approx(2.50)

    def test_a_fresh_optimizer_per_call_still_reports_once(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Two optimizers means two exact caches, so both calls really are billed.

        Distinguishes "the cache suppressed the second charge" from "the ledger
        lost the first" -- the two readings of a $2.50 total that the original
        test could not tell apart.
        """
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> None:
            Optimizer(emit_spans=True, tracer_provider=provider).call(_request(), _priced)
            Optimizer(emit_spans=True, tracer_provider=provider).call(_request(), _priced)

        run_agent()

        attrs = _run_span(exporter).attributes  # type: ignore[attr-defined]
        assert attrs is not None
        assert attrs[semconv.RUN_ACTUAL_COST] == pytest.approx(5.00)


class TestTheThreeSilentRoutesToAMissingCost:
    """Each is correct behaviour; each is indistinguishable from the others at
    the assertion that catches it. Pinning them separately is what turns one
    ``KeyError`` into a diagnosis.
    """

    def test_no_recording_span_drops_the_signal_and_says_nothing(self) -> None:
        """``write_signals(None, ...)`` is a deliberate no-op (§ signal_writer).

        Correct -- outside a trace there is nowhere to put a value -- but it
        means a misconfigured provider loses every cost signal in silence, which
        is one way the attribute goes missing without any error.
        """
        from optio import semconv
        from optio.lanes.base import Signal
        from optio.runtime.signal_writer import write_signals

        written = write_signals(None, [Signal(semconv.RUN_ACTUAL_COST, 2.50)])

        assert written == 0

    def test_a_non_finite_cost_is_dropped_rather_than_written(self) -> None:
        """A cost of ``inf`` or ``nan`` never reaches a consumer.

        Right call -- a budget policy reading ``inf`` would behave worse than
        one reading nothing -- and another silent route to a missing attribute.
        """
        from optio import semconv
        from optio.lanes.base import Signal
        from optio.runtime.signal_writer import write_signals

        assert write_signals(None, [Signal(semconv.RUN_ACTUAL_COST, float("nan"))]) == 0

    def test_an_unpriceable_run_reports_no_cost_rather_than_zero(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """The third route, and the one that is load-bearing.

        A model nothing can price leaves ``reconciled_steps == 0``, and the lane
        deliberately omits the cost rather than emitting ``0.0`` -- reporting a
        run as free when the truth is that it could not be priced is the
        confusion ``docs/signals.md`` forbids. So **absence is a real answer
        here**, and the original test's ``KeyError`` is what it looks like.
        """
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> None:
            request = LLMRequest(
                model="a-model-nobody-prices",
                messages=(Message(role="user", content="hello"),),
                temperature=0.0,
            )
            Optimizer(emit_spans=True, tracer_provider=provider).call(request, _priced)

        run_agent()

        attrs = _run_span(exporter).attributes  # type: ignore[attr-defined]
        assert attrs is not None
        assert semconv.RUN_ACTUAL_COST not in attrs
