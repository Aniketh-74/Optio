"""ADR-014: optio_optimize integrates by emitting spans, not by calling optio.

The real claim under test is the last class here: a span this package emits
reaches ``optio``'s own cost lane and gets priced, with zero code changed on
the ``optio`` side. Everything above it tests the mechanism in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio_optimize import telemetry
from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.optimize


def _request(model: str = "gpt-4o") -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(Message(role="user", content="hi"),),
        temperature=0.0,
    )


def _response(model: str = "gpt-4o", **overrides: object) -> LLMResponse:
    defaults: dict[str, object] = {
        "content": "hello",
        "input_tokens": 10,
        "output_tokens": 5,
        "model": model,
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """An in-memory exporter, isolated per test."""
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> Iterator[TracerProvider]:
    """A tracer provider local to one test -- never the OTel global default."""
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield tracer_provider
    tracer_provider.shutdown()


class TestConstantsStayInStepWithOptio:
    """The duplication ADR-014 accepts, checked so drift is a test failure."""

    def test_the_consumed_attribute_names_match_optios_semconv(self) -> None:
        from optio import semconv

        assert telemetry.GEN_AI_OPERATION_NAME == semconv.GEN_AI_OPERATION_NAME
        assert telemetry.GEN_AI_REQUEST_MODEL == semconv.GEN_AI_REQUEST_MODEL
        assert telemetry.GEN_AI_RESPONSE_MODEL == semconv.GEN_AI_RESPONSE_MODEL
        assert telemetry.GEN_AI_USAGE_INPUT_TOKENS == semconv.GEN_AI_USAGE_INPUT_TOKENS
        assert telemetry.GEN_AI_USAGE_OUTPUT_TOKENS == semconv.GEN_AI_USAGE_OUTPUT_TOKENS
        assert telemetry.GEN_AI_RUN_ID == semconv.RUN_ID

    def test_optios_own_attributes_are_not_reused_for_this_packages_data(self) -> None:
        # optio_optimize.* must never collide with a name optio's frozen
        # signal contract (§7.2) claims, or a consumer could misread one
        # package's data as the other's.
        from optio import semconv

        own = {
            telemetry.OPTIMIZE_STAGE,
            telemetry.OPTIMIZE_SAVED_INPUT_TOKENS,
            telemetry.OPTIMIZE_SAVED_OUTPUT_TOKENS,
            telemetry.OPTIMIZE_SHORT_CIRCUITED,
        }
        assert own.isdisjoint(semconv.EMITTED_SIGNALS)


class TestRecordSpanEmitsTheDocumentedAttributes:
    def test_a_normal_call_carries_usage_and_savings(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        telemetry.record_span(
            _request(),
            _response(input_tokens=100, output_tokens=20),
            stages=["prefix_cache"],
            saved_input_tokens=7,
            saved_output_tokens=3,
            short_circuited=False,
            tracer_provider=provider,
        )

        (span,) = exporter.get_finished_spans()
        attrs = span.attributes
        assert attrs is not None
        assert attrs[telemetry.GEN_AI_REQUEST_MODEL] == "gpt-4o"
        assert attrs[telemetry.GEN_AI_RESPONSE_MODEL] == "gpt-4o"
        assert attrs[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[telemetry.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert attrs[telemetry.OPTIMIZE_SAVED_INPUT_TOKENS] == 7
        assert attrs[telemetry.OPTIMIZE_SAVED_OUTPUT_TOKENS] == 3
        assert attrs[telemetry.OPTIMIZE_SHORT_CIRCUITED] is False
        assert attrs[telemetry.OPTIMIZE_STAGE] == "prefix_cache"

    def test_a_short_circuit_carries_zeroed_usage_honestly(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """A cache hit's zeroed tokens must reach the span unchanged.

        This is the whole mechanism ADR-014 relies on: optio's ledger prices
        whatever the span says, so an honest zero here is what makes a cache
        hit cost $0 downstream with no special-casing on either side.
        """
        cached = _response(input_tokens=0, output_tokens=0, served_from="exact_cache")

        telemetry.record_span(
            _request(),
            cached,
            stages=["exact_cache"],
            saved_input_tokens=100,
            saved_output_tokens=20,
            short_circuited=True,
            tracer_provider=provider,
        )

        (span,) = exporter.get_finished_spans()
        attrs = span.attributes
        assert attrs is not None
        assert attrs[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 0
        assert attrs[telemetry.GEN_AI_USAGE_OUTPUT_TOKENS] == 0
        assert attrs[telemetry.OPTIMIZE_SHORT_CIRCUITED] is True

    def test_run_id_is_written_when_given(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        telemetry.record_span(
            _request(),
            _response(),
            stages=[],
            saved_input_tokens=0,
            saved_output_tokens=0,
            short_circuited=False,
            run_id="run-abc123",
            tracer_provider=provider,
        )

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes[telemetry.GEN_AI_RUN_ID] == "run-abc123"

    def test_run_id_is_absent_when_not_given(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        telemetry.record_span(
            _request(),
            _response(),
            stages=[],
            saved_input_tokens=0,
            saved_output_tokens=0,
            short_circuited=False,
            tracer_provider=provider,
        )

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert telemetry.GEN_AI_RUN_ID not in span.attributes

    def test_no_stages_omits_the_stage_attribute_rather_than_writing_empty(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        telemetry.record_span(
            _request(),
            _response(),
            stages=[],
            saved_input_tokens=0,
            saved_output_tokens=0,
            short_circuited=False,
            tracer_provider=provider,
        )

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert telemetry.OPTIMIZE_STAGE not in span.attributes

    def test_multiple_stages_are_joined(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        telemetry.record_span(
            _request(),
            _response(),
            stages=["deduplicate", "prune_retrieval"],
            saved_input_tokens=1,
            saved_output_tokens=0,
            short_circuited=False,
            tracer_provider=provider,
        )

        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes[telemetry.OPTIMIZE_STAGE] == "deduplicate,prune_retrieval"


class TestFailOpen:
    """A broken exporter or SDK must never reach the caller (ADR-013 rule 1)."""

    def test_an_exception_inside_span_creation_never_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ExplodingTracer:
            def start_as_current_span(self, *_args: object, **_kwargs: object) -> object:
                raise RuntimeError("tracer is broken")

        class ExplodingTrace:
            @staticmethod
            def get_tracer(*_args: object, **_kwargs: object) -> ExplodingTracer:
                return ExplodingTracer()

        import opentelemetry

        monkeypatch.setattr(opentelemetry, "trace", ExplodingTrace)

        # Must not raise.
        telemetry.record_span(
            _request(),
            _response(),
            stages=[],
            saved_input_tokens=0,
            saved_output_tokens=0,
            short_circuited=False,
        )


class TestPipelineIntegration:
    def test_no_span_is_created_when_emit_spans_is_off(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The default. A pre-existing Optimizer() caller must see no new
        # side effect from this feature landing.
        optimizer = Optimizer(exact_cache=False, prefix_cache=False, tracer_provider=provider)

        optimizer.call(_request(), lambda r: _response(model=r.model))

        assert exporter.get_finished_spans() == ()

    def test_a_span_is_created_per_call_when_emit_spans_is_on(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        optimizer = Optimizer(
            emit_spans=True,
            exact_cache=False,
            prefix_cache=False,
            tracer_provider=provider,
        )

        optimizer.call(_request(), lambda r: _response(model=r.model))
        optimizer.call(_request("gpt-4o-mini"), lambda r: _response(model=r.model))

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

    def test_a_short_circuiting_stage_still_emits_a_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        optimizer = Optimizer(emit_spans=True, tracer_provider=provider)
        request = _request()
        calls: list[LLMRequest] = []

        def call(r: LLMRequest) -> LLMResponse:
            calls.append(r)
            return _response(model=r.model, input_tokens=50, output_tokens=10)

        optimizer.call(request, call)  # miss: populates the cache
        optimizer.call(request, call)  # hit: short-circuits

        assert len(calls) == 1  # the cache actually avoided the second call
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        hit = spans[-1]
        assert hit.attributes is not None
        assert hit.attributes[telemetry.OPTIMIZE_SHORT_CIRCUITED] is True
        assert hit.attributes[telemetry.GEN_AI_USAGE_INPUT_TOKENS] == 0


class TestOptioPricesASpanThisPackageEmits:
    """The actual proof ADR-014 exists to deliver.

    No optio code is touched anywhere in this test. optio's own span tap
    (installed the standard way, via @meter) observes the span
    optio_optimize.telemetry emits and prices it -- because it looks exactly
    like any framework adapter's span to a processor that was designed to
    consume any GenAI span, regardless of what produced it.
    """

    def test_a_real_call_is_priced_by_optios_cost_lane(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> str:
            optimizer = Optimizer(
                emit_spans=True,
                exact_cache=False,
                prefix_cache=False,
                tracer_provider=provider,
            )
            response = optimizer.call(
                _request(),
                lambda r: _response(model=r.model, input_tokens=1_000_000, output_tokens=0),
            )
            return response.content

        assert run_agent() == "hello"

        run_span = next(s for s in exporter.get_finished_spans() if s.name.startswith("optio.run"))
        attrs = run_span.attributes
        assert attrs is not None
        # $2.50 per million input tokens on gpt-4o, priced by optio -- not
        # asserted, computed by CostLane from the span this test never
        # touched directly.
        assert attrs[semconv.RUN_ACTUAL_COST] == pytest.approx(2.50)

    def test_a_cache_hit_is_priced_at_zero_by_optios_own_ledger(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        from optio import meter, semconv

        @meter(provider=provider)
        def run_agent() -> None:
            optimizer = Optimizer(emit_spans=True, tracer_provider=provider)
            request = _request()

            def call(r: LLMRequest) -> LLMResponse:
                return _response(model=r.model, input_tokens=1_000_000, output_tokens=0)

            optimizer.call(request, call)  # priced normally
            optimizer.call(request, call)  # served from cache: must add $0

        run_agent()

        run_span = next(s for s in exporter.get_finished_spans() if s.name.startswith("optio.run"))
        attrs = run_span.attributes
        assert attrs is not None
        # One priced call at $2.50, one cache hit contributing nothing --
        # not 2x $2.50, which is what a span carrying stale/reused token
        # counts would have produced.
        assert attrs[semconv.RUN_ACTUAL_COST] == pytest.approx(2.50)

    def test_run_id_reaches_the_span_optio_never_needed_it_for_pricing(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Correlation is ambient (OTel context), not run_id string matching.

        Pricing already worked in the tests above with run_id never passed.
        This only confirms the debugging convenience actually lands.
        """
        from optio import current_run, meter

        @meter(provider=provider)
        def run_agent() -> None:
            run = current_run()
            assert run is not None
            optimizer = Optimizer(emit_spans=True, tracer_provider=provider)
            optimizer.call(
                _request(),
                lambda r: _response(model=r.model),
                run_id=run.run_id,
            )

        run_agent()

        step_span = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("chat ")
        )
        assert step_span.attributes is not None
        assert step_span.attributes[telemetry.GEN_AI_RUN_ID]
