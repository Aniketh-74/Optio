"""Behavior signals from a real instrumented run (M3-3 gate).

Runs through the real OTel SDK, the real span tap, and the real signal writer.
That last one matters most: the writer validates `loop_state` against
`semconv.LOOP_STATES` and silently drops anything off-contract. A detector
returning a value the writer rejects would pass every unit test in
`tests/unit/` and emit nothing at all in production.

Also verified here: cost and behavior run side by side without interfering. The
two lanes are independent by design (Section 3.1) and this is the only place
that independence is exercised rather than asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from optio import meter, semconv
from optio.config import default_config
from optio.runtime import failopen, installer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    failopen.reset_activations()
    installer.reset_installations()
    yield
    failopen.reset_activations()
    installer.reset_installations()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _run_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    return next(s for s in exporter.get_finished_spans() if s.name.startswith("optio.run"))


def _tool_call(tracer: object, tool: str, *, errored: bool = False) -> None:
    """Emit one GenAI tool span."""
    with tracer.start_as_current_span("tool") as span:  # type: ignore[attr-defined]
        span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
        span.set_attribute(semconv.GEN_AI_TOOL_NAME, tool)
        span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
        span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 100)
        span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 50)
        if errored:
            span.set_status(Status(StatusCode.ERROR))


class TestSignalsReachTheSpan:
    def test_a_stuck_agent_is_reported_as_looping(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def stuck_agent() -> None:
            for _ in range(10):
                _tool_call(tracer, "search")

        stuck_agent()

        attributes = _run_span(exporter).attributes or {}
        assert attributes[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_LOOPING
        assert attributes[semconv.RUN_REPEAT_COUNT] == 10

    def test_a_healthy_agent_is_reported_as_healthy(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def working_agent() -> None:
            for n in range(10):
                _tool_call(tracer, f"tool_{n}")

        working_agent()

        attributes = _run_span(exporter).attributes or {}
        assert attributes[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_HEALTHY

    def test_a_failing_dependency_is_reported_as_a_retry_storm(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def blocked_agent() -> None:
            for _ in range(6):
                _tool_call(tracer, "call_api", errored=True)
            _tool_call(tracer, "search")

        blocked_agent()

        attributes = _run_span(exporter).attributes or {}
        assert attributes[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_RETRY_STORM

    def test_the_emitted_state_survives_writer_validation(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The writer drops an off-contract loop_state silently. Asserting the
        # attribute is *present* is what catches that.
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def agent() -> None:
            for n in range(8):
                _tool_call(tracer, f"t{n % 3}")

        agent()

        attributes = _run_span(exporter).attributes or {}
        assert semconv.RUN_LOOP_STATE in attributes
        assert attributes[semconv.RUN_LOOP_STATE] in semconv.LOOP_STATES
        assert isinstance(attributes[semconv.RUN_REPEAT_COUNT], int)


class TestLanesCoexist:
    def test_cost_and_behavior_signals_appear_together(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def agent() -> None:
            for _ in range(6):
                _tool_call(tracer, "search")

        agent()

        attributes = _run_span(exporter).attributes or {}
        assert semconv.RUN_ACTUAL_COST in attributes
        assert semconv.RUN_LOOP_STATE in attributes

        cost = attributes[semconv.RUN_ACTUAL_COST]
        assert isinstance(cost, float) and cost > 0

    def test_behavior_signals_do_not_feed_back_as_input(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # loop_state lands on the run span, which then ends and returns through
        # the tap. If it were treated as GenAI input the run would meter itself.
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def agent() -> None:
            for _ in range(6):
                _tool_call(tracer, "search")

        agent()

        attributes = _run_span(exporter).attributes or {}
        assert attributes[semconv.RUN_REPEAT_COUNT] == 6, "an extra step was counted"

    def test_no_fail_open_activations_on_the_happy_path(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The guard makes lane bugs invisible by design. Without this, a lane
        # that raised on every span would still leave a green suite.
        tracer = provider.get_tracer("test")

        @meter(provider=provider)
        def agent() -> None:
            for n in range(10):
                _tool_call(tracer, f"t{n % 4}", errored=n % 5 == 0)

        agent()

        assert failopen.activation_count("behavior") == 0
        assert failopen.activation_count("cost") == 0


class TestNoRunNoSignals:
    def test_spans_outside_a_run_are_ignored(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        installer.install_tap(default_config(), provider=provider)
        tracer = provider.get_tracer("test")

        for _ in range(10):
            _tool_call(tracer, "search")

        for span in exporter.get_finished_spans():
            attributes = span.attributes or {}
            assert semconv.RUN_LOOP_STATE not in attributes
