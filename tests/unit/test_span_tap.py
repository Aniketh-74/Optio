"""Span ingestion and lane dispatch (M1-3).

The acceptance criteria: only ``gen_ai.*`` spans are processed, unknown spans
are ignored, and all dispatch is wrapped by fail-open.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentmeter import semconv
from agentmeter.config import Config, default_config
from agentmeter.lanes.base import Lane, Signal
from agentmeter.runtime import failopen
from agentmeter.runtime.run_context import RunContext
from agentmeter.runtime.span_tap import AgentMeterSpanTap, is_genai_span

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan


@pytest.fixture(autouse=True)
def _clean_activations() -> Iterator[None]:
    """Isolate guard activation counters between tests."""
    failopen.reset_activations()
    yield
    failopen.reset_activations()


class _RecordingLane(Lane):
    """A lane that records what it was handed and emits one signal."""

    name = "recording"

    def __init__(self, config: Config | None = None) -> None:
        """Start with empty call logs."""
        super().__init__(config or default_config())
        self.spans: list[ReadableSpan] = []
        self.run_ends: int = 0

    def process_span(self, span: ReadableSpan, run: object) -> list[Signal]:
        """Record the span and emit a cost signal."""
        self.spans.append(span)
        return [Signal(semconv.RUN_ACTUAL_COST, 0.25)]

    def on_run_end(self, run: object) -> list[Signal]:
        """Record the run end and emit a success signal."""
        self.run_ends += 1
        return [Signal(semconv.RUN_SUCCESS, True)]


class _BrokenLane(Lane):
    """A lane that raises on every call."""

    name = "broken"

    def __init__(self) -> None:
        """Build with default config."""
        super().__init__(default_config())

    def process_span(self, span: ReadableSpan, run: object) -> list[Signal]:
        """Raise unconditionally."""
        raise RuntimeError("lane bug")

    def on_run_end(self, run: object) -> list[Signal]:
        """Raise unconditionally."""
        raise RuntimeError("lane bug at run end")


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """An in-memory exporter capturing finished spans."""
    return InMemorySpanExporter()


def _provider(exporter: InMemorySpanExporter, tap: AgentMeterSpanTap) -> TracerProvider:
    """Build a provider with the tap installed ahead of the exporter."""
    provider = TracerProvider()
    provider.add_span_processor(tap)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


class TestSpanClassification:
    def test_genai_span_is_recognised(self, exporter: InMemorySpanExporter) -> None:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("llm") as span:
            span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        (finished,) = exporter.get_finished_spans()
        assert is_genai_span(finished) is True

    def test_non_genai_span_is_rejected(self, exporter: InMemorySpanExporter) -> None:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("http") as span:
            span.set_attribute("http.method", "GET")

        (finished,) = exporter.get_finished_spans()
        assert is_genai_span(finished) is False

    def test_a_span_carrying_only_our_own_signals_is_not_input(
        self, exporter: InMemorySpanExporter
    ) -> None:
        # Writing signals to the run span makes it match gen_ai.*, so it comes
        # back through the tap when it ends. Treating it as input would let a
        # run's own cost be re-ingested as a fresh step and double counted
        # (R-TECH-1) -- silent, and in the direction that makes the core number
        # wrong.
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("run") as span:
            span.set_attribute(semconv.RUN_ACTUAL_COST, 0.25)

        (finished,) = exporter.get_finished_spans()
        assert is_genai_span(finished) is False

    def test_a_span_with_both_input_and_our_signals_is_still_input(
        self, exporter: InMemorySpanExporter
    ) -> None:
        # A framework span we annotated in place must not become invisible.
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("llm") as span:
            span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
            span.set_attribute(semconv.RUN_ACTUAL_COST, 0.25)

        (finished,) = exporter.get_finished_spans()
        assert is_genai_span(finished) is True

    def test_span_without_attributes_is_rejected(self, exporter: InMemorySpanExporter) -> None:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("bare"):
            pass

        (finished,) = exporter.get_finished_spans()
        assert is_genai_span(finished) is False


class TestDispatch:
    def test_genai_span_reaches_the_lane(self, exporter: InMemorySpanExporter) -> None:
        lane = _RecordingLane()
        tracer = _provider(exporter, AgentMeterSpanTap(default_config(), [lane])).get_tracer("test")

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("llm") as step,
        ):
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        assert len(lane.spans) == 1

    def test_non_genai_spans_never_reach_the_lane(self, exporter: InMemorySpanExporter) -> None:
        lane = _RecordingLane()
        tracer = _provider(exporter, AgentMeterSpanTap(default_config(), [lane])).get_tracer("test")

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("http") as step,
        ):
            step.set_attribute("http.method", "GET")

        assert lane.spans == []

    def test_span_outside_a_run_is_dropped(self, exporter: InMemorySpanExporter) -> None:
        # Signals are run-scoped, so a GenAI span with no active run has nowhere
        # to be attributed.
        lane = _RecordingLane()
        tracer = _provider(exporter, AgentMeterSpanTap(default_config(), [lane])).get_tracer("test")

        with tracer.start_as_current_span("llm") as step:
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        assert lane.spans == []

    def test_every_lane_receives_the_span(self, exporter: InMemorySpanExporter) -> None:
        first, second = _RecordingLane(), _RecordingLane()
        tap = AgentMeterSpanTap(default_config(), [first, second])
        tracer = _provider(exporter, tap).get_tracer("test")

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("llm") as step,
        ):
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        assert len(first.spans) == 1
        assert len(second.spans) == 1

    def test_no_lanes_is_a_cheap_no_op(self, exporter: InMemorySpanExporter) -> None:
        tap = AgentMeterSpanTap(default_config(), [])
        tracer = _provider(exporter, tap).get_tracer("test")

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("llm") as step,
        ):
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        assert failopen.activation_count() == 0


class TestSignalsLandOnTheRunSpan:
    """A processor cannot annotate the span it observes; see the module docstring."""

    def test_lane_signals_reach_the_enclosing_run_span(
        self, exporter: InMemorySpanExporter
    ) -> None:
        lane = _RecordingLane()
        tracer = _provider(exporter, AgentMeterSpanTap(default_config(), [lane])).get_tracer("test")

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("llm") as step,
        ):
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        run_span = next(s for s in exporter.get_finished_spans() if s.name == "run")
        assert run_span.attributes is not None
        assert run_span.attributes[semconv.RUN_ACTUAL_COST] == pytest.approx(0.25)

    def test_an_ended_span_cannot_be_annotated(self, exporter: InMemorySpanExporter) -> None:
        # Locks in the SDK behaviour the design depends on: by the time a
        # processor sees a span it is ended, and `on_end` hands out a
        # ReadableSpan with no `set_attribute` at all. If a future SDK version
        # changes this, writing signals back onto the step span becomes possible
        # and this test is the prompt to reconsider.
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("llm"):
            pass

        (finished,) = exporter.get_finished_spans()
        assert not hasattr(finished, "set_attribute")


class TestFailOpen:
    def test_a_broken_lane_does_not_break_the_agent(self, exporter: InMemorySpanExporter) -> None:
        tap = AgentMeterSpanTap(default_config(), [_BrokenLane()])
        tracer = _provider(exporter, tap).get_tracer("test")
        agent_completed = False

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("llm") as step,
        ):
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
            agent_completed = True

        assert agent_completed is True
        assert failopen.activation_count("broken") == 1

    def test_a_working_lane_still_runs_when_another_is_broken(
        self, exporter: InMemorySpanExporter
    ) -> None:
        # Lane independence (§3.1 rule 11) has to hold at dispatch time too.
        working = _RecordingLane()
        tap = AgentMeterSpanTap(default_config(), [_BrokenLane(), working])
        tracer = _provider(exporter, tap).get_tracer("test")

        with (
            RunContext().start(),
            tracer.start_as_current_span("run"),
            tracer.start_as_current_span("llm") as step,
        ):
            step.set_attribute(semconv.GEN_AI_SYSTEM, "openai")

        assert len(working.spans) == 1

    def test_a_span_that_cannot_be_classified_is_contained(self) -> None:
        class ExplodingSpan:
            @property
            def attributes(self) -> dict[str, str]:
                raise RuntimeError("attributes blew up")

        tap = AgentMeterSpanTap(default_config(), [_RecordingLane()])
        tap.on_end(ExplodingSpan())  # type: ignore[arg-type]

        assert failopen.activation_count("span_tap") == 1


class TestRunEnd:
    def test_run_end_signals_are_collected(self, exporter: InMemorySpanExporter) -> None:
        lane = _RecordingLane()
        tap = AgentMeterSpanTap(default_config(), [lane])
        tracer = _provider(exporter, tap).get_tracer("test")
        run = RunContext()

        with run.start(), tracer.start_as_current_span("run"):
            tap.on_run_end(run)

        assert lane.run_ends == 1
        run_span = next(s for s in exporter.get_finished_spans() if s.name == "run")
        assert run_span.attributes is not None
        assert run_span.attributes[semconv.RUN_SUCCESS] is True

    def test_a_broken_lane_at_run_end_is_contained(self) -> None:
        tap = AgentMeterSpanTap(default_config(), [_BrokenLane()])
        run = RunContext().start()

        tap.on_run_end(run)

        assert failopen.activation_count("broken") == 1


class TestProcessorProtocol:
    def test_shutdown_and_flush_are_safe(self) -> None:
        tap = AgentMeterSpanTap(default_config(), [])
        tap.shutdown()
        assert tap.force_flush() is True

    def test_on_start_is_a_no_op(self, exporter: InMemorySpanExporter) -> None:
        lane = _RecordingLane()
        tap = AgentMeterSpanTap(default_config(), [lane])
        tracer = _provider(exporter, tap).get_tracer("test")

        # A GenAI span at start time: no usage attributes exist yet, so nothing
        # is dispatched until it ends.
        with RunContext().start(), tracer.start_as_current_span("llm") as span:
            span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
            assert lane.spans == []

        assert failopen.activation_count() == 0

    def test_lanes_are_resolved_from_config_when_omitted(self) -> None:
        # M0/M1 wire no concrete lanes yet; the point is that the tap asks
        # `enabled_lanes` rather than importing lanes itself (§3.1).
        tap = AgentMeterSpanTap(default_config())
        assert tap.lanes == []
