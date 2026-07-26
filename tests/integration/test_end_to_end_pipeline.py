"""End-to-end pipeline: instrument, run, observe (M1 gate).

The M1 exit criterion is that a real run produces spans that reach the lanes.
This exercises the whole path -- ``instrument()`` to adapter to installer to tap
to lane to signal writer -- rather than any single component, because every bug
found in M1 lived in the seams between them rather than inside them.

A stub lane stands in for the real ones, which land in M2, M3 and M5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentmeter import instrument, meter, semconv
from agentmeter.config import default_config
from agentmeter.lanes.base import Lane, Signal
from agentmeter.runtime import failopen, installer
from agentmeter.runtime.run_context import RunContext, current_run
from agentmeter.runtime.span_tap import AgentMeterSpanTap

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    """Reset guard counters and tap installs between tests."""
    failopen.reset_activations()
    installer.reset_installations()
    yield
    failopen.reset_activations()
    installer.reset_installations()


class _StubCostLane(Lane):
    """Stands in for the M2 cost lane: counts steps, emits a running total."""

    name = "stub_cost"

    def __init__(self) -> None:
        """Start with an empty ledger."""
        super().__init__(default_config())
        self.total = 0.0
        self.steps = 0
        self.run_ids: list[str] = []

    def process_span(self, span: ReadableSpan, run: object) -> list[Signal]:
        """Charge a flat rate per step and report the running total."""
        self.steps += 1
        self.total += 0.01
        run_id = getattr(run, "run_id", None)
        if run_id is not None:
            self.run_ids.append(run_id)
        return [Signal(semconv.RUN_ACTUAL_COST, self.total)]

    def on_run_end(self, run: object) -> list[Signal]:
        """Report the run as successful."""
        return [Signal(semconv.RUN_SUCCESS, True)]


class _FakeGraph:
    """A stand-in for a compiled LangGraph graph.

    Its ``invoke`` emits GenAI spans the way an instrumented LangChain would, so
    the pipeline sees realistic input without depending on LangGraph itself.
    """

    def __init__(self, tracer: object, steps: int = 3) -> None:
        self._tracer = tracer
        self._steps = steps

    def invoke(self, prompt: str) -> str:
        """Run the agent, emitting one GenAI span per step."""
        for i in range(self._steps):
            with self._tracer.start_as_current_span(f"llm-{i}") as span:  # type: ignore[attr-defined]
                span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 100)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 50)
        return prompt.upper()

    def stream(self, prompt: str) -> Iterator[str]:
        """Present the graph interface; unused here."""
        yield prompt


_FakeGraph.__module__ = "langgraph.graph.state"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """An in-memory exporter capturing finished spans."""
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    """A tracer provider wired to the in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


class TestOneLineInstall:
    def test_instrument_returns_the_same_graph(self, provider: TracerProvider) -> None:
        graph = _FakeGraph(provider.get_tracer("langgraph"))
        assert instrument(graph, provider=provider) is graph

    def test_instrument_installs_the_tap(self, provider: TracerProvider) -> None:
        graph = _FakeGraph(provider.get_tracer("langgraph"))
        instrument(graph, provider=provider)

        assert installer.installed_tap(provider) is not None

    def test_instrumenting_twice_installs_one_tap(self, provider: TracerProvider) -> None:
        # Two taps means every span dispatched twice, i.e. doubled cost in M2.
        graph = _FakeGraph(provider.get_tracer("langgraph"))
        instrument(graph, provider=provider)
        instrument(graph, provider=provider)

        tap = installer.installed_tap(provider)
        assert tap is not None
        assert sum(1 for _ in _taps_on(provider)) == 1


def _taps_on(provider: TracerProvider) -> Iterator[AgentMeterSpanTap]:
    """Yield every agentmeter tap registered on a provider."""
    processors = provider._active_span_processor._span_processors
    for processor in processors:
        if isinstance(processor, AgentMeterSpanTap):
            yield processor


class TestSpansReachTheLanes:
    def test_every_step_span_reaches_the_lane(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        lane = _StubCostLane()
        tap = AgentMeterSpanTap(default_config(), [lane])
        provider.add_span_processor(tap)
        tracer = provider.get_tracer("langgraph")
        graph = _FakeGraph(tracer, steps=3)

        with RunContext().start(), tracer.start_as_current_span("agent-run"):
            graph.invoke("hello")

        assert lane.steps == 3

    def test_signals_land_on_the_run_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        lane = _StubCostLane()
        provider.add_span_processor(AgentMeterSpanTap(default_config(), [lane]))
        tracer = provider.get_tracer("langgraph")
        graph = _FakeGraph(tracer, steps=3)

        with RunContext().start(), tracer.start_as_current_span("agent-run"):
            graph.invoke("hello")

        run_span = next(s for s in exporter.get_finished_spans() if s.name == "agent-run")
        assert run_span.attributes is not None
        # Three steps at 0.01 each, the last write winning.
        assert run_span.attributes[semconv.RUN_ACTUAL_COST] == pytest.approx(0.03)

    def test_all_steps_share_one_run_id(self, provider: TracerProvider) -> None:
        # run_id is the join key across lanes and the store; if it drifted
        # mid-run, per-run totals would silently split in two.
        lane = _StubCostLane()
        provider.add_span_processor(AgentMeterSpanTap(default_config(), [lane]))
        tracer = provider.get_tracer("langgraph")
        graph = _FakeGraph(tracer, steps=4)

        with RunContext() as run, tracer.start_as_current_span("agent-run"):
            graph.invoke("hello")

        assert lane.run_ids == [run.run_id] * 4

    def test_the_agents_own_spans_are_ignored(self, provider: TracerProvider) -> None:
        lane = _StubCostLane()
        provider.add_span_processor(AgentMeterSpanTap(default_config(), [lane]))
        tracer = provider.get_tracer("app")

        with (
            RunContext().start(),
            tracer.start_as_current_span("agent-run"),
            tracer.start_as_current_span("db-query") as span,
        ):
            span.set_attribute("db.system", "postgresql")

        assert lane.steps == 0


class TestMeterDecorator:
    def test_decorated_function_runs_and_returns_normally(self, provider: TracerProvider) -> None:
        @meter(budget="$0.50", provider=provider)
        def run_agent(prompt: str) -> str:
            return prompt.upper()

        assert run_agent("hello") == "HELLO"

    def test_decorated_function_opens_a_run(self, provider: TracerProvider) -> None:
        seen: list[str] = []

        @meter(provider=provider)
        def run_agent() -> None:
            run = current_run()
            assert run is not None
            seen.append(run.run_id)

        run_agent()
        run_agent()

        assert len(seen) == 2
        assert seen[0] != seen[1]

    def test_run_span_is_exported(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        @meter(provider=provider)
        def run_agent() -> None:
            return None

        run_agent()

        names = [s.name for s in exporter.get_finished_spans()]
        assert "agentmeter.run.run_agent" in names

    def test_budget_is_attached_to_the_run(self, provider: TracerProvider) -> None:
        captured: list[float | None] = []

        @meter(budget="$0.50", provider=provider)
        def run_agent() -> None:
            run = current_run()
            assert run is not None
            captured.append(run.budget.limit_usd if run.budget else None)

        run_agent()

        assert captured == [pytest.approx(0.50)]

    def test_an_agent_exception_propagates(self, provider: TracerProvider) -> None:
        # agentmeter observes runs; it must never swallow the user's errors.
        @meter(provider=provider)
        def failing_agent() -> None:
            raise ValueError("agent failed")

        with pytest.raises(ValueError, match="agent failed"):
            failing_agent()

        assert current_run() is None


class TestFailOpenEndToEnd:
    def test_a_broken_lane_does_not_break_the_run(self, provider: TracerProvider) -> None:
        class _BrokenLane(Lane):
            name = "broken"

            def __init__(self) -> None:
                super().__init__(default_config())

            def process_span(self, span: ReadableSpan, run: object) -> list[Signal]:
                raise RuntimeError("lane bug")

        provider.add_span_processor(AgentMeterSpanTap(default_config(), [_BrokenLane()]))
        tracer = provider.get_tracer("langgraph")
        graph = _FakeGraph(tracer, steps=3)

        with RunContext().start(), tracer.start_as_current_span("agent-run"):
            result = graph.invoke("hello")

        assert result == "HELLO"
        assert failopen.activation_count("broken") == 3
