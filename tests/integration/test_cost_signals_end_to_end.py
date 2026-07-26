"""Cost signals from a real instrumented run (M2-4 gate).

The M2 exit criterion is that cost is visible in a backend within five minutes
of install (SC-1). This is that path, minus the backend: instrument, run, read
the attributes off the exported run span.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentmeter import meter, semconv
from agentmeter.config import default_config
from agentmeter.errors import LedgerInvariantError
from agentmeter.lanes.cost.lane import CostLane
from agentmeter.runtime import failopen, installer
from agentmeter.runtime.run_context import current_run

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


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """An in-memory exporter."""
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    """A tracer provider wired to the exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _run_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    """Return the agentmeter run span."""
    return next(s for s in exporter.get_finished_spans() if s.name.startswith("agentmeter.run"))


class TestCostReachesTheSpan:
    def test_a_metered_run_reports_its_cost(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("agent")

        @meter(provider=provider)
        def run_agent() -> str:
            for _ in range(3):
                with tracer.start_as_current_span("llm") as span:
                    span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                    span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)
                    span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 0)
            return "done"

        assert run_agent() == "done"

        attributes = _run_span(exporter).attributes
        assert attributes is not None
        # Three steps at $2.50 per million input tokens.
        assert attributes[semconv.RUN_ACTUAL_COST] == pytest.approx(7.50)

    def test_budget_remaining_reaches_the_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("agent")

        @meter(budget="$10.00", provider=provider)
        def run_agent() -> None:
            with tracer.start_as_current_span("llm") as span:
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 0)

        run_agent()

        attributes = _run_span(exporter).attributes
        assert attributes is not None
        assert attributes[semconv.RUN_BUDGET_REMAINING] == pytest.approx(7.50)

    def test_only_declared_signal_names_are_emitted(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The names are a hard contract with strangers' policies (ADR-001), so
        # nothing outside the frozen set may reach a span.
        tracer = provider.get_tracer("agent")

        @meter(budget="$10.00", provider=provider)
        def run_agent() -> None:
            with tracer.start_as_current_span("llm") as span:
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1000)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 500)

        run_agent()

        attributes = _run_span(exporter).attributes or {}
        emitted = {key for key in attributes if key.startswith(f"{semconv.GENAI_NAMESPACE}.")}
        assert emitted <= semconv.EMITTED_SIGNALS

    def test_an_unpriceable_model_emits_no_cost(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # Absent, not zero. A policy reading zero would think the run was free.
        tracer = provider.get_tracer("agent")

        @meter(provider=provider)
        def run_agent() -> None:
            with tracer.start_as_current_span("llm") as span:
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "not-a-real-model")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 0)

        run_agent()

        attributes = _run_span(exporter).attributes or {}
        assert semconv.RUN_ACTUAL_COST not in attributes


class TestRunEndFires:
    def test_run_end_closes_the_ledger(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # Run end is driven by RunContext, not the OTel SDK, so this verifies
        # the observer wiring actually fires.
        tracer = provider.get_tracer("agent")
        seen_run_ids: list[str] = []

        @meter(provider=provider)
        def run_agent() -> None:
            run = current_run()
            assert run is not None
            seen_run_ids.append(run.run_id)
            with tracer.start_as_current_span("llm") as span:
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 0)

        run_agent()
        (run_id,) = seen_run_ids

        tap = installer.installed_tap(provider)
        assert tap is not None
        cost_lane = next(lane for lane in tap.lanes if isinstance(lane, CostLane))

        # State is released at run end. Agents are long-lived processes, so
        # retaining every run seen would be an unbounded leak (~368 bytes each).
        assert cost_lane.ledger.run_count() == 0

        # Finality outlives eviction: a straggling callback must not start a new
        # total under a run id whose cost was already reported (ADR-010).
        with pytest.raises(LedgerInvariantError, match="closed run"):
            cost_lane.ledger.reserve(run_id, "late-step", 1.0)

        # A genuinely new run is unaffected.
        cost_lane.ledger.reserve("a-different-run", "s", 1.0)

    def test_separate_runs_report_separate_costs(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        tracer = provider.get_tracer("agent")

        @meter(provider=provider)
        def run_agent(steps: int) -> None:
            for _ in range(steps):
                with tracer.start_as_current_span("llm") as span:
                    span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                    span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)
                    span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 0)

        run_agent(1)
        run_agent(2)

        run_spans = [
            s for s in exporter.get_finished_spans() if s.name.startswith("agentmeter.run")
        ]
        costs = sorted(
            float(s.attributes[semconv.RUN_ACTUAL_COST])  # type: ignore[index,arg-type]
            for s in run_spans
        )
        assert costs == [pytest.approx(2.50), pytest.approx(5.00)]


class TestFailOpenStillHolds:
    def test_a_broken_pricing_provider_does_not_break_the_agent(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The cost lane is now on the hot path for real; ADR-004 has to survive
        # that.
        tracer = provider.get_tracer("agent")

        tap = installer.install_tap(default_config(), provider)
        assert tap is not None

        class ExplodingProvider:
            def price_for(self, model: str) -> None:
                raise RuntimeError("pricing blew up")

        for lane in tap.lanes:
            if lane.name == "cost":
                lane.pricing = ExplodingProvider()  # type: ignore[attr-defined]

        @meter(provider=provider)
        def run_agent() -> str:
            with tracer.start_as_current_span("llm") as span:
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1000)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 500)
            return "agent still works"

        assert run_agent() == "agent still works"
        assert failopen.activation_count("cost") >= 1
