"""The single signal write path (M1-5).

Two properties carry the weight here:

* Names are validated against the frozen ``semconv`` contract, because a name
  that escapes review becomes a contract with strangers' policies.
* Absence is preserved. Coercing an uncomputable value to zero would be the
  worst possible failure: a policy cannot distinguish it from a real zero, so a
  broken lane would read as a run that cost nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentmeter import semconv
from agentmeter.lanes.base import Signal
from agentmeter.runtime import failopen
from agentmeter.runtime.signal_writer import write_signal, write_signals

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clean_activations() -> Iterator[None]:
    """Isolate guard activation counters between tests."""
    failopen.reset_activations()
    yield
    failopen.reset_activations()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """An in-memory exporter capturing finished spans."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> trace.Tracer:
    """A tracer wired to the in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


class TestWriting:
    def test_declared_signal_lands_on_the_span(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter
    ) -> None:
        with tracer.start_as_current_span("run") as span:
            written = write_signal(span, Signal(semconv.RUN_ACTUAL_COST, 0.42))

        assert written is True
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert finished.attributes[semconv.RUN_ACTUAL_COST] == pytest.approx(0.42)

    def test_batch_writes_every_valid_signal(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter
    ) -> None:
        signals = [
            Signal(semconv.RUN_ACTUAL_COST, 0.10),
            Signal(semconv.RUN_PROJECTED_COST, 0.25),
            Signal(semconv.RUN_LOOP_STATE, semconv.LOOP_STATE_HEALTHY),
        ]

        with tracer.start_as_current_span("run") as span:
            count = write_signals(span, signals)

        assert count == 3
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert finished.attributes[semconv.RUN_LOOP_STATE] == semconv.LOOP_STATE_HEALTHY

    def test_one_bad_signal_does_not_discard_its_neighbours(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter
    ) -> None:
        signals = [
            Signal(semconv.RUN_ACTUAL_COST, 0.10),
            Signal(semconv.RUN_PROJECTED_COST, float("nan")),
            Signal(semconv.RUN_BUDGET_REMAINING, 0.90),
        ]

        with tracer.start_as_current_span("run") as span:
            count = write_signals(span, signals)

        assert count == 2
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert semconv.RUN_PROJECTED_COST not in finished.attributes

    def test_boolean_signals_are_written(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter
    ) -> None:
        # bool subclasses int, so it must be handled before the finite check.
        with tracer.start_as_current_span("run") as span:
            assert write_signal(span, Signal(semconv.RUN_SUCCESS, True)) is True

        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert finished.attributes[semconv.RUN_SUCCESS] is True


class TestAbsenceIsPreserved:
    """Uncomputable values are omitted, never coerced to a plausible number."""

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_non_finite_values_are_omitted(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter, value: float
    ) -> None:
        # cost / 0 successful tasks produces these. Exporting one would poison a
        # downstream average silently; omitting says "unknown", which is true.
        with tracer.start_as_current_span("run") as span:
            written = write_signal(span, Signal(semconv.RUN_ACTUAL_COST, value))

        assert written is False
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert semconv.RUN_ACTUAL_COST not in finished.attributes

    def test_none_is_omitted_not_written_as_null(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter
    ) -> None:
        # A lane returning None means "I could not compute this". Writing a null
        # would make it indistinguishable from a computed absence downstream.
        with tracer.start_as_current_span("run") as span:
            written = write_signal(
                span,
                Signal(semconv.RUN_ACTUAL_COST, None),  # type: ignore[arg-type]
            )

        assert written is False
        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert semconv.RUN_ACTUAL_COST not in finished.attributes

    def test_zero_is_still_written(
        self, tracer: trace.Tracer, exporter: InMemorySpanExporter
    ) -> None:
        # A real zero must survive: the omission rule exists to keep "unknown"
        # and "zero" distinguishable, which fails if zero is dropped too.
        with tracer.start_as_current_span("run") as span:
            assert write_signal(span, Signal(semconv.RUN_ACTUAL_COST, 0.0)) is True

        (finished,) = exporter.get_finished_spans()
        assert finished.attributes is not None
        assert semconv.RUN_ACTUAL_COST in finished.attributes


class TestContractEnforcement:
    def test_undeclared_names_are_refused(self, tracer: trace.Tracer) -> None:
        # A name outside semconv means a lane bypassed the contract. It must not
        # reach a consumer, but it also must not break the agent -- so the guard
        # absorbs it and the activation counter records the lane bug.
        with tracer.start_as_current_span("run") as span:
            written = write_signal(span, Signal("gen_ai.run.invented_metric", 1.0))

        assert written is False
        assert failopen.activation_count("signal_writer") == 1

    def test_invalid_loop_state_is_refused(self, tracer: trace.Tracer) -> None:
        # An undeclared enum value matches nobody's policy and would read as
        # healthy to a consumer checking `!= "looping"`.
        with tracer.start_as_current_span("run") as span:
            written = write_signal(span, Signal(semconv.RUN_LOOP_STATE, "on_fire"))

        assert written is False
        assert failopen.activation_count("signal_writer") == 1

    def test_every_declared_loop_state_is_accepted(self, tracer: trace.Tracer) -> None:
        with tracer.start_as_current_span("run") as span:
            for state in semconv.LOOP_STATES:
                assert write_signal(span, Signal(semconv.RUN_LOOP_STATE, state)) is True

        assert failopen.activation_count("signal_writer") == 0


class TestNeverRaises:
    """Writing sits on the agent's critical path (ADR-004)."""

    def test_no_span_is_a_no_op(self) -> None:
        assert write_signal(None, Signal(semconv.RUN_ACTUAL_COST, 1.0)) is False

    def test_non_recording_span_is_a_no_op(self) -> None:
        # Outside a trace, or when sampled out, there is nowhere to put the
        # value. Normal, not an error -- so it must not count as a failure.
        assert write_signal(trace.INVALID_SPAN, Signal(semconv.RUN_ACTUAL_COST, 1.0)) is False
        assert failopen.activation_count("signal_writer") == 0

    def test_a_span_that_raises_does_not_break_the_agent(self) -> None:
        class ExplodingSpan:
            def is_recording(self) -> bool:
                return True

            def set_attribute(self, key: str, value: object) -> None:
                raise RuntimeError("exporter is broken")

        written = write_signal(
            ExplodingSpan(),  # type: ignore[arg-type]
            Signal(semconv.RUN_ACTUAL_COST, 1.0),
        )

        assert written is False
        assert failopen.activation_count("signal_writer") == 1

    def test_a_failing_signal_iterable_does_not_break_the_agent(self, tracer: trace.Tracer) -> None:
        def exploding_signals() -> Iterator[Signal]:
            yield Signal(semconv.RUN_ACTUAL_COST, 0.1)
            raise RuntimeError("lane generator blew up")

        with tracer.start_as_current_span("run") as span:
            count = write_signals(span, exploding_signals())

        # The batch is abandoned, the agent proceeds, and the bug is counted.
        assert count == 0
        assert failopen.activation_count("signal_writer") == 1
