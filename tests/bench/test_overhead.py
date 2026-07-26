"""Overhead benchmarks against SC-5.

    *Added wall-clock latency per governed step **< 5 ms p99** for cost+behavior
    lanes (quality lane excluded; it is opt-in/async).*

Section 11 is explicit that these are **design budgets to be verified by
benchmark, not measured results** -- so this file is what turns the claim into
evidence, and publishing the real numbers is itself a deliverable.

Measured as a *difference*: the same span workload with and without the tap
installed. Timing the tap alone would flatter it by excluding the SDK work that
happens either way.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentmeter import semconv
from agentmeter.config import BudgetPolicy, default_config
from agentmeter.runtime.run_context import RunContext
from agentmeter.runtime.span_tap import AgentMeterSpanTap

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

pytestmark = pytest.mark.bench

#: SC-5, in seconds.
BUDGET_P99_SECONDS = 0.005

#: Enough samples for a meaningful p99 without making the suite slow.
SAMPLES = 2000


def _percentile(values: list[float], pct: float) -> float:
    """Return the given percentile of a sample."""
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[index]


def _time_steps(tracer: object, steps: int) -> list[float]:
    """Emit GenAI spans, returning per-step wall-clock durations.

    Each step names a distinct tool. That is the *expensive* path for the
    behavior lane: distinct signatures fill the window with unique entries, so
    the classifier's counter does the most work it ever will. Repeating one
    tool would collapse the counter to a single key and flatter the numbers.
    """
    timings: list[float] = []
    for step in range(steps):
        start = time.perf_counter()
        with tracer.start_as_current_span("llm") as span:  # type: ignore[attr-defined]
            span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
            span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
            span.set_attribute(semconv.GEN_AI_TOOL_NAME, f"tool_{step % 64}")
            span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1500)
            span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 300)
        timings.append(time.perf_counter() - start)
    return timings


def _build(with_tap: bool) -> tuple[object, InMemorySpanExporter]:
    """Build a provider, optionally with the agentmeter tap installed."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    if with_tap:
        provider.add_span_processor(AgentMeterSpanTap(default_config()))
    return provider.get_tracer("bench"), exporter


def _measure_overhead() -> tuple[float, float, float]:
    """Return (mean, p95, p99) added seconds per step."""
    baseline_tracer, _ = _build(with_tap=False)
    metered_tracer, _ = _build(with_tap=True)

    # Warm up both paths so import and first-call costs land outside the sample.
    _time_steps(baseline_tracer, 200)
    with RunContext(budget=BudgetPolicy(limit_usd=100.0, max_steps=10_000)):
        _time_steps(metered_tracer, 200)

    baseline = _time_steps(baseline_tracer, SAMPLES)
    with RunContext(budget=BudgetPolicy(limit_usd=100.0, max_steps=10_000)):
        metered = _time_steps(metered_tracer, SAMPLES)

    mean = statistics.mean(metered) - statistics.mean(baseline)
    p95 = _percentile(metered, 0.95) - _percentile(baseline, 0.95)
    p99 = _percentile(metered, 0.99) - _percentile(baseline, 0.99)
    return mean, p95, p99


def test_lane_overhead_is_within_budget() -> None:
    """Added per-step latency stays under the SC-5 p99 budget.

    Covers cost + behavior together, which is exactly the combination SC-5
    names (the quality lane is excluded there as opt-in and async).
    """
    mean, p95, p99 = _measure_overhead()

    print(
        f"\noverhead per step -- mean {mean * 1e6:.1f}us  "
        f"p95 {p95 * 1e6:.1f}us  p99 {p99 * 1e6:.1f}us  "
        f"(budget {BUDGET_P99_SECONDS * 1e3:.0f}ms p99)"
    )

    assert p99 < BUDGET_P99_SECONDS, (
        f"p99 overhead {p99 * 1e3:.3f}ms exceeds the SC-5 budget of "
        f"{BUDGET_P99_SECONDS * 1e3:.0f}ms"
    )


def test_overhead_does_not_grow_with_run_length() -> None:
    """Per-step cost stays flat as a run accumulates steps.

    The ledger derives totals by summing open reservations on every read. That
    is deliberate -- a separately-maintained total can drift silently -- but it
    means an accidental O(n) would show up as a run that gets slower the longer
    it lasts, which no unit test would catch.
    """
    tracer, _ = _build(with_tap=True)

    with RunContext(budget=BudgetPolicy(limit_usd=1000.0, max_steps=100_000)):
        _time_steps(tracer, 200)  # warm up
        early = _time_steps(tracer, 500)
        _time_steps(tracer, 4000)  # accumulate history
        late = _time_steps(tracer, 500)

    early_p95 = _percentile(early, 0.95)
    late_p95 = _percentile(late, 0.95)

    print(
        f"\nstep p95 early {early_p95 * 1e6:.1f}us -> "
        f"late (after 4700 steps) {late_p95 * 1e6:.1f}us"
    )

    # Generous factor: this catches O(n) growth, not ordinary jitter.
    assert late_p95 < early_p95 * 5 + 0.001, (
        f"per-step cost grew from {early_p95 * 1e6:.1f}us to "
        f"{late_p95 * 1e6:.1f}us -- suspect O(n) work per step"
    )


def test_snapshot_is_flat_when_reservations_close_normally() -> None:
    """The normal case must not scale with run length.

    ``snapshot`` sums open reservations on every read. Because reservations
    close immediately in normal operation, that sum stays tiny no matter how
    long the run gets -- confirmed here so a change to the accounting cannot
    quietly turn per-step cost into O(n).
    """
    from agentmeter.lanes.cost.ledger import CostLedger

    def snapshot_cost(steps: int) -> float:
        ledger = CostLedger()
        for i in range(steps):
            ledger.reserve("r", f"s{i}", 1.0)
            ledger.reconcile("r", f"s{i}", 1.0)
        start = time.perf_counter()
        for _ in range(500):
            ledger.snapshot("r")
        return (time.perf_counter() - start) / 500

    small = snapshot_cost(100)
    large = snapshot_cost(10_000)

    print(f"\nsnapshot p50: 100 steps {small * 1e6:.2f}us -> 10k steps {large * 1e6:.2f}us")
    assert large < small * 5 + 1e-5, "snapshot scales with completed steps"


def test_snapshot_with_many_open_reservations_stays_within_budget() -> None:
    """The pathological case: every step unpriceable, so nothing closes.

    Here ``snapshot`` really is O(open reservations), which makes the run
    quadratic overall. It stays far inside SC-5, but this pins the actual number
    so a regression shows up as a failure rather than as a slow agent.
    """
    from agentmeter.lanes.cost.ledger import CostLedger

    ledger = CostLedger()
    for i in range(10_000):
        ledger.reserve("leaky", f"s{i}", 0.0)  # never reconciled

    start = time.perf_counter()
    for _ in range(200):
        ledger.snapshot("leaky")
    per_call = (time.perf_counter() - start) / 200

    print(f"\nsnapshot with 10k OPEN reservations: {per_call * 1e6:.1f}us")
    assert per_call < BUDGET_P99_SECONDS


def test_classification_is_flat_in_run_length() -> None:
    """The behavior lane must not get slower as a run gets longer.

    ``classify`` builds a Counter over the whole window on every step, so its
    cost is O(window) -- bounded by config, not by run length. An accidental
    change to O(total steps) would make a long run quadratic, and would show up
    to a user as an agent that mysteriously degrades over hours rather than as
    any test failure.
    """
    from agentmeter.config import Config
    from agentmeter.lanes.behavior.lane import BehaviorLane

    class _Run:
        run_id = "bench"
        budget = None

    lane = BehaviorLane(Config(behavior_window_size=50))
    run = _Run()

    def step_cost(after: int) -> float:
        for n in range(after):
            lane.process_span(_span_stub(f"tool_{n % 8}"), run)
        start = time.perf_counter()
        for n in range(500):
            lane.process_span(_span_stub(f"tool_{n % 8}"), run)
        return (time.perf_counter() - start) / 500

    early = step_cost(200)
    late = step_cost(10_000)

    print(f"\nbehavior step: early {early * 1e6:.2f}us -> after 10k steps {late * 1e6:.2f}us")
    assert late < early * 5 + 1e-5, "behavior lane cost grows with run length"
    assert late < BUDGET_P99_SECONDS


def _span_stub(tool: str) -> ReadableSpan:
    """A minimal span-like object for lane-level timing."""
    from unittest.mock import Mock

    span = Mock()
    span.name = "tool"
    span.attributes = {
        semconv.GEN_AI_TOOL_NAME: tool,
        semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
        semconv.GEN_AI_USAGE_INPUT_TOKENS: 1500,
    }
    span.status = None
    return cast("ReadableSpan", span)


def test_disabled_lanes_cost_almost_nothing() -> None:
    """With every lane off, the tap should be close to free."""
    from agentmeter.config import Config

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(AgentMeterSpanTap(Config(cost_lane=False, behavior_lane=False)))
    tracer = provider.get_tracer("bench")

    baseline_tracer, _ = _build(with_tap=False)
    _time_steps(baseline_tracer, 200)
    _time_steps(tracer, 200)

    baseline = _percentile(_time_steps(baseline_tracer, SAMPLES), 0.99)
    disabled = _percentile(_time_steps(tracer, SAMPLES), 0.99)
    overhead = disabled - baseline

    print(f"\ndisabled-lane p99 overhead {overhead * 1e6:.1f}us")
    assert overhead < BUDGET_P99_SECONDS
