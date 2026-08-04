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
from typing import TYPE_CHECKING, Any, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio import semconv
from optio.config import BudgetPolicy, default_config
from optio.runtime.run_context import RunContext
from optio.runtime.span_tap import OptioSpanTap

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

    Cycles through 64 tool names rather than repeating one. Per-step cost in the
    behavior lane tracks how many *distinct* calls the window holds, so a single
    repeated tool would collapse the counts to one key and flatter the numbers;
    64 is a realistic upper end for an agent's toolset. The extreme -- every step
    unique -- is measured separately by
    ``test_the_cost_driver_is_call_diversity_not_window_size``.
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
    """Build a provider, optionally with the optio tap installed."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    if with_tap:
        provider.add_span_processor(OptioSpanTap(default_config()))
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
    from optio.lanes.cost.ledger import CostLedger

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
    from optio.lanes.cost.ledger import CostLedger

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

    Per-step cost is bounded by ``behavior_window_size``, not by run length. An
    accidental change to O(total steps) would make a long run quadratic, and
    would show up to a user as an agent that mysteriously degrades over hours
    rather than as any test failure.
    """
    from optio.config import Config
    from optio.lanes.behavior.lane import BehaviorLane

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


#: Window sizes the flatness claim is published at. The README states
#: "37 us at 50, 38 us at 1000, flat in window" as a *measured* number, so a
#: change that made it grow would falsify a documented figure rather than merely
#: slow something down.
_WINDOWS = (50, 200, 1000)

#: Distinct call identities held constant while the window varies.
#:
#: This separation is the point of the two tests below. Per-step cost is driven
#: by how many *distinct* calls a window holds, not by how many entries it can
#: hold -- both backends must scan the live counts to find the top k. Letting
#: every step name a new tool makes ``distinct == window`` and conflates the two
#: axes, which is how a first version of this test "found" a regression that was
#: really the workload.
_DISTINCT_TOOLS = 64

#: Ratio allowed between the widest and narrowest window. Generous, because this
#: asserts a *shape* -- flat rather than linear -- on a machine also running the
#: rest of the suite. Twenty times the window for at most double the cost is
#: unambiguously not O(window); the pre-incremental implementation was 7x and
#: would fail this by a wide margin.
_FLATNESS_RATIO = 2.0


def _store_step_cost(
    store: object,
    maxlen: int,
    distinct: int = _DISTINCT_TOOLS,
    warmup: int = 400,
    samples: int = 400,
) -> float:
    """Time one ``record`` call at a given window size and call diversity.

    Args:
        store: A ``BehaviorStore``.
        maxlen: Window bound to measure at.
        distinct: How many distinct call identities the workload cycles through.
        warmup: Steps to run before timing, so the window is full.
        samples: Steps to time.

    Returns:
        Mean seconds per ``record`` call.
    """
    record = cast("Any", store).record
    run = f"bench-{maxlen}-{distinct}"
    for n in range(warmup):
        record(run, ("tool", f"t{n % distinct}"), False, maxlen=maxlen, k=2)

    start = time.perf_counter()
    for n in range(samples):
        record(run, ("tool", f"t{n % distinct}"), False, maxlen=maxlen, k=2)
    elapsed = (time.perf_counter() - start) / samples

    cast("Any", store).close_run(run, k=2)
    return elapsed


def _assert_flat(costs: dict[int, float], label: str) -> None:
    """Assert per-step cost did not grow with the window, and print the row."""
    print(f"\n{label}: " + ", ".join(f"window {w}={c * 1e6:.1f}us" for w, c in costs.items()))
    narrow, wide = costs[_WINDOWS[0]], costs[_WINDOWS[-1]]
    assert wide < narrow * _FLATNESS_RATIO + 1e-6, (
        f"{label} per-step cost grew with window size: {narrow * 1e6:.1f}us at "
        f"{_WINDOWS[0]} to {wide * 1e6:.1f}us at {_WINDOWS[-1]}"
    )
    assert wide < BUDGET_P99_SECONDS


def test_classification_is_flat_in_window_size_in_memory() -> None:
    """Widening the window must cost memory, never per-step latency.

    This is the property that made the incremental counter worth its complexity.
    Before it, the default window cost 53 us per step and a 1000-entry window
    cost 370 us, so anyone widening the window to catch longer cycles paid for
    it in latency without being told. Kept as a test because the numbers are
    published as measured and nothing else would notice them regressing.
    """
    from optio.lanes.behavior.store_memory import InMemoryBehaviorStore

    store = InMemoryBehaviorStore()
    _assert_flat({w: _store_step_cost(store, w) for w in _WINDOWS}, "in-memory record")


@pytest.mark.redis
def test_classification_is_flat_in_window_size_over_redis() -> None:
    """The same shape across a network, which is why the Lua reduces server-side.

    Returning the counter would have been the obvious port and would have put up
    to ``maxlen`` entries on the wire per step. That is flat in *time* on a
    loopback and linear in bytes on any real deployment, so it is measured
    against a real server rather than reasoned about.

    The absolute numbers here are dominated by the round trip, not by anything
    optio does -- see ``test_a_shared_step_is_mostly_round_trip``.
    """
    from optio.lanes.behavior.store_redis import RedisBehaviorStore
    from tests.integration.test_redis_ledger import connect_or_skip

    client = connect_or_skip(timeout_ms=2000)
    store = RedisBehaviorStore(client, ttl_seconds=60.0)
    try:
        _assert_flat({w: _store_step_cost(store, w) for w in _WINDOWS}, "redis record")
    finally:
        client.close()


def test_the_cost_driver_is_call_diversity_not_window_size() -> None:
    """Pins the axis that *does* scale, so it is measured rather than surprising.

    Finding the top ``k`` counts means scanning the live ones, on both backends
    -- ``Counter.most_common(k)`` over the keys, or ``HVALS`` plus a bounded
    insertion in Lua. So per-step cost is O(distinct calls in the window), and
    "flat in window size" holds because widening the window does not by itself
    add distinct calls.

    The growth is real but points the safe way: it peaks when every step is
    different, which is the case with no loop to detect, and it is at its
    cheapest on the tight cycles the lane exists to catch. Measured here so that
    stays a stated property with a number rather than an assumption.
    """
    from optio.lanes.behavior.store_memory import InMemoryBehaviorStore

    store = InMemoryBehaviorStore()
    costs = {d: _store_step_cost(store, 1000, distinct=d) for d in (8, 1000)}

    print(
        "\nin-memory record: "
        + ", ".join(f"{d} distinct={c * 1e6:.1f}us" for d, c in costs.items())
    )
    assert costs[1000] > costs[8], (
        "cost no longer tracks call diversity -- the docs describing it are now wrong"
    )
    # The whole point: even the worst case is two orders of magnitude inside the
    # per-step budget, so this is a documentation obligation, not a regression.
    assert costs[1000] < BUDGET_P99_SECONDS


@pytest.mark.redis
def test_a_shared_step_is_mostly_round_trip() -> None:
    """What the Redis numbers above are actually measuring.

    A ``record`` is one script and one round trip by construction, so its cost
    is a bare PING plus the server's work. Stated as a ratio rather than an
    absolute so it means the same thing on a developer's laptop and in CI --
    where the loopback is much faster than Docker Desktop's NAT.

    If this ever fails, the script started doing real work and the reduction is
    the place to look.
    """
    from optio.lanes.behavior.store_redis import RedisBehaviorStore
    from tests.integration.test_redis_ledger import connect_or_skip

    client = connect_or_skip(timeout_ms=2000)
    try:
        client.ping()
        start = time.perf_counter()
        for _ in range(200):
            client.ping()
        ping = (time.perf_counter() - start) / 200

        step = _store_step_cost(RedisBehaviorStore(client, ttl_seconds=60.0), 1000)
    finally:
        client.close()

    print(f"\nredis: bare PING {ping * 1e6:.0f}us, record {step * 1e6:.0f}us")
    assert step < ping * 3, (
        f"a record costs {step / ping:.1f} round trips ({step * 1e6:.0f}us against a "
        f"{ping * 1e6:.0f}us PING); the script is doing more than reduce"
    )


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
    from optio.config import Config

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(OptioSpanTap(Config(cost_lane=False, behavior_lane=False)))
    tracer = provider.get_tracer("bench")

    baseline_tracer, _ = _build(with_tap=False)
    _time_steps(baseline_tracer, 200)
    _time_steps(tracer, 200)

    baseline = _percentile(_time_steps(baseline_tracer, SAMPLES), 0.99)
    disabled = _percentile(_time_steps(tracer, SAMPLES), 0.99)
    overhead = disabled - baseline

    print(f"\ndisabled-lane p99 overhead {overhead * 1e6:.1f}us")
    assert overhead < BUDGET_P99_SECONDS


#: Section 11's separate budget for the quality lane running its inline
#: heuristic. Looser than SC-5 because this lane is opt-in: a user who turns it
#: on has accepted some cost, whereas cost+behavior are on by default.
QUALITY_BUDGET_P99_SECONDS = 0.010


def test_quality_lane_inline_overhead_is_within_budget() -> None:
    """The heuristic tier stays within Section 11's 10 ms p99 budget.

    Measured with the lane *on* and no judge supplied, which is the inline path:
    every run scored, no model call. The judge tier is deliberately not measured
    against a latency budget -- it runs off the hot path by construction (M5-3),
    and its cost is the user's own model call, not ours.
    """
    from optio.config import Config

    quality_config = Config(quality_lane=True, quality_sample_rate=0.0)

    baseline_tracer, _ = _build(with_tap=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(OptioSpanTap(quality_config))
    metered_tracer = provider.get_tracer("bench")

    _time_steps(baseline_tracer, 200)
    with RunContext(config=quality_config):
        _time_steps(metered_tracer, 200)

    baseline = _time_steps(baseline_tracer, SAMPLES)
    with RunContext(config=quality_config):
        metered = _time_steps(metered_tracer, SAMPLES)

    mean = statistics.mean(metered) - statistics.mean(baseline)
    p99 = _percentile(metered, 0.99) - _percentile(baseline, 0.99)

    print(
        f"\nquality lane (inline) -- mean {mean * 1e6:.1f}us  "
        f"p99 {p99 * 1e6:.1f}us  (budget 10ms p99)"
    )
    assert p99 < QUALITY_BUDGET_P99_SECONDS, (
        f"quality lane p99 {p99 * 1e3:.2f}ms exceeds the 10ms budget"
    )


def test_a_slow_judge_does_not_slow_the_run() -> None:
    """A judge taking 200 ms must not add 200 ms to the run (M5-3).

    The guarantee that makes the judge tier usable at all: it is dispatched to a
    worker and the run does not wait. If this regressed, every sampled run would
    pay full model latency at run end -- and the failure would look like "the
    agent got slower", far from its cause.
    """
    import time as _time

    from optio.config import Config
    from optio.lanes.quality.judge import JudgeRequest, JudgeScores
    from optio.lanes.quality.lane import QualityLane

    judge_seconds = 0.2

    def slow_judge(_request: JudgeRequest) -> JudgeScores:
        _time.sleep(judge_seconds)
        return JudgeScores(task_success=1.0)

    class Run:
        run_id = "bench-run"
        budget = None
        sampled = True
        successes = None

    config = Config(quality_lane=True, quality_sample_rate=1.0, judge=slow_judge)
    lane = QualityLane(config, judge=slow_judge)

    from unittest.mock import Mock

    from opentelemetry.trace import StatusCode

    span = Mock()
    span.name = "step"
    span.attributes = {semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 20}
    span.status = Mock(status_code=StatusCode.OK)

    run = Run()
    lane.process_span(cast("ReadableSpan", span), run)

    start = _time.perf_counter()
    lane.on_run_end(run)
    elapsed = _time.perf_counter() - start
    lane.shutdown()

    print(f"\nrun end with a {judge_seconds * 1e3:.0f}ms judge: {elapsed * 1e3:.2f}ms")
    assert elapsed < judge_seconds / 2, (
        f"run end took {elapsed * 1e3:.1f}ms against a {judge_seconds * 1e3:.0f}ms "
        f"judge -- the judge is blocking the run"
    )
