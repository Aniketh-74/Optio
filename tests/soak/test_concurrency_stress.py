"""The whole pipeline under real thread contention.

`tests/property/test_ledger_invariant.py` already hammers the ledger directly,
which is right -- it holds the invariant a lost update would silently break. But
the ledger is not the only shared state on the hot path. A step passing through
`instrument()` touches the span tap, the lane registry, the behavior window, the
fail-open activation table and the self-observability instruments, each with its
own locking, and nothing exercised them together under contention.

That gap matters because agent frameworks are overwhelmingly concurrent: LangGraph
fans out to parallel branches, CrewAI runs agents on a pool, and async frameworks
interleave steps across tasks in one thread. Single-threaded correctness says
almost nothing about how the library behaves in the environment it actually runs
in.

The failures being hunted here are the quiet ones:

* **Lost updates** -- a total that is too low, with no error anywhere.
* **Cross-run contamination** -- run A's cost landing on run B's span, which is
  the worst outcome because both numbers look plausible.
* **Deadlock** -- every test here has a timeout, because a hung agent is a
  worse outcome than a wrong signal and would otherwise show as a stalled suite.
* **Torn state under eviction** -- runs ending while others start, exercising
  the paths where state is released.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio import meter, semconv
from optio.config import BudgetPolicy, Config
from optio.runtime import failopen, installer
from optio.runtime.installer import install_tap

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

pytestmark = pytest.mark.soak

#: Every test is bounded. A deadlock would otherwise hang CI rather than fail it,
#: and "the suite stopped" is a much harder signal to act on than "this failed".
TIMEOUT_SECONDS = 120.0

COST_PER_STEP = 2.50  # 1M input tokens on gpt-4o


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    failopen.reset_activations()
    installer.reset_installations()
    yield
    failopen.reset_activations()
    installer.reset_installations()


def _pipeline(**config: object) -> tuple[TracerProvider, InMemorySpanExporter, object]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    install_tap(Config(**config), provider)  # type: ignore[arg-type]
    return provider, exporter, provider.get_tracer("stress")


def _step(tracer: object, tool: str = "search") -> None:
    with tracer.start_as_current_span("gen_ai.chat") as span:  # type: ignore[attr-defined]
        span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
        span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)
        span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 0)
        span.set_attribute(semconv.GEN_AI_TOOL_NAME, tool)


def _run_spans(exporter: InMemorySpanExporter) -> list[object]:
    return [s for s in exporter.get_finished_spans() if s.name.startswith("optio.run")]


class TestConcurrentRunsDoNotContaminateEachOther:
    """The worst failure mode: two plausible numbers, both wrong."""

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_16_concurrent_runs_each_report_their_own_cost(self) -> None:
        provider, exporter, tracer = _pipeline()
        runs, steps = 16, 12
        barrier = threading.Barrier(runs)

        @meter(budget=BudgetPolicy(limit_usd=1e6), provider=provider)
        def governed() -> None:
            for _ in range(steps):
                _step(tracer)

        def one_run(index: int) -> None:
            # Start together, so the runs genuinely overlap rather than
            # queueing behind each other's setup.
            barrier.wait()
            governed()

        try:
            with ThreadPoolExecutor(max_workers=runs) as pool:
                for future in as_completed(pool.submit(one_run, i) for i in range(runs)):
                    future.result()  # re-raise anything a worker hit
        finally:
            provider.shutdown()

        spans = _run_spans(exporter)
        assert len(spans) == runs

        expected = steps * COST_PER_STEP
        for span in spans:
            attributes = dict(span.attributes or {})  # type: ignore[attr-defined]
            cost = attributes.get(semconv.RUN_ACTUAL_COST)
            assert cost == pytest.approx(expected), (
                f"run reported {cost} instead of {expected}: concurrent runs are "
                "sharing state, so a step from one run was counted against another"
            )

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_runs_starting_and_ending_at_random_offsets_stay_isolated(self) -> None:
        # Staggered rather than synchronised: this exercises the eviction paths,
        # where one run is releasing state while others are mid-flight.
        provider, exporter, tracer = _pipeline()
        runs, steps = 12, 8

        @meter(budget=BudgetPolicy(limit_usd=1e6), provider=provider)
        def governed(offset: int) -> None:
            for i in range(steps):
                _step(tracer)
                if i == offset % steps:
                    # Yield at a different point in each run.
                    threading.Event().wait(0.001)

        def one_run(index: int) -> None:
            governed(index)

        try:
            with ThreadPoolExecutor(max_workers=runs) as pool:
                for future in as_completed(pool.submit(one_run, i) for i in range(runs)):
                    future.result()
        finally:
            provider.shutdown()

        expected = steps * COST_PER_STEP
        for span in _run_spans(exporter):
            attributes = dict(span.attributes or {})  # type: ignore[attr-defined]
            assert attributes.get(semconv.RUN_ACTUAL_COST) == pytest.approx(expected)


class TestAllLanesTogetherUnderContention:
    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_every_lane_enabled_across_many_threads(self) -> None:
        # Cost, behavior and quality all writing to the same run spans at once.
        # Each lane locks its own state; nothing verified they compose.
        provider, exporter, tracer = _pipeline(quality_lane=True)
        runs, steps = 12, 10

        @meter(budget=BudgetPolicy(limit_usd=1e6, max_steps=100), provider=provider)
        def governed() -> None:
            for i in range(steps):
                _step(tracer, tool=f"tool_{i % 3}")

        def one_run(index: int) -> None:
            governed()

        try:
            with ThreadPoolExecutor(max_workers=runs) as pool:
                for future in as_completed(pool.submit(one_run, i) for i in range(runs)):
                    future.result()
        finally:
            provider.shutdown()

        spans = _run_spans(exporter)
        assert len(spans) == runs

        for span in spans:
            attributes = dict(span.attributes or {})  # type: ignore[attr-defined]
            # Every lane should have contributed to every run.
            assert semconv.RUN_ACTUAL_COST in attributes
            assert semconv.RUN_LOOP_STATE in attributes
            assert attributes[semconv.RUN_ACTUAL_COST] == pytest.approx(steps * COST_PER_STEP)

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_no_lane_failure_was_silently_absorbed(self) -> None:
        # Fail-open means a lock bug under contention would show up as a
        # *missing* signal rather than an exception -- invisible unless the
        # activation counter is checked. This is that check.
        provider, _exporter, tracer = _pipeline(quality_lane=True)
        failopen.reset_activations()

        @meter(budget=BudgetPolicy(limit_usd=1e6, max_steps=100), provider=provider)
        def governed() -> None:
            for _ in range(15):
                _step(tracer)

        def one_run(_index: int) -> None:
            governed()

        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                for future in as_completed(pool.submit(one_run, i) for i in range(16)):
                    future.result()
        finally:
            provider.shutdown()

        activations = failopen.activation_count()
        assert activations == 0, (
            f"the guard absorbed {activations} failure(s) under contention. "
            "Fail-open hid them from the agent, which is correct, and they are "
            "still bugs -- run with the guard disabled to see the traceback."
        )


class TestSharedRuntimeStateIsThreadSafe:
    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_the_activation_table_counts_every_failure(self) -> None:
        # _record runs while the guard is already handling a failure, so it must
        # be both total and thread-safe. A lost increment here would understate
        # how broken a lane is.
        failopen.reset_activations()
        threads, per_thread = 12, 500

        def worker() -> None:
            for _ in range(per_thread):
                try:
                    raise ValueError("stress")
                except ValueError as error:
                    failopen._record("cost", error)

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=TIMEOUT_SECONDS)
            assert not thread.is_alive(), "activation recording deadlocked"

        assert failopen.activation_count("cost") == threads * per_thread

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_installing_the_tap_concurrently_yields_one_tap(self) -> None:
        # Frameworks call instrument() from wherever they initialise, and two
        # taps on one provider would double-count every step.
        installer.reset_installations()
        provider = TracerProvider()
        barrier = threading.Barrier(8)
        results = []
        lock = threading.Lock()

        def install() -> None:
            barrier.wait()
            tap = install_tap(Config(), provider)
            with lock:
                results.append(tap)

        workers = [threading.Thread(target=install) for _ in range(8)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=TIMEOUT_SECONDS)
            assert not thread.is_alive(), "tap installation deadlocked"

        assert len(results) == 8
        assert len({id(tap) for tap in results}) == 1, (
            "concurrent install_tap calls produced different taps; each span "
            "would be processed once per tap and every cost doubled"
        )
        provider.shutdown()


class TestTheLedgerLockIsLoadBearing:
    """A race that is reproducible, unlike the ones people usually write.

    Worth stating what was learned building this, because it changes what a
    concurrency test is even for. Removing the ledger's lock entirely and
    hammering it with 64 threads at a 100 ns switch interval produces **no lost
    update**: CPython's GIL makes the individual dict writes atomic, so the
    obvious "many threads, check the total" test passes whether the lock exists
    or not. It proves nothing.

    What the lock actually guards is the *composite* sequence in ``close_run``:
    it reads ``len(ledger.open)``, then writes ``leaked_steps`` and ``closed``
    from that read. A ``reserve`` landing between the read and the write yields
    a leak count that disagrees with the ledger's own state -- and the leak
    count is what tells an operator whether a reported cost is measured spend or
    a reserved worst case.

    Measured: with the lock removed this fails ~75 times in 400 rounds; with it,
    zero. That ratio is the argument for the test existing.
    """

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_closing_a_run_while_it_is_still_reserving_stays_consistent(self) -> None:
        from optio.errors import LedgerInvariantError
        from optio.lanes.cost.ledger import CostLedger, LedgerSnapshot

        def one_round(round_id: int) -> None:
            # A function rather than a loop body: closures over loop variables
            # bind late, so the threads would race against the *next* round's
            # ledger. Correct here only by accident of timing, which is not a
            # property to rely on in a test about timing.
            ledger = CostLedger()
            run = f"race-{round_id}"
            for i in range(20):
                ledger.reserve(run, f"seed{i}", 1.0)

            gate = threading.Barrier(2)
            closed: dict[str, object] = {}

            def close_it() -> None:
                gate.wait()
                closed["snapshot"] = ledger.close_run(run)

            def keep_reserving() -> None:
                gate.wait()
                for i in range(20):
                    try:
                        ledger.reserve(run, f"late{i}", 1.0)
                    except LedgerInvariantError:
                        break  # correct: the run is closed (ADR-010)

            threads = [
                threading.Thread(target=close_it),
                threading.Thread(target=keep_reserving),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=TIMEOUT_SECONDS)
                assert not thread.is_alive(), "close/reserve deadlocked"

            snapshot = closed["snapshot"]
            assert isinstance(snapshot, LedgerSnapshot)
            final = ledger.snapshot(run)
            if final.open_steps > 0:
                assert snapshot.leaked_steps == final.open_steps, (
                    f"close_run reported {snapshot.leaked_steps} leaked steps "
                    f"but the ledger holds {final.open_steps}: the read of "
                    "len(open) and the write of leaked_steps were not atomic, "
                    "so the run's cost is labelled measured when it is an estimate"
                )

        original_interval = sys.getswitchinterval()
        # Force frequent thread switches so the interleaving is actually
        # explored rather than left to chance on a fast machine.
        sys.setswitchinterval(0.0000001)
        try:
            for round_id in range(150):
                one_round(round_id)
        finally:
            sys.setswitchinterval(original_interval)

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_a_closed_run_rejects_late_reservations_from_any_thread(self) -> None:
        # ADR-010: closing is final. Under contention the check and the close
        # must not interleave such that a straggler restarts a finished run --
        # that would begin a second total under an id already reported.
        from optio.errors import LedgerInvariantError
        from optio.lanes.cost.ledger import CostLedger

        ledger = CostLedger()
        run = "final"
        ledger.reserve(run, "s0", 1.0)
        ledger.reconcile(run, "s0", 1.0)
        ledger.close_run(run)

        rejected = 0
        lock = threading.Lock()

        def straggler(worker: int) -> None:
            nonlocal rejected
            for i in range(50):
                try:
                    ledger.reserve(run, f"w{worker}-{i}", 5.0)
                except LedgerInvariantError:
                    with lock:
                        rejected += 1

        workers = [threading.Thread(target=straggler, args=(n,)) for n in range(8)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=TIMEOUT_SECONDS)
            assert not thread.is_alive(), "late reservation deadlocked"

        assert rejected == 8 * 50, f"only {rejected} of 400 late reservations were refused"
        assert ledger.snapshot(run).actual == pytest.approx(1.0), (
            "a late reservation changed the cost of a run already reported"
        )


class TestTheBehaviorLaneLockIsLoadBearing:
    """The window's incremental counts are a read-modify-write, and need the lock.

    `BehaviorWindow.add` decrements the evicted call's count and increments the
    new one. That is the same shape as the ledger's `close_run` race -- and
    unlike a plain dict write, the GIL does not make it atomic.

    Two distinct failures appear when the lane's lock is removed, verified by
    removing it:

    * The counts drift from the deque, so every later verdict is computed from
      a window that never existed. Silent: the deque stays correct, so nothing
      else looks wrong.
    * `classify` raises `RuntimeError: dictionary changed size during iteration`
      from `Counter.most_common`, because it iterates the counter while another
      thread mutates it. The fail-open guard absorbs that (ADR-004), so the
      agent survives -- but the behavior signal vanishes for the affected steps
      and only the `optio.internal.lane_errors` metric would say why.

    Both are contained today because `add` and `classify` are called inside the
    same lock. This test exists so that narrowing that lock for performance --
    an entirely reasonable-looking change, since classify is now O(1) -- fails
    here instead of in a user's agent.
    """

    @pytest.mark.timeout(TIMEOUT_SECONDS)
    def test_counts_stay_consistent_under_contention(self) -> None:
        from collections import Counter

        from optio.lanes.behavior.lane import BehaviorLane

        class _Run:
            run_id = "shared"
            budget = None

        lane = BehaviorLane(Config(behavior_window_size=32))
        run = _Run()
        threads, per_thread = 16, 400
        gate = threading.Barrier(threads)
        errors: list[BaseException] = []

        original = sys.getswitchinterval()
        # Force preemption mid-sequence; at the default interval the whole
        # read-modify-write usually completes inside one time slice and the
        # race never appears.
        sys.setswitchinterval(0.0000001)
        try:

            def worker(worker_id: int) -> None:
                gate.wait()
                try:
                    for i in range(per_thread):
                        lane.process_span(_span_for(f"t{(worker_id + i) % 6}"), run)
                except BaseException as exc:  # noqa: BLE001 - recorded, then asserted
                    errors.append(exc)

            pool = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join(timeout=TIMEOUT_SECONDS)
                assert not thread.is_alive(), "behavior lane deadlocked"
        finally:
            sys.setswitchinterval(original)

        assert not errors, f"classification raised under contention: {errors[:3]}"

        window = lane._windows["shared"]
        assert window.call_counts == Counter(step.call for step in window), (
            "maintained call counts drifted from the window's actual contents"
        )
        assert window.error_count == sum(1 for step in window if step.errored), (
            "maintained error count drifted from the window's actual contents"
        )


def _span_for(tool: str) -> ReadableSpan:
    """A finished-span stub carrying just what the behavior lane reads."""
    from unittest.mock import Mock

    span = Mock()
    span.name = "gen_ai.chat"
    span.attributes = {
        semconv.GEN_AI_TOOL_NAME: tool,
        semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
    }
    span.status = None
    return cast("ReadableSpan", span)
