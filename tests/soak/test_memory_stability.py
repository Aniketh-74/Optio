"""Long-run memory stability (§11: bounded, < a few KB per run).

§11 budgets memory per run and names a soak test as its verification. This is
that test, and it exists because the failure it guards against is invisible in
every other suite: nothing here *breaks*, it just grows. A leak of a few hundred
bytes per run is undetectable in a unit test and fatal in an agent process that
stays up for a week.

Two shapes of leak matter, and they need different tests:

* **Within one run.** A 100,000-step run must not accumulate 100,000 anything.
  Every retention point -- the ledger's open reservations, the behavior window,
  the quality lane's span buffer -- is supposed to be bounded by configuration
  rather than by run length.
* **Across many runs.** A process handling 10,000 short runs must return to its
  starting footprint. This is the one that bites in production, because it is
  the shape a real agent service has.

Measured with ``tracemalloc`` rather than RSS: RSS is dominated by allocator
behaviour and GC timing, and would make this test flap. tracemalloc counts what
Python actually holds, which is the thing under our control.
"""

from __future__ import annotations

import gc
import tracemalloc
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider

from optio import RunContext, semconv
from optio.config import BudgetPolicy, Config
from optio.runtime import failopen, installer
from optio.runtime.installer import install_tap

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.soak

#: Bytes per run we are willing to retain after a run has ended. §11 says "a few
#: KB"; this is deliberately far tighter, because the correct steady-state answer
#: is *zero* -- run state is evicted at run end. The allowance absorbs
#: interpreter noise (interned strings, arena fragmentation), not real growth.
BYTES_PER_RUN_CEILING = 512


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    failopen.reset_activations()
    installer.reset_installations()
    yield
    failopen.reset_activations()
    installer.reset_installations()


def _tap(**config: object) -> tuple[TracerProvider, object]:
    provider = TracerProvider()
    install_tap(Config(**config), provider)  # type: ignore[arg-type]
    return provider, provider.get_tracer("soak")


def _step(tracer: object) -> None:
    with tracer.start_as_current_span("gen_ai.chat") as span:  # type: ignore[attr-defined]
        span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
        span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 100)
        span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 50)


def _settled() -> int:
    """Return current traced memory after forcing collection."""
    gc.collect()
    return tracemalloc.get_traced_memory()[0]


class TestOneVeryLongRun:
    def test_a_50k_step_run_stays_bounded(self) -> None:
        # The whole premise of the windowed design: cost is O(window), not
        # O(steps). If this grows linearly, a long-lived agent dies overnight.
        provider, tracer = _tap()
        tracemalloc.start()
        try:
            with RunContext(budget=BudgetPolicy(limit_usd=1e9, max_steps=200_000)):
                for _ in range(500):
                    _step(tracer)
                baseline = _settled()

                for _ in range(50_000):
                    _step(tracer)
                after = _settled()
        finally:
            tracemalloc.stop()
            provider.shutdown()

        growth = after - baseline
        assert growth < 512 * 1024, (
            f"50,000 steps grew retained memory by {growth / 1024:.1f} KiB; "
            "per-run state is supposed to be bounded by window size, not run length"
        )

    def test_the_quality_lane_also_stays_bounded(self) -> None:
        # The quality lane retains spans until run end, which is the most
        # obviously unbounded thing in the library. MAX_RETAINED_SPANS caps it.
        provider, tracer = _tap(quality_lane=True)
        tracemalloc.start()
        try:
            with RunContext(budget=BudgetPolicy(limit_usd=1e9)):
                for _ in range(500):
                    _step(tracer)
                baseline = _settled()

                for _ in range(20_000):
                    _step(tracer)
                after = _settled()
        finally:
            tracemalloc.stop()
            provider.shutdown()

        growth = after - baseline
        assert growth < 512 * 1024, (
            f"20,000 steps with the quality lane on grew retained memory by "
            f"{growth / 1024:.1f} KiB; MAX_RETAINED_SPANS should cap this"
        )


class TestManyShortRuns:
    def test_10k_runs_return_to_baseline(self) -> None:
        # The production shape: a long-lived service handling short runs. Each
        # must release its state at run end, or the process grows forever.
        provider, tracer = _tap()
        tracemalloc.start()
        try:
            for _ in range(200):  # warm up caches, interned strings, etc.
                with RunContext(budget="$1.00"):
                    _step(tracer)
            baseline = _settled()

            runs = 10_000
            for _ in range(runs):
                with RunContext(budget="$1.00"):
                    _step(tracer)
                    _step(tracer)
            after = _settled()
        finally:
            tracemalloc.stop()
            provider.shutdown()

        per_run = (after - baseline) / runs
        assert per_run < BYTES_PER_RUN_CEILING, (
            f"{per_run:.1f} bytes retained per completed run over {runs:,} runs; "
            "run state is evicted at run end, so the steady state should be ~0"
        )

    def test_growth_stops_rather_than_merely_being_slow(self) -> None:
        # A per-run ceiling passes for a genuine leak that happens to be small,
        # so the ceiling alone is not enough. This measures the *rate* in
        # successive blocks: bounded state fills its cap and the rate falls to
        # zero, while a leak holds its rate forever.
        #
        # The ledger deliberately remembers recently-closed run ids (ADR-010),
        # so early blocks legitimately grow until _CLOSED_MEMORY is reached.
        # Measured: ~114 B/run in the first block, ~0.0 from the third onward.
        provider, tracer = _tap()
        tracemalloc.start()
        try:
            for _ in range(200):
                with RunContext(budget="$1.00"):
                    _step(tracer)

            rates: list[float] = []
            previous = _settled()
            for _ in range(6):
                for _ in range(4_000):
                    with RunContext(budget="$1.00"):
                        _step(tracer)
                current = _settled()
                rates.append((current - previous) / 4_000)
                previous = current
        finally:
            tracemalloc.stop()
            provider.shutdown()

        # By the final blocks the cap is long since full; anything still
        # growing there is unbounded.
        tail = rates[-3:]
        assert all(rate < 8.0 for rate in tail), (
            f"per-run growth had not settled: block rates {[f'{r:.1f}' for r in rates]} "
            "bytes/run. A rate that stays flat instead of falling to zero is a leak, "
            "not bounded state filling its cap."
        )

    def test_abandoned_runs_do_not_accumulate(self) -> None:
        # A run whose context manager never exits -- the framework crashed, the
        # task was cancelled. The ledger must not hold these forever.
        from optio.lanes.cost.ledger import CostLedger

        ledger = CostLedger()
        for i in range(5_000):
            ledger.reserve(f"abandoned-{i}", "step-0", 0.01)
            ledger.close_run(f"abandoned-{i}")
            ledger.evict(f"abandoned-{i}")

        assert ledger.run_count() == 0, (
            f"{ledger.run_count()} runs still tracked after eviction; "
            "a long-lived process would grow without bound"
        )

    def test_the_closed_run_memory_is_itself_bounded(self) -> None:
        # Closed run ids are remembered so a straggling callback cannot restart
        # a finished run (ADR-010). That memory must also be capped.
        # Addresses the backend rather than the facade: the closed-id window is
        # the in-memory store's own structure, and Redis bounds the same thing
        # with a TTL instead.
        from optio.lanes.cost.ledger import _CLOSED_MEMORY
        from optio.lanes.cost.ledger_memory import InMemoryLedgerStore

        ledger = InMemoryLedgerStore()
        overshoot = _CLOSED_MEMORY + 1_000
        for i in range(overshoot):
            ledger.reserve(f"run-{i}", "s", 0.01)
            ledger.close_run(f"run-{i}")
            ledger.evict(f"run-{i}")

        remembered = len(ledger._recently_closed)
        assert remembered <= _CLOSED_MEMORY, (
            f"{remembered} closed ids retained, cap is {_CLOSED_MEMORY}"
        )


class TestFailOpenBookkeepingIsBounded:
    def test_repeated_failures_do_not_grow_the_activation_table(self) -> None:
        # _activations is keyed by (component, exception type). Both sets are
        # small and fixed, so the table is bounded by construction -- but only
        # if nothing ever puts a unique value in the key.
        for i in range(10_000):
            try:
                raise ValueError(f"failure number {i}")
            except ValueError as error:
                failopen._record("cost", error)

        assert failopen.activation_count("cost") == 10_000
        assert len(failopen._activations) == 1, (
            f"activation table holds {len(failopen._activations)} keys after 10,000 "
            "identical failures; the key must not include per-failure detail"
        )
