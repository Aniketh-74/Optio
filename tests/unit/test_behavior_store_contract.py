"""One suite, every behaviour backend, so "interchangeable" is a checked claim.

Parametrised over the backends: a rule that holds in memory and not in Redis
fails here rather than in production.

The eviction cases are the load-bearing half. A window's aggregates are
maintained incrementally on both backends -- a ``Counter`` in Python, a hash in
Lua -- so the arithmetic that runs when a step *leaves* the window exists twice,
in two languages, and a divergence there produces a different verdict rather
than an error.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from optio.lanes.behavior.store import BehaviorStore
from optio.lanes.behavior.store_memory import InMemoryBehaviorStore
from optio.lanes.behavior.store_redis import RedisBehaviorStore
from tests.integration.test_redis_ledger import connect_or_skip


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> Iterator[BehaviorStore]:
    """A backend under test.

    Redis skips when no server is reachable, which keeps this suite runnable on
    a laptop -- and the gate sets ``OPTIO_REQUIRE_REDIS`` so the same skip is a
    failure there, because a job whose purpose is running these must not pass
    by not running them.
    """
    if request.param == "memory":
        yield InMemoryBehaviorStore()
        return

    client = connect_or_skip(timeout_ms=1000)
    client._redis.flushdb()
    yield RedisBehaviorStore(client, ttl_seconds=60.0)
    client._redis.flushdb()
    client.close()


class TestRecordingBuildsAWindow:
    def test_the_first_step_yields_a_size_of_one(self, store: BehaviorStore) -> None:
        state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.size == 1
        assert state.errors == 0
        assert state.distinct_calls == 1
        assert state.top_counts == (1,)

    def test_repeats_accumulate(self, store: BehaviorStore) -> None:
        for _ in range(4):
            state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.size == 4
        assert state.distinct_calls == 1
        assert state.top_counts[0] == 4

    def test_errors_are_counted(self, store: BehaviorStore) -> None:
        store.record("run", ("tool", "read"), True, maxlen=50, k=2)
        state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.errors == 1

    def test_an_errored_retry_is_the_same_call(self, store: BehaviorStore) -> None:
        """What separates a retry storm from ordinary repetition: the outcome
        is tallied separately from the call identity, so two attempts at one
        failing operation still count as one repeated call."""
        store.record("run", ("tool", "read"), True, maxlen=50, k=2)
        state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.distinct_calls == 1
        assert state.top_counts == (2,)

    def test_runs_do_not_bleed(self, store: BehaviorStore) -> None:
        store.record("a", ("tool", "read"), False, maxlen=50, k=2)
        state = store.record("b", ("tool", "write"), False, maxlen=50, k=2)

        assert state.size == 1
        assert store.run_count() == 2

    def test_top_counts_are_descending(self, store: BehaviorStore) -> None:
        """``classify`` reads ``top_counts[0]`` as the repeat count and never
        sorts, so ordering is the store's promise, not the detector's."""
        store.record("run", ("tool", "rare"), False, maxlen=50, k=2)
        for _ in range(3):
            state = store.record("run", ("tool", "common"), False, maxlen=50, k=2)

        assert state.top_counts == (3, 1)

    def test_the_payload_is_capped_at_k(self, store: BehaviorStore) -> None:
        """The guarantee a naive shared backend breaks silently: the summary is
        a fixed shape, so a wider window costs memory, never bytes per step."""
        for n in range(20):
            state = store.record("run", ("tool", f"c{n}"), False, maxlen=50, k=2)

        assert len(state.top_counts) == 2
        assert state.distinct_calls == 20


class TestTheWindowIsBounded:
    def test_size_never_exceeds_maxlen(self, store: BehaviorStore) -> None:
        for n in range(20):
            state = store.record("run", ("tool", f"c{n}"), False, maxlen=5, k=2)

        assert state.size == 5

    def test_evicted_calls_leave_the_counts(self, store: BehaviorStore) -> None:
        """The subtle half: a count reaching zero must drop its key, because
        ``distinct_calls`` is the number of keys and a zeroed one inflates it
        for the rest of the run -- pushing a stuck agent's window back over
        ``LOOP_MAX_DISTINCT`` and hiding the loop."""
        for _ in range(5):
            store.record("run", ("tool", "old"), False, maxlen=5, k=2)
        for _ in range(5):
            state = store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert state.distinct_calls == 1
        assert state.top_counts == (5,)

    def test_evicted_errors_leave_the_tally(self, store: BehaviorStore) -> None:
        for _ in range(5):
            store.record("run", ("tool", "old"), True, maxlen=5, k=2)
        for _ in range(5):
            state = store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert state.errors == 0

    def test_a_partially_evicted_call_keeps_the_rest_of_its_count(
        self, store: BehaviorStore
    ) -> None:
        """Eviction decrements; it does not forget. Dropping the key on the
        first eviction would erase four repeats that are still in the window."""
        for _ in range(5):
            store.record("run", ("tool", "old"), False, maxlen=5, k=2)
        state = store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert state.distinct_calls == 2
        assert state.top_counts == (4, 1)


class TestLifecycle:
    def test_closing_returns_the_final_state(self, store: BehaviorStore) -> None:
        """Run end emits a last verdict, so closing has to hand back the state
        it is releasing. Reading then closing would be two round trips with a
        race between them."""
        for _ in range(3):
            store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        state = store.close_run("run", k=2)

        assert state is not None
        assert state.size == 3

    def test_closing_releases_the_window(self, store: BehaviorStore) -> None:
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store.close_run("run", k=2)

        assert store.run_count() == 0

    def test_closing_twice_reports_nothing_the_second_time(self, store: BehaviorStore) -> None:
        """Run end can fire more than once (M1-2). A second close that
        re-derived a verdict from an absent window would emit ``healthy`` with
        no repeats, overwriting a real ``looping`` verdict on the run span --
        the same failure the cost lane hit, in the direction that hides a
        pathology. Absence is reported as absence."""
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store.close_run("run", k=2)

        assert store.close_run("run", k=2) is None

    def test_closing_an_unknown_run_is_not_an_error(self, store: BehaviorStore) -> None:
        assert store.close_run("never", k=2) is None
        assert store.run_count() == 0

    def test_closing_one_run_leaves_the_others(self, store: BehaviorStore) -> None:
        store.record("a", ("tool", "read"), False, maxlen=50, k=2)
        store.record("b", ("tool", "read"), False, maxlen=50, k=2)

        store.close_run("a", k=2)

        assert store.run_count() == 1

    def test_recording_after_a_close_starts_a_fresh_window(self, store: BehaviorStore) -> None:
        """Unlike the ledger, closing is not final here. A re-opened window is
        short again, so at worst the lane under-reports -- and failing toward
        ``healthy`` is the bias Section 6.4 requires. It cannot invent a
        pathology or corrupt a published total the way a reopened cost run
        could (ADR-010)."""
        for _ in range(5):
            store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store.close_run("run", k=2)

        state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.size == 1
