"""In-memory state store (ADR-005 default backend).

The concurrency tests are the point. A lost update in ``incr`` would not raise;
it would produce a cost total that is quietly too low, which is the failure mode
R-TECH-1 exists to prevent.
"""

from __future__ import annotations

import threading

import pytest

from agentmeter.errors import StateStoreError
from agentmeter.store.base import StateStore
from agentmeter.store.memory import InMemoryStateStore


@pytest.fixture
def store() -> InMemoryStateStore:
    """A fresh store."""
    return InMemoryStateStore()


class TestContract:
    def test_satisfies_the_state_store_interface(self, store: InMemoryStateStore) -> None:
        # Structural conformance: the ledger and the Redis backend must be
        # swappable, which fails at wiring time if the shape drifts.
        for method in ("get", "set", "incr", "delete"):
            assert callable(getattr(store, method))
            assert hasattr(StateStore, method)


class TestBasicOperations:
    def test_missing_key_reads_as_none(self, store: InMemoryStateStore) -> None:
        assert store.get("run", "missing") is None

    def test_set_then_get(self, store: InMemoryStateStore) -> None:
        store.set("run", "key", 42)
        assert store.get("run", "key") == 42

    def test_runs_are_isolated(self, store: InMemoryStateStore) -> None:
        store.set("run-a", "key", "a")
        store.set("run-b", "key", "b")

        assert store.get("run-a", "key") == "a"
        assert store.get("run-b", "key") == "b"

    def test_incr_starts_from_zero(self, store: InMemoryStateStore) -> None:
        assert store.incr("run", "cost", 1.5) == pytest.approx(1.5)

    def test_incr_accumulates(self, store: InMemoryStateStore) -> None:
        store.incr("run", "cost", 1.5)
        assert store.incr("run", "cost", 2.5) == pytest.approx(4.0)

    def test_incr_accepts_negative_deltas(self, store: InMemoryStateStore) -> None:
        store.incr("run", "cost", 5.0)
        assert store.incr("run", "cost", -2.0) == pytest.approx(3.0)

    def test_incr_on_a_non_numeric_value_is_rejected(self, store: InMemoryStateStore) -> None:
        store.set("run", "key", "not a number")
        with pytest.raises(StateStoreError, match="non-numeric"):
            store.incr("run", "key", 1.0)

    def test_incr_rejects_booleans(self, store: InMemoryStateStore) -> None:
        # bool subclasses int, so `True + 1` would silently succeed as 2.
        store.set("run", "flag", True)
        with pytest.raises(StateStoreError, match="non-numeric"):
            store.incr("run", "flag", 1.0)


class TestEviction:
    def test_delete_removes_run_state(self, store: InMemoryStateStore) -> None:
        store.set("run", "key", 1)
        store.delete("run")
        assert store.get("run", "key") is None

    def test_delete_is_idempotent(self, store: InMemoryStateStore) -> None:
        # Run end can fire more than once (M1-2).
        store.delete("never-seen")
        store.delete("never-seen")

    def test_delete_leaves_other_runs_alone(self, store: InMemoryStateStore) -> None:
        store.set("run-a", "key", "a")
        store.set("run-b", "key", "b")
        store.delete("run-a")

        assert store.get("run-b", "key") == "b"

    def test_run_count_tracks_retained_state(self, store: InMemoryStateStore) -> None:
        # Growth without bound means runs are not being evicted.
        store.set("run-a", "key", 1)
        store.set("run-b", "key", 1)
        assert store.run_count() == 2

        store.delete("run-a")
        assert store.run_count() == 1

    def test_repr_reports_run_count(self, store: InMemoryStateStore) -> None:
        store.set("run", "key", 1)
        assert "runs=1" in repr(store)


class TestConcurrency:
    def test_concurrent_incr_loses_no_updates(self, store: InMemoryStateStore) -> None:
        # The failure this guards is silent: a lost update produces a cost total
        # that is quietly too low, not an error.
        threads, per_thread = 8, 500

        def worker() -> None:
            for _ in range(per_thread):
                store.incr("run", "cost", 1.0)

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        assert store.get("run", "cost") == pytest.approx(threads * per_thread)

    def test_concurrent_writes_to_separate_runs_stay_isolated(
        self, store: InMemoryStateStore
    ) -> None:
        def worker(run_id: int) -> None:
            for _ in range(200):
                store.incr(f"run-{run_id}", "cost", 1.0)

        workers = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()

        for n in range(6):
            assert store.get(f"run-{n}", "cost") == pytest.approx(200.0)
