"""One suite, every backend. This is what makes "interchangeable" a checked claim.

Parametrised over the backends so a behaviour that holds in memory and not in
Redis fails here rather than in production. Redis joins the parameter list in a
later task; the shape is fixed now so adding it is one line and every test
below immediately applies to it.

Deliberately tests the **store**, not ``CostLedger``. The facade's own
behaviour is covered by the existing ledger tests, which stayed untouched
through the extraction and are the regression net for it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from optio.errors import LedgerInvariantError
from optio.lanes.cost.ledger_memory import InMemoryLedgerStore
from optio.lanes.cost.ledger_redis import RedisLedgerStore
from optio.lanes.cost.ledger_store import LedgerStore
from optio.store.redis_client import RedisClient, StoreUnavailableError

REDIS_URL = os.environ.get("OPTIO_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> Iterator[LedgerStore]:
    """A backend under test.

    Redis skips when no server is reachable, which keeps this suite runnable on
    a laptop -- and the gate runs ``-m redis`` explicitly against a service
    container, where a skip would be a silent hole rather than a convenience.
    """
    if request.param == "memory":
        yield InMemoryLedgerStore()
        return

    client = RedisClient(REDIS_URL, timeout_ms=1000)
    try:
        client.ping()
    except StoreUnavailableError:
        pytest.skip(f"no Redis at {REDIS_URL}")
    client._redis.flushdb()
    yield RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    client._redis.flushdb()
    client.close()


class TestReserveAndReconcile:
    def test_a_reconciled_step_reports_its_actual_cost(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.reconcile("run", "step", 0.25)

        snap = store.snapshot("run")

        assert snap.actual == pytest.approx(0.25)
        assert snap.reserved == pytest.approx(0.0)
        assert snap.reconciled_steps == 1

    def test_an_open_reservation_counts_as_reserved_not_actual(self, store: LedgerStore) -> None:
        """Pre-spend gating is the whole reason reserve exists: the worst case
        has to be visible before the step runs."""
        store.reserve("run", "step", 1.0)

        snap = store.snapshot("run")

        assert snap.reserved == pytest.approx(1.0)
        assert snap.actual == pytest.approx(0.0)
        assert snap.committed == pytest.approx(1.0)

    def test_re_reserving_replaces_rather_than_stacks(self, store: LedgerStore) -> None:
        """Frameworks retry steps and reuse ids; stacking would inflate reserved
        and under-report budget_remaining for the rest of the run."""
        store.reserve("run", "step", 1.0)
        store.reserve("run", "step", 2.0)

        assert store.snapshot("run").reserved == pytest.approx(2.0)

    def test_a_double_reconcile_raises(self, store: LedgerStore) -> None:
        """Exactly-once, or the total double-counts money a policy gates on."""
        store.reserve("run", "step", 1.0)
        store.reconcile("run", "step", 0.25)

        with pytest.raises(LedgerInvariantError):
            store.reconcile("run", "step", 0.25)

    def test_reconciling_without_reserving_raises(self, store: LedgerStore) -> None:
        store.reserve("run", "other", 1.0)

        with pytest.raises(LedgerInvariantError):
            store.reconcile("run", "never-reserved", 0.25)

    def test_a_negative_reservation_raises(self, store: LedgerStore) -> None:
        with pytest.raises(LedgerInvariantError):
            store.reserve("run", "step", -1.0)

    def test_a_negative_reconcile_raises(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)

        with pytest.raises(LedgerInvariantError):
            store.reconcile("run", "step", -0.25)

    def test_runs_do_not_bleed_into_each_other(self, store: LedgerStore) -> None:
        store.reserve("run-a", "step", 1.0)
        store.reserve("run-b", "step", 2.0)
        store.reconcile("run-a", "step", 0.5)

        assert store.snapshot("run-a").actual == pytest.approx(0.5)
        assert store.snapshot("run-b").actual == pytest.approx(0.0)
        assert store.snapshot("run-b").reserved == pytest.approx(2.0)


class TestTheThreeStates:
    """`unknown`, `known`, and `finalised` are distinct, and collapsing any two
    of them produces a confidently wrong signal (ADR-044)."""

    def test_an_unseen_run_is_not_known(self, store: LedgerStore) -> None:
        assert store.knows("nobody") is False

    def test_a_metered_run_is_known(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)

        assert store.knows("run") is True

    def test_an_unseen_run_is_not_finalised(self, store: LedgerStore) -> None:
        assert store.is_finalised("never") is False

    def test_a_closed_run_is_finalised(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.close_run("run")

        assert store.is_finalised("run") is True

    def test_finality_outlives_eviction(self, store: LedgerStore) -> None:
        """State is released to bound memory; the id is not, or a straggling
        callback starts a fresh total under a run already reported."""
        store.reserve("run", "step", 1.0)
        store.close_run("run")
        store.evict("run")

        assert store.is_finalised("run") is True

    def test_an_unknown_run_snapshots_as_all_zero(self, store: LedgerStore) -> None:
        snap = store.snapshot("nobody")

        assert snap.reserved == pytest.approx(0.0)
        assert snap.actual == pytest.approx(0.0)
        assert snap.open_steps == 0


class TestClosing:
    def test_closing_is_idempotent(self, store: LedgerStore) -> None:
        """Run end can fire more than once (M1-2)."""
        store.reserve("run", "step", 1.0)
        store.reconcile("run", "step", 0.25)
        first = store.close_run("run")
        second = store.close_run("run")

        assert first.actual == pytest.approx(second.actual)
        assert first.leaked_steps == second.leaked_steps

    def test_a_leak_is_kept_as_the_reserved_worst_case(self, store: LedgerStore) -> None:
        """Dropping it would make the run look cheaper than the evidence
        supports, and under-reporting is the direction that lets an over-budget
        run through."""
        store.reserve("run", "step", 1.5)

        snap = store.close_run("run")

        assert snap.leaked_steps == 1
        assert snap.reserved == pytest.approx(1.5)

    def test_reserving_on_a_closed_run_raises(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.close_run("run")

        with pytest.raises(LedgerInvariantError):
            store.reserve("run", "another", 1.0)

    def test_reconciling_on_a_closed_run_raises(self, store: LedgerStore) -> None:
        """A late callback must not change a number a policy may have acted on."""
        store.reserve("run", "step", 1.0)
        store.close_run("run")

        with pytest.raises(LedgerInvariantError):
            store.reconcile("run", "step", 0.25)


class TestEviction:
    def test_evicting_releases_state(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.evict("run")

        assert store.run_count() == 0

    def test_evicting_an_unknown_run_is_not_an_error(self, store: LedgerStore) -> None:
        store.evict("never-existed")

        assert store.run_count() == 0
