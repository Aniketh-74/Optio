"""The Redis backend against a real server.

A fake would be faster and would prove less. Lua is what carries the atomicity
here, and a fake's Lua implementation is exactly where it diverges from the
real thing -- so the two properties below, which only a real server can
demonstrate, are tested against one.

Both are properties the in-memory backend gets for free and Redis has to
arrange deliberately: expiry that does not fire mid-run, and finality that
outlives the data it was derived from.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator

import pytest

from optio.errors import LedgerInvariantError
from optio.lanes.cost.ledger_redis import RedisLedgerStore
from optio.store.redis_client import RedisClient, StoreUnavailableError

#: Both markers, deliberately. ``integration`` is the directory's convention and
#: a contract test enforces it; ``redis`` is what the gate selects on to run
#: these against a service container, where a skip would be a silent hole.
pytestmark = [pytest.mark.integration, pytest.mark.redis]

#: Both read at import. ``tests/conftest.py`` strips every ``OPTIO_*`` variable
#: in an autouse fixture, so anything consulted from inside a fixture body would
#: always look unset -- the same trap that sent this suite's spawned workers to
#: the wrong server.
REDIS_URL = os.environ.get("OPTIO_TEST_REDIS_URL", "redis://localhost:6379/15")

#: Set in CI. Skipping when no server is reachable is right on a laptop and is a
#: silent hole in a gate whose entire purpose is running these -- a
#: misconfigured service container is simply "unreachable", and the job would go
#: green having proved nothing. Mirrors ``OPTIO_REQUIRE_PROVIDER_SDKS``.
REQUIRE_REDIS = bool(os.environ.get("OPTIO_REQUIRE_REDIS"))

#: Whether the optional driver is installed at all -- a different condition
#: from "a server answers". The floor job installs the declared minimums and
#: nothing else, so it has neither, and a test needing only the driver still
#: has to say which of the two it is missing.
HAS_REDIS_DRIVER = importlib.util.find_spec("redis") is not None

needs_driver = pytest.mark.skipif(not HAS_REDIS_DRIVER, reason="the redis driver is not installed")


def connect_or_skip(timeout_ms: int = 2000) -> RedisClient:
    """Return a live client, or skip -- unless CI demanded one.

    Catches ``ImportError`` as well as ``StoreUnavailableError`` because they
    are different problems with the same symptom. ``redis`` is an optional
    runtime extra, so a job that installs only the declared minimums -- the
    floor job does exactly that -- has no driver at all, and
    ``RedisClient.__init__`` raises ``ImportError`` before there is anything to
    ping. Letting that propagate turns "this environment cannot run these
    tests" into a failed build.
    """
    try:
        client = RedisClient(REDIS_URL, timeout_ms=timeout_ms)
        client.ping()
    except ImportError:
        if REQUIRE_REDIS:
            pytest.fail("OPTIO_REQUIRE_REDIS is set but the redis driver is not installed")
        pytest.skip("the redis driver is not installed")
    except StoreUnavailableError:
        if REQUIRE_REDIS:
            pytest.fail(f"OPTIO_REQUIRE_REDIS is set but no Redis answered at {REDIS_URL}")
        pytest.skip(f"no Redis at {REDIS_URL}")
    return client


@pytest.fixture
def store() -> Iterator[RedisLedgerStore]:
    """A store against a flushed database.

    Database 15 by convention, and flushed rather than assumed empty: a test
    that inherits another run's keys fails in ways that look like logic bugs.
    """
    client = connect_or_skip(timeout_ms=1000)
    client._redis.flushdb()
    yield RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    client._redis.flushdb()
    client.close()


class TestTtlIsAnIdleTimeoutNotADeadline:
    def test_a_second_write_refreshes_the_expiry(self, store: RedisLedgerStore) -> None:
        """An absolute expiry would be a time bomb on a long run.

        State would vanish mid-flight, open reservations with it, and
        ``budget_remaining`` would jump back to full -- ADR-044's failure
        arriving on a timer rather than through a bug. So every write pushes
        the expiry out.
        """
        short = RedisLedgerStore(store._client, ttl_seconds=5.0, tombstone_ttl_seconds=300.0)
        short.reserve("run", "a", 1.0)
        first = short.ttl_seconds_remaining("run")

        # A longer-lived store writing to the same run must extend it, which is
        # only observable because the first write set a short expiry.
        long = RedisLedgerStore(store._client, ttl_seconds=120.0, tombstone_ttl_seconds=300.0)
        long.reserve("run", "b", 1.0)
        second = long.ttl_seconds_remaining("run")

        assert first <= 5.0
        assert second > first, "a later write did not refresh the TTL"

    def test_reconciling_also_refreshes(self, store: RedisLedgerStore) -> None:
        """Reconcile is a write too. A run that only reconciles -- the tail of
        any run -- must not age out while it is still reporting."""
        short = RedisLedgerStore(store._client, ttl_seconds=5.0, tombstone_ttl_seconds=300.0)
        short.reserve("run", "a", 1.0)

        long = RedisLedgerStore(store._client, ttl_seconds=120.0, tombstone_ttl_seconds=300.0)
        long.reconcile("run", "a", 0.5)

        assert long.ttl_seconds_remaining("run") > 5.0


class TestTheTombstoneOutlivesThePayload:
    def test_a_closed_run_stays_finalised_after_its_payload_is_gone(
        self, store: RedisLedgerStore
    ) -> None:
        """Without this, a late span arriving after expiry starts a *fresh*
        run record -- resurrecting a run ADR-010 declares final."""
        store.reserve("run", "a", 1.0)
        store.close_run("run")

        # Simulate the payload expiring while the tombstone has not. Deleting
        # the keys directly is the only way to reach that state without waiting
        # out a real TTL, and it is the state the TTL split exists to survive.
        open_key, totals_key, _ = store._keys("run")
        store._client._redis.delete(open_key, totals_key)

        assert store.is_finalised("run") is True

    def test_a_resurrected_run_is_still_refused_after_expiry(self, store: RedisLedgerStore) -> None:
        """The tombstone is not decoration: it has to actually reject writes."""
        store.reserve("run", "a", 1.0)
        store.close_run("run")
        open_key, totals_key, _ = store._keys("run")
        store._client._redis.delete(open_key, totals_key)

        with pytest.raises(LedgerInvariantError):
            store.reserve("run", "b", 1.0)

    def test_a_run_that_never_existed_is_not_finalised(self, store: RedisLedgerStore) -> None:
        assert store.is_finalised("never") is False


class TestTheConfiguredBackendActuallyReachesTheLane:
    """ADR-005's addendum exists because a setting was read by nobody.

    ``store_backend='redis'`` was accepted, validated, given an extra and an
    env var, and no lane ever looked at it. Asserting the config parses would
    have passed then too, so these assert the object graph instead.
    """

    def test_redis_config_builds_a_redis_backed_cost_lane(self) -> None:
        from optio.config import Config
        from optio.lanes.cost.lane import CostLane
        from optio.lanes.registry import enabled_lanes

        # Takes no `store` fixture -- it builds its own lanes -- so it has to
        # ask for the skip itself. Without this it *fails* rather than skips
        # wherever Redis is absent, which is every leg of the test matrix.
        connect_or_skip().close()

        lanes = enabled_lanes(Config(store_backend="redis", redis_url=REDIS_URL))

        cost = next(lane for lane in lanes if isinstance(lane, CostLane))
        assert isinstance(cost.ledger._store, RedisLedgerStore)

    def test_the_default_config_stays_in_process(self) -> None:
        """The zero-infrastructure default is the reason SC-1 is achievable;
        a change that quietly networked it would be a regression."""
        from optio.config import Config
        from optio.lanes.cost.lane import CostLane
        from optio.lanes.cost.ledger_memory import InMemoryLedgerStore
        from optio.lanes.registry import enabled_lanes

        lanes = enabled_lanes(Config())

        cost = next(lane for lane in lanes if isinstance(lane, CostLane))
        assert isinstance(cost.ledger._store, InMemoryLedgerStore)

    @needs_driver
    def test_an_unreachable_redis_fails_at_setup_not_at_runtime(self) -> None:
        """Loud here, fail-open later. Configuration that cannot do what it
        claims is a setup error (Section 4.2), and discovering it on the first
        billed step instead is how ADR-005's addendum got written.
        """
        from optio.config import Config
        from optio.lanes.registry import enabled_lanes

        # Port 1 is reserved and never listening, so this is a connection
        # refusal rather than a timeout -- fast and deterministic.
        config = Config(
            store_backend="redis",
            redis_url="redis://localhost:1/15",
            store_timeout_ms=200,
        )

        with pytest.raises(StoreUnavailableError):
            enabled_lanes(config)


class TestConcurrentReconcilesDoNotLoseUpdates:
    def test_many_threads_reconciling_produce_the_exact_total(
        self, store: RedisLedgerStore
    ) -> None:
        """The reason reconcile is one script rather than three round trips.

        Threads here share a process, but they contend on the *server* exactly
        as separate processes would -- so a read-modify-write implementation
        loses updates here too, and the total comes out plausibly low.
        """
        import threading

        steps = 200
        for n in range(steps):
            store.reserve("run", f"s{n}", 0.01)

        def work(start: int) -> None:
            for n in range(start, steps, 4):
                store.reconcile("run", f"s{n}", 0.01)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            assert not t.is_alive(), "a reconcile thread hung"

        snap = store.snapshot("run")

        assert snap.actual == pytest.approx(steps * 0.01)
        assert snap.reconciled_steps == steps
        assert snap.open_steps == 0
