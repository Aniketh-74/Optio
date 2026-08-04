"""The Redis behaviour backend against a real server.

A fake would be faster and would prove less. The eviction arithmetic lives in
Lua here, and a fake's Lua is exactly where it diverges from the real thing.

What this file covers is what only a real server can show: that the reduction
really happens server-side, that the idle TTL refreshes rather than expiring
mid-run, and that the window is bounded in Redis's own data structures rather
than in the client. The behavioural rules the two backends share are in
``tests/unit/test_behavior_store_contract.py``, which runs against this backend
too.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from optio.lanes.behavior.store_redis import _PREFIX, RedisBehaviorStore
from tests.integration.test_redis_ledger import connect_or_skip, needs_driver

#: Both markers, deliberately. ``integration`` is the directory's convention and
#: a contract test enforces it; ``redis`` is what the gate selects on.
pytestmark = [pytest.mark.integration, pytest.mark.redis]


@pytest.fixture
def store() -> Iterator[RedisBehaviorStore]:
    """A backend against a live server, on a flushed database."""
    client = connect_or_skip(timeout_ms=1000)
    client._redis.flushdb()
    yield RedisBehaviorStore(client, ttl_seconds=60.0)
    client._redis.flushdb()
    client.close()


class TestTheReductionHappensOnTheServer:
    def test_the_payload_does_not_grow_with_the_window(self, store: RedisBehaviorStore) -> None:
        """The guarantee a naive port breaks silently.

        Two hundred distinct calls in a window of a thousand still return two
        counts. Shipping the counter instead would have passed every other test
        in this file while turning a published O(1)-in-window-size claim into
        O(window) in bytes on every step.
        """
        for n in range(200):
            state = store.record("run", ("tool", f"c{n}"), False, maxlen=1000, k=2)

        assert len(state.top_counts) == 2
        assert state.distinct_calls == 200

    def test_the_reply_is_four_fields_whatever_the_window(self, store: RedisBehaviorStore) -> None:
        """Asserted against the wire, not the parsed object.

        ``WindowState`` has a fixed shape by construction, so a backend that
        shipped the counter and reduced it in Python would satisfy every other
        assertion in this file. This one reads what the script itself returned.
        """
        for n in range(300):
            store.record("run", ("tool", f"c{n}"), False, maxlen=1000, k=2)

        raw = store._client.run_script(
            "record", store._keys("run"), ["tool\x00extra", "0", "1000", "2", "60000"]
        )

        assert len(raw) == 4
        assert raw[3] == "1,1", "the server sent more than the top k"

    def test_the_top_counts_come_back_descending(self, store: RedisBehaviorStore) -> None:
        """``classify_state`` reads ``top_counts[0]`` as the repeat count and
        never sorts. Redis hashes have no order, so the ordering is the
        script's promise."""
        store.record("run", ("tool", "rare"), False, maxlen=50, k=2)
        for _ in range(3):
            state = store.record("run", ("tool", "common"), False, maxlen=50, k=2)

        assert state.top_counts == (3, 1)


class TestTheWindowIsBoundedInRedis:
    def test_the_step_list_is_trimmed_server_side(self, store: RedisBehaviorStore) -> None:
        """Bounded memory is structural here as it is in the deque. A client
        that stopped calling would otherwise leave an unbounded list behind."""
        for n in range(50):
            store.record("run", ("tool", f"c{n}"), False, maxlen=5, k=2)

        assert store._client._redis.llen(f"{_PREFIX}:run:steps") == 5

    def test_a_zeroed_count_leaves_no_field_behind(self, store: RedisBehaviorStore) -> None:
        """Read from the hash rather than from ``distinct_calls``.

        The count is what ``distinct_calls`` is derived from, so asserting on
        the derived number cannot tell "the field was deleted" from "the
        reduction skipped it". A field left at zero would inflate the distinct
        count for the rest of the run and hide a loop.
        """
        for _ in range(5):
            store.record("run", ("tool", "old"), False, maxlen=5, k=2)
        for _ in range(5):
            store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert store._client._redis.hgetall(f"{_PREFIX}:run:counts") == {"tool\x00new": "5"}

    def test_an_error_that_leaves_the_window_leaves_the_tally(
        self, store: RedisBehaviorStore
    ) -> None:
        for _ in range(5):
            store.record("run", ("tool", "old"), True, maxlen=5, k=2)
        for _ in range(5):
            state = store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert state.errors == 0
        assert store._client.hget(f"{_PREFIX}:run:meta", "errors") == "0"


class TestExpiry:
    def test_every_key_carries_a_ttl(self, store: RedisBehaviorStore) -> None:
        """A window with no expiry is a leak that outlives the process that
        made it -- the same unbounded growth the in-memory lane closes by
        evicting at run end, expressed as a TTL for state nobody may return
        to."""
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        for key in store._keys("run"):
            assert store._client.pttl(key) > 0, f"{key} has no expiry"

    def test_the_ttl_refreshes_on_every_step(self, store: RedisBehaviorStore) -> None:
        """An idle timeout, not an absolute one. An absolute expiry would empty
        a long run's window mid-flight and reset its verdict to ``healthy`` --
        ADR-044's failure arriving on a timer."""
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store._client._redis.pexpire(f"{_PREFIX}:run:steps", 5_000)

        store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert store.ttl_seconds_remaining("run") > 50.0

    def test_the_error_tally_exists_before_the_first_error(self, store: RedisBehaviorStore) -> None:
        """Why the tally is seeded rather than created by the error branch.

        ``PEXPIRE`` on a key Redis has not created is a silent no-op. A meta
        key that appeared only with the first error would therefore be written
        at a point where the script's expiry calls had already run, leaving one
        key of the three unexpiring for runs that error late. Seeding it makes
        the "every key carries a TTL" invariant hold from step one.
        """
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert store._client.hget(f"{_PREFIX}:run:meta", "errors") == "0"
        assert store._client.pttl(f"{_PREFIX}:run:meta") > 0


class TestLifecycle:
    def test_closing_removes_every_key(self, store: RedisBehaviorStore) -> None:
        """Eviction has to release all three keys. Leaving the counts behind
        would let a re-opened run inherit counts for steps nobody can see."""
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        store.close_run("run", k=2)

        for key in store._keys("run"):
            assert not store._client.exists(key), f"{key} survived close"

    def test_closing_an_orphaned_counter_set_clears_it(self, store: RedisBehaviorStore) -> None:
        """The list can outlive nothing, but the counters can outlive the list.

        Only if something deleted the list alone -- an operator, an eviction
        policy. Left behind, they would give the next window counts for
        signatures nobody can see, inflating ``distinct_calls`` and repeat
        counts for a run that has barely started.
        """
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store._client.delete(f"{_PREFIX}:run:steps")

        assert store.close_run("run", k=2) is None
        assert not store._client.exists(f"{_PREFIX}:run:counts")

    def test_run_count_sees_only_this_backend(self, store: RedisBehaviorStore) -> None:
        """The ledger shares this Redis. A count that swept ``optio:*`` would
        report cost runs as behaviour runs and make leak detection useless."""
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store._client._redis.hset("optio:other:totals", "actual", "1.0")

        assert store.run_count() == 1


class TestTheConfiguredBackendActuallyReachesTheLane:
    """ADR-005's addendum exists because a setting was read by nobody.

    ``store_backend='redis'`` was accepted, validated, given an extra and an
    env var, and no lane ever looked at it. Asserting the config parses would
    have passed then too, so these assert the object graph instead -- and the
    behaviour lane is the second lane to be wired to it, which is exactly when
    the same omission is easiest to repeat.
    """

    def test_redis_config_builds_a_redis_backed_behavior_lane(self) -> None:
        from optio.config import Config
        from optio.lanes.behavior.lane import BehaviorLane
        from optio.lanes.registry import enabled_lanes
        from tests.integration.test_redis_ledger import REDIS_URL

        # Takes no `store` fixture -- it builds its own lanes -- so it has to
        # ask for the skip itself. Without this it *fails* rather than skips
        # wherever Redis is absent, which is every leg of the test matrix.
        connect_or_skip().close()

        lanes = enabled_lanes(Config(store_backend="redis", redis_url=REDIS_URL))

        behavior = next(lane for lane in lanes if isinstance(lane, BehaviorLane))
        assert isinstance(behavior._store, RedisBehaviorStore)

    def test_the_default_config_stays_in_process(self) -> None:
        """The zero-infrastructure default is the reason SC-1 is achievable;
        a change that quietly networked it would be a regression."""
        from optio.config import Config
        from optio.lanes.behavior.lane import BehaviorLane
        from optio.lanes.behavior.store_memory import InMemoryBehaviorStore
        from optio.lanes.registry import enabled_lanes

        lanes = enabled_lanes(Config())

        behavior = next(lane for lane in lanes if isinstance(lane, BehaviorLane))
        assert isinstance(behavior._store, InMemoryBehaviorStore)

    def test_each_lane_gets_its_own_client(self) -> None:
        """Independent lifecycles, by contract (Section 3.1).

        A shared connection would make the behaviour lane's timeout budget
        depend on whether the cost lane happened to be enabled, and closing one
        lane's client would silently blind the other.
        """
        from optio.config import Config
        from optio.lanes.behavior.lane import BehaviorLane
        from optio.lanes.cost.lane import CostLane
        from optio.lanes.cost.ledger_redis import RedisLedgerStore
        from optio.lanes.registry import enabled_lanes
        from tests.integration.test_redis_ledger import REDIS_URL

        connect_or_skip().close()

        lanes = enabled_lanes(Config(store_backend="redis", redis_url=REDIS_URL))

        behavior = next(lane for lane in lanes if isinstance(lane, BehaviorLane))
        cost = next(lane for lane in lanes if isinstance(lane, CostLane))
        assert isinstance(behavior._store, RedisBehaviorStore)
        assert isinstance(cost.ledger._store, RedisLedgerStore)
        assert behavior._store._client is not cost.ledger._store._client

    @needs_driver
    def test_an_unreachable_redis_fails_at_setup_not_at_runtime(self) -> None:
        """Loud here, fail-open later. A backend that cannot be reached at
        wiring time is a configuration error (Section 4.2); discovering it on
        the first step instead is how ADR-005's addendum got written.

        Covers the behaviour lane specifically: with the cost lane disabled,
        nothing else in the graph would touch Redis, so a lane that skipped its
        ping would reach the agent's path before failing.
        """
        from optio.config import Config
        from optio.lanes.registry import enabled_lanes
        from optio.store.redis_client import StoreUnavailableError

        # Port 1 is reserved and never listening, so this is a connection
        # refusal rather than a timeout -- fast and deterministic.
        config = Config(
            store_backend="redis",
            redis_url="redis://localhost:1/15",
            store_timeout_ms=200,
            cost_lane=False,
        )

        with pytest.raises(StoreUnavailableError):
            enabled_lanes(config)
