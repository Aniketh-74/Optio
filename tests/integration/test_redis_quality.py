"""The Redis quality backend against a real server.

What only a real server can show: that the state expires as an idle timeout
rather than a deadline, that closing releases the key, that the keyspace stays
disjoint from the other two lanes', and that a value with no string form --
``None`` -- survives a hash that can only hold strings.

The behavioural rules the two backends share live in
``tests/unit/test_quality_store_contract.py``, which runs against this backend
too.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from optio.lanes.quality.store import QualityStep
from optio.lanes.quality.store_redis import _PREFIX, RedisQualityStore
from tests.integration.test_redis_ledger import connect_or_skip, needs_driver, reset_optio_keys

pytestmark = [pytest.mark.integration, pytest.mark.redis]


@pytest.fixture
def store() -> Iterator[RedisQualityStore]:
    """A backend against a live server, on a clean optio keyspace."""
    client = connect_or_skip(timeout_ms=1000)
    reset_optio_keys(client)
    yield RedisQualityStore(client, ttl_seconds=60.0)
    reset_optio_keys(client)
    client.close()


def _step(*, tokens: int | None = 50) -> QualityStep:
    return QualityStep(errored=False, finish_reasons=(), output_tokens=tokens)


class TestTheStateDoesNotGrowWithTheRun:
    def test_a_long_run_holds_one_hash_of_four_fields(self, store: RedisQualityStore) -> None:
        """The reason the span buffer could go entirely.

        Ten thousand steps cost the same as one, because run end reads a count
        and the final step. A backend that buffered steps to be drained later
        would pass every behavioural test in the contract suite and grow without
        bound here.
        """
        for n in range(10_000):
            store.record("run", _step(tokens=n))

        assert store._client._redis.hlen(f"{_PREFIX}:run") == 4

    def test_the_write_payload_is_fixed_size(self, store: RedisQualityStore) -> None:
        """Corollary on the wire: a step costs the same however long the run
        has been going, so per-step latency cannot drift upward over hours."""
        for n in range(500):
            store.record("run", _step(tokens=n))

        summary = store.close_run("run")

        assert summary is not None
        assert summary.step_count == 500
        assert summary.last == _step(tokens=499)


class TestAbsenceSurvivesAHashThatOnlyHoldsStrings:
    def test_an_unreported_token_count_comes_back_as_none(self, store: RedisQualityStore) -> None:
        """A Redis hash field is a string and cannot hold ``None``. Storing
        ``"0"`` would turn "the framework did not report a count" into "the
        model produced nothing" -- abstain becomes declare-failed."""
        store.record("run", _step(tokens=None))

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last is not None
        assert summary.last.output_tokens is None

    def test_a_genuine_zero_is_not_confused_with_absence(self, store: RedisQualityStore) -> None:
        """The other half. Zero output tokens *is* evidence of failure, and the
        sentinel must not swallow it."""
        store.record("run", _step(tokens=0))

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last is not None
        assert summary.last.output_tokens == 0

    def test_finish_reasons_round_trip_through_the_hash(self, store: RedisQualityStore) -> None:
        store.record(
            "run", QualityStep(errored=True, finish_reasons=("length", "stop"), output_tokens=None)
        )

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last == QualityStep(
            errored=True, finish_reasons=("length", "stop"), output_tokens=None
        )

    def test_no_finish_reasons_round_trips_as_empty(self, store: RedisQualityStore) -> None:
        store.record("run", _step())

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last is not None
        assert summary.last.finish_reasons == ()


class TestExpiry:
    def test_the_key_carries_a_ttl(self, store: RedisQualityStore) -> None:
        """State with no expiry outlives the process that made it. The
        in-memory lane releases at run end; a run whose end never arrives needs
        the server to forget it."""
        store.record("run", _step())

        assert store._client.pttl(f"{_PREFIX}:run") > 0

    def test_the_ttl_refreshes_on_every_step(self, store: RedisQualityStore) -> None:
        """An idle timeout, not an absolute one. An absolute expiry would drop a
        long run's state mid-flight and score it from nothing -- ADR-044's
        failure arriving on a timer."""
        store.record("run", _step())
        store._client._redis.pexpire(f"{_PREFIX}:run", 5_000)

        store.record("run", _step())

        assert store.ttl_seconds_remaining("run") > 50.0


class TestLifecycle:
    def test_closing_removes_the_key(self, store: RedisQualityStore) -> None:
        store.record("run", _step())

        store.close_run("run")

        assert not store._client.exists(f"{_PREFIX}:run")

    def test_run_count_sees_only_this_backend(self, store: RedisQualityStore) -> None:
        """All three lanes share this Redis. A count that swept ``optio:*``
        would report cost and behaviour runs as quality runs and make leak
        detection useless."""
        store.record("run", _step())
        store._client._redis.hset("optio:other:totals", "actual", "1.0")
        store._client._redis.rpush("optio:b:other:steps", "0tool")

        assert store.run_count() == 1


class TestTheConfiguredBackendActuallyReachesTheLane:
    """ADR-005's addendum exists because a setting was read by nobody.

    ``store_backend='redis'`` was accepted, validated, given an extra and an
    env var, and no lane ever looked at it. Asserting the config parses would
    have passed then too, so these assert the object graph. This is the third
    and last lane wired to that setting, and the one most exposed to the same
    omission: quality is off by default, so nothing in a default configuration
    would have noticed the branch missing.
    """

    def test_redis_config_builds_a_redis_backed_quality_lane(self) -> None:
        from optio.config import Config
        from optio.lanes.quality.lane import QualityLane
        from optio.lanes.registry import enabled_lanes
        from tests.integration.test_redis_ledger import REDIS_URL

        connect_or_skip().close()

        lanes = enabled_lanes(Config(quality_lane=True, store_backend="redis", redis_url=REDIS_URL))

        quality = next(lane for lane in lanes if isinstance(lane, QualityLane))
        assert isinstance(quality._store, RedisQualityStore)

    def test_the_default_config_stays_in_process(self) -> None:
        """The zero-infrastructure default is why SC-1 is achievable; a change
        that quietly networked it would be a regression."""
        from optio.config import Config
        from optio.lanes.quality.lane import QualityLane
        from optio.lanes.quality.store_memory import InMemoryQualityStore
        from optio.lanes.registry import enabled_lanes

        lanes = enabled_lanes(Config(quality_lane=True))

        quality = next(lane for lane in lanes if isinstance(lane, QualityLane))
        assert isinstance(quality._store, InMemoryQualityStore)

    def test_all_three_lanes_get_their_own_client(self) -> None:
        """Independent lifecycles, by contract (Section 3.1).

        A shared connection would make each lane's timeout budget depend on
        which other lanes happened to be enabled, and closing one lane's client
        would silently blind the others.
        """
        from optio.config import Config
        from optio.lanes.behavior.lane import BehaviorLane
        from optio.lanes.behavior.store_redis import RedisBehaviorStore
        from optio.lanes.cost.lane import CostLane
        from optio.lanes.cost.ledger_redis import RedisLedgerStore
        from optio.lanes.quality.lane import QualityLane
        from optio.lanes.registry import enabled_lanes
        from tests.integration.test_redis_ledger import REDIS_URL

        connect_or_skip().close()

        lanes = enabled_lanes(Config(quality_lane=True, store_backend="redis", redis_url=REDIS_URL))

        quality = next(lane for lane in lanes if isinstance(lane, QualityLane))
        behavior = next(lane for lane in lanes if isinstance(lane, BehaviorLane))
        cost = next(lane for lane in lanes if isinstance(lane, CostLane))
        assert isinstance(quality._store, RedisQualityStore)
        assert isinstance(behavior._store, RedisBehaviorStore)
        assert isinstance(cost.ledger._store, RedisLedgerStore)

        clients = [quality._store._client, behavior._store._client, cost.ledger._store._client]
        assert len({id(client) for client in clients}) == 3

    def test_a_disabled_quality_lane_opens_no_connection(self) -> None:
        """Off by default (ADR-003), and that has to mean *no infrastructure
        touched*. Building a client for a lane nobody enabled would make an
        unreachable Redis fail setup for users who never asked for scoring."""
        from optio.config import Config
        from optio.lanes.registry import enabled_lanes

        # Port 1 is reserved and never listening. With quality off and the other
        # two lanes off, nothing should reach for it.
        config = Config(
            quality_lane=False,
            cost_lane=False,
            behavior_lane=False,
            store_backend="redis",
            redis_url="redis://localhost:1/15",
            store_timeout_ms=200,
        )

        assert enabled_lanes(config) == []

    @needs_driver
    def test_an_unreachable_redis_fails_at_setup_not_at_runtime(self) -> None:
        """Loud here, fail-open later (Section 4.2).

        Covers the quality lane specifically: with the other two disabled,
        nothing else in the graph would touch Redis, so a lane that skipped its
        ping would reach the agent's path before failing.
        """
        from optio.config import Config
        from optio.lanes.registry import enabled_lanes
        from optio.store.redis_client import StoreUnavailableError

        config = Config(
            quality_lane=True,
            cost_lane=False,
            behavior_lane=False,
            store_backend="redis",
            redis_url="redis://localhost:1/15",
            store_timeout_ms=200,
        )

        with pytest.raises(StoreUnavailableError):
            enabled_lanes(config)
