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
from tests.integration.test_redis_ledger import connect_or_skip, reset_optio_keys

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
