"""Four processes, one run, one correct total.

This is the test the milestone exists for, and the only one that demonstrates
the bug rather than describing it. Against the in-memory backend each worker
meters into its own dictionary, so a ``$0.50`` budget admits ``$2.00`` while
every process's arithmetic is internally consistent -- R-TECH-1's worst case,
wrong rather than broken.

Real processes, not threads. Threads share a heap, so a thread-based version
would pass against the in-memory backend too and prove nothing about the thing
that actually fails.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from optio.lanes.cost.ledger_memory import InMemoryLedgerStore
from optio.lanes.cost.ledger_redis import RedisLedgerStore
from optio.store.redis_client import RedisClient
from tests.integration.test_redis_ledger import REDIS_URL, connect_or_skip, reset_optio_keys

pytestmark = [pytest.mark.integration, pytest.mark.redis]

WORKERS = 4
STEPS_PER_WORKER = 25
COST_PER_STEP = 0.01
RUN_ID = "shared-run"

#: What four workers spend between them. The number a budget policy would gate
#: on, and the number each process reports on its own if state is not shared.
EXPECTED_TOTAL = WORKERS * STEPS_PER_WORKER * COST_PER_STEP


def _meter(worker: int, redis_url: str) -> None:
    """Reserve and reconcile a fixed number of steps against the shared run.

    Runs in a spawned process, so it builds its own client -- a connection
    cannot be inherited across ``spawn``.

    **The URL is an argument, not read from the environment**, and that is not
    style. ``tests/conftest.py`` strips every ``OPTIO_*`` variable in an autouse
    fixture so a developer's shell cannot skew results. The parent read
    ``REDIS_URL`` at import, before that ran; a spawned child re-imports this
    module *after* it, gets no variable, and silently falls back to the default
    port -- writing its share to whatever Redis happens to be there while the
    parent reads an empty one. That is exactly how this test failed the first
    time it ran, and passing the value removes the coupling rather than working
    around it.
    """
    client = RedisClient(redis_url, timeout_ms=2000)
    store = RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    for step in range(STEPS_PER_WORKER):
        step_id = f"w{worker}-s{step}"
        store.reserve(RUN_ID, step_id, COST_PER_STEP)
        store.reconcile(RUN_ID, step_id, COST_PER_STEP)
    client.close()


@pytest.fixture
def client() -> RedisClient:
    """A client on a clean optio keyspace, or a skip."""
    conn = connect_or_skip()
    reset_optio_keys(conn)
    return conn


def test_four_processes_produce_one_correct_total(client: RedisClient) -> None:
    """The milestone's success criterion, as an assertion."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_meter, args=(w, REDIS_URL)) for w in range(WORKERS)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert not proc.is_alive(), "a worker hung"
        assert proc.exitcode == 0, f"worker exited {proc.exitcode}"

    store = RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    snap = store.snapshot(RUN_ID)

    assert snap.actual == pytest.approx(EXPECTED_TOTAL), (
        f"four processes metered {snap.actual} against an expected "
        f"{EXPECTED_TOTAL}; the shared ledger lost or duplicated updates"
    )
    assert snap.reconciled_steps == WORKERS * STEPS_PER_WORKER
    assert snap.reserved == pytest.approx(0.0), "reservations were left open"

    reset_optio_keys(client)
    client.close()


def test_the_in_memory_backend_cannot_do_this_and_says_so() -> None:
    """Documents the limitation as a measurement rather than a sentence.

    Two in-memory stores are what two worker processes have: separate objects,
    separate dictionaries. Each sees a quarter of the spend and reports it as
    the whole, which is why a budget multiplies by worker count. Asserting it
    here means the README's claim about the default backend is checked rather
    than believed.
    """
    workers = [InMemoryLedgerStore() for _ in range(WORKERS)]
    for index, store in enumerate(workers):
        for step in range(STEPS_PER_WORKER):
            step_id = f"w{index}-s{step}"
            store.reserve(RUN_ID, step_id, COST_PER_STEP)
            store.reconcile(RUN_ID, step_id, COST_PER_STEP)

    per_worker = [store.snapshot(RUN_ID).actual for store in workers]

    assert all(seen == pytest.approx(EXPECTED_TOTAL / WORKERS) for seen in per_worker), (
        "each in-memory worker should see exactly its own share"
    )
    assert sum(per_worker) == pytest.approx(EXPECTED_TOTAL), (
        "the shares should add up to the truth no single worker can see"
    )
