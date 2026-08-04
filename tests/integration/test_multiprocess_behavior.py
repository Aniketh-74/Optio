"""Four processes, one run, one verdict.

The behaviour lane's version of the budget bug, and the same shape: each worker
classifies from its own slice of the steps and every slice is internally
consistent. The difference is which way it fails. A sharded budget *over*-spends
because four quarters each look affordable; a sharded window *under*-reports,
because four quarters each fall below ``MIN_STEPS_FOR_VERDICT`` and the detector
is required to answer ``healthy`` when the evidence is thin (Section 6.4).

So the agent is stuck, the operator sees ``healthy`` from every worker, and
nothing is logged -- the quiet direction. Nothing raises, no metric moves, and
the lane is doing exactly what it was told.

Real processes, not threads. Threads share a heap, so a thread-based version
would pass against the in-memory backend too and prove nothing about the thing
that actually fails.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from optio import semconv
from optio.lanes.behavior.detectors import (
    LOOP_MAX_DISTINCT,
    MIN_STEPS_FOR_VERDICT,
    classify_state,
)
from optio.lanes.behavior.store import WindowState
from optio.lanes.behavior.store_memory import InMemoryBehaviorStore
from optio.lanes.behavior.store_redis import RedisBehaviorStore
from optio.store.redis_client import RedisClient
from tests.integration.test_redis_ledger import REDIS_URL, connect_or_skip, reset_optio_keys

pytestmark = [pytest.mark.integration, pytest.mark.redis]

WORKERS = 4
STEPS_PER_WORKER = 3
RUN_ID = "shared-behavior-run"
WINDOW = 50

#: The textbook stuck agent: read, think, read, think, forever. Two calls, so
#: neither holds a majority alone -- which is why dominance is measured over the
#: top ``LOOP_MAX_DISTINCT`` rather than over the single most frequent call.
CYCLE = [("tool", "read"), ("tool", "think")]

#: Three steps each. Deliberately below MIN_STEPS_FOR_VERDICT so a worker acting
#: alone *must* answer healthy, and twelve together so the shared window clears
#: it comfortably. If this ever stops being below the threshold the test proves
#: nothing, so the arithmetic is asserted rather than assumed.
TOTAL_STEPS = WORKERS * STEPS_PER_WORKER


def _step(worker: int, redis_url: str) -> None:
    """Add this worker's share of the cycle to the shared run.

    Runs in a spawned process, so it builds its own client -- a connection
    cannot be inherited across ``spawn``.

    **The URL is an argument, not read from the environment.**
    ``tests/conftest.py`` strips every ``OPTIO_*`` variable in an autouse
    fixture. The parent read ``REDIS_URL`` at import, before that ran; a spawned
    child re-imports after it, gets no variable, and silently falls back to the
    default port -- writing its share to whatever Redis happens to be there
    while the parent reads an empty one. That is how the budget version of this
    test failed the first time it ran.
    """
    client = RedisClient(redis_url, timeout_ms=2000)
    store = RedisBehaviorStore(client, ttl_seconds=60.0)
    for step in range(STEPS_PER_WORKER):
        tool, digest = CYCLE[(worker + step) % len(CYCLE)]
        store.record(RUN_ID, (tool, digest), False, maxlen=WINDOW, k=LOOP_MAX_DISTINCT)
    client.close()


@pytest.fixture
def client() -> RedisClient:
    """A client on a clean optio keyspace, or a skip."""
    conn = connect_or_skip()
    reset_optio_keys(conn)
    return conn


def test_the_workers_alone_could_not_reach_a_verdict() -> None:
    """The premise, asserted rather than assumed.

    If ``STEPS_PER_WORKER`` ever rises above the evidence floor, the test below
    would pass whether or not state was shared -- and would look like a proof.
    """
    assert STEPS_PER_WORKER < MIN_STEPS_FOR_VERDICT
    assert TOTAL_STEPS >= MIN_STEPS_FOR_VERDICT


def test_four_processes_produce_one_looping_verdict(client: RedisClient) -> None:
    """The milestone's success criterion for this lane, as an assertion."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_step, args=(w, REDIS_URL)) for w in range(WORKERS)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert not proc.is_alive(), "a worker hung"
        assert proc.exitcode == 0, f"worker exited {proc.exitcode}"

    store = RedisBehaviorStore(client, ttl_seconds=60.0)
    state = store.close_run(RUN_ID, k=LOOP_MAX_DISTINCT)

    assert state is not None, "the shared window was empty after four workers wrote to it"
    assert state.size == TOTAL_STEPS, (
        f"the shared window holds {state.size} of {TOTAL_STEPS} steps; a worker's steps were lost"
    )
    assert state.distinct_calls == len(CYCLE)
    assert classify_state(state).state == semconv.LOOP_STATE_LOOPING

    reset_optio_keys(client)
    client.close()


def test_the_in_memory_backend_cannot_do_this_and_says_so() -> None:
    """Documents the limitation as a measurement rather than a sentence.

    Four in-memory stores are what four worker processes have: separate
    objects, separate dictionaries. Each holds three steps, which is below the
    evidence floor, so each answers ``healthy`` -- correctly, given what it can
    see. Nothing is wrong with any worker's reasoning, which is precisely why
    the failure is silent.

    Asserting it here means the README's claim about the default backend is
    checked rather than believed.
    """

    def _record(store: InMemoryBehaviorStore, worker: int, step: int) -> WindowState:
        call = CYCLE[(worker + step) % len(CYCLE)]
        return store.record(RUN_ID, call, False, maxlen=WINDOW, k=LOOP_MAX_DISTINCT)

    sharded = [InMemoryBehaviorStore() for _ in range(WORKERS)]
    verdicts = [
        classify_state(_record(store, worker, step))
        for worker, store in enumerate(sharded)
        for step in range(STEPS_PER_WORKER)
    ]

    assert all(v.state == semconv.LOOP_STATE_HEALTHY for v in verdicts), (
        "each isolated worker should report healthy -- that is the bug"
    )

    # The same twelve steps into one store. This is what separates "the steps
    # were not pathological" from "the sharding hid it", and only the second is
    # a bug worth a milestone.
    together = InMemoryBehaviorStore()
    for worker in range(WORKERS):
        for step in range(STEPS_PER_WORKER):
            state = _record(together, worker, step)

    assert classify_state(state).state == semconv.LOOP_STATE_LOOPING, (
        "the very same steps are a loop once one window can see all of them"
    )
