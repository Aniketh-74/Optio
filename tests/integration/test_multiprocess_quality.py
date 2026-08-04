"""Four processes, one run, one score.

The third instance of the milestone's shape, and the one where a wrong answer is
most likely to be believed. A cost total is arithmetic a reader could check
against their invoice; a loop verdict has a step count behind it. A quality score
has neither -- it is a number between 0 and 1 that looks equally plausible
whether it was computed from the whole run or from a quarter of it.

Two things break when the state is process-local:

**The judge is told the wrong run length.** ``step_count`` is documented as how
many steps the run took, and ``docs/quality.md`` shows users passing it straight
into their own evaluator. Four workers means each reports a quarter, and a judge
that scales its rubric by run length scores a long run as if it were short.

**The heuristic scores the wrong step.** It reads the run's *final* step, which
is where the answer is. Sharded, each worker has its own last step, and the
verdict comes from whichever worker happened to observe run end -- so a run that
ended in an error can be scored from a healthy step that finished earlier
somewhere else.

Real processes, not threads. Threads share a heap, so a thread-based version
would pass against the in-memory backend too and prove nothing about the thing
that actually fails.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from optio.lanes.quality import heuristic
from optio.lanes.quality.store import QualityStep
from optio.lanes.quality.store_memory import InMemoryQualityStore
from optio.lanes.quality.store_redis import RedisQualityStore
from optio.store.redis_client import RedisClient
from tests.integration.test_redis_ledger import REDIS_URL, connect_or_skip, reset_optio_keys

pytestmark = [pytest.mark.integration, pytest.mark.redis]

WORKERS = 4
STEPS_PER_WORKER = 25
RUN_ID = "shared-quality-run"

#: What the judge should be told the run took. The number each worker reports on
#: its own is this divided by four.
TOTAL_STEPS = WORKERS * STEPS_PER_WORKER

#: The last worker to write. Its final step is the one the run should be scored
#: from, and it is the only worker whose step says the run failed.
FAILING_WORKER = WORKERS - 1


def _healthy() -> QualityStep:
    """A step that looks like a completed generation."""
    return QualityStep(errored=False, finish_reasons=("stop",), output_tokens=50)


def _failed() -> QualityStep:
    """A step that ended in error -- the only conclusive failure signal."""
    return QualityStep(errored=True, finish_reasons=(), output_tokens=0)


def _work(worker: int, redis_url: str, gate: object) -> None:
    """Record this worker's share of the run.

    Runs in a spawned process, so it builds its own client -- a connection
    cannot be inherited across ``spawn``.

    **The URL is an argument, not read from the environment.**
    ``tests/conftest.py`` strips every ``OPTIO_*`` variable in an autouse
    fixture. The parent read ``REDIS_URL`` at import, before that ran; a spawned
    child re-imports after it, gets no variable, and silently falls back to the
    default port -- writing its share to whatever Redis happens to be there
    while the parent reads an empty one.

    Args:
        worker: This worker's index.
        redis_url: Where the shared store lives.
        gate: A barrier the failing worker waits on, so its step is provably
            last rather than last by luck.
    """
    client = RedisClient(redis_url, timeout_ms=2000)
    store = RedisQualityStore(client, ttl_seconds=60.0)

    if worker == FAILING_WORKER:
        # Wait for the others to finish before writing the run's final step.
        # Without this the "last step wins" assertion would be a race, and a
        # test that passes three times in four is worse than no test.
        gate.wait()  # type: ignore[attr-defined]
        for _ in range(STEPS_PER_WORKER - 1):
            store.record(RUN_ID, _healthy())
        store.record(RUN_ID, _failed())
    else:
        for _ in range(STEPS_PER_WORKER):
            store.record(RUN_ID, _healthy())
        gate.wait()  # type: ignore[attr-defined]

    client.close()


@pytest.fixture
def client() -> RedisClient:
    """A client on a clean optio keyspace, or a skip."""
    conn = connect_or_skip()
    reset_optio_keys(conn)
    return conn


def test_four_processes_produce_one_score(client: RedisClient) -> None:
    """The milestone's success criterion for this lane, as an assertion."""
    ctx = mp.get_context("spawn")
    gate = ctx.Barrier(WORKERS)
    procs = [ctx.Process(target=_work, args=(w, REDIS_URL, gate)) for w in range(WORKERS)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert not proc.is_alive(), "a worker hung"
        assert proc.exitcode == 0, f"worker exited {proc.exitcode}"

    store = RedisQualityStore(client, ttl_seconds=60.0)
    summary = store.close_run(RUN_ID)

    assert summary is not None, "the shared run was empty after four workers wrote to it"
    assert summary.step_count == TOTAL_STEPS, (
        f"the judge would be told {summary.step_count} of {TOTAL_STEPS} steps"
    )
    assert summary.last == _failed(), "the run was not scored from the step that finished last"
    assert heuristic.score(summary.last).failed is True

    reset_optio_keys(client)
    client.close()


def test_the_in_memory_backend_cannot_do_this_and_says_so() -> None:
    """Documents the limitation as a measurement rather than a sentence.

    Four in-memory stores are what four worker processes have. Each reports a
    quarter of the run's length, and three of the four end on a healthy step --
    so whichever worker observes run end decides the verdict, and three of the
    four possible outcomes are wrong.

    Asserting it here means the README's claim about the default backend is
    checked rather than believed.
    """
    workers = [InMemoryQualityStore() for _ in range(WORKERS)]
    for index, store in enumerate(workers):
        for step in range(STEPS_PER_WORKER):
            last = index == FAILING_WORKER and step == STEPS_PER_WORKER - 1
            store.record(RUN_ID, _failed() if last else _healthy())

    summaries = [store.close_run(RUN_ID) for store in workers]

    assert all(s is not None and s.step_count == STEPS_PER_WORKER for s in summaries), (
        "each isolated worker should see exactly its own share"
    )
    assert sum(s.step_count for s in summaries if s is not None) == TOTAL_STEPS, (
        "the shares add up to the run length no single worker can see"
    )

    verdicts = [heuristic.score(s.last if s else None).failed for s in summaries]
    assert verdicts.count(True) == 1, "only the worker holding the failing step can see it"
    assert verdicts.count(False) == WORKERS - 1, (
        "the other three would score the run from a step that was not its last"
    )
