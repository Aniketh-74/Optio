"""The eviction arithmetic exists twice; this is what keeps it identical.

``BehaviorWindow.add`` decrements a departing call's count and deletes the key
at zero, and the Lua in ``store_redis`` does the same thing in another language
against a Redis hash. A divergence between them would not raise. It would
produce a slightly different ``WindowState`` on one backend, and therefore a
different verdict for the same run depending on where it ran -- the
silent-wrongness class this project treats as its worst failure mode.

Asserting on the state rather than on the verdict is deliberate and strictly
stronger: ``classify_state`` is a pure function of these four numbers, so equal
states imply equal verdicts, while equal verdicts would tolerate two backends
disagreeing about ``distinct_calls`` anywhere the thresholds happen to absorb
it. Absorbed today, load-bearing after one threshold change.

The generators are tuned to make eviction the common case rather than the edge:
a short window, a four-word call alphabet so counts collide and reach zero, and
errors mixed in so the tally evicts too. ``k`` varies because the reduction's
cap is separate logic from the counting, and a bug at ``k`` of one or three
would be invisible at the two the detector actually uses.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optio.lanes.behavior.store_memory import InMemoryBehaviorStore
from optio.lanes.behavior.store_redis import RedisBehaviorStore
from tests.integration.test_redis_ledger import connect_or_skip

#: ``redis`` is what the gate selects on, so these run against the service
#: container rather than skipping there.
pytestmark = [pytest.mark.property, pytest.mark.redis]

#: One run id, reused. Each example closes it first, which deletes every key --
#: so examples cannot contaminate each other and the keyspace stays at three
#: keys however many examples Hypothesis draws.
_RUN = "prop"

_CALLS = st.sampled_from(["read", "write", "search", "plan"])
_STEPS = st.lists(st.tuples(_CALLS, st.booleans()), min_size=1, max_size=80)


@pytest.fixture(scope="module")
def redis_store() -> Iterator[RedisBehaviorStore]:
    """One connection for the whole module.

    Module-scoped because a per-example connection would dominate the runtime
    and because a function-scoped fixture under ``@given`` is reused across
    examples anyway -- which Hypothesis flags, since it makes the fixture's
    setup silently a no-op after the first draw.
    """
    client = connect_or_skip(timeout_ms=1000)
    client._redis.flushdb()
    yield RedisBehaviorStore(client, ttl_seconds=60.0)
    client._redis.flushdb()
    client.close()


@given(
    steps=_STEPS,
    maxlen=st.integers(min_value=2, max_value=12),
    k=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=40)
def test_both_backends_agree_step_for_step(
    redis_store: RedisBehaviorStore,
    steps: list[tuple[str, bool]],
    maxlen: int,
    k: int,
) -> None:
    """Compared after every step, not at the end.

    A divergence that appears mid-window and is later evicted away would be
    invisible in a final comparison -- and a verdict is read on every step, so
    a state that was wrong only in the middle is a wrong signal that was
    genuinely emitted.
    """
    memory = InMemoryBehaviorStore()
    redis_store.close_run(_RUN, k=k)

    for index, (call, errored) in enumerate(steps):
        want = memory.record(_RUN, ("tool", call), errored, maxlen=maxlen, k=k)
        got = redis_store.record(_RUN, ("tool", call), errored, maxlen=maxlen, k=k)

        assert got == want, f"backends diverged at step {index} ({call!r}, errored={errored})"


@given(
    steps=_STEPS,
    maxlen=st.integers(min_value=2, max_value=12),
    k=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=25)
def test_both_backends_agree_on_the_closing_state(
    redis_store: RedisBehaviorStore,
    steps: list[tuple[str, bool]],
    maxlen: int,
    k: int,
) -> None:
    """The close path reduces the window a second time, in its own script.

    ``record`` and ``close`` share the reduction source, so this looks
    redundant -- until someone edits one script. It is also the state that
    reaches the run span, which is the verdict a user actually keeps.
    """
    memory = InMemoryBehaviorStore()
    redis_store.close_run(_RUN, k=k)

    for call, errored in steps:
        memory.record(_RUN, ("tool", call), errored, maxlen=maxlen, k=k)
        redis_store.record(_RUN, ("tool", call), errored, maxlen=maxlen, k=k)

    assert redis_store.close_run(_RUN, k=k) == memory.close_run(_RUN, k=k)


@given(k=st.integers(min_value=1, max_value=3))
@settings(max_examples=5)
def test_both_backends_report_absence_the_same_way(redis_store: RedisBehaviorStore, k: int) -> None:
    """``None`` on both, for a run neither has seen.

    The asymmetry worth guarding: an empty ``WindowState`` from one backend and
    ``None`` from the other would let a second run-end emit ``healthy`` over a
    real ``looping`` verdict on exactly one deployment (ADR-044).
    """
    redis_store.close_run(_RUN, k=k)

    assert redis_store.close_run(_RUN, k=k) is None
    assert InMemoryBehaviorStore().close_run(_RUN, k=k) is None
