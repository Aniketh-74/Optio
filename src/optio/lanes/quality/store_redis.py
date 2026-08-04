"""The Redis quality backend -- one logical run, however many processes.

The in-memory backend keeps state per process, and a process only sees its own
steps. A run sharded across four workers is scored by whichever worker happened
to observe run end, from the steps that landed in *that* worker: the judge is
told a quarter of the run's length, and the heuristic scores a step that was
last in one process rather than last overall.

Both failures are quiet. A judge given the wrong step count returns a plausible
score; a heuristic given the wrong final step returns a plausible verdict. This
is the third instance of the milestone's shape, and the one where the wrong
answer is most likely to be believed, because a quality score carries no
arithmetic a reader could check.

**One hash per run**, ``optio:q:{run}``:

===========  ==============================================================
field        holds
===========  ==============================================================
``steps``    how many steps the run has taken
``errored``  ``1``/``0`` for the latest step
``finish``   JSON array of the latest step's finish reasons
``tokens``   the latest step's output-token count, or ``-`` for absent
===========  ==============================================================

``-`` rather than an empty string or a ``0``, because absence is unknown and
zero is positive evidence the model produced nothing -- they lead to opposite
verdicts, and a hash cannot hold ``None``.

**The TTL is an idle timeout, refreshed on every step.** An absolute expiry
would drop a long run's state mid-flight and score it from nothing. There is no
tombstone: unlike a cost run, closing is not final here. A re-opened run counts
from zero, which reports what it can account for rather than resuming a total
whose steps were already scored.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from optio.lanes.quality.store import QualityStep, QualitySummary

if TYPE_CHECKING:
    from optio.store.redis_client import RedisClient

#: Namespace for every key this backend owns. The ``:q:`` segment keeps the
#: quality keyspace disjoint from the ledger's ``optio:{run}:*`` and the
#: behaviour lane's ``optio:b:*``, so a shared Redis stays legible and each
#: backend's ``run_count`` scans only its own.
_PREFIX: Final = "optio:q"

#: Stand-in for an unreported token count. A Redis hash field is a string and
#: cannot hold ``None``; using ``"0"`` would turn "the framework did not say"
#: into "the model produced nothing", which is the difference between abstaining
#: and declaring the run failed.
_ABSENT: Final = "-"

#: Record one step: increment the count, overwrite the latest step, refresh the
#: expiry. One script and one round trip, with a fixed-size payload -- the state
#: does not grow with the run, so neither does the write.
#:
#: Overwriting rather than appending is what makes "last" mean *last across all
#: workers*: whichever step reaches the server last is the one that stays, which
#: is the same rule the in-memory backend follows within one process.
_RECORD: Final = """
redis.call('HINCRBY', KEYS[1], 'steps', 1)
redis.call('HSET', KEYS[1], 'errored', ARGV[1], 'finish', ARGV[2], 'tokens', ARGV[3])
redis.call('PEXPIRE', KEYS[1], ARGV[4])
return 1
"""

#: Close: read the run's state and release it as one event. Reading then
#: deleting would be two round trips with a gap another worker's step can land
#: in, so the summary would describe a run that had already moved on.
#:
#: Returns Lua ``false`` -- Python ``None`` -- when there is no state, which is
#: what run end firing twice looks like (M1-2). Reporting an empty summary
#: instead would let the lane score from no evidence and emit a weaker verdict
#: over a judge result the user paid for.
_CLOSE: Final = """
if redis.call('EXISTS', KEYS[1]) == 0 then return false end
local state = redis.call('HMGET', KEYS[1], 'steps', 'errored', 'finish', 'tokens')
redis.call('DEL', KEYS[1])
return state
"""


class RedisQualityStore:
    """Per-run scoring state shared across processes.

    Structurally implements :class:`~optio.lanes.quality.store.QualityStore`.
    """

    def __init__(self, client: RedisClient, *, ttl_seconds: float) -> None:
        """Build the backend and register its scripts.

        Args:
            client: Connected Redis client.
            ttl_seconds: Idle expiry for a run's state, refreshed on each step.

        Raises:
            StoreUnavailableError: If the scripts cannot be loaded. Loud, at
                setup, rather than on the first step (Section 4.2).
        """
        self._client: Final = client
        self._ttl_ms: Final = int(ttl_seconds * 1000)
        for name, source in (("q_record", _RECORD), ("q_close", _CLOSE)):
            client.register_script(name, source)

    def _keys(self, run_id: str) -> list[str]:
        """The single key for a run, as the scripts expect it."""
        return [f"{_PREFIX}:{run_id}"]

    def record(self, run_id: str, step: QualityStep) -> None:
        """Note that a step happened, and that it was the latest.

        Args:
            run_id: The run's identifier.
            step: The step's projection. Never a span -- see
                :mod:`optio.lanes.quality.store`.

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        self._client.run_script(
            "q_record",
            self._keys(run_id),
            [
                "1" if step.errored else "0",
                json.dumps(step.finish_reasons),
                _ABSENT if step.output_tokens is None else str(step.output_tokens),
                str(self._ttl_ms),
            ],
        )

    def close_run(self, run_id: str) -> QualitySummary | None:
        """Release a run's state and return what it held.

        Args:
            run_id: The run's identifier.

        Returns:
            The summary, or ``None`` if the run held no state -- either it
            never recorded a step, or run end already fired once (M1-2).

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        raw = self._client.run_script("q_close", self._keys(run_id), [])
        if raw is None:
            return None

        steps, errored, finish, tokens = raw
        return QualitySummary(
            step_count=int(steps),
            last=QualityStep(
                errored=errored == "1",
                finish_reasons=tuple(json.loads(finish)),
                output_tokens=None if tokens == _ABSENT else int(tokens),
            ),
        )

    def run_count(self) -> int:
        """Return how many runs currently hold state.

        Scans this backend's own keyspace, so it is O(keyspace) rather than
        O(1). Diagnostic only -- it exists for leak detection, the same reason
        the in-memory backend exposes it, and is never on a request path.

        Returns:
            Number of runs with retained state.
        """
        return self._client.count_keys(f"{_PREFIX}:*")

    def ttl_seconds_remaining(self, run_id: str) -> float:
        """Seconds before this run's state expires, or ``0.0`` if it is gone.

        Args:
            run_id: The run's identifier.

        Returns:
            Remaining lifetime in seconds.
        """
        return self._client.pttl(self._keys(run_id)[0]) / 1000.0

    def __repr__(self) -> str:
        """Return a debug representation with the tracked run count."""
        return f"<RedisQualityStore runs={self.run_count()}>"
