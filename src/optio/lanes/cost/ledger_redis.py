"""The Redis ledger backend -- one logical run, however many processes.

The in-memory backend guarantees exactly-once reconciliation with a lock, and a
lock says nothing to another process. Four workers metering one run each hold a
quarter of the truth and each believes it holds all of it, so a ``$0.50`` budget
admits ``$2.00`` -- silently, because every process's arithmetic is internally
consistent (R-TECH-1's worst case).

**Every compound operation is one Lua script.** Not for speed: for atomicity.
``reconcile`` has to check a reservation is open, remove it, and fold the cost
into the total, and those three steps must not interleave with another worker's.
Done as separate round trips they can, and the result is a total that is wrong
rather than missing.

Three keys per run, because they expire on different schedules:

===========================  ======  ==============================================
key                          type    holds
===========================  ======  ==============================================
``optio:{run}:open``         hash    ``step_id -> projected``
``optio:{run}:totals``       hash    ``actual``, ``reconciled``, ``leaked``, ``closed``
``optio:{run}:done``         string  tombstone; outlives the other two
===========================  ======  ==============================================

**The TTL is an idle timeout, refreshed on every write.** An absolute expiry
would drop a long run's open reservations mid-flight and send
``budget_remaining`` back to full -- ADR-044's failure arriving on a timer.

**The tombstone outlives the payload** so :meth:`is_finalised` keeps answering
after the run's data expires. Without it a late callback would find no state,
be treated as a new run, and start a second total under an id that was already
reported -- exactly what the in-memory backend's ``_recently_closed`` window
prevents, expressed as a TTL instead of a bounded FIFO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from optio.errors import LedgerInvariantError
from optio.lanes.cost.ledger_store import LedgerSnapshot

if TYPE_CHECKING:
    from optio.store.redis_client import RedisClient

#: Namespace for every key this backend owns, so a shared Redis stays legible
#: and :meth:`RedisLedgerStore.run_count` can scan only its own keyspace.
_PREFIX: Final = "optio"

#: Reserve. Refuses a closed run, then writes the field -- replacing any
#: previous value for this step, because frameworks retry steps and reuse ids.
#:
#: One script rather than EXISTS-then-HSET: between two round trips another
#: worker can close the run, and the reservation would land on a total that has
#: already been reported.
_RESERVE: Final = """
if redis.call('EXISTS', KEYS[3]) == 1 then return 'CLOSED' end
if redis.call('HGET', KEYS[2], 'closed') == '1' then return 'CLOSED' end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HSETNX', KEYS[2], 'actual', '0')
redis.call('HSETNX', KEYS[2], 'reconciled', '0')
redis.call('HSETNX', KEYS[2], 'leaked', '0')
redis.call('PEXPIRE', KEYS[1], ARGV[3])
redis.call('PEXPIRE', KEYS[2], ARGV[3])
return 'OK'
"""

#: Reconcile -- the reason this file exists. Check the reservation is open,
#: remove it, and fold the cost into the total, atomically. Split across round
#: trips, two workers interleave between the check and the delete and the same
#: cost is counted twice, or neither is counted at all.
_RECONCILE: Final = """
if redis.call('EXISTS', KEYS[3]) == 1 then return 'CLOSED' end
if redis.call('HGET', KEYS[2], 'closed') == '1' then return 'CLOSED' end
if redis.call('EXISTS', KEYS[2]) == 0 then return 'UNKNOWN' end
if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 0 then return 'NOTOPEN' end
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('HINCRBYFLOAT', KEYS[2], 'actual', ARGV[2])
redis.call('HINCRBY', KEYS[2], 'reconciled', 1)
redis.call('PEXPIRE', KEYS[1], ARGV[3])
redis.call('PEXPIRE', KEYS[2], ARGV[3])
return 'OK'
"""

#: Snapshot. Derives ``reserved`` by summing the open reservations rather than
#: reading a maintained total, matching the in-memory backend: a separately
#: accumulated total is one missed decrement from drifting, and the drift would
#: look plausible. A script so the sum and the totals come from one instant.
_SNAPSHOT: Final = """
local reserved = 0
local vals = redis.call('HVALS', KEYS[1])
for i = 1, #vals do reserved = reserved + tonumber(vals[i]) end
local actual = redis.call('HGET', KEYS[2], 'actual')
local reconciled = redis.call('HGET', KEYS[2], 'reconciled')
local leaked = redis.call('HGET', KEYS[2], 'leaked')
return {
  tostring(reserved),
  actual or '0',
  reconciled or '0',
  leaked or '0',
  tostring(redis.call('HLEN', KEYS[1]))
}
"""

#: Close. Counts what is still open as leaked -- assigned, not accumulated, so
#: a repeated close cannot drift the number -- and writes the tombstone.
_CLOSE: Final = """
if redis.call('HGET', KEYS[2], 'closed') ~= '1' then
  redis.call('HSET', KEYS[2], 'leaked', redis.call('HLEN', KEYS[1]))
  redis.call('HSET', KEYS[2], 'closed', '1')
  redis.call('HSETNX', KEYS[2], 'actual', '0')
  redis.call('HSETNX', KEYS[2], 'reconciled', '0')
end
redis.call('SET', KEYS[3], '1', 'PX', ARGV[1])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('PEXPIRE', KEYS[2], ARGV[2])
return 'OK'
"""


class RedisLedgerStore:
    """Reserve/reconcile accounting shared across processes.

    Structurally implements
    :class:`~optio.lanes.cost.ledger_store.LedgerStore`.
    """

    def __init__(
        self,
        client: RedisClient,
        *,
        ttl_seconds: float,
        tombstone_ttl_seconds: float,
    ) -> None:
        """Build the backend and register its scripts.

        Args:
            client: Connected Redis client.
            ttl_seconds: Idle expiry for a run's state, refreshed on each write.
            tombstone_ttl_seconds: How long finality is remembered after the
                state itself expires. Longer than ``ttl_seconds``, or a late
                callback could resurrect a closed run.

        Raises:
            StoreUnavailableError: If the scripts cannot be loaded.
        """
        self._client: Final = client
        self._ttl_ms: Final = int(ttl_seconds * 1000)
        self._tombstone_ttl_ms: Final = int(tombstone_ttl_seconds * 1000)
        for name, source in (
            ("reserve", _RESERVE),
            ("reconcile", _RECONCILE),
            ("snapshot", _SNAPSHOT),
            ("close", _CLOSE),
        ):
            client.register_script(name, source)

    def _keys(self, run_id: str) -> list[str]:
        """The three keys for a run, in the order every script expects."""
        return [
            f"{_PREFIX}:{run_id}:open",
            f"{_PREFIX}:{run_id}:totals",
            f"{_PREFIX}:{run_id}:done",
        ]

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record the worst-case cost of a step before it runs.

        Args:
            run_id: The run's identifier.
            step_id: Identifier for this step, stable across a retry.
            projected: Worst-case cost in USD, finite and non-negative.

        Raises:
            LedgerInvariantError: If ``projected`` is negative, or the run is
                closed.
            StoreUnavailableError: If Redis cannot be reached.
        """
        if projected < 0:
            raise LedgerInvariantError(
                f"cannot reserve a negative cost ({projected}) for {run_id}/{step_id}"
            )

        result = self._client.run_script(
            "reserve",
            self._keys(run_id),
            [step_id, repr(float(projected)), str(self._ttl_ms)],
        )
        if result == "CLOSED":
            raise LedgerInvariantError(
                f"cannot reserve on closed run {run_id!r}; the run's cost has already been reported"
            )

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost.

        Args:
            run_id: The run's identifier.
            step_id: The step being reconciled.
            actual: Actual cost in USD.

        Raises:
            LedgerInvariantError: On a double reconcile, a reconcile with no
                matching reservation, a negative cost, or a closed run.
            StoreUnavailableError: If Redis cannot be reached.
        """
        if actual < 0:
            raise LedgerInvariantError(
                f"cannot reconcile a negative cost ({actual}) for {run_id}/{step_id}"
            )

        result = self._client.run_script(
            "reconcile",
            self._keys(run_id),
            [step_id, repr(float(actual)), str(self._ttl_ms)],
        )
        if result == "CLOSED":
            raise LedgerInvariantError(
                f"cannot reconcile {run_id}/{step_id} on a closed run; "
                f"the run's cost has already been reported"
            )
        if result == "UNKNOWN":
            raise LedgerInvariantError(
                f"reconcile for unknown run {run_id!r}; reserve must come first"
            )
        if result == "NOTOPEN":
            raise LedgerInvariantError(
                f"no open reservation for {run_id}/{step_id}; "
                f"either reconciled twice or never reserved"
            )

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        """Return a consistent view of a run's cost state.

        Args:
            run_id: The run's identifier.

        Returns:
            The snapshot. An unknown run yields an all-zero snapshot, since a
            run that has spent nothing and a run that does not exist are the
            same thing to a consumer.
        """
        raw = self._client.run_script("snapshot", self._keys(run_id), [])
        reserved = float(raw[0])
        actual = float(raw[1])
        return LedgerSnapshot(
            reserved=reserved,
            actual=actual,
            committed=reserved + actual,
            open_steps=int(raw[4]),
            reconciled_steps=int(raw[2]),
            leaked_steps=int(raw[3]),
        )

    def close_run(self, run_id: str) -> LedgerSnapshot:
        """Finalise a run, recording any leaked reservations.

        A reservation still open at run end never got its actual cost, and is
        kept in the snapshot rather than discarded: dropping it would make the
        run look cheaper than the evidence supports, and under-reporting is the
        direction that lets an over-budget run through.

        Idempotent -- run end can fire more than once (M1-2).

        Args:
            run_id: The run's identifier.

        Returns:
            The final snapshot, with ``leaked_steps`` set.
        """
        self._client.run_script(
            "close",
            self._keys(run_id),
            [str(self._tombstone_ttl_ms), str(self._ttl_ms)],
        )
        return self.snapshot(run_id)

    def is_finalised(self, run_id: str) -> bool:
        """Whether this run has already been closed.

        Answers from the tombstone first, so finality survives the expiry of
        the run's own state.

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run was closed.
        """
        _, totals_key, done_key = self._keys(run_id)
        if self._client.exists(done_key):
            return True
        return self._client.hget(totals_key, "closed") == "1"

    def knows(self, run_id: str) -> bool:
        """Whether this store has ever recorded anything for a run.

        Distinct from :meth:`is_finalised`: an all-zero snapshot for a run
        nobody metered is indistinguishable from one for a run that has not
        started, and only the first is a lie (ADR-044).

        Args:
            run_id: The run's identifier.

        Returns:
            ``True`` if the run has state here.
        """
        _, totals_key, _ = self._keys(run_id)
        return self._client.exists(totals_key)

    def evict(self, run_id: str) -> None:
        """Drop a run's state, keeping the tombstone. Idempotent.

        Args:
            run_id: The run's identifier.
        """
        open_key, totals_key, _ = self._keys(run_id)
        self._client.delete(open_key, totals_key)

    def run_count(self) -> int:
        """Return how many runs currently hold state.

        Scans this backend's own keyspace, so it is O(keys) rather than O(1).
        Diagnostic only -- it exists for leak detection, the same reason the
        in-memory backend exposes it, and is not on any request path.

        Returns:
            Number of runs with retained state.
        """
        return self._client.count_keys(f"{_PREFIX}:*:totals")

    def ttl_seconds_remaining(self, run_id: str) -> float:
        """Seconds before this run's state expires, or ``0.0`` if it is gone.

        Args:
            run_id: The run's identifier.

        Returns:
            Remaining lifetime in seconds.
        """
        _, totals_key, _ = self._keys(run_id)
        return self._client.pttl(totals_key) / 1000.0

    def __repr__(self) -> str:
        """Return a debug representation with the tracked run count."""
        return f"<RedisLedgerStore runs={self.run_count()}>"
