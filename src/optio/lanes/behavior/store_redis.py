"""The Redis behaviour backend -- one logical run, however many processes.

The in-memory backend keeps a window per process, and a process only sees its
own steps. Four workers sharding one run each hold a quarter of it, and a
quarter of a loop is not a smaller loop: below
:data:`~optio.lanes.behavior.detectors.MIN_STEPS_FOR_VERDICT` every worker
reports ``healthy`` for an agent that is visibly stuck. That is the same shape
as the budget bug the cost lane had, in the direction that hides a pathology.

**The reduction happens on the server.** ``classify_state`` reads five numbers,
so five numbers are what cross the network -- not the counter. Returning the
counter would put up to ``behavior_window_size`` entries on the wire *per step*
(1,000 at the documented ceiling), converting the O(1)-in-window-size guarantee
the README publishes as measured into O(window) in bytes, and no existing test
would have noticed.

**One script per operation, for atomicity rather than speed.** A step has to
append, evict what fell out, decrement that call's count, and read the result,
and those must not interleave with another worker's step. Split across round
trips they can, and the result is a verdict computed from a window that never
existed.

Three keys per run:

===========================  ======  ==============================================
key                          type    holds
===========================  ======  ==============================================
``optio:b:{run}:steps``      list    window entries, ``LTRIM``-equivalent bounded
``optio:b:{run}:counts``     hash    call identity -> count within the window
``optio:b:{run}:meta``       hash    ``errors``
===========================  ======  ==============================================

**The TTL is an idle timeout, refreshed on every step.** An absolute expiry
would empty a long run's window mid-flight and reset its verdict to ``healthy``
-- ADR-044's failure arriving on a timer. There is no tombstone: unlike a cost
run, closing a window is not final. A re-opened window is short again, so the
lane under-reports at worst, and failing toward ``healthy`` is the bias Section
6.4 requires (ADR-010 covers why the ledger cannot do the same).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from optio.lanes.behavior.store import WindowState

if TYPE_CHECKING:
    from optio.store.redis_client import RedisClient

#: Namespace for every key this backend owns. The ``:b:`` segment keeps the
#: behaviour keyspace disjoint from the ledger's ``optio:{run}:*``, so a shared
#: Redis stays legible and each backend's ``run_count`` scans only its own.
_PREFIX: Final = "optio:b"

#: Separator between a call identity's two halves inside a hash field name.
#: Redis field names are binary-safe and the Lua never splits them -- the field
#: is opaque to the script -- so this only has to round-trip through Python.
_CALL_SEP: Final = "\x00"

#: Reduce the window's aggregates to what a verdict needs. Shared verbatim by
#: both scripts below rather than written twice: this is the arithmetic that
#: already exists a second time in Python, and a third copy would be a third
#: place for the two to drift apart.
#:
#: The top-``k`` selection is a bounded insertion rather than ``table.sort``.
#: Sorting would be O(distinct log distinct) of server CPU per step, which grows
#: with the window; this is O(distinct * k) with ``k`` fixed at 2, and it never
#: materialises more than ``k`` values.
_REDUCE: Final = """
local counts = redis.call('HVALS', KEYS[2])
local top = {}
for i = 1, #counts do
  local v = tonumber(counts[i])
  local pos = #top + 1
  for j = 1, #top do
    if v > top[j] then pos = j break end
  end
  if pos <= k then
    table.insert(top, pos, v)
    if #top > k then table.remove(top) end
  end
end
-- `or '0'` is unreachable while HSETNX seeds the tally on every step, and is
-- kept because a Lua `false` at index 2 becomes a nil that truncates the whole
-- multi-bulk reply -- a wire-level failure far from its cause.
local reduced = {
  tostring(redis.call('LLEN', KEYS[1])),
  redis.call('HGET', KEYS[3], 'errors') or '0',
  tostring(#counts),
  table.concat(top, ',')
}
"""

#: Record one step: append, evict what overflowed, and reduce -- one round trip.
#:
#: The entry is the error flag as a **fixed one-character prefix** followed by
#: the call identity, so eviction splits it by position. A separator would have
#: to be a byte that cannot appear in a tool name, and there is no such byte.
_RECORD: Final = (
    """
local call, flag = ARGV[1], ARGV[2]
local maxlen, k, ttl = tonumber(ARGV[3]), tonumber(ARGV[4]), ARGV[5]

redis.call('RPUSH', KEYS[1], flag .. call)
redis.call('HINCRBY', KEYS[2], call, 1)
-- Seed the tally so the meta key exists from the first step. Created lazily by
-- the error branch instead, it would not exist when PEXPIRE runs below, and
-- PEXPIRE on an absent key is a silent no-op -- so a run whose first error
-- arrived late would hold an unexpiring key. Making the key unconditional makes
-- "every key this backend writes carries an expiry" structural rather than an
-- argument about statement order.
redis.call('HSETNX', KEYS[3], 'errors', '0')
if flag == '1' then redis.call('HINCRBY', KEYS[3], 'errors', 1) end

-- Evict from the head while over the bound, undoing what each departing step
-- contributed. A count reaching zero has its field DELETED, not left at zero:
-- `distinct_calls` is the number of fields, and a stale zero would inflate it
-- for the rest of the run -- pushing a stuck agent's window back over
-- LOOP_MAX_DISTINCT and hiding the loop. `BehaviorWindow.add` deletes the key
-- for the same reason, and the property test holds the two to each other.
while redis.call('LLEN', KEYS[1]) > maxlen do
  local gone = redis.call('LPOP', KEYS[1])
  local gone_call = string.sub(gone, 2)
  if redis.call('HINCRBY', KEYS[2], gone_call, -1) <= 0 then
    redis.call('HDEL', KEYS[2], gone_call)
  end
  if string.sub(gone, 1, 1) == '1' then
    redis.call('HINCRBY', KEYS[3], 'errors', -1)
  end
end
"""
    + _REDUCE
    + """
redis.call('PEXPIRE', KEYS[1], ttl)
redis.call('PEXPIRE', KEYS[2], ttl)
redis.call('PEXPIRE', KEYS[3], ttl)
return reduced
"""
)

#: Close: read the final state and release the window as one event. Reading then
#: deleting would be two round trips with a gap another worker's step can land
#: in, so the final verdict would describe a window that no longer existed.
#:
#: Returns Lua ``false`` -- Python ``None`` -- when there is no window, which is
#: what run end firing twice looks like (M1-2). Reporting an empty window
#: instead would let the lane emit ``healthy`` with no repeats over a real
#: ``looping`` verdict on the run span.
_CLOSE: Final = (
    """
local k = tonumber(ARGV[1])
if redis.call('EXISTS', KEYS[1]) == 0 then
  -- The counters can only outlive the steps if something deleted the list
  -- alone; clear them so a later step starts from a clean window rather than
  -- inheriting counts for signatures nobody can see.
  redis.call('DEL', KEYS[2], KEYS[3])
  return false
end
"""
    + _REDUCE
    + """
redis.call('DEL', KEYS[1], KEYS[2], KEYS[3])
return reduced
"""
)


class RedisBehaviorStore:
    """Per-run step windows shared across processes.

    Structurally implements
    :class:`~optio.lanes.behavior.store.BehaviorStore`.
    """

    def __init__(self, client: RedisClient, *, ttl_seconds: float) -> None:
        """Build the backend and register its scripts.

        Args:
            client: Connected Redis client.
            ttl_seconds: Idle expiry for a run's window, refreshed on each step.

        Raises:
            StoreUnavailableError: If the scripts cannot be loaded. Loud, at
                setup, rather than on the first step (Section 4.2).
        """
        self._client: Final = client
        self._ttl_ms: Final = int(ttl_seconds * 1000)
        for name, source in (("record", _RECORD), ("close", _CLOSE)):
            client.register_script(name, source)

    def _keys(self, run_id: str) -> list[str]:
        """The three keys for a run, in the order both scripts expect."""
        return [
            f"{_PREFIX}:{run_id}:steps",
            f"{_PREFIX}:{run_id}:counts",
            f"{_PREFIX}:{run_id}:meta",
        ]

    def record(
        self,
        run_id: str,
        signature_call: tuple[str, str],
        errored: bool,
        maxlen: int,
        k: int,
    ) -> WindowState:
        """Add one step to a run's window and return the resulting summary.

        Args:
            run_id: The run's identifier.
            signature_call: The step's call identity -- ``(tool, args_digest)``,
                never the arguments themselves (Section 10).
            errored: Whether the step ended in error.
            maxlen: Window bound, from ``Config.behavior_window_size``.
            k: How many top counts to return.

        Returns:
            The window's state after adding the step.

        Raises:
            ValueError: If ``maxlen`` is not positive. Checked here rather than
                left to the script, which would quietly evict every step and
                report an always-empty window where the in-memory backend
                raises. Config validates this at setup (Section 4.2).
            StoreUnavailableError: If Redis cannot be reached.
        """
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")

        raw = self._client.run_script(
            "record",
            self._keys(run_id),
            [
                _CALL_SEP.join(signature_call),
                "1" if errored else "0",
                str(maxlen),
                str(k),
                str(self._ttl_ms),
            ],
        )
        return _state_from(raw)

    def close_run(self, run_id: str, k: int) -> WindowState | None:
        """Release a run's window and return the state it held.

        Args:
            run_id: The run's identifier.
            k: How many top counts to return.

        Returns:
            The final state, or ``None`` if the run held no window -- either it
            never recorded a step, or run end already fired once (M1-2).

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        raw = self._client.run_script("close", self._keys(run_id), [str(k)])
        # Lua `false` arrives as `None`. It means no window, which is not the
        # same as an empty one (ADR-044).
        return None if raw is None else _state_from(raw)

    def run_count(self) -> int:
        """Return how many runs currently hold a window.

        Scans this backend's own keyspace, so it is O(keyspace) rather than
        O(1). Diagnostic only -- it exists for leak detection, the same reason
        the in-memory backend exposes it, and is never on a request path.

        Returns:
            Number of runs with a retained window.
        """
        return self._client.count_keys(f"{_PREFIX}:*:steps")

    def ttl_seconds_remaining(self, run_id: str) -> float:
        """Seconds before this run's window expires, or ``0.0`` if it is gone.

        Args:
            run_id: The run's identifier.

        Returns:
            Remaining lifetime in seconds.
        """
        steps_key, _, _ = self._keys(run_id)
        return self._client.pttl(steps_key) / 1000.0

    def __repr__(self) -> str:
        """Return a debug representation with the tracked run count."""
        return f"<RedisBehaviorStore runs={self.run_count()}>"


def _state_from(raw: list[str]) -> WindowState:
    """Rebuild a :class:`WindowState` from the script's four-element reply.

    Args:
        raw: ``[size, errors, distinct_calls, comma-joined top counts]``.

    Returns:
        The state. The counts field is empty for an empty window, which
        ``"".split(",")`` would turn into ``('',)`` rather than ``()``.
    """
    top = raw[3]
    return WindowState(
        size=int(raw[0]),
        errors=int(raw[1]),
        distinct_calls=int(raw[2]),
        top_counts=tuple(int(part) for part in top.split(",")) if top else (),
    )
