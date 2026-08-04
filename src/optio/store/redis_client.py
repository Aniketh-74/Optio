"""Redis access, with no idea what it is storing.

Lives in ``optio.store`` because the import-linter's layering forbids this
package from importing any lane -- so it must stay domain-free. The ledger's Lua
lives with the ledger; only the mechanics of loading and invoking a script live
here. That split is what lets the behaviour and quality lanes reuse this without
either lane importing the other (their independence is its own contract).

**Timeouts are the point.** A hung Redis must never add latency to an agent, so
connect and read timeouts are short by default and there are no retries on the
hot path: a retry is latency paid for a signal the caller can live without
(ADR-004).

Every failure is normalised to :class:`StoreUnavailableError`, a
:class:`~optio.errors.StateStoreError`, so the existing fail-open guard absorbs
it without learning redis-py's exception tree.
"""

from __future__ import annotations

from typing import Any, Final

from optio.errors import StateStoreError

#: Connect and read timeout. Short because this sits on the agent's critical
#: path, where a dropped signal is cheaper than a stalled step.
DEFAULT_TIMEOUT_MS: Final = 50

#: Marker Redis returns when a SHA is no longer in its script cache. Matched as
#: a substring because the full message carries the SHA and varies by version.
_NOSCRIPT: Final = "NOSCRIPT"


class StoreUnavailableError(StateStoreError):
    """The store could not answer.

    A subclass of :class:`~optio.errors.StateStoreError` so the fail-open guard
    absorbs it without learning a new type. Callers that must distinguish
    "unreachable" from "the caller broke an invariant" catch this specifically;
    everything else can keep catching the base.
    """


class RedisClient:
    """A Redis connection plus a cache of server-side scripts."""

    def __init__(self, url: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        """Connect to Redis.

        Args:
            url: Redis connection string.
            timeout_ms: Connect and read timeout in milliseconds.

        Raises:
            ImportError: If the ``redis`` driver is not installed. Loud, at
                construction: configuration naming a backend whose driver is
                absent is a setup error, and §4.2 says setup fails loudly.
        """
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - needs the extra uninstalled
            raise ImportError(
                "store_backend='redis' needs the redis driver: pip install 'optio[redis]'"
            ) from exc

        seconds = timeout_ms / 1000.0
        self._redis = redis.Redis.from_url(
            url,
            socket_connect_timeout=seconds,
            socket_timeout=seconds,
            # Scripts return strings and the ledger compares them against
            # sentinels; decoding here keeps that comparison from silently
            # failing against bytes.
            decode_responses=True,
        )
        self._shas: dict[str, str] = {}
        self._sources: dict[str, str] = {}

    def register_script(self, name: str, source: str) -> None:
        """Load a Lua script and remember its SHA under ``name``.

        The source is retained so the script can be reloaded if Redis drops its
        cache.

        Args:
            name: Local name used to invoke the script.
            source: The Lua source.

        Raises:
            StoreUnavailableError: If the script cannot be loaded.
        """
        self._sources[name] = source
        try:
            self._shas[name] = self._redis.script_load(source)
        except Exception as exc:
            raise StoreUnavailableError(f"could not load script {name!r}") from exc

    def run_script(self, name: str, keys: list[str], args: list[str]) -> Any:
        """Invoke a registered script by SHA.

        Keys and arguments are passed separately because Redis needs the key
        count to route in a cluster -- a script reading ``ARGV`` where it meant
        ``KEYS`` works on one node and breaks on many.

        Reloads and retries **once** on ``NOSCRIPT``. Redis drops its script
        cache on restart and failover, and treating that as fatal would turn an
        ordinary restart into a run with no cost signals. Once, because a second
        ``NOSCRIPT`` means something other than a cold cache.

        Args:
            name: Name the script was registered under.
            keys: Redis keys the script operates on.
            args: Non-key arguments.

        Returns:
            Whatever the script returned.

        Raises:
            StoreUnavailableError: If Redis cannot be reached, or the script fails.
        """
        try:
            return self._redis.evalsha(self._shas[name], len(keys), *keys, *args)
        except Exception as exc:
            if _NOSCRIPT not in str(exc):
                raise StoreUnavailableError(f"redis unavailable running {name!r}") from exc

        try:
            self._shas[name] = self._redis.script_load(self._sources[name])
            return self._redis.evalsha(self._shas[name], len(keys), *keys, *args)
        except Exception as exc:
            raise StoreUnavailableError(f"redis script {name!r} failed after reload") from exc

    def exists(self, key: str) -> bool:
        """Whether a key is present.

        Args:
            key: The key to check.

        Returns:
            ``True`` if it exists.

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        try:
            return bool(self._redis.exists(key))
        except Exception as exc:
            raise StoreUnavailableError(f"redis unavailable checking {key!r}") from exc

    def hget(self, key: str, field: str) -> str | None:
        """Read one field of a hash.

        Args:
            key: The hash key.
            field: Field within the hash.

        Returns:
            The value, or ``None`` if the key or field is absent.

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        try:
            value = self._redis.hget(key, field)
        except Exception as exc:
            raise StoreUnavailableError(f"redis unavailable reading {key!r}") from exc
        return None if value is None else str(value)

    def delete(self, *keys: str) -> None:
        """Remove keys. Idempotent -- deleting an absent key is not an error.

        Args:
            *keys: Keys to remove.

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        if not keys:
            return
        try:
            self._redis.delete(*keys)
        except Exception as exc:
            raise StoreUnavailableError("redis unavailable deleting keys") from exc

    def pttl(self, key: str) -> int:
        """Milliseconds until a key expires.

        Args:
            key: The key to inspect.

        Returns:
            Remaining lifetime in milliseconds; ``0`` when the key is gone or
            carries no expiry. Redis signals both with negative values, and
            collapsing them to zero keeps callers from doing arithmetic on a
            sentinel.

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        try:
            remaining = int(self._redis.pttl(key))
        except Exception as exc:
            raise StoreUnavailableError(f"redis unavailable reading ttl of {key!r}") from exc
        return max(remaining, 0)

    def count_keys(self, pattern: str) -> int:
        """Count keys matching a glob pattern.

        Uses ``SCAN`` rather than ``KEYS``: the latter blocks the server for
        the length of the keyspace, which is unacceptable on a Redis shared
        with anything else. Still O(keyspace), so this is for diagnostics and
        never for a request path.

        Args:
            pattern: Glob pattern, e.g. ``"optio:*:totals"``.

        Returns:
            Number of matching keys.

        Raises:
            StoreUnavailableError: If Redis cannot be reached.
        """
        try:
            return sum(1 for _ in self._redis.scan_iter(match=pattern, count=500))
        except Exception as exc:
            raise StoreUnavailableError(f"redis unavailable scanning {pattern!r}") from exc

    def ping(self) -> None:
        """Verify the server answers.

        Called at setup so an unreachable Redis fails loudly there rather than
        silently on the runtime path.

        Raises:
            StoreUnavailableError: If the server does not answer.
        """
        try:
            self._redis.ping()
        except Exception as exc:
            raise StoreUnavailableError("redis did not answer ping") from exc

    def close(self) -> None:
        """Release the connection. Idempotent."""
        self._redis.close()
