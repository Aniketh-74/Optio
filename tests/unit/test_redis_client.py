"""Redis access with the two properties the ledger depends on.

The client is deliberately domain-free: it knows about connections, scripts and
timeouts, and nothing about reservations. That is what lets it sit in
``optio.store``, which the import-linter forbids from importing any lane -- and
``LedgerSnapshot`` lives in a lane.

Both behaviours here are ones a mock cannot be trusted to describe later:
script caching is a performance property that silently degrades, and NOSCRIPT
recovery only ever happens on a Redis restart, which no test would reproduce by
accident.
"""

from __future__ import annotations

import pytest

from optio.errors import StateStoreError
from optio.store.redis_client import RedisClient, StoreUnavailableError


class _FakeRedis:
    """Records calls and can be told to fail the way a real server does."""

    def __init__(self) -> None:
        self.loaded: dict[str, str] = {}
        self.evalsha_calls: list[tuple[str, list[str], list[str]]] = []
        self.raise_noscript_once = False
        self.unreachable = False
        self.closed = False

    def script_load(self, source: str) -> str:
        sha = f"sha-{len(self.loaded)}"
        self.loaded[sha] = source
        return sha

    def evalsha(self, sha: str, numkeys: int, *rest: str) -> str:
        if self.unreachable:
            raise ConnectionError("no route to host")
        keys = list(rest[:numkeys])
        args = list(rest[numkeys:])
        self.evalsha_calls.append((sha, keys, args))
        if self.raise_noscript_once:
            self.raise_noscript_once = False
            raise RuntimeError("NOSCRIPT No matching script")
        return "ok"

    def ping(self) -> bool:
        if self.unreachable:
            raise ConnectionError("no route to host")
        return True

    def close(self) -> None:
        self.closed = True


def _client(fake: _FakeRedis) -> RedisClient:
    """A client wired to ``fake``, bypassing the real connection.

    ``__init__`` builds a socket; these tests are about what happens after one
    exists, so the constructor is skipped rather than mocked at import level.
    """
    client = RedisClient.__new__(RedisClient)
    client._redis = fake  # type: ignore[assignment]
    client._shas = {}
    client._sources = {}
    return client


class TestScriptCaching:
    def test_a_script_is_loaded_once_and_reused(self) -> None:
        """Loading per call would put a second round trip on every step.

        The whole reason the ledger's compound operations are scripts is that
        each costs one round trip; re-loading would quietly double that and
        nothing would fail.
        """
        fake = _FakeRedis()
        client = _client(fake)
        client.register_script("bump", "return 1")

        client.run_script("bump", ["k"], ["1"])
        client.run_script("bump", ["k"], ["2"])

        assert len(fake.loaded) == 1, "the script was re-loaded instead of cached"
        assert len(fake.evalsha_calls) == 2

    def test_the_script_receives_its_keys_and_args_separately(self) -> None:
        """Redis needs the key count to route in a cluster, and a script that
        reads ARGV where it meant KEYS works on one node and breaks on many."""
        fake = _FakeRedis()
        client = _client(fake)
        client.register_script("bump", "return 1")

        client.run_script("bump", ["k1", "k2"], ["a1"])

        _, keys, args = fake.evalsha_calls[0]
        assert keys == ["k1", "k2"]
        assert args == ["a1"]


class TestAFlushedScriptCacheIsRecoverable:
    def test_noscript_reloads_and_retries_once(self) -> None:
        """Redis drops its script cache on restart and on failover.

        Treating that as fatal would turn an ordinary restart into a run with no
        cost signals, so the client reloads and retries exactly once -- once,
        because a second NOSCRIPT means something other than a cold cache.
        """
        fake = _FakeRedis()
        client = _client(fake)
        client.register_script("bump", "return 1")
        fake.raise_noscript_once = True

        result = client.run_script("bump", ["k"], ["1"])

        assert result == "ok"
        assert len(fake.loaded) == 2, "the script was not reloaded after NOSCRIPT"


class TestUnavailability:
    def test_an_unreachable_redis_raises_store_unavailable(self) -> None:
        """One exception type, so callers need not know redis-py's tree."""
        fake = _FakeRedis()
        fake.unreachable = True
        client = _client(fake)
        client.register_script("bump", "return 1")

        with pytest.raises(StoreUnavailableError):
            client.run_script("bump", ["k"], ["1"])

    def test_ping_raises_store_unavailable(self) -> None:
        """Used at setup, where an unreachable backend must fail loudly."""
        fake = _FakeRedis()
        fake.unreachable = True
        client = _client(fake)

        with pytest.raises(StoreUnavailableError):
            client.ping()

    def test_store_unavailable_is_a_state_store_error(self) -> None:
        """The fail-open guard already absorbs StateStoreError.

        A new unrelated exception type would sail straight past it and break the
        agent -- the one thing ADR-004 forbids.
        """
        assert issubclass(StoreUnavailableError, StateStoreError)
