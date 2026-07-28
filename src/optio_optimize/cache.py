"""Cache backends and key construction.

Correctness here is entirely a question of what goes into the key. A key that
omits a field which changes the answer serves a wrong response — permanently,
and to every caller, which makes it far worse than a slow cache. So the key
includes every request field that can alter output, and the code says so
explicitly rather than relying on a reader to notice what is absent.

Memory is bounded by construction. This runs in a long-lived agent process, and
an unbounded response cache is the same leak the core's soak tests were written
to catch (§11). The default backend is an LRU with an entry ceiling.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest, LLMResponse

#: Entries retained by the default backend. 512 full responses is a few MB --
#: large enough to be useful across a long agent run, small enough that nobody
#: needs to think about it.
DEFAULT_MAX_ENTRIES = 512

#: Default entry lifetime. Model outputs do not spoil, but the *world* does: a
#: cached answer about "today's date" or a fetched document goes stale silently.
#: An hour bounds that without making the cache useless.
DEFAULT_TTL_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A stored response and when it was written."""

    response: LLMResponse
    stored_at: float

    def is_fresh(self, ttl_seconds: float, now: float) -> bool:
        """Whether this entry is still within its lifetime."""
        return (now - self.stored_at) < ttl_seconds


class CacheBackend(Protocol):
    """Storage for cached responses.

    Implementations must be thread-safe: agent frameworks fan out across
    threads, and the core's concurrency work established that read-modify-write
    sequences here are not protected by the GIL.
    """

    def get(self, key: str) -> LLMResponse | None:
        """Return a fresh entry, or ``None``."""
        ...

    def put(self, key: str, response: LLMResponse) -> None:
        """Store a response."""
        ...

    def clear(self) -> None:
        """Drop everything."""
        ...

    def __len__(self) -> int:
        """Number of entries currently held."""
        ...


class MemoryCache:
    """Bounded, thread-safe, in-process LRU with TTL.

    The eviction and expiry logic is a composite read-modify-write, so it is
    lock-guarded rather than relying on individual dict operations being atomic
    — the distinction the core's concurrency testing made concrete.
    """

    __slots__ = ("_entries", "_hits", "_lock", "_max_entries", "_misses", "_ttl")

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Build a cache.

        Args:
            max_entries: Ceiling on retained responses.
            ttl_seconds: Entry lifetime.

        Raises:
            ValueError: If either bound is not positive. Setup fails loudly;
                a zero ceiling would make every lookup a miss and present as
                "caching does not work".
        """
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> LLMResponse | None:
        """Return a fresh response for ``key``, or ``None``."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if not entry.is_fresh(self._ttl, now):
                # Drop rather than leave it to the LRU: a stale entry occupying
                # a slot is worse than an empty one, and returning it is worse
                # than both.
                del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.response

    def put(self, key: str, response: LLMResponse) -> None:
        """Store ``response`` under ``key``, evicting the least recent if full."""
        with self._lock:
            self._entries[key] = CacheEntry(response=response, stored_at=time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry and reset counters."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        """Number of entries held."""
        with self._lock:
            return len(self._entries)

    @property
    def hit_rate(self) -> float | None:
        """Fraction of lookups that hit, or ``None`` before any lookup.

        ``None`` rather than ``0.0`` on no data: a zero would read as "the cache
        is useless" when the truth is "nothing has been asked of it yet".
        """
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total else None


def request_key(request: LLMRequest) -> str:
    """Build the exact-match cache key for a request.

    Every field that can change the model's output is included. Listing them
    explicitly, rather than hashing the whole object, means adding a field to
    :class:`~optio_optimize.types.LLMRequest` does not silently start ignoring
    it — a new field is simply absent here until someone decides where it
    belongs, and the omission is visible in this function rather than invisible
    in a serializer.

    Deliberately included:

    * ``model`` — different models, different answers.
    * every message role and content, in order.
    * ``tools`` — a different toolset changes what the model may do.
    * ``temperature`` — though only ``0`` is ever cached, see below.
    * ``response_format`` — a schema changes the shape of the reply.

    Deliberately excluded:

    * ``max_tokens`` — it truncates a reply rather than changing it, and
      including it would miss cache hits between otherwise identical calls.
      The stage compensates by never serving an entry whose stored response was
      truncated (``finish_reason == "length"``).
    * ``Message.cacheable`` — a marker this library placed, not caller input.
    * ``extra`` — provider transport details, not semantics.

    Args:
        request: The request to key.

    Returns:
        A hex digest. Never the prompt itself: cache keys end up in logs and
        metrics, and §10's content rule applies to this package too.
    """
    payload = {
        "model": request.model,
        "messages": [[m.role, m.content, m.name] for m in request.messages],
        "tools": [json.dumps(t, sort_keys=True, separators=(",", ":")) for t in request.tools],
        "temperature": request.temperature,
        "response_format": json.dumps(request.response_format, sort_keys=True)
        if request.response_format
        else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8", "replace")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()
