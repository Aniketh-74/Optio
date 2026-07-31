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

from optio_optimize.wire import RAW_CONTENT_KEY, canonical_block, is_text_block

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest, LLMResponse, Message

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


#: Request fields deliberately *excluded* from the cache key, with the reason.
#:
#: Machine-checked against :class:`~optio_optimize.types.LLMRequest`'s own field
#: list, the same way :data:`~optio_optimize.wire.UNSENT_FIELDS` is, and for a
#: reason paid for in a real defect: :func:`request_key`'s docstring listed
#: ``stop`` and ``thinking_budget`` as keyed, with a paragraph of justification
#: each, while the payload contained neither. Two requests differing only in
#: their stop sequence hashed identically, so ``exact_cache`` -- on by default --
#: could serve a caller who asked generation to halt at a delimiter an answer
#: produced without one.
#:
#: A prose list in a docstring is not a decision anything can verify. This is.
UNKEYED_FIELDS: dict[str, str] = {
    "max_tokens": (
        "truncates a reply rather than changing it, and keying on it would miss "
        "hits between otherwise identical calls; ExactCacheStage compensates by "
        "never serving a stored response whose finish_reason was `length`"
    ),
    "extra": (
        "request-level provider transport details, not semantics. Message.extra "
        "is a different question and was answered wrongly until ADR-022: image, "
        "tool_use and tool_result blocks live there, and non_text_digest now "
        "keys them"
    ),
}


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
    * ``stop`` — a stop sequence halts generation mid-answer, so two otherwise
      identical requests with different stop sequences have genuinely different
      replies. Unlike ``max_tokens`` this cannot be compensated for by checking
      ``finish_reason``: a completion that stopped at a delimiter reports
      ``"stop"`` exactly as a naturally-finished one does, so a shared entry
      would be indistinguishable from a correct hit.
    * ``thinking_budget`` — reasoning tokens are how a model reaches its answer,
      and a smaller allowance can change what it concludes, not merely how long
      it takes to get there.

    Deliberately excluded:

    * ``max_tokens`` — it truncates a reply rather than changing it, and
      including it would miss cache hits between otherwise identical calls.
      The stage compensates by never serving an entry whose stored response was
      truncated (:attr:`~optio_optimize.types.LLMResponse.was_truncated`, which
      knows each vendor's spelling -- Anthropic says ``max_tokens``, and a
      comparison against ``"length"`` alone left this compensation inert for
      every Anthropic caller until ADR-033).
    * ``Message.cacheable`` — a marker this library placed, not caller input.
    * ``LLMRequest.extra`` — request-level provider transport details, not
      semantics. **Note the level.** ``Message.extra`` is a different matter:
      until ADR-022 it was excluded on this same "not semantics" reasoning,
      which is false for the image, ``tool_use`` and ``tool_result`` blocks that
      live there — see :func:`non_text_digest`, which keys their identity.

    Args:
        request: The request to key.

    Returns:
        A hex digest. Never the prompt itself: cache keys end up in logs and
        metrics, and §10's content rule applies to this package too.
    """
    payload = {
        "model": request.model,
        # The digest is appended only when there is non-text content, so a
        # text-only request's key is byte-identical to what it was before
        # ADR-022 -- a trailing `null` on every message would have changed every
        # key in the package to record the absence of an image.
        "messages": [_keyed_message(m) for m in request.messages],
        "tools": [json.dumps(t, sort_keys=True, separators=(",", ":")) for t in request.tools],
        "temperature": request.temperature,
        "response_format": json.dumps(request.response_format, sort_keys=True)
        if request.response_format
        else None,
        # The three below were documented as keyed for a long time and were not
        # in this payload -- see UNKEYED_FIELDS for what that cost.
        "stop": list(request.stop),
        "thinking_budget": request.thinking_budget,
        "reasoning_effort": request.reasoning_effort,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8", "replace")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _keyed_message(message: Message) -> list[str | None]:
    """The key payload for one message: role, text, name, and non-text identity."""
    digest = non_text_digest(message)
    base: list[str | None] = [message.role, message.content, message.name]
    return base if digest is None else [*base, digest]


def non_text_digest(message: Message) -> str | None:
    """A digest of everything in ``message`` that is not text, or ``None``.

    ``Message.content`` holds only the *text* an adapter could extract, so an
    image block, a ``tool_use``'s arguments and a ``tool_result``'s payload live
    on ``extra[RAW_CONTENT_KEY]`` and were invisible to :func:`request_key`.
    Two vision requests with the same prompt and different images hashed
    identically -- verified, both `277f0c84dc88772471a95b2f9cbe7846` -- and
    ``exact_cache`` is on by default at ``temperature == 0``, exactly the setting
    OCR and screenshot analysis use. That served the first image's answer for the
    second (ADR-022).

    **Text blocks are excluded, and that is load-bearing.** Their text is already
    keyed through ``content``; digesting them again would make an ordinary
    text-only conversation's key depend on which adapter normalized it, and on
    block wrappers that carry no meaning.

    **Everything else is included, including block types nobody has seen.** The
    bug came from deciding a block was semantically irrelevant, so the fix does
    not get to make that call again for a future block type.

    Returns:
        A hex digest, or ``None`` when the message carries no non-text content --
        ``None`` so that a text-only request's key is byte-identical to what it
        was before this existed. Never the content itself: §10's rule covers an
        image at least as squarely as prose, and an image is more identifying
        than most text.
    """
    raw = message.extra.get(RAW_CONTENT_KEY)
    if not isinstance(raw, dict):
        return None
    content = raw.get("content")
    if not isinstance(content, list):
        return None
    parts = [canonical_block(b) for b in content if not is_text_block(b)]
    if not parts:
        return None
    joined = "\x00".join(parts).encode("utf-8", "replace")
    return hashlib.blake2b(joined, digest_size=16).hexdigest()
