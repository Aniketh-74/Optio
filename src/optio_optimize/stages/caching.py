"""Exact-match response caching, and provider prefix-cache markers.

Two different mechanisms with the same goal, worth understanding as distinct:

**Exact cache** avoids the call entirely. Saving is 100% of the request when it
hits. Hit rate on agent workloads is usually *low* — prompts carry accumulated
history, so consecutive steps rarely match byte-for-byte — but retries, fan-out
over identical sub-tasks, and re-runs during development hit constantly.

**Prefix cache** does not avoid tokens at all. It marks the stable head of the
prompt so the *provider* bills it at a discount (typically 10% of the input
rate on Anthropic and OpenAI). Tokens sent are unchanged; the bill is not. On a
long agent run with a large system prompt and tool schemas, this is usually the
single largest lossless saving available, because that prefix is resent on
every step and is identical every time.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING

from optio_optimize.cache import MemoryCache, request_key
from optio_optimize.stages.base import PREFIX_IS_UNSTABLE, Fidelity, Stage, StageResult
from optio_optimize.tokens import count_request, count_tools

if TYPE_CHECKING:
    from collections.abc import Callable

    from optio_optimize.cache import CacheBackend
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest, LLMResponse, Message

#: Scratch key under which the exact stage carries its key to ``after``.
_KEY = "exact_cache_key"

#: Minimum tokens before a prefix marker is worth placing. Below a provider's
#: floor the marker is ignored, so requesting one is pure noise.
#:
#: **One number cannot express this and 1024 is wrong in both directions.**
#: Anthropic's floor is per-model, and as of 2026-07-30 spans a factor of eight:
#: Opus 5 at 512, Sonnet 5 / Sonnet 4.6 / 4.5 / Opus 4.8 at 1,024, Opus 4.7 and
#: Haiku 3.5 at 2,048, **Haiku 4.5 and Opus 4.6 / 4.5 at 4,096**. So this
#: constant is too high for Opus 5 -- declining a breakpoint that would have
#: worked -- and too low for four models, where it places one the provider
#: silently discards while the stage's note claims a breakpoint.
#:
#: **Now the fallback for an unrecognized model only** (ADR-027). It was the
#: single floor for every model until 2026-07-31, left that way on ADR-016's
#: grounds that a per-model table changes what every Anthropic caller sends. What
#: forced the change: the benchmark's default model became ``claude-haiku-4-5``,
#: whose floor is 4,096, and **eleven of its twelve workloads sit below that** --
#: so the live run reported zero cache reads everywhere and nothing said why.
#:
#: Kept as the unknown-model default rather than the lowest or highest floor: the
#: lowest would place markers four known models discard, the highest would
#: decline breakpoints that work on most, and this value leaves an unrecognized
#: name behaving exactly as it did before.
MIN_PREFIX_TOKENS = 1024

#: Anthropic's published minimum cacheable prefix, by model-name prefix.
#:
#: Matched longest-first, so a dated id (``claude-haiku-4-5-20251001``) resolves
#: through its alias -- the API reports the dated form back on every response, so
#: a table knowing only the alias would return the wrong floor for half the
#: lookups.
#:
#: Data, auditable against the vendor's page, and stale the moment Anthropic
#: changes it. Same standing caveat as ``PRICING``, and the same mitigation: this
#: is one dict, not logic.
MIN_PREFIX_TOKENS_BY_MODEL: dict[str, int] = {
    "claude-opus-5": 512,
    "claude-sonnet-5": 1024,
    "claude-opus-4-8": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-7": 2048,
    "claude-haiku-3-5": 2048,
    "claude-haiku-4-5": 4096,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
}


def model_floor_note(model: str, floor: int) -> str:
    """A decline reason naming both the model and the floor it missed."""
    return f"{model}'s {floor}-token cacheable minimum"


def min_prefix_tokens_for(model: str) -> int:
    """Return the minimum cacheable prefix for ``model``.

    Longest-prefix match against :data:`MIN_PREFIX_TOKENS_BY_MODEL`, falling
    back to :data:`MIN_PREFIX_TOKENS` for anything unrecognized -- including
    every OpenAI model, where the marker is inert anyway because that provider
    caches automatically.
    """
    best = ""
    for name in MIN_PREFIX_TOKENS_BY_MODEL:
        if model.startswith(name) and len(name) > len(best):
            best = name
    return MIN_PREFIX_TOKENS_BY_MODEL[best] if best else MIN_PREFIX_TOKENS


#: How long Anthropic's default cache entry lives. A gap longer than this means
#: the entry the previous call wrote is gone, so the next call pays a fresh
#: 1.25x write on a prefix it just wrote -- which is the whole case for the
#: one-hour TTL (ADR-021).
FIVE_MINUTE_WINDOW_SECONDS = 300.0

#: Prefixes tracked for TTL selection. Bounded because this lives for the
#: process lifetime in a long-running agent (§11), and an agent that rotates
#: system prompts would otherwise grow the map without limit. 1,024 prefixes is
#: a few hundred kilobytes of hex digests and covers far more distinct prompts
#: than a single process realistically holds.
MAX_TRACKED_PREFIXES = 1024

#: The value sent when observation says the entry will have expired.
ONE_HOUR_TTL = "1h"

#: Scratch key recording that ``PrefixCacheStage.before`` actually marked this
#: request, so ``after`` knows the provider's cache numbers are about a
#: breakpoint we placed.
_MARKED = "prefix_cache_marked"

#: Consecutive marked requests that write and read nothing before the stage
#: stops marking.
#:
#: Three, from the economics rather than from taste. A wasted write costs 0.25x
#: base per prefix token -- 1.25x paid where 1.0x would have done -- while a
#: *missed* cache costs 0.9x, since 1.0x is paid where 0.1x would have done.
#: Declining wrongly is therefore about 3.6x worse per request than writing
#: wrongly, so this waits for repeated unambiguous evidence rather than reacting
#: to a single miss, and any read at all resets it.
#:
#: The first write is always unrewarded -- nothing can be read from a cache that
#: was never written -- so the floor cannot be lower than two without disabling
#: the stage on the opening turn of every conversation.
MAX_UNREWARDED_WRITES = 3


class ExactCacheStage(Stage):
    """Serve byte-identical deterministic requests from memory.

    Only ``temperature == 0`` requests are cached. Caching a sampled request
    would replace the variety the caller explicitly asked for with one frozen
    answer — cheaper, and wrong in a way that takes a long time to notice
    because every individual response looks plausible.
    """

    # The stored response *is* the model's answer to this exact prompt, so
    # serving it is identical by construction rather than by approximation.
    fidelity = Fidelity.IDENTICAL

    def __init__(self, backend: CacheBackend | None = None) -> None:
        """Build the stage.

        Args:
            backend: Storage to use. A bounded in-memory LRU by default.
        """
        self.backend: CacheBackend = backend if backend is not None else MemoryCache()

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "exact_cache"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Return a cached response when this exact request was seen before."""
        if not request.is_deterministic:
            return self.declines(request)

        key = request_key(request)
        ctx.scratch[_KEY] = key

        hit = self.backend.get(key)
        if hit is None:
            return self.declines(request)

        # A truncated reply is not a complete answer. Serving one to a caller
        # who allowed more output would silently cap them at whatever ceiling
        # happened to apply the first time -- and max_tokens is excluded from
        # the key precisely so those calls share an entry.
        if hit.finish_reason == "length":
            return self.declines(request)

        # Prefer what the original call was actually billed over our own
        # estimate of the same request. The stored counts are ground truth from
        # the provider for this exact prompt; the estimator is a model of it.
        # Falling back only when the response carried no usage data keeps the
        # figure honest for providers that omit it.
        saved_input = hit.input_tokens or count_request(request, ctx.counter)
        return StageResult(
            request=request,
            response=served_from_cache(hit, self.name),
            saved_input_tokens=saved_input,
            saved_output_tokens=hit.output_tokens,
            note="exact hit",
        )

    def after(self, request: LLMRequest, response: LLMResponse, ctx: StageContext) -> None:
        """Store a fresh response under the key computed in ``before``."""
        if response.served_from is not None:
            return  # Came from a cache; storing it again is a no-op write.
        key = ctx.scratch.get(_KEY)
        if isinstance(key, str):
            self.backend.put(key, response)


def served_from_cache(response: LLMResponse, stage: str) -> LLMResponse:
    """Return a copy of ``response`` marked as served by ``stage``.

    Token counts are zeroed. They describe what the *original* call cost, and
    leaving them would re-bill that spend in the report every time the entry is
    served -- turning a cache that saves money into one that appears to spend it
    repeatedly. ``served_from`` preserves the provenance, so the saving stays
    attributable to the right stage.
    """
    return replace(
        response,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        served_from=stage,
    )


class PrefixCacheStage(Stage):
    """Mark the stable head of the prompt for provider-side prefix caching.

    Finds the longest leading run of messages that will be identical on the next
    call — system prompt, tool schemas, few-shot examples, early turns — and
    marks the last of them. Provider adapters translate the marker into the
    vendor's own mechanism (Anthropic ``cache_control``; OpenAI caches
    automatically and needs only stable ordering).

    Reports **no token saving**, because it avoids none: the same tokens are
    sent. The saving is a lower *price* per token, which the report captures
    through ``cached_input_tokens`` on the response rather than as an avoided
    count. Claiming avoided tokens here would inflate every report.

    **Its value depends entirely on the provider, and on OpenAI it is zero.**
    Vendors split into two camps:

    * *Automatic* (OpenAI): any prompt prefix over ~1024 tokens is cached with
      no cooperation from the caller. Measured live: 1280 of 1401 tokens served
      from cache on a repeat call with no marker sent. This stage changes
      nothing there -- it costs nothing either, but a user should not expect a
      discount they were already getting.
    * *Explicit* (Anthropic): nothing is cached without a ``cache_control``
      breakpoint. The marker this stage places is the difference between a
      discounted prefix and none. **Measured live 2026-07-30**, six-turn
      conversation on ``claude-haiku-4-5`` through
      :func:`~optio_optimize.adapters.anthropic.wrap_anthropic_client`, this
      stage isolated: 23,023 of 30,113 input tokens served from cache (76.5%)
      against 0 for the disabled arm, for a **50.1% cost reduction on identical
      token counts** -- 30,111 versus 30,113 sent, which is noise. The stage
      avoids no tokens at all and halves the bill by changing what they cost.
      This replaces an earlier "roughly 30%" that had no run behind it.

      **That number was first published as 53.7%, and the difference is a bug
      this stage's own accounting had.** Cache *writes* -- 5,487 of them, the
      tokens Anthropic charges a 1.25x premium to store -- were priced at the
      base rate, because :class:`~optio_optimize.config.ModelPricing` had no
      write rate at all. The measurement's token counts were right; only their
      price was wrong, always in the direction that flatters this package.

    **The floor is per-model and :data:`MIN_PREFIX_TOKENS` cannot express it.**
    See that constant: Anthropic's minimum spans 512 to 4,096 depending on the
    model, so a single 1024 both declines breakpoints that would work and places
    breakpoints that are discarded. The first run of the measurement above hit
    the second case: a 1,449-token prompt cleared the constant, missed Haiku
    4.5's real 4,096 floor, and reported zero cache reads in both arms -- which
    reads like "the stage does nothing" and meant "the stage was never given a
    chance". Recorded in ``docs/optimize-benchmarks.md`` with the full table.

    Worth stating plainly because the benchmark got it wrong first: modelling
    only the explicit style credited this library with a 36.3% saving on
    ``multi_turn_chat`` that OpenAI grants unconditionally. The live run
    measured -1.8%.
    """

    # A marker changes what the provider *bills*, never what it generates.
    fidelity = Fidelity.IDENTICAL

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        """Build the stage.

        Args:
            clock: Monotonic time source, injectable for tests. TTL selection
                turns on a five-minute threshold, and a test that proves it by
                sleeping for five minutes is a test nobody runs -- so the clock
                is a seam rather than a call to ``time.monotonic`` inline. The
                live script does the real waiting, because only the provider can
                confirm an entry actually expired.
        """
        self._clock = clock if clock is not None else time.monotonic
        #: Why the last call declined, for reports and diagnostics. Never
        #: contains prompt content -- only a model name and a token count.
        self.last_decline_reason = ""
        # Prefix digest -> when it was last sent. An OrderedDict so the bound
        # below evicts the least recently seen rather than an arbitrary entry.
        self._last_seen: OrderedDict[str, float] = OrderedDict()
        # Prefixes observed to have outlived a five-minute entry. Once true it
        # stays true: reverting on the next quick call would re-write the prefix
        # at 1.25x while a live one-hour entry sat there unread.
        self._expires: set[str] = set()
        # Consecutive marked requests where the provider wrote and never read.
        # Reset by any read (ADR-030 amendment).
        self._unrewarded_writes = 0

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "prefix_cache"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Mark the cacheable prefix boundary, if one is worth marking."""
        # Before size, and deliberately so. A cache write is an investment
        # against a future read; where the prefix has been observed changing on
        # effectively every request, there will be no read and the write is a
        # pure loss at 1.25x base rate. A live Sonnet 4.5 run on
        # `timestamped_agent` wrote 20,333 tokens, read back none of them, and
        # came in 23.5% more expensive than doing nothing -- an ADR-013 rule 1
        # violation this stage caused while `unstable_prefix` was reporting the
        # exact cause on the same run (ADR-030).
        #
        # Order matters: counting tools below can lift a prefix back over the
        # floor, and doing that first would reinstate the very write nobody
        # reads.
        # The provider's own verdict, and the stronger of the two signals: it
        # measures the outcome instead of inferring it from prompt digests, and
        # it converges in about three requests where `unstable_prefix` needs
        # ten. The digest guard alone still left `timestamped_agent` at -17.1%
        # live, because ten of its twelve requests had already paid the premium
        # by the time the window filled.
        #
        # It also covers a case the digests cannot see: a byte-stable prefix
        # that always expires before reuse writes at 1.25x forever and reads
        # nothing, and the digests look perfect throughout.
        if self._unrewarded_writes >= MAX_UNREWARDED_WRITES:
            self.last_decline_reason = (
                f"the provider wrote this prefix {self._unrewarded_writes} times in a row "
                "and read it back none of them, so the write premium is buying nothing"
            )
            return self.declines(request)

        if ctx.scratch.get(PREFIX_IS_UNSTABLE):
            self.last_decline_reason = (
                "prefix is unstable -- it differed on nearly every recent request, so a "
                "cache write here can never be read back. Move the varying part below the "
                "stable instructions to restore the discount."
            )
            return self.declines(request)

        boundary = self._stable_prefix_length(request)
        if boundary == 0:
            return self.declines(request)

        # Tools count. Anthropic caches tools -> system -> messages, so a
        # breakpoint in the system block caches every schema ahead of it, and
        # measuring the messages alone understates every tool-carrying request.
        # `large_system_agent` reported "~1715 tokens" for a 5,186-token prefix
        # and was declined on the 4,096 tier; the live `mcp_agent` run read
        # ~2,839 tokens per request against a stable *message* prefix of ~1,387,
        # which is the provider confirming what it counts (ADR-030).
        prefix_tokens = sum(
            ctx.counter.count_text(m.content, request.model) for m in request.messages[:boundary]
        ) + count_tools(request.tools, ctx.counter, request.model)
        floor = min_prefix_tokens_for(request.model)
        if prefix_tokens < floor:
            # Below the provider's floor the marker is ignored. Placing it
            # anyway would show up in reports as work done for no effect.
            #
            # The reason is recorded rather than silent (ADR-027): the live
            # Anthropic run reported zero cache reads on eleven of twelve
            # workloads and said nothing about why, which reads like a broken
            # stage instead of a prompt that was never eligible.
            self.last_decline_reason = (
                f"prefix is ~{prefix_tokens} tokens, below {model_floor_note(request.model, floor)}"
            )
            return self.declines(request)

        ttl = self._ttl_for(request, boundary, enabled=ctx.config.cache_ttl_selection)
        # Only a request we actually marked can tell `after` anything about
        # marking. Below-floor requests report reads 0 / writes 0 forever, and
        # counting those would disable the stage on workloads it never tried.
        ctx.scratch[_MARKED] = True

        marked = list(request.messages)
        marked[boundary - 1] = _with_cacheable(marked[boundary - 1], ttl)
        note = f"prefix marked at message {boundary} (~{prefix_tokens} tokens)"
        if ttl is not None:
            note += f", ttl {ttl}"
        return StageResult(request=request.with_messages(tuple(marked)), note=note)

    def after(self, request: LLMRequest, response: LLMResponse, ctx: StageContext) -> None:
        """Learn from what the provider actually served (ADR-030 amendment).

        A marked request that produced a cache write and no read did not pay
        for itself: 1.25x base was billed where 1.0x would have done. Repeated,
        that is a standing loss, and this is the outcome rather than a proxy for
        it -- which is why it needs three observations where the digest-based
        detector needs ten.

        A read resets the run outright. It is direct proof the prefix is
        cacheable, and the counter is consecutive rather than lifetime so an
        agent that goes quiet past the TTL and resumes cannot accumulate its way
        to a permanent decline.
        """
        del request
        if not ctx.scratch.get(_MARKED):
            return
        if response.cached_input_tokens > 0:
            self._unrewarded_writes = 0
            return
        # A response carrying no write count is not evidence either way.
        # OpenAI populates its cache automatically and reports no write, and
        # reading that silence as a wasted premium would switch the marker off
        # for a provider where it was never the mechanism.
        if response.cache_write_tokens > 0:
            self._unrewarded_writes += 1

    def _ttl_for(self, request: LLMRequest, boundary: int, *, enabled: bool) -> str | None:
        """Pick a cache lifetime, and record this sighting either way.

        ``None`` means "send no ``ttl`` field", which the provider reads as its
        five-minute default. That is the answer unless expiry has actually been
        *observed* for this prefix -- the same prefix seen again after a gap
        longer than :data:`FIVE_MINUTE_WINDOW_SECONDS`.

        Nothing here predicts a gap. A one-hour write costs 2x base input against
        1.25x, so guessing wrong raises the bill, and ADR-013's rule 1 forbids a
        cost-reduction library doing that. Reacting to an expiry that already
        happened is free of that risk in a way any predictor would not be.

        The sighting is recorded even when the feature is off, so enabling the
        flag mid-process does not start from a blank history -- and so the
        bookkeeping is exercised by the default configuration rather than only by
        the opt-in path.
        """
        digest = _prefix_digest(request, boundary)
        now = self._clock()
        previous = self._last_seen.get(digest)

        if previous is not None and (now - previous) > FIVE_MINUTE_WINDOW_SECONDS:
            self._expires.add(digest)

        self._last_seen[digest] = now
        self._last_seen.move_to_end(digest)
        while len(self._last_seen) > MAX_TRACKED_PREFIXES:
            evicted, _ = self._last_seen.popitem(last=False)
            self._expires.discard(evicted)

        if not enabled:
            return None
        return ONE_HOUR_TTL if digest in self._expires else None

    @staticmethod
    def _stable_prefix_length(request: LLMRequest) -> int:
        """How many leading messages will still be identical next call.

        System messages qualify unconditionally. Beyond them, the *oldest* turns
        of a conversation are stable too — a chat only ever appends — so the
        run extends through history while leaving the final turns unmarked,
        since those are what changes.

        Returns:
            Count of leading messages forming a stable prefix; ``0`` when
            marking would not help.
        """
        messages = request.messages
        if not messages:
            return 0

        boundary = 0
        while boundary < len(messages) and messages[boundary].role == "system":
            boundary += 1

        # Extend through history, holding back the last exchange: marking right
        # up to the newest message would invalidate the cached prefix on the
        # very next call, which is the classic way to get zero benefit while
        # believing the feature is on.
        stable_history = max(0, len(messages) - 2)
        return max(boundary, stable_history) if stable_history > boundary else boundary


def _with_cacheable(message: Message, ttl: str | None = None) -> Message:
    """Return a copy of ``message`` flagged as a prefix-cache boundary."""
    return replace(message, cacheable=True, cache_ttl=ttl)


def _prefix_digest(request: LLMRequest, boundary: int) -> str:
    """Identify a prefix without carrying it.

    A hash, for the reason :func:`~optio_optimize.cache.request_key` gives: these
    values live in a long-running process and reach debug output, and §10's rule
    that this package never emits prompt content applies to an identifier derived
    from a prompt just as much as to the prompt.

    The model is part of the digest. Two workloads with the same system prompt on
    different models have separate cache entries at the provider, so one
    observing expiry says nothing about the other.
    """
    parts = [request.model, *(m.content for m in request.messages[:boundary])]
    encoded = "\x00".join(parts).encode("utf-8", "replace")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()
