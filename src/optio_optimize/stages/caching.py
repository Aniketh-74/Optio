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

from dataclasses import replace
from typing import TYPE_CHECKING

from optio_optimize.cache import MemoryCache, request_key
from optio_optimize.stages.base import Fidelity, Stage, StageResult
from optio_optimize.tokens import count_request

if TYPE_CHECKING:
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
#: Left as one number deliberately: a per-model table changes what every
#: Anthropic caller sends, and ADR-016 does not let one measurement carry that.
#: The cost of being wrong is a marker that does nothing, never a wrong answer.
#: ``docs/optimize-benchmarks.md`` records the full table.
MIN_PREFIX_TOKENS = 1024


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

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "prefix_cache"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Mark the cacheable prefix boundary, if one is worth marking."""
        boundary = self._stable_prefix_length(request)
        if boundary == 0:
            return self.declines(request)

        prefix_tokens = sum(
            ctx.counter.count_text(m.content, request.model) for m in request.messages[:boundary]
        )
        if prefix_tokens < MIN_PREFIX_TOKENS:
            # Below the provider's floor the marker is ignored. Placing it
            # anyway would show up in reports as work done for no effect.
            return self.declines(request)

        marked = list(request.messages)
        marked[boundary - 1] = _with_cacheable(marked[boundary - 1])
        return StageResult(
            request=request.with_messages(tuple(marked)),
            note=f"prefix marked at message {boundary} (~{prefix_tokens} tokens)",
        )

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


def _with_cacheable(message: Message) -> Message:
    """Return a copy of ``message`` flagged as a prefix-cache boundary."""
    return replace(message, cacheable=True)
