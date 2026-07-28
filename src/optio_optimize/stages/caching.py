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

#: Minimum tokens before a prefix marker is worth placing. Providers impose
#: their own floors (Anthropic 1024 for most models, 2048 for the smallest) and
#: below those a marker is ignored, so requesting one is pure noise.
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
