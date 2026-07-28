"""Configuration for the optimization pipeline.

Two principles shape the defaults.

**Lossless by default.** Every stage that can change what the model would have
produced is off until switched on by name. Someone who installs this package and
calls it with no arguments gets caching, prefix markers and token ceilings --
optimizations whose output is provably identical. Cheaper-but-different is a
choice a user makes, never one they inherit from a default.

**Setup fails loudly.** A misspelled stage name or a nonsensical threshold
raises at construction, matching §4.2 in the core. The alternative is an
optimizer that silently does nothing, which presents as "this library doesn't
work" long after the typo.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from optio_optimize.errors import OptimizeConfigError

#: Whole-pipeline latency ceiling in milliseconds (ADR-013 rule 4). Deliberately
#: not SC-5's 5 ms: that budget exists because optio is pure overhead, whereas a
#: stage that removes 60% of a prompt has earned some latency back. A stage
#: whose own budget would breach this is skipped for that request.
DEFAULT_LATENCY_BUDGET_MS = 100.0

#: Conversation turns kept verbatim before trimming or summarization applies.
#: Six is two or three exchanges -- enough for local coherence, which is where
#: most of the value in recent history is.
DEFAULT_RECENT_TURNS = 6

#: Cosine similarity above which a semantic cache entry counts as a hit.
#: 0.97 is deliberately severe. Public GPTCache-style examples often use 0.8-0.9;
#: at those thresholds "what is 2+2" and "what is 2+3" collide, which for an
#: agent making decisions is a wrong answer served instantly. Raising this is
#: how you trade correctness for hit rate, and it should be a conscious edit.
DEFAULT_SEMANTIC_THRESHOLD = 0.97


@dataclass(frozen=True, slots=True)
class OptimizeConfig:
    """Pipeline configuration.

    Attributes:
        enabled: Master switch. ``False`` passes every request through
            untouched, which is the control arm for A/B measurement.
        exact_cache: Serve byte-identical deterministic requests from cache.
            Lossless.
        prefix_cache: Place provider prefix-cache markers. Lossless, and the
            single highest-value lossless optimization on most workloads.
        adaptive_max_tokens: Cap output length from observed history. Lossless
            in the sense that it does not alter a response that would have
            fitted; it can truncate one that would have run longer.
        structured_output: Prefer JSON schemas over free prose where the caller
            supplied one.
        trim_history: Drop old turns beyond the recent window.
        deduplicate: Remove repeated identical context blocks.
        prune_retrieval: Drop retrieved chunks that do not earn their tokens.
        summarize_history: Replace old turns with a model-written summary.
            Lossy: costs a small model call, and the summary is not the history.
        route_models: Send easy steps to a cheaper model. Lossy.
        semantic_cache: Serve near-matching requests from cache. **Lossy in the
            strongest sense** -- returns text the model never produced for this
            prompt. Off by default; see :data:`DEFAULT_SEMANTIC_THRESHOLD`.
        compress_prompt: Drop low-information tokens from the prompt. Lossy.
        recent_turns: Turns kept verbatim by trimming and summarization.
        semantic_threshold: Similarity required for a semantic cache hit.
        latency_budget_ms: Whole-pipeline ceiling.
        context_limit: Model context window, when the caller knows it.
        cheap_model: Model the routing stage may downgrade to.
        disabled_stages: Stage names to skip regardless of other settings. The
            escape hatch for "this one misbehaves on my workload".
    """

    enabled: bool = True

    # Lossless -- on by default.
    exact_cache: bool = True
    prefix_cache: bool = True
    adaptive_max_tokens: bool = True
    structured_output: bool = True

    # Bounded-risk -- on by default, they drop context rather than invent it.
    trim_history: bool = True
    deduplicate: bool = True
    prune_retrieval: bool = True

    # Lossy -- off by default (ADR-013).
    summarize_history: bool = False
    route_models: bool = False
    semantic_cache: bool = False
    compress_prompt: bool = False

    recent_turns: int = DEFAULT_RECENT_TURNS
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS
    context_limit: int | None = None
    cheap_model: str | None = None
    disabled_stages: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate at construction (§4.2: setup fails loudly).

        Raises:
            OptimizeConfigError: On any invalid or incoherent setting.
        """
        if self.recent_turns < 1:
            raise OptimizeConfigError(
                f"recent_turns must be at least 1, got {self.recent_turns}. "
                "Zero would discard the current exchange, not just history."
            )
        if not 0.0 < self.semantic_threshold <= 1.0:
            raise OptimizeConfigError(
                f"semantic_threshold must be in (0.0, 1.0], got {self.semantic_threshold}"
            )
        if self.semantic_cache and self.semantic_threshold < 0.9:
            # Not a hard floor -- the owner may know their workload -- but below
            # 0.9 unrelated prompts collide, and silently serving those is the
            # failure this package must not cause by accident.
            raise OptimizeConfigError(
                f"semantic_threshold={self.semantic_threshold} is below 0.9, at which "
                "unrelated prompts match and the cache returns wrong answers. Set it "
                "to 0.9 or above, or disable semantic_cache."
            )
        if self.latency_budget_ms <= 0:
            raise OptimizeConfigError(
                f"latency_budget_ms must be positive, got {self.latency_budget_ms}"
            )
        if self.context_limit is not None and self.context_limit < 1:
            raise OptimizeConfigError(f"context_limit must be positive, got {self.context_limit}")
        if self.route_models and not self.cheap_model:
            raise OptimizeConfigError(
                "route_models is on but cheap_model is unset; there is nothing to route to"
            )

    @property
    def lossy_enabled(self) -> tuple[str, ...]:
        """Names of the enabled stages that can change model output.

        Used by the eval gate and reported at startup, so that running in a
        cheaper-but-different mode is never an unnoticed state.
        """
        candidates = {
            "summarize_history": self.summarize_history,
            "route_models": self.route_models,
            "semantic_cache": self.semantic_cache,
            "compress_prompt": self.compress_prompt,
        }
        return tuple(sorted(name for name, on in candidates.items() if on))


def config_from_mapping(values: dict[str, Any]) -> OptimizeConfig:
    """Build a config from a mapping, rejecting unknown keys.

    Args:
        values: Field names to values.

    Returns:
        A validated configuration.

    Raises:
        OptimizeConfigError: If a key is not a configuration field. A silently
            ignored typo means an optimization the user believes is on and is
            not -- indistinguishable from the library not working.
    """
    known = {f.name for f in fields(OptimizeConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise OptimizeConfigError(
            f"unknown config option(s): {', '.join(unknown)}. Valid options: "
            f"{', '.join(sorted(known))}"
        )
    payload = dict(values)
    if "disabled_stages" in payload:
        payload["disabled_stages"] = frozenset(payload["disabled_stages"])
    return OptimizeConfig(**payload)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per million tokens, for translating token savings into money.

    Attributes:
        input_usd_per_m: Prompt token price.
        output_usd_per_m: Completion token price.
        cached_input_usd_per_m: Price of a provider prefix-cache hit. Typically
            a tenth of the input rate; ``None`` means the provider does not
            discount, in which case the prefix stage reports no saving rather
            than an assumed one.
    """

    input_usd_per_m: float
    output_usd_per_m: float
    cached_input_usd_per_m: float | None = None


#: Prices for the models the router and reporter know about. Shares the
#: staleness problem of the core's table and the same mitigation: it is data,
#: auditable against the vendor's page, and overridable.
PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    "claude-opus-4": ModelPricing(15.00, 75.00, 1.50),
    "claude-sonnet-4": ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4": ModelPricing(0.80, 4.00, 0.08),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40, 0.025),
}

#: Cheap counterpart per model family, used when routing is on and no explicit
#: ``cheap_model`` fits. Only same-vendor pairs: crossing vendors changes
#: tokenizer, tool-call format and refusal behaviour all at once.
CHEAP_COUNTERPART: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "claude-opus-4": "claude-haiku-4",
    "claude-sonnet-4": "claude-haiku-4",
}
