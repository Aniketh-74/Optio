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
from optio_optimize.stages.tools import DEFAULT_MAX_TOOL_RESULT_TOKENS

#: Whole-pipeline latency ceiling in milliseconds (ADR-013 rule 4). Deliberately
#: not SC-5's 5 ms: that budget exists because optio is pure overhead, whereas a
#: stage that removes 60% of a prompt has earned some latency back. A stage
#: whose own budget would breach this is skipped for that request.
DEFAULT_LATENCY_BUDGET_MS = 100.0

#: Conversation turns kept verbatim before trimming or summarization applies.
#: Six is two or three exchanges -- enough for local coherence, which is where
#: most of the value in recent history is.
DEFAULT_RECENT_TURNS = 6

#: Oldest turns an anchored trim keeps. **Zero: the shipped behaviour is
#: unchanged**, because changing what every existing caller's prompt looks
#: like on a hypothesis is what ADR-016 exists to prevent -- and the hypothesis
#: here is a good one, which is exactly when that rule is easiest to break.
#:
#: The suggested setting when turning it on is ``2``: one user message and one
#: reply, the opening exchange where a task statement, a budget figure or a
#: constraint gets stated and then never repeated. Small on purpose -- the
#: anchor is billed on every request, and its whole argument is that it sits
#: inside the region a provider already serves from cache, so it should be
#: nearly free where it works at all.
DEFAULT_ANCHOR_TURNS = 0

#: Similarity above which a semantic cache entry counts as a hit -- against
#: whatever :data:`~optio_optimize.stages.semantic_cache.SimilarityFn` the
#: stage is given, cosine or otherwise. 0.97 is deliberately severe. Public
#: GPTCache-style examples often use cosine thresholds of 0.8-0.9; at those,
#: "what is 2+2" and "what is 2+3" collide, which for an agent making
#: decisions is a wrong answer served instantly. The packaged default
#: similarity function is lexical word-overlap, not embeddings (see
#: :mod:`optio_optimize.similarity`) -- coarser than cosine, so this
#: threshold does more of the safety work by itself there than it would
#: against a real embedding metric. Raising it is how you trade correctness
#: for hit rate, and it should be a conscious edit either way.
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
        concision: Suppress chat scaffolding -- restatement, self-summary,
            follow-up offers. Costs a one-sentence instruction in input to
            save output billed at several times the rate.
        cache_ttl_selection: Ask Anthropic for a **one-hour** cache entry once a
            prefix has been observed outliving the five-minute default. A
            5-minute write costs 1.25x base input, an hour costs 2x, and a read
            costs 0.1x from either -- so one further use inside the hour pays for
            the upgrade, and an agent step slower than five minutes otherwise
            re-writes a prefix it just wrote. Off by default despite changing
            nothing about the request: it is the only setting here that can
            *raise* a bill, on a prefix that turns out never to be re-used.
            Measured live at **-30.9%** over four rounds five minutes apart --
            but **+29.9% after only two**, because the upgrade write costs 2x
            and pays back on the round after. Worth turning on for an agent loop
            with steps slower than five minutes; not worth defaulting on for a
            prefix that might be seen twice and then dropped (ADR-021).
        detect_unstable_prefix: Watch for prompts whose cacheable head changes
            between calls, and report why. Observes only -- it never modifies a
            request, and it is the one setting here that cannot cost anything.
        minify_tools: Strip annotation-only keys from tool schemas. Removes
            nothing the model reads.
        cap_tool_results: Bound how many tokens one tool result may add. An
            oversized payload is billed again on every later turn, so the
            ceiling protects the rest of the conversation, not just this call.
        trim_history: Drop old turns beyond the recent window.
        deduplicate: Remove repeated identical context blocks.
        prune_retrieval: Drop retrieved chunks that do not earn their tokens.
        reorder_context: Position the strongest retrieved blocks at the edges
            of the context, weakest in the middle. Saves no tokens; buys
            headroom for the stages that do. Off by default because it
            invalidates provider caching below the reordered region.
        prune_tools: Drop tools unrelated to the conversation. **Lossy**: a
            wrongly-pruned tool is a capability the agent silently loses.
        summarize_history: Replace old turns with a model-written summary.
            Lossy: costs a small model call, and the summary is not the history.
        route_models: Send easy steps to a cheaper model. Lossy.
        semantic_cache: Serve near-matching requests from cache. **Lossy in the
            strongest sense** -- returns text the model never produced for this
            prompt. Off by default; see :data:`DEFAULT_SEMANTIC_THRESHOLD`.
        compress_prompt: Drop low-information tokens from the prompt. Lossy.
        chain_of_draft: Ask for shorthand reasoning rather than prose.
            Lossy: it changes how the model reasons, not how it presents.
        reasoning_budget: Lower a caller-set reasoning budget toward what this
            workload has been observed to use. **Lossy, and the tier matters
            more here than anywhere else** (ADR-018): reasoning tokens bill at
            the completion rate, so this is the most expensive lever in the
            package -- and a budget that binds degrades exactly on the hard
            questions a reasoning model was chosen for, leaving no trace in the
            prompt or the report. Never raises a budget and never sets one
            where the caller set none.
        recent_turns: Turns kept verbatim by trimming and summarization.
        anchor_turns: Oldest turns trimming must keep, dropping the middle
            instead of the front. The cached region a provider serves is
            ``system + oldest turns``, and the recall audit found the facts
            that matter stated in the first exchange -- so a front cut
            discards the cheapest and most valuable context at once. ``0``
            restores the plain sliding window.
        compact_at_tokens: Hold trimming until the prompt reaches this size,
            then cut in one go ("append-then-compact"). ``None`` trims every
            turn. The trade is a stable prompt head -- and so a live provider
            prefix cache -- against carrying more tokens between compactions;
            see :class:`~optio_optimize.stages.history.TrimHistoryStage`.
        max_tool_result_tokens: Ceiling applied by ``cap_tool_results``.
        semantic_threshold: Similarity required for a semantic cache hit.
        latency_budget_ms: Whole-pipeline ceiling.
        context_limit: Model context window, when the caller knows it.
        cheap_model: Model the routing stage may downgrade to.
        disabled_stages: Stage names to skip regardless of other settings. The
            escape hatch for "this one misbehaves on my workload".
        emit_spans: Emit one OTel GenAI span per request/response cycle
            (ADR-014), so a live ``optio`` install prices and classifies
            optimizer activity through its existing span tap -- no call into
            ``optio`` required, and none made; this package still imports
            nothing from it. Off by default: a new, previously-absent side
            effect for every existing caller, even though a span with no
            configured exporter costs close to nothing. Failures here are
            swallowed, never raised -- see :mod:`optio_optimize.telemetry`.
    """

    enabled: bool = True

    # Lossless -- on by default.
    exact_cache: bool = True
    prefix_cache: bool = True
    adaptive_max_tokens: bool = True
    structured_output: bool = True
    minify_tools: bool = True

    # Cache economics -- off by default because it is the only flag here that
    # can raise a bill rather than lower one (ADR-021).
    cache_ttl_selection: bool = False

    # Diagnostic -- transforms nothing, so there is no risk tier to place it in.
    detect_unstable_prefix: bool = True

    # Bounded-risk -- on by default, they drop context rather than invent it.
    trim_history: bool = True
    deduplicate: bool = True
    prune_retrieval: bool = True
    cap_tool_results: bool = True

    # Quality-only, and it costs provider caching below the region it moves.
    reorder_context: bool = False

    # Off by default despite being SHAPED, because the saving is unproven.
    # `concision` spends input tokens on every request to save output tokens on
    # some, and only a live run can see the second half of that trade. The
    # adversarial `unique_questions` workload measured the visible half alone:
    # -14.8% token reduction, i.e. a cost *increase*, because twelve short
    # prompts each grew by a one-sentence instruction and the simulator returns
    # a fixed-length completion no instruction can shorten. Turning it on by
    # default on the strength of a published 30-50% figure is exactly what
    # ADR-016 says not to do; it flips only if `--live` says it should.
    concision: bool = False

    # Lossy -- off by default (ADR-013).
    summarize_history: bool = False
    route_models: bool = False
    semantic_cache: bool = False
    compress_prompt: bool = False
    prune_tools: bool = False
    chain_of_draft: bool = False
    reasoning_budget: bool = False

    recent_turns: int = DEFAULT_RECENT_TURNS
    anchor_turns: int = DEFAULT_ANCHOR_TURNS
    compact_at_tokens: int | None = None
    max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS
    context_limit: int | None = None
    cheap_model: str | None = None
    disabled_stages: frozenset[str] = frozenset()
    emit_spans: bool = False

    def __post_init__(self) -> None:
        """Validate at construction (§4.2: setup fails loudly).

        Raises:
            OptimizeConfigError: On any invalid or incoherent setting.
        """
        if self.anchor_turns < 0:
            raise OptimizeConfigError(
                f"anchor_turns cannot be negative, got {self.anchor_turns}. "
                "Use 0 for a plain sliding window."
            )
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
        if self.compact_at_tokens is not None and self.compact_at_tokens < 1:
            raise OptimizeConfigError(
                f"compact_at_tokens must be positive, got {self.compact_at_tokens}. "
                "Use None to trim on every turn rather than 0."
            )
        if self.max_tool_result_tokens < 1:
            raise OptimizeConfigError(
                f"max_tool_result_tokens must be at least 1, got {self.max_tool_result_tokens}. "
                "Zero would erase every tool result rather than bounding it."
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
            "prune_tools": self.prune_tools,
            "chain_of_draft": self.chain_of_draft,
            "reasoning_budget": self.reasoning_budget,
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
        cache_write_usd_per_m: Price of *populating* the provider's cache with a
            **5-minute** entry. This is one of the two rates here that are
            **higher** than the base input rate -- Anthropic charges 1.25x -- and
            they are the only ones whose absence flatters this library rather
            than penalizing it. ``None`` means writes cost the base rate, which
            is correct for providers that populate their cache for free (OpenAI).
        cache_write_1h_usd_per_m: Price of populating a **one-hour** entry, which
            Anthropic charges at **2x** base input. Separate from the field above
            rather than folded into it, because the gap between them is 60% of a
            write and a single rate would have to be wrong for one band or the
            other. ``None`` falls back to the 5-minute rate -- never to the base
            rate, since a one-hour write is certainly not *cheaper* than a
            five-minute one and guessing low is the direction that inflates this
            package's reported saving (ADR-021).
    """

    input_usd_per_m: float
    output_usd_per_m: float
    cached_input_usd_per_m: float | None = None
    cache_write_usd_per_m: float | None = None
    cache_write_1h_usd_per_m: float | None = None


#: Prices for the models the router and reporter know about. Shares the
#: staleness problem of the core's table and the same mitigation: it is data,
#: auditable against the vendor's page, and overridable.
PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(2.50, 10.00, 1.25),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 0.075),
    # Every Anthropic row carries **two** cache-write rates: 1.25x its input rate
    # for a 5-minute entry and 2x for a one-hour one, both published premiums.
    # The second was added with ADR-021, before anything could request a
    # one-hour TTL -- with a single write rate, asking for an hour would bill the
    # most expensive band in the request at the cheaper band's price and
    # understate it by 37.5%, in the direction that inflates this package's
    # headline. The row below already records what that class of error cost once.
    #
    # OpenAI rows leave both None: OpenAI populates its cache for free and offers
    # no TTL control, and stating an explicit 1.0x multiple would imply someone
    # had checked a rate that does not exist.
    "claude-opus-4": ModelPricing(15.00, 75.00, 1.50, 18.75, 30.00),
    "claude-sonnet-4": ModelPricing(3.00, 15.00, 0.30, 3.75, 6.00),
    "claude-haiku-4": ModelPricing(0.80, 4.00, 0.08, 1.00, 1.60),
    # Haiku 4.5, added 2026-07-30 for the Anthropic prefix-cache measurement.
    # Both the alias and the dated id: callers write the alias, and the API
    # reports the dated one back on every response, so a table with only one of
    # them makes half the lookups return None. Rates are the published list
    # price at time of writing and share this table's standing staleness
    # caveat -- it is data, auditable against the vendor's page, and
    # overridable. The measurement that motivated the entry is a *token* count
    # (`cache_read_input_tokens`); the dollar figure derived from it is only as
    # current as this row.
    #
    # The fourth rate is the cache-write premium, and it was missing when this
    # row was first added -- so the measurement it exists to price charged 5,487
    # write tokens at 1.00 instead of 1.25 and reported a 53.7% saving where the
    # true figure is 50.1%. An error in the direction that flatters the library.
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, 0.10, 1.25, 2.00),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00, 0.10, 1.25, 2.00),
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
