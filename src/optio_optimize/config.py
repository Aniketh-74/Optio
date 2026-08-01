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

import re
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import date
from enum import Enum
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
        detect_window_pressure: Watch for prompts approaching the model's
            context window and report before the provider rejects them. Observes
            only, like ``detect_unstable_prefix``. Silent on any model whose
            window is neither in
            :data:`~optio_optimize.config.CONTEXT_WINDOW` nor given as
            ``context_limit`` (ADR-037).
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
        cascade_routing: Try the cheaper model first, verify the answer, and
            escalate to the requested model on failure (ADR-023). Lossy, but
            bounded by the verifier in a way static ``route_models`` is not.
            Needs ``cheap_model``; mutually exclusive with ``route_models``
            (both retarget ``.model``, so running them together would route
            twice). Not a stage -- it wraps the provider call, so it does not
            appear in ``stage_names``; its activity is reported via
            ``Optimizer.cascade_stats``.
        cascade_structured_output: Let cascade also attempt requests carrying a
            ``response_format`` (ADR-023 step 1). Static ``route_models``
            declines these because it cannot check the schema was honoured;
            cascade can, because the requested JSON *is* a verifier -- the
            built-in ``default_verifier`` escalates when the cheap answer does
            not parse as JSON or drops a required key. Off by default; requires
            ``cascade_routing``.
        cascade_max_tokens: Prompt-token ceiling for cascade attempts, raising
            the static ``route_models`` limit of 500 (ADR-023 step 2). ``None``
            keeps that default. Because cascade escalates a cheap answer it
            cannot verify, the ceiling is a *cost* knob rather than a safety
            one: a higher limit lets longer prompts try the cheap model, and
            only stops paying if enough of them escalate that the wasted cheap
            attempts outweigh the wins. Requires ``cascade_routing``.
        cascade_tools: Let cascade also attempt requests carrying ``tools``
            (ADR-023 step 3). Safe only because the model returns a *proposed*
            call, not an executed one: the built-in ``default_verifier`` vets it
            (escalates on an unknown tool name or non-JSON arguments) before the
            agent runs anything. Requires that the provider adapter surface the
            proposed call in ``response.extra["tool_calls"]``; until it does, a
            tool request escalates rather than mis-routing, so enabling this is
            never worse than static behaviour. Off by default; requires
            ``cascade_routing``.
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
        context_limit: Model context window, when the caller knows it. Read by
            ``detect_window_pressure``, which reports a prompt approaching the
            limit that will reject it, and overrides
            :data:`~optio_optimize.config.CONTEXT_WINDOW` for the model in the
            request. Worth setting for any model this package has not measured
            -- without it the diagnostic stays silent rather than guessing.

            It binds the **prompt only**. ``prompt + max_tokens`` over the
            window is not an error: 158,965 prompt tokens plus a 21,000 ceiling
            against a 200,000 window generated normally, so nothing here lowers
            a reply ceiling on account of window pressure (ADR-037). The limit
            that binds the reply is a different one, and it is not configurable
            because the provider states it exactly -- see
            :data:`~optio_optimize.config.MAX_OUTPUT_TOKENS`.
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
    # Off since 2026-07-31 (ADR-024). It was on, and the first end-to-end live
    # agent run found it raising cost on two of four scenarios and helping none
    # -- while the report claimed 10.0% and 13.2% savings on the two it made
    # more expensive. ADR-013 rule 1 forbids the increase, and its benefit has
    # never been measured on a request that actually carries a schema.
    structured_output: bool = False
    minify_tools: bool = True

    # Cache economics -- off by default because it is the only flag here that
    # can raise a bill rather than lower one (ADR-021).
    cache_ttl_selection: bool = False

    # Diagnostic -- transforms nothing, so there is no risk tier to place it in.
    detect_unstable_prefix: bool = True
    # On by default for the same reason as the line above: it observes, changes
    # nothing, and cannot cost anything. Unlike that one it is also silent
    # unless it has a measured window to compare against, so on an unmeasured
    # model it is not merely harmless but inert (ADR-037).
    detect_window_pressure: bool = True

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
    cascade_routing: bool = False
    cascade_structured_output: bool = False
    cascade_tools: bool = False
    semantic_cache: bool = False
    compress_prompt: bool = False
    prune_tools: bool = False
    chain_of_draft: bool = False
    reasoning_budget: bool = False

    cascade_max_tokens: int | None = None
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
        if self.cascade_routing and not self.cheap_model:
            raise OptimizeConfigError(
                "cascade_routing is on but cheap_model is unset; there is nothing to route to"
            )
        if self.cascade_routing and self.route_models:
            raise OptimizeConfigError(
                "cascade_routing and route_models are both on; both retarget the model, so "
                "running them together would route twice. Enable one -- cascade_routing is the "
                "one that verifies before it commits (ADR-023)."
            )
        if self.cascade_structured_output and not self.cascade_routing:
            raise OptimizeConfigError(
                "cascade_structured_output is on but cascade_routing is off; it widens what "
                "cascade attempts and does nothing without the cascade itself (ADR-023 step 1)."
            )
        if self.cascade_max_tokens is not None:
            if not self.cascade_routing:
                raise OptimizeConfigError(
                    "cascade_max_tokens is set but cascade_routing is off; it only bounds "
                    "cascade attempts and does nothing without the cascade itself (ADR-023 step 2)."
                )
            if self.cascade_max_tokens < 1:
                raise OptimizeConfigError(
                    f"cascade_max_tokens must be positive, got {self.cascade_max_tokens}. "
                    "Use None to keep the default 500-token ceiling."
                )
        if self.cascade_tools and not self.cascade_routing:
            raise OptimizeConfigError(
                "cascade_tools is on but cascade_routing is off; it widens what cascade "
                "attempts and does nothing without the cascade itself (ADR-023 step 3)."
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
    # Anthropic rows are keyed by the alias a caller writes. The API reports a
    # *dated* id back on every response, and `pricing_for` resolves one to the
    # other -- a second row per model was the previous answer and it scaled
    # badly, since every model needs one and a missing one is invisible.
    #
    # `claude-opus-4`, `claude-sonnet-4` and `claude-haiku-4` used to sit here.
    # All three return `404 not_found_error`: they were family keys, not model
    # ids, and one of them being the benchmark's default meant no live
    # Anthropic run had ever completed a call (ADR-029).
    #
    # Rates are the published list price at time of writing and share this
    # table's standing staleness caveat -- data, auditable against the vendor's
    # page, overridable. Seven models Anthropic currently serves are
    # deliberately absent, because nobody here has read their rates off that
    # page: opus-5, sonnet-5, fable-5, opus-4-6/4-7/4-8, sonnet-4-6.
    # Every band written out rather than computed from the base rate, even
    # though all sixteen published rows follow 1.25x / 2.00x / 0.10x exactly. A
    # multiplier holding across every model today is a fact about today's price
    # list, not a law: the first model to break the pattern would be silently
    # mispriced by a formula and visibly wrong in a table (ADR-031).
    "claude-fable-5": ModelPricing(10.00, 50.00, 1.00, 12.50, 20.00),
    "claude-mythos-5": ModelPricing(10.00, 50.00, 1.00, 12.50, 20.00),
    "claude-opus-5": ModelPricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-8": ModelPricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-7": ModelPricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-6": ModelPricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-5": ModelPricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-1": ModelPricing(15.00, 75.00, 1.50, 18.75, 30.00),
    # The rate in force *before* the first entry in `_SCHEDULED_PRICING`.
    "claude-sonnet-5": ModelPricing(2.00, 10.00, 0.20, 2.50, 4.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00, 0.30, 3.75, 6.00),
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00, 0.30, 3.75, 6.00),
    "claude-haiku-3-5": ModelPricing(0.80, 4.00, 0.08, 1.00, 1.60),
    # Haiku 4.5, added 2026-07-30 for the Anthropic prefix-cache measurement.
    # The measurement that motivated the entry is a *token* count
    # (`cache_read_input_tokens`); the dollar figure derived from it is only as
    # current as this row.
    #
    # The fourth rate is the cache-write premium, and it was missing when this
    # row was first added -- so the measurement it exists to price charged 5,487
    # write tokens at 1.00 instead of 1.25 and reported a 53.7% saving where the
    # true figure is 50.1%. An error in the direction that flatters the library.
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, 0.10, 1.25, 2.00),
    # **Google lists this model as "Shut down"**, under "Previous models" on
    # https://ai.google.dev/gemini-api/docs/models -- read 2026-08-02 while
    # looking up its token limits for the table below, which is why it carries
    # none. The row is left in place rather than deleted: removing a price
    # silently changes what every historical report meant, and this package has
    # exactly one Google model, so dropping it would also drop the only evidence
    # that a third vendor was ever priced here.
    #
    # It should not be used for new work, and no live provider here serves it.
    # This is the ADR-029 shape -- a table naming a model the API will not
    # serve -- caught by reading the vendor's page rather than by a 404.
    "gemini-2.0-flash": ModelPricing(0.10, 0.40, 0.025),
}

#: A suffix that names the same model rather than a newer one: a release date
#: or a Bedrock revision tag. The rule, and the reasoning behind the four-digit
#: discriminator, is ADR-029's -- `-20251101` is a snapshot of one model and
#: `-5` is the next one. Duplicated from `optio.lanes.cost.pricing` rather than
#: imported: `optio_optimize` has no dependency on `optio` and ADR-013 exists to
#: keep it that way, so five lines of regex is the cheaper of the two prices.
_SAME_MODEL_SUFFIX = re.compile(r"^-(?:\d{4,}|v\d+(?::\d+)?)(?:-|$)")

#: Published price changes with a known effective date, newest last.
#:
#: The vendor's page lists Sonnet 5 twice -- ``2 / 10`` "through Aug 31, 2026"
#: and ``3 / 15`` "from Sep 1, 2026". Whichever single number were written here
#: would be wrong on one side of the boundary and wrong by 50%, which is larger
#: than most savings this package reports. Recording a dated commitment from the
#: vendor's own page is not the prediction this project has a rule against; it is
#: the same auditable data as the row above it (ADR-031).
#:
#: Duplicated from ``optio.lanes.cost.pricing`` rather than imported, for the
#: same reason ``_SAME_MODEL_SUFFIX`` is: ADR-013 keeps this package free of any
#: dependency on ``optio``.
_SCHEDULED_PRICING: dict[str, tuple[tuple[date, ModelPricing], ...]] = {
    "claude-sonnet-5": ((date(2026, 9, 1), ModelPricing(3.00, 15.00, 0.30, 3.75, 6.00)),),
}


def pricing_for(model: str, *, today: Callable[[], date] | None = None) -> ModelPricing | None:
    """Look up a model's rates, or ``None`` rather than something close.

    Exact match first, then a prefix match restricted to suffixes that denote
    the same model. Of the eleven ids ``models.list`` returned on 2026-07-31,
    exact matching alone priced one.

    Args:
        model: Model id, as a caller wrote it or as the API reported it back.
        today: Source of the current date, for resolving a scheduled price
            change. Injectable so tests can stand on both sides of an effective
            date without waiting for the calendar.

    Returns:
        The rates, or ``None`` for a model this table does not carry. Never a
        neighbouring generation's row: Anthropic cut Opus list pricing at 4.5,
        so inferring across generations is how a $10 bill gets reported as $30
        (ADR-029).
    """
    if not model:
        return None
    matched = _row_for(model)
    if matched is None:
        return None
    name, pricing = matched
    # Resolved per call rather than at import, so a process running across an
    # effective date does not keep serving the stale rate (ADR-031).
    now = today() if today is not None else date.today()
    for effective, scheduled in _SCHEDULED_PRICING.get(name, ()):
        if now >= effective:
            pricing = scheduled
    return pricing


def _row_for(model: str) -> tuple[str, ModelPricing] | None:
    """The table row for ``model``, with the key that matched it."""
    exact = PRICING.get(model)
    if exact is not None:
        return model, exact
    for name in sorted(PRICING, key=len, reverse=True):
        if model.startswith(name) and _SAME_MODEL_SUFFIX.match(model[len(name) :]):
            return name, PRICING[name]
    return None


class Evidence(Enum):
    """How a model limit in this module came to be known.

    Two ways, and they are not equally strong. Keeping them apart is what lets
    this package carry a vendor it has never billed without pretending to have
    measured it.

    Attributes:
        MEASURED: This project sent a request and read the provider's answer --
            usually its own 400, which is the provider's arithmetic rather than
            a page about it (the reasoning ADR-036 used for ``count_tokens``).
            Costs a key and some money. Cannot go stale silently, because the
            probe is re-runnable.
        PUBLISHED: The vendor documents it and this project has not observed it.
            Costs nothing and covers every vendor with a documentation page. It
            is **not a guess** -- it is a citable claim, which is why a
            published entry must carry a URL. It can go stale without anything
            here noticing, which is why it must also carry a date.
    """

    MEASURED = "measured"
    PUBLISHED = "published"


@dataclass(frozen=True, slots=True)
class Limit:
    """A model limit together with why anyone should believe it.

    The type exists to make the citation structural. These tables previously
    held bare ``int``s, and an ``int`` cannot say where it came from -- so the
    only entries anyone could justify adding were ones this project had paid to
    measure, and **coverage became a function of whose API key was to hand.**
    That is the wrong thing for a multi-vendor library's coverage to depend on.

    Attributes:
        tokens: The limit itself.
        evidence: Which of the two ways this is known. See :class:`Evidence`.
        source: For a measured limit, the probe that produced it. For a
            published one, a URL a reader can open -- the only thing separating
            "the vendor states this" from "somebody typed a number".
        checked: ISO date the source last said so. A published figure with no
            date cannot be told from a current one, which is the same reason a
            recording carries the date it was made (ADR-039).
    """

    tokens: int
    evidence: Evidence
    source: str
    checked: str


#: The probe behind every measured entry below. Named once so the tables stay
#: readable and so re-running it is one obvious command.
_PROBE = "scripts/measure_window_pressure.py"

#: OpenAI's model pages, which state both limits directly. Read 2026-08-02.
#: These are the first non-Anthropic rows either table has ever carried, and
#: they cost nothing to obtain -- which is the entire argument for
#: :attr:`Evidence.PUBLISHED` existing.
_OPENAI_GPT_4O = "https://developers.openai.com/api/docs/models/gpt-4o"
_OPENAI_GPT_4O_MINI = "https://developers.openai.com/api/docs/models/gpt-4o-mini"


#: Context window per model, in tokens: the limit that binds the **prompt**.
#:
#: Exceeding it is a hard 400 before any generation --
#: ``prompt is too long: 217570 tokens > 200000 maximum`` -- and no stage in
#: this package can rescue such a request.
#:
#: Every value here was read out of that error message rather than off a
#: documentation page, which is what makes it the provider's own arithmetic
#: (the reasoning ADR-036 used for ``count_tokens``).
#:
#: **Seven models this package prices are deliberately absent.** Opus 5, Fable 5,
#: Opus 4-6/4-7/4-8, Sonnet 5 and Sonnet 4-6 *accepted* a 217,554-token probe
#: instead of rejecting it, which establishes ``window > 217,554`` and nothing
#: further. Recording 1,000,000 for them would be an inference across a
#: generation boundary -- the error ADR-029 exists because of -- and a
#: diagnostic that guesses a window is worse than one that stays quiet.
#: Every value here was read out of that error message rather than off a
#: documentation page, and :class:`Limit` is what lets a value say which.
CONTEXT_WINDOW: dict[str, Limit] = {
    "claude-opus-4-5": Limit(200_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-4-1": Limit(200_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-sonnet-4-5": Limit(200_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-haiku-4-5": Limit(200_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "gpt-4o": Limit(128_000, Evidence.PUBLISHED, _OPENAI_GPT_4O, "2026-08-02"),
    "gpt-4o-mini": Limit(128_000, Evidence.PUBLISHED, _OPENAI_GPT_4O_MINI, "2026-08-02"),
}

#: Maximum completion tokens per model: the limit that binds the **reply**.
#:
#: A different limit from :data:`CONTEXT_WINDOW`, not derivable from it, and one
#: this package had no concept of before ADR-037. Three models share a 200,000
#: window with caps of 32,000 and 64,000, and the 128,000-cap models have a
#: window larger than either and still unmeasured.
#:
#: Exceeding it is a hard 400 naming the cap: ``max_tokens: 1000000 > 64000,
#: which is the maximum allowed number of output tokens``. That matters here
#: because ``AdaptiveMaxTokensStage`` *sets* ``max_tokens`` on requests that
#: carried none, and its ceiling can land above a cap this low.
MAX_OUTPUT_TOKENS: dict[str, Limit] = {
    "claude-fable-5": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-5": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-4-8": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-4-7": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-4-6": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-4-5": Limit(64_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-opus-4-1": Limit(32_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-sonnet-5": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-sonnet-4-6": Limit(128_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-sonnet-4-5": Limit(64_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    "claude-haiku-4-5": Limit(64_000, Evidence.MEASURED, _PROBE, "2026-08-01"),
    # 16,384 against a 128,000 window: the narrowest cap-to-window ratio in
    # either table by a wide margin, and a reminder that the two limits are
    # genuinely independent (ADR-037). `AdaptiveMaxTokensStage` now clamps to
    # this on OpenAI, where before it had no cap to clamp to at all.
    "gpt-4o": Limit(16_384, Evidence.PUBLISHED, _OPENAI_GPT_4O, "2026-08-02"),
    "gpt-4o-mini": Limit(16_384, Evidence.PUBLISHED, _OPENAI_GPT_4O_MINI, "2026-08-02"),
}


def _limit_for(table: dict[str, Limit], model: str) -> Limit | None:
    """Look up a per-model limit, resolving a dated id to its alias.

    Returns ``None`` for a model the table does not carry -- never a
    neighbouring generation's figure. ``claude-opus-4-1`` caps output at 32,000
    and ``claude-opus-4-5`` at 64,000, so inheriting across that boundary would
    hand back double the real limit and produce the 400 the lookup exists to
    avoid.
    """
    if not model:
        return None
    exact = table.get(model)
    if exact is not None:
        return exact
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name) and _SAME_MODEL_SUFFIX.match(model[len(name) :]):
            return table[name]
    return None


def context_window_for(model: str) -> int | None:
    """The model's prompt limit in tokens, or ``None`` if this package has none.

    Args:
        model: Model id, as a caller wrote it or as the API reported it back.

    Returns:
        The window, or ``None``. Absence means "no evidence either way", never
        "unlimited" and never a guess -- see :data:`CONTEXT_WINDOW` for the
        seven models that sit in exactly that position. Use
        :func:`context_window_provenance` to tell an observed figure from a
        documented one.
    """
    limit = _limit_for(CONTEXT_WINDOW, model)
    return limit.tokens if limit is not None else None


def context_window_provenance(model: str) -> Evidence | None:
    """How the window in :func:`context_window_for` is known, if it is.

    Args:
        model: Model id.

    Returns:
        :attr:`Evidence.MEASURED`, :attr:`Evidence.PUBLISHED`, or ``None`` when
        this package carries no window for the model at all. The third case is
        distinct from the second on purpose: "we have not looked" and "the
        vendor says so" are different claims, and collapsing them is how the
        table came to look Anthropic-only.
    """
    limit = _limit_for(CONTEXT_WINDOW, model)
    return limit.evidence if limit is not None else None


def max_output_tokens_for(model: str) -> int | None:
    """The model's reply limit in tokens, or ``None`` if this package has none.

    Args:
        model: Model id, as a caller wrote it or as the API reported it back.

    Returns:
        The cap, or ``None`` for a model this table does not carry.
    """
    limit = _limit_for(MAX_OUTPUT_TOKENS, model)
    return limit.tokens if limit is not None else None


def max_output_tokens_provenance(model: str) -> Evidence | None:
    """How the cap in :func:`max_output_tokens_for` is known, if it is.

    Args:
        model: Model id.

    Returns:
        The evidence class, or ``None`` when no cap is carried.
    """
    limit = _limit_for(MAX_OUTPUT_TOKENS, model)
    return limit.evidence if limit is not None else None


#: Cheap counterpart per model family, used when routing is on and no explicit
#: ``cheap_model`` fits. Only same-vendor pairs: crossing vendors changes
#: tokenizer, tool-call format and refusal behaviour all at once.
CHEAP_COUNTERPART: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    # Both sides used to name models that do not exist -- Opus and Sonnet
    # mapped to `claude-haiku-4`, which 404s, so every Anthropic routing run
    # this table configured would have failed on its first downgraded call
    # (ADR-029). Benchmark-only: production routing reads `config.cheap_model`,
    # which the caller sets.
    #
    # Every pair is now checkable against real rates rather than asserted: the
    # target must be strictly cheaper on both input and output (ADR-031).
    "claude-fable-5": "claude-haiku-4-5",
    "claude-mythos-5": "claude-haiku-4-5",
    "claude-opus-5": "claude-haiku-4-5",
    "claude-opus-4-8": "claude-haiku-4-5",
    "claude-opus-4-7": "claude-haiku-4-5",
    "claude-opus-4-6": "claude-haiku-4-5",
    "claude-sonnet-5": "claude-haiku-4-5",
    "claude-sonnet-4-6": "claude-haiku-4-5",
    "claude-opus-4-5": "claude-haiku-4-5",
    "claude-opus-4-1": "claude-haiku-4-5",
    "claude-sonnet-4-5": "claude-haiku-4-5",
}
