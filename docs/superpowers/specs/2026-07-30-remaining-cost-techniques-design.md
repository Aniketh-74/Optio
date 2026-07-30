# The remaining cost techniques, in value order

**Status:** Accepted
**Date:** 2026-07-30
**Related:** ADR-013 (the package exists), ADR-015 (evidence bar), ADR-016 (in-scope test),
ADR-017 (a second surface)

## Context

`optio_optimize` ships 19 stages plus batch dispatch. That is close to exhaustive on the **prompt
side** and nearly empty on the **output side** — and the output side is where the remaining money
is, because reasoning tokens bill at completion rates.

An audit on 2026-07-30 found the gaps cluster in three places:

1. **Output-side control** is barely built. `LLMRequest.thinking_budget` exists, is typed,
   documented, and part of the cache key — and sits in `wire.UNSENT_FIELDS`, so it has never
   reached a provider. No stage sets `stop` either, though the field is modelled and sent.
2. **Non-text modalities are unmodelled.** No stage touches images and `tokens.py` does not count
   them, so multimodal reports undercount input as well as missing the saving.
3. **The cache economics this package just measured are not exploited.** The 1.25x write premium
   was only modelled on 2026-07-30; two techniques fall directly out of knowing it.

## The organizing thesis

**These techniques are not additive, and the provider cache is why.** Grouping by what a technique
touches predicts which combinations compose:

| group | members | touches the bytes the cache keys on? |
|---|---|---|
| **Exits** | `exact_cache`, `semantic_cache` | no — the call never happens |
| **Output side** | reasoning budget, `stop`, `adaptive_max_tokens`, `chain_of_draft` | **no** |
| **Cache economics** | `prefix_cache`, TTL choice, warm-up ordering | it *is* the cache |
| **Prompt shrinking** | `trim_history`, `deduplicate`, `prune_retrieval`, `compress_prompt` | **yes — fights it** |

The output-side group is the only one that stacks cleanly, because it never changes the prompt. It
is also the group this package has barely built. Prompt shrinking has the largest apparent numbers
and the worst interaction profile — already demonstrated four times: `trim_history` defeating
`prefix_cache`, `concision` evicting more cache than it saved, `reorder_context` breaking prefix
caching by construction, and `structured_output`/`concision` being mutually exclusive outright.

**This thesis is a prediction, not a result.** It orders the work below; each item still has to
earn its place by measurement.

## Non-goals

- **Prompt optimization** (DSPy/MIPRO-style search over prompt variants against a metric). Out of
  scope permanently, not deferred. It needs a labelled eval set and hundreds of calls to converge,
  it is a build-time activity, and it permanently rewrites *the caller's own source prompt* —
  where every stage here is a per-request transform that leaves caller code untouched. It is a
  different product with a different lifecycle. `compress_prompt`'s finding is also a warning:
  collapsing a 9x-repeated instruction to one statement is information-preserving by any
  reasonable definition, and the model stopped obeying it.
- **A per-model `MIN_PREFIX_TOKENS` table.** Recorded as a known issue; ADR-016 does not let one
  measurement change what every Anthropic caller sends.
- **The pairwise stage-interaction matrix.** Considered and deliberately not built first (see
  "Accepted risk").

## Accepted risk

`build_stages` encodes six ordering rules, each derived from a real failure, and has **no tests**.
Its own docstring says misordering "is subtle rather than loud… it just silently never hits the
provider cache." Interactions grow quadratically: 19 stages is 171 pairs, 26 is 325.

The alternative plan was to make interactions declarative and machine-checked before adding
anything. That was weighed and rejected in favour of shipping techniques first, on the grounds
that the framework delivers no saving on its own.

**The mitigation, per technique rather than in general:** each item below names the specific
existing stages it plausibly fights, and its gate includes a check against those. This is narrower
than a matrix and cheaper; it will miss conflicts nobody predicted, which is the risk being
accepted.

## The sequence

Every item is gated by ADR-016 (is this in scope?) and, where the tier is `ALTERED`, by ADR-015
(isolated live evidence, one technique at a time, before it ships enabled). An item that fails its
gate ships off by default or does not ship — three of the last four stages went that way.

### 1. Reasoning-budget control — the largest single lever

**Mechanism.** Reasoning tokens bill at *completion* rates and routinely exceed the visible answer
several-fold. Anthropic takes `thinking: {type, budget_tokens}`; OpenAI takes `reasoning_effort`.
Lowering it on steps that do not need deliberation is a direct cut to the most expensive tokens in
a request.

**Why first.** No other item touches output tokens at this magnitude, and the field is already
modelled — only the wire path and the deciding stage are missing.

**Fidelity: `ALTERED`.** This is the most consequential fidelity call in the package. Unlike
trimming, which drops context a caller can see is gone, a reduced budget silently degrades
correctness on precisely the hard problems that justified a reasoning model. Needs an ADR of its
own before implementation.

**Interaction risk: low, one real hazard.** Touches neither messages nor tools, so no cache
interaction. But it overlaps `chain_of_draft` (both target reasoning verbosity — likely
double-counted savings, the `minify_tools`/`prune_tools` problem again) and `adaptive_max_tokens`
(both cap output; a low budget plus a tight ceiling can truncate before the answer starts).

**Gate.** Isolated live run on a reasoning model, easy and hard task sets, measuring cost *and*
correctness. Ships off by default unless correctness holds.

**Done, 2026-07-30 — and the gate was not met.** Both hazards above turned out to be real, and the
`adaptive_max_tokens` one was a live bug rather than a risk: Anthropic rejects a `max_tokens` at or
below `thinking.budget_tokens`, so a default-on ceiling derived from observed output turned a working
reasoning call into a 400 and a fail-open call at full price. Fixed with an answer-headroom floor and
ordering rule 7. The overlap with `chain_of_draft` is settled by the stage claiming no saving at all.

The live run measured **−21.9%** against the mean of two bracketing controls with accuracy unchanged
— and then showed the mechanism is not the one the stage was built on: zero of forty control calls
came near the ceiling, so `budget_tokens` shapes the trace as a *target*, not just a cap. The
accuracy half of the gate was satisfied in form only, because every arm scored 100% on both sets. The
flag stays off. See the ADR-018 amendment for the numbers and the two limits on them.

### 2. Streaming — make existing savings reachable

**Mechanism.** `stream=True` currently bypasses the wrapper entirely, so a streaming caller gets
*nothing* — including exact-cache hits, which need no streaming machinery to serve, and prefix
markers, which are request-side only.

**Why second.** Not a new saving: it makes the savings already built reachable from the dominant
production mode for anything user-facing. A user enables the library, sees zero, and cannot tell
why. That is a plug-and-play defect more than a missing technique.

**Fidelity: `IDENTICAL`** for the cache-hit replay and the marker; the response bytes are the same.

**Interaction risk: medium.** Stage `after` hooks expect a complete response, so anything that
stores or measures a reply cannot run until the stream finishes. Scope to request-side stages plus
cache-hit replay; do not pretend the full pipeline works on a stream.

**Gate.** ADR-016 in-scope test, plus live confirmation that a streamed Anthropic call carries the
`cache_control` breakpoint.

### 3. Fan-out cache warm-up ordering

**Mechanism.** N cold parallel calls sharing a prefix each pay the 1.25x write premium, because
none can see another's cache. Issue one first, then the remaining N−1 → one write plus N−1 reads
at 0.1x.

**Fidelity: `IDENTICAL`.** No request changes at all; only dispatch order does.

**Shape.** Like batch (ADR-017), this answers *when and by whom*, not *what should this request
look like* — so it is not a `Stage`. Needs an ADR to settle whether it is an `Optimizer` option or
a third surface.

**Interaction risk: low.** Composes with `prefix_cache` by design; the hazard is latency, since
serializing the first call delays the batch.

**Gate.** Isolated live run on Anthropic against the existing `fan_out` workload.

### 4. Cache TTL selection

**Mechanism.** A 5-minute write costs 1.25x, a one-hour write 2x, reads 0.1x either way. Break-even
is 1.6 — if a prefix would otherwise be re-written more than 1.6 times within an hour, the 1h TTL
is cheaper. Agent loops with gaps over five minutes currently pay a fresh write every step.

**Fidelity: `IDENTICAL`.**

**Interaction risk: low.** One real constraint: never overwrite a caller's own `cache_control`,
already enforced in the Anthropic adapter as of 2026-07-30.

**Gate.** Arithmetic plus one live confirmation that the longer TTL is honoured.

### 5. Multimodal token cost

**Mechanism.** Two separate things. **Counting** images at all — currently zero, so every
multimodal report undercounts input. And **reducing** them: OpenAI's `detail: "low"` is ~85 tokens
against thousands, and pre-upload downscaling is near-lossless for the screenshots agent workloads
increasingly carry.

**Fidelity.** Counting is `IDENTICAL` (a measurement, not a transform). Detail downgrade is
`ALTERED` — it degrades vision accuracy, and needs its own evidence.

**Interaction risk: high, and specific.** Image blocks live in content *lists*, the exact code path
whose corruption was fixed on 2026-07-30 (block content rebuilt as `""`). Any change here must
carry the round-trip tests added with that fix.

**Gate.** Token-count correctness against the vendors' published tile formulas, then a vision
accuracy probe before any downgrade ships on.

### 6. Cascade routing

**Mechanism.** `route_models` guesses cheap-vs-expensive *before* the call from prompt length. Its
own docstring records the failure: *"What is 17 times 24, minus 89?"* — eight words, no tools, so
it routes, and the cheap model answers 329 against 319. A cascade instead calls cheap, verifies,
and escalates on failure: worst case slow-and-right, never wrong-and-cheap.

**Fidelity: `ALTERED`, but bounded** in a way static routing is not.

**Interaction risk: high, and a genuine conflict.** With `exact_cache` on, a *rejected* cheap
answer would be stored and could be served later as a hit. The cascade must either not cache
rejected attempts or key them by model. This is exactly the class of conflict the matrix would
have caught and this plan will not.

**Gate.** Live comparison with graded correctness, against static routing as the baseline.

### 7. The minor items

Each cheap, each small, in rough value order: truncation-retry avoidance
(`finish_reason == "length"` means the call was wasted and will be repeated at full double cost);
cross-turn tool-result deduplication (`cap_tool_results` truncates by size but never notices the
same tool with the same arguments returned the same output three steps ago); Anthropic's
token-efficient tool-use header; OpenAI Predicted Outputs for edit-heavy work.

**`stop` sequences are deliberately not on this list, and the audit that produced it was wrong to
suggest them.** ADR-016 already settled the question: a correct stop sequence is a fact about the
caller's output format, so it is a request *field* — which `LLMRequest.stop` already is, and which
`wire` already sends — not something a stage can infer. Guessing one would truncate a legitimate
answer that happened to contain the token. The audit found "no stage sets `stop`" and read a gap
where there was a decision.

## Testing strategy

Per technique: unit tests against real SDK shapes with mocked transport, the way
`test_adapters_anthropic.py` does; a wire test proving the new field actually reaches the provider,
because "reported but never sent" has now happened three times in this package; and for anything
`ALTERED`, an isolated live run per ADR-015 with correctness measured, not assumed.

Each technique also gets a **named** check against the specific stages listed under its
interaction risk. Not a matrix — a short list of pairs chosen because there is a reason to suspect
them.

## Success criteria

1. `thinking_budget` reaches both providers and a test proves it, replacing its `UNSENT_FIELDS`
   entry.
2. A streaming call gets an exact-cache hit and a prefix marker; both verified live.
3. Every technique that ships enabled has an isolated live measurement with a date, a model, and a
   method in `docs/optimize-benchmarks.md` — including the ones the measurement argues against.
4. The full gate stays green: ruff, `mypy --strict`, import contracts, and the suite.
5. Every new stage declares which existing stages it may fight, and carries a test for each.
