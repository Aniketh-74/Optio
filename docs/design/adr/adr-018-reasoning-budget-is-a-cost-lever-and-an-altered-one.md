# ADR-018 — Reasoning budget is a cost lever, and an `ALTERED` one

**Status:** Accepted
**Date:** 2026-07-30
**Related:** ADR-013 (the package exists), ADR-015 (evidence bar for `ALTERED`), ADR-016 (the
in-scope test), ADR-004 (fail-open is absolute)

## Context

`LLMRequest.thinking_budget` has existed since the type was written. It is typed, documented, and
included in `request_key` — `cache.py` explains why at length: *"reasoning tokens are how a model
reaches its answer"*, so two requests differing only in budget are not the same request.

It has never reached a provider. `wire.UNSENT_FIELDS` lists it with the reason:

> provider-specific shape (Anthropic nests it under `thinking`, OpenAI under `reasoning`); carried
> in `extra` until an adapter needs it

That was a reasonable deferral when no adapter needed it. It is no longer, for a reason the package
only recently had the vocabulary to state: **reasoning tokens bill at completion rates**, and
completion rates are 4–5x input rates on every model in `PRICING`. On a reasoning model the thinking
trace routinely exceeds the visible answer several times over. So the single most expensive tokens
in a request are the ones this package has no ability to influence — while it ships nineteen stages
aimed at the cheaper half of the bill.

Two facts make this the highest-value remaining item rather than merely a missing feature:

- Every prompt-side stage fights the provider cache (`trim_history` versus `prefix_cache` is the
  documented case). Reasoning budget touches neither messages nor tools, so it composes with
  everything.
- The lever already exists on the provider side and costs nothing to pull. Unlike batch dispatch,
  there is no latency trade; unlike routing, there is no second model.

## The decision this ADR has to make

Not *whether* to expose it — that follows from ADR-016. The question is **what fidelity tier a
reduced reasoning budget sits in**, because that determines whether it can ever ship on by default.

The tempting answer is `IDENTICAL`. A budget is a ceiling, not a rewrite: the prompt is unchanged
byte for byte, no context is dropped, nothing is invented. On any request that would not have used
the budget anyway, the response is genuinely identical.

That reasoning is wrong, and the way it is wrong matters more than the conclusion.

## Decision

**Reasoning-budget control is `Fidelity.ALTERED`, and it ships off by default.**

`ALTERED` is the tier ADR-015 gates behind isolated live evidence. This stage earns it despite
changing not one byte of the prompt, because fidelity is a claim about *the response*, not about
the request. Three consequences follow:

1. **It can change the answer, and it degrades exactly where it hurts most.** A lower budget is
   free on easy steps and wrong on hard ones — and "hard" is precisely why someone chose a
   reasoning model. Every other `SHAPED` stage in this package drops context a caller could notice
   is missing; this one silently reduces the model's capacity to think while the prompt looks
   untouched. `trim_history` at least leaves evidence in the message list.

2. **The failure is invisible in the report.** A truncated reasoning trace still produces a
   confident, well-formed answer. `route_models` has the same property and the same defence — its
   docstring records the cheap model answering 329 for 319 — but routing at least leaves a
   different model name in the response. A reduced budget leaves nothing at all.

3. **Cheaper and wrong is the worst outcome this package can produce.** It is also the one the
   savings report is structurally unable to detect, because the report measures tokens.

**`thinking_budget` leaves `UNSENT_FIELDS` and gets a real wire path.** That is a separate decision
from the stage and a safe one: sending a field the caller explicitly set is not an optimization, it
is the adapter doing its job. A caller who sets `thinking_budget` today has it silently discarded,
which is its own defect.

### Amendment, 2026-07-30: two fields, because the vendors do not take the same kind of thing

This ADR first said the single field would reach "Anthropic `thinking: {…budget_tokens: N}`, OpenAI
`reasoning_effort`", as though one value fit both. Checking the installed SDKs before writing any
code showed it does not:

- Anthropic takes a **token count**: `thinking={"type": "enabled", "budget_tokens": int}`.
- OpenAI takes a **category**: `reasoning_effort` is a `Literal["none", "minimal", "low", "medium",
  "high", "xhigh", "max"]`.

There is no honest conversion. Whether a 2,000-token budget is "low" or "medium" depends on the
model, and any threshold table would be invented — the precise species of unevidenced number that
has already cost this project a 36.3% claim that measured −1.8%, and a 53.7% figure that was 50.1%.
`UNSENT_FIELDS` was right that the shape is provider-specific; it was only wrong that this made the
field unsendable.

**So `LLMRequest` carries both, and each adapter sends the one its provider accepts and ignores the
other.** `thinking_budget: int | None` stays as Anthropic's token ceiling; `reasoning_effort:
str | None` is added for OpenAI's category. Neither is derived from the other. A caller targeting
both vendors sets both, which is more typing than a fabricated mapping and is the only version that
does not silently mean something different on each provider.

Both are in `request_key`, for the reason `cache.py` already gives about `thinking_budget`:
reasoning tokens are how a model reaches its answer, so two requests that differ in how hard the
model was told to think are not the same request.

The cost of the amendment is one more public field on an exported type. The alternative cost was a
threshold table nobody measured, silently changing what every OpenAI reasoning call does.

**The stage never raises a budget, only lowers it, and never sets one where the caller set none.**
Raising it would spend money the caller did not ask to spend — a cost-reduction library causing a
cost increase, the outcome ADR-013's rule 1 exists to forbid. Setting one where there was none
would impose a ceiling the provider's default did not have.

## Consequences

- The most valuable technique in the package ships **off**, and stays off unless a live run shows
  correctness holding. That is the same bargain `concision`, `summarize_history`, `route_models`,
  `compress_prompt` and `chain_of_draft` already took, and three of those measured badly enough
  that the measurement is the reason they are off. This one may too.
- Correctness has to be measured, not asserted — which means a task set with checkable answers at
  two difficulty levels, and the cost saving reported beside the accuracy, never alone. A cost
  number without an accuracy number is not evidence for this stage; it is evidence for the half of
  the trade that flatters it.
- `chain_of_draft` overlaps: both target reasoning verbosity, so enabling both risks the
  double-counted saving that rule 5 of `stages/__init__.py` exists to prevent between
  `minify_tools` and `prune_tools`. The two must not both credit the same avoided tokens.
- `adaptive_max_tokens` interacts for real: a low budget plus a tight completion ceiling can
  exhaust the ceiling on thinking and truncate before the answer begins. Ordering and a floor are
  required, not optional.
- Fail-open is unchanged and absolute (ADR-004): a provider that rejects the field, or a model that
  has no reasoning mode, must produce an ordinary unoptimized call rather than an error.

## Alternatives considered

**Ship it `SHAPED` and on by default.** Rejected. `SHAPED` in this package means "drops context
rather than inventing it", and the tier's defence is that the loss is visible in what was sent.
Nothing about a reduced budget is visible in what was sent.

**Expose the field only, with no stage.** Tempting — it is the safe half, and callers who know
their workload could set it themselves. Rejected as the *whole* answer because it is not plug and
play: it moves the decision to the caller for the one lever they are least equipped to tune by
hand. Adopted as the *first* half, shipped ahead of the stage, so the wire path is proven
independently of the policy that uses it.

**Infer the budget from prompt complexity, as `route_models` does from length.** Rejected for now.
That is the same weak proxy, applied to a higher-stakes decision, and `route_models`' one live
measurement already found the proxy failing on an eight-word arithmetic prompt. If a heuristic
ships at all it needs its own evidence, separately from the mechanism.
