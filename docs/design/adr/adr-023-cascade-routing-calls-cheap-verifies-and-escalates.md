# ADR-023 — Cascade routing calls cheap, verifies, and escalates

**Status:** Accepted — implemented; first live gate run 2026-07-30 (see addendum); remains off by default
**Date:** 2026-07-30

## Context

`route_models` (ADR-015, `stages/routing.py`) is the package's one static router. It reads
`request.model`, and when the request is short (≤`MAX_ROUTABLE_TOKENS`), carries no tools, and has
no `response_format`, it swaps in `cheap_model` and sends. The swap is a *guess made before the
call*, from prompt length — and the stage's own docstring records the guess failing: *"What is 17
times 24, minus 89?"*, eight words, no tools, well inside the ceiling, so it routes. `gpt-4o`
answers 319; `gpt-4o-mini` answers 329. Reproduced identically three times. The stage is
`ALTERED` precisely because this can happen, and it ships off by default because nothing in the
request path can tell this prompt apart from a genuine lookup of the same length.

Length is a proxy for difficulty, and ADR-018 already rejected a proxy of exactly this shape for
reasoning-budget control: "a reduced budget is free on easy steps and wrong on hard ones, and
'hard' is why someone chose a reasoning model in the first place." Static routing has the same
defect and the same silent failure — a fluent, well-formed, wrong answer that no cost-and-latency
dashboard and no token-identity check can catch.

Cascade routing removes the guess. Instead of deciding cheap-vs-expensive *before* the call, it
**calls cheap, verifies the answer, and escalates to the expensive model only on failure.** The
worst case stops being wrong-and-cheap and becomes slow-and-right: a routed-and-rejected request
pays for the cheap call *plus* the expensive one, but it never returns a silently degraded answer.
That is a fidelity improvement over static routing bought with a latency-and-cost tail on the
requests that escalate.

The reason this needs an ADR rather than a patch to `routing.py` is structural. Every optimization
in this package is a `Stage`, and a stage's contract (`stages/base.py`) is that it does exactly one
of three things — transform the request, short-circuit with a response, or decline — and then the
pipeline makes **one** provider call (`pipeline.py`: `before → call → after`). A stage never holds
the provider call; it hands a request back and the pipeline sends it. Cascade routing must call the
provider, look at what came back, and possibly call again. That is the same wall ADR-017 hit with
batch dispatch — "a stage's contract is that a response comes back on the same stack frame, and
this one comes back tomorrow" — and it forces the same conclusion: **cascade routing cannot be a
stage.** It has to own the provider call.

## Decision

Cascade routing is a **provider-call wrapper**, not a stage. The pipeline's `execute`/`aexecute`
already take the provider call as a parameter (`ProviderCall = Callable[[LLMRequest],
LLMResponse]`). Cascade wraps that callable: given the caller's `call`, it produces a new callable
that (1) sends the request to `cheap_model`, (2) runs a verifier over the response, (3) on pass
returns the cheap response, (4) on fail re-sends the *original* request to its original model and
returns that. The stage pipeline runs around this wrapper unchanged — every existing `before`/
`after` hook still fires exactly once against whichever request actually gets sent last.

This keeps the wall ADR-017 drew intact: the stage contract stays single-call, and the one
technique that needs two calls lives outside it, wired at the same layer the caller already passes
`call` in. It reuses the `route_models` eligibility rule verbatim (short, no tools, no
`response_format`, not already cheap) to decide *whether to attempt the cascade at all* — the
gate that says "this request is even a candidate" is unchanged; only the decision that used to
follow it ("so send it cheap and hope") is replaced by "so try cheap, and verify."

Three decisions each a real fork; the resolution chosen is stated under each, and the code in
`cascade.py` / `optimizer.py` implements them.

### DECISION 1 — the verifier → *caller-supplied, with a structural default*

The whole technique rests on "verify the answer," and the spec (`2026-07-30-remaining-cost-
techniques-design.md` §6) does not pin the mechanism. The constraint that rules out the obvious
answer: we cannot grade the cheap answer by asking the expensive model, because that spends the
expensive call on *every* routed request and deletes the saving we are trying to book. The verifier
has to be cheaper than the escalation it triggers. Candidates, in rough order of how much I'd trust
them:

- **(a) Cheap self-verification** — a second, small, cheap-model call that grades the first
  ("is this answer complete and correct for the question? yes/no"). Costs a second cheap call on
  every routed request, catches semantic failures, but a weak model grading a weak model is a known
  soft spot.
- **(b) Structural / heuristic validator** — cheap deterministic checks: did it refuse, is it
  empty, did it hit `finish_reason == "length"`, does it satisfy the caller's `response_format`
  schema when one exists. Nearly free, catches obvious failures, catches *none* of the
  "confidently wrong arithmetic" case that is the whole motivating example.
- **(c) Caller-supplied verifier** — the caller passes a `verify(request, response) -> bool`, the
  same shape ADR's summarizer decision took for `summarize_history` (refuse to run the technique
  without one rather than ship a stub that measures the stub). Most honest, most correct, most
  friction; punts the hard problem to whoever knows their own traffic.

**Chosen: (c) as the contract with (b) as a built-in default verifier.** `Optimizer` accepts a
`cascade_verifier` callable; when none is given, `default_verifier` escalates on empty and
truncated answers and accepts everything else. This makes the technique usable out of the box on
the failures a heuristic *can* catch, and correct when a caller who cares supplies a real judge.
The default explicitly does not catch the "17×24−89" case — its docstring says so — because a
structural check cannot, and pretending otherwise is the trap this decision exists to avoid.

### DECISION 2 — cache interaction → *never cache rejected attempts*

The spec flags this directly and it is the same class of bug ADR-022 just closed. With `exact_cache`
on, if the cheap call runs *through* the normal pipeline, a **rejected** cheap answer gets written
to the cache under the request's key — and served later as a hit, silently, to a request the
cascade would have escalated. The rejected answer becomes permanent. Two ways out:

- **(a) Cascade owns caching for its attempts** — the cheap attempt bypasses the cache write
  entirely; only the *accepted* final answer (cheap-passed or escalated) is cached. Simplest to
  reason about; means the cheap attempt can't be a cache *hit* either, which is a small missed
  saving.
- **(b) Key cheap attempts by model** — the cache key includes the model, so a cheap answer and an
  expensive answer for the same prompt are distinct entries. A rejected cheap answer can still be
  cached, but can only ever be served to another cheap attempt, never to an expensive-model
  request. More capable, more surface, and it touches the cache key that ADR-022 just spent a whole
  ADR getting right.

**Chosen: (a) — do not cache rejected attempts at all.** It is the option that cannot resurface as
a correctness bug, and ADR-022 is a fresh reminder of what the other kind costs. The mechanism is
structural rather than a special case: the cheap attempt is made *inside the wrapper*, against the
raw provider call, so it never passes through the stage pipeline and no cache stage's `after` hook
ever sees a rejected answer. Only the accepted final answer reaches the pipeline and the cache. A
test (`test_a_rejected_cheap_answer_is_never_cached`) pins this: a rejected cheap answer, then a
second identical request that returns the *escalated* answer from cache, never the rejected one.

### DECISION 3 — the gate requires live spend → *deferred; ships off until run*

ADR-015's `route_models` acceptance criteria apply here and then some: an isolated live run, a
workload with *both* genuinely-easy and genuinely-hard short requests (the failure only shows in
the second category), per-request graded correctness, and — new for cascade — a measured
**escalation rate** (exposed as `CascadeStats.escalation_rate`), because a cascade that escalates
everything saves nothing and one that escalates nothing is just static routing with extra latency.
The gate is a live A/B against static routing as the baseline, with graded correctness on both
arms. That needs real API keys and real spend, which has **not** been run. The mechanism is
implemented and its logic is covered by the offline test suite, but per ADR-015 it ships **off by
default** and stays there until the live gate is run and its numbers recorded — the same bar every
`ALTERED` promotion clears. Turning it on before then is a caller's informed choice, not this
package's default.

## Alternatives

- **Patch `route_models` in place to escalate.** Rejected: it breaks the stage contract for every
  stage. Once one stage is allowed to hold the provider call and call twice, "a stage does one of
  three things and the pipeline calls the provider once" is no longer true for anyone, and the
  fail-open guarantee (ADR-013 rule 1, `pipeline.py`'s "last known-good request") gets materially
  harder to reason about. The wall ADR-017 built exists for this reason.
- **Make cascade a second public surface like `BatchDispatch` (ADR-017).** Plausible, and if the
  wrapper grows its own configuration and reporting it may end up here anyway. Rejected *for now*
  as heavier than the mechanism needs: batch dispatch is a second surface because its response
  comes back on another day; cascade's comes back on the same stack frame, just after up to two
  calls, so a call wrapper composes with the existing pipeline without a new entry point. Revisit
  if the wrapper accretes surface.
- **Ship static `route_models` as-is and call it done.** Rejected: ADR-015 already recorded that
  static routing has zero live evidence and a known silent-failure mode, and the spec ranks cascade
  as the fix. Shipping the guess as the final answer is the outcome this sequence exists to avoid.
- **No verifier — escalate on a confidence signal from the cheap model.** Rejected as OPEN
  DECISION 1's problem in disguise: provider confidence signals are uneven across models and absent
  from most, so it is a verifier that works for some model pairs and silently no-ops for others,
  which is worse than an honest heuristic because its coverage is invisible.

## Consequences

- **A latency-and-cost tail on escalated requests.** A routed request that fails verification pays
  cheap + verifier + expensive, and returns later than a direct expensive call would have. The
  saving is real only if the escalation rate stays well under 100% and the cheap+verifier cost is
  small against the expensive call. The gate (OPEN DECISION 3) has to *measure* the escalation rate,
  not assume it; a cascade is a loss on any workload that escalates most of the time.
- **The verifier is now part of the fidelity claim.** Whatever OPEN DECISION 1 settles on, the
  technique is only as good as its verifier — a verifier that passes the "17×24−89" failure lets
  exactly the motivating bug through. This is strictly better than static routing (which has no
  check at all), but "cascade routing" is not a fidelity guarantee on its own; the verifier is the
  guarantee, and it should be named wherever the saving is reported, the way ADR-017 requires batch
  savings to name that they come from a published figure.
- **One more thing that is off by default and gated.** Like every `ALTERED` technique since
  ADR-015, this ships disabled and stays disabled until live evidence clears the bar. The package's
  default behavior does not change.
- **The stage contract stays single-call.** The deliberate cost of the wrapper approach is that
  cascade logic lives in a different place from the other nineteen optimizations, so "read
  `stages/` to see everything the package does" stops being complete — there is now one technique
  outside that directory. ADR-017 already spent this; cascade is the second withdrawal against it,
  and a third would be a reason to design a real "multi-call technique" home rather than keep
  adding wrappers one at a time.

## Addendum — live gate result (2026-07-30)

The gate DECISION 3 describes was run once, live, through the repo's reviewed `route_models` audit
harness (`bench/routing.py`) under a $0.50 `SpendGuard` cap. Pair: **gpt-4o** vs **gpt-4o-mini**
(16.7x cheaper input), the twelve-probe workload (four easy lookups, eight short-but-hard), graded
against known answers. Total spend: **$0.0013** across 24 calls.

Accuracy: easy, both models 100%. Hard, gpt-4o 100%, gpt-4o-mini 88% (7 of 8). The single
regression was the motivating case, reproduced live: *"What is 17 times 24, minus 89?"* — gpt-4o
answered 319, gpt-4o-mini answered 329. All four decline guards held against real request shapes.

Interpreted for cascade, treating a request as escalated exactly when the cheap answer is wrong (an
*oracle* verifier, to isolate the mechanism and the capability gap from verifier quality):

| approach | accuracy | escalation rate | approx cost vs all-expensive |
|---|---|---|---|
| static routing (always cheap) | 92% (11/12) | — | 6% |
| cascade (oracle verifier) | 100% (12/12) | 8% (1/12) | 14% |

So on this workload cascade recovers the one point static routing loses, at ~14% of the
all-expensive cost — the accuracy of the expensive model at a fraction of the price, *given a
verifier that catches the bad answer*. That proviso is the whole result: the built-in
`default_verifier` rejects only empty and truncated answers, and "329" is neither, so with the
default verifier cascade would accept it and score the same 92% as static routing. The gap between
the two rows is precisely what a real caller-supplied verifier buys, and confirms DECISION 1's
framing rather than softening it.

**What this does and does not justify.** It is a floor, not a characterization: one model pair,
twelve single-turn answer-checkable probes, an oracle stand-in for the verifier. It reproduces the
failure and demonstrates the mechanism closes it; it does not measure a real verifier on real
traffic, which is the caller's evidence to gather on their own workload. Cascade therefore **stays
off by default** — this run clears the mechanism, not the judgment call ADR-015 reserves for
whoever supplies `cheap_model`.

## Addendum — eligibility expansion (sub-project, in sequence)

The guardrails cascade inherited from `route_models` (no tools, no `response_format`, ≤500 tokens)
exist because static routing had no recovery from a bad guess. Cascade's verify-and-escalate *is*
that recovery, so each guardrail can be reconsidered — with its own verifier, off by default, and
its own evidence. Being done in order: structured output, then the length ceiling, then tools.

### Step 1 — structured-output requests (`cascade_structured_output`) — implemented

`is_routable` gains `allow_response_format`; static `route_models` keeps the default `False`,
cascade passes `True` when `cascade_structured_output` is on. The requested JSON *is* the verifier:
`default_verifier` now escalates when the cheap answer does not parse as JSON, or parses but drops
a top-level key a `json_schema` marked `required`. Deliberately shallow — full JSON Schema
validation would need a dependency this package does not carry, and the parse-plus-required-keys
check catches the common "returned JSON but dropped a field" failure without pretending to validate
the whole shape. Covered by `TestStructuredOutput` and `TestDefaultVerifierConformance`; ships off
by default and requires `cascade_routing`.

### Step 2 — length ceiling (`cascade_max_tokens`) — implemented

`is_routable` gains a `max_tokens` parameter (default `MAX_ROUTABLE_TOKENS`); cascade passes
`cascade_max_tokens` when set, so longer prompts than static routing's 500-token limit can attempt
the cheap model. No new verifier: the escalation net already recovers a long prompt the cheap model
fumbles, so the ceiling becomes a *cost* knob, not a safety one — raising it only stops paying once
enough long prompts escalate that the wasted cheap attempts outweigh the wins, which is a
per-workload measurement, not a fixed answer. Covered by `TestLengthCeiling`; off by default
(`None`), requires `cascade_routing`.

### Step 3 — tool-using requests (`cascade_tools`) — implemented

`is_routable` gains `allow_tools`; cascade passes it when `cascade_tools` is on. This is safe only
because a chat-completions-style provider call returns a *proposed* tool call, not an executed one
— the agent runs the tool afterward — so the cheap model's proposal can be vetted and escalated
before anything with a side effect happens. `default_verifier` gains a tool branch: for a request
with `tools`, it reads the proposed call from `response.extra["tool_calls"]` (the same convention
the request side uses, per `wire`), and escalates when the call names a tool absent from the
request or carries `arguments` that are not valid JSON. A tool request answered in plain text falls
through to the text checks; an empty answer with no proposed call escalates. Covered by
`TestToolRequests` and the tool cases in `TestDefaultVerifierConformance`.

**The provider surfacing that makes this live is now done.** `OpenAIProvider`, `AnthropicProvider`,
and the OpenAI-Agents adapter (`_response_from_completion`) now populate `response.extra["tool_calls"]`
from the model's proposed call — additively, so the native object each already returned is
unchanged. Anthropic's `tool_use` block (whose `input` is a dict) is normalised to the OpenAI-shaped
convention (`arguments` a JSON string) so the verifier reads one shape; the OpenAI union's
custom-tool variant, which has no `.function`, is skipped rather than assumed. Verified with the
real SDK types under a mocked HTTP transport (no network, no spend) in `test_providers.py`, plus an
end-to-end `Optimizer` test (`test_cascade_tools_vets_a_live_proposed_call_end_to_end`) proving a
proposed known tool call now reaches the verifier and is accepted without escalation. Streaming
adapters remain out of scope: a streamed tool call is assembled from deltas and wants its own
handling.

### Live end-to-end confirmation (2026-07-30)

The actual cascade wrapper — all three flags on — was run once against the live OpenAI API
(gpt-4o requested, gpt-4o-mini as cheap), three representative requests, $0.50 cap, total spend
**$0.00004**:

- routing: cheap answered "Tokyo." → accepted;
- structured output: cheap returned valid JSON with both requested keys → conformance check passed → accepted;
- tools: cheap proposed `web_search({"query": "Tokyo weather today"})` → the call was surfaced into `extra["tool_calls"]`, its name matched the request's tool and its arguments parsed → accepted.

`cascade_stats`: attempted 3, cheap_passed 3, escalated 0, skipped 0 — every request was eligible
under the widened rules and the cheap model was accepted, so gpt-4o was never called. This is the
first time the cascade code itself (not a direct model comparison) ran live, and it confirms steps
1 and 3 engage end-to-end rather than only in unit tests. It does **not** exercise the escalation
path live (the cheap model happened to succeed on all three) or use real production traffic; those
remain covered offline and by the caller's own workload respectively.

## Addendum — post-review improvements (2026-07-30)

A review after the live runs found five weaknesses; all five are now addressed.

1. **Cascade accounts for its own cost.** Cascade wraps the provider call, so the stage
   `SavingsReport` — correctly — only ever saw the single final response, which meant an escalated
   request's *wasted* cheap attempt was spent at the provider and counted nowhere. `CascadeStats`
   now records the tokens of every cheap attempt (split into accepted vs wasted) and every
   escalation call, and exposes `cost_summary(expensive, cheap) -> CascadeCost`: measured cheap
   spend, measured escalation spend, measured escalation *waste*, and a net saving against an
   all-expensive baseline. Only the accepted-cheap side of that baseline is a projection (the
   expensive model was never called for those) and is labelled as such — everything else is a
   receipt. This keeps the stage report's "baseline = actual + saved, same source" invariant intact
   while making cascade's true economics, including the loss case, visible.

2. **Model-verifier cost is measured.** A verifier that calls a model spends money cascade used to
   ignore, so "saved X" was gross, not net. A verifier may now expose
   `pop_cost() -> (in_tokens, out_tokens, usd)`; cascade folds it into `CascadeStats.verifier_usd`.
   A ready-made `ModelJudge` verifier implements this, priced from `PRICING`, and its docstring
   names the trap it exposes: a judge as expensive as the escalation it guards erases the saving.

3. **Cascade latency is visible.** `CascadeStats` now times each phase (`cheap_ms`, `verifier_ms`,
   `escalation_ms`, `total_latency_ms`), so the extra round trips cascade adds — doubled or tripled
   on escalation — are measured rather than silent.

4. **Tool-call vetting checks required arguments.** The verifier previously accepted any valid-JSON
   arguments for a known tool; a cheap model could emit `{}` for a tool needing a `query` and pass.
   It now reads the tool schema's `required` list (both OpenAI `parameters` and Anthropic
   `input_schema` shapes) and escalates when a required argument is missing.

5. **Streaming and SDK robustness.** The streaming bypass is now stated explicitly in the module
   docstring (cascade needs the whole cheap answer to verify before escalating, which a stream
   cannot give without buffering away its own benefit). Anthropic tool-call surfacing detects
   `tool_use` blocks by their `type` discriminator via `getattr` rather than importing a class name,
   so it survives SDK versions that place or spell it differently.

All five are covered by tests (`TestCostAccounting`, `TestVerifierCost`, `TestLatencyTracking`, the
required-argument cases, and the provider surfacing tests) and ship without changing any default.
