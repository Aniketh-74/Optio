# ADR-020 — Fan-out warm-up is an async dispatch order, not a stage

**Status:** Accepted
**Date:** 2026-07-30
**Related:** ADR-013 (the package exists, fail-open is rule 1), ADR-016 (the in-scope test),
ADR-017 (batch is a second surface because it answers *when and by whom*), ADR-019 (`prepare`/
`complete` reused a third time)

## Context

N parallel calls sharing a prompt prefix each pay to populate the provider's cache, because none of
them can see another's write. Issue one first and the remaining N−1 read what it wrote.

The arithmetic is the strongest in the remaining queue. On Anthropic, a 5-minute cache write costs
1.25x the base input rate and a read costs 0.1x. For a fan-out of five over a shared prefix:

| dispatch | cost of the shared prefix, in units of base input rate |
|---|---|
| all five cold, in parallel | 5 × 1.25 = **6.25** |
| one first, then four | 1.25 + 4 × 0.1 = **1.65** |

**74% off the shared prefix**, with no request changed and no answer altered. It also helps on
OpenAI, which is unusual for anything in this package: OpenAI caches long prefixes automatically at
a 50% input discount, but *something still has to go first*, so N cold parallel calls all pay full
rate there too.

The cost is latency. One call's round trip is prepended to the batch's wall time — for a fan-out of
short calls, close to a doubling.

## Decision

### 1. Not a stage. A method on `Optimizer`

A stage answers *what should this request look like*, and this changes nothing about any request. It
answers *when and by whom*, which is ADR-017's test for a second surface. But unlike batch it is not
a second *surface*: responses come back on the same stack frame, in the same call, from the same
pipeline and into the same report. A class would advertise a submit/poll lifecycle that does not
exist.

So: `Optimizer.afan_out(requests, provider) -> list[LLMResponse]`, results in the caller's own order
because they correlate by index.

### 2. Async only, and the reason is not effort

A synchronous caller issuing five calls in a loop **already gets this for free** — the first call
populates the cache and the next four read it, because sequential execution is warm-up ordering. The
defect only exists where calls are genuinely concurrent.

A sync caller with a `ThreadPoolExecutor` has the concurrency and therefore the problem, and serving
them would mean this package owning a thread pool. That is infrastructure the caller must stand up
and reason about, which ADR-016's second test excludes. `asyncio` needs no such thing.

So there is no `fan_out`, only `afan_out`, and the docstring says why rather than leaving a reader to
assume the sync version was forgotten.

### 3. Opt-in, never inferred

Nothing reorders a dispatch the caller did not hand to this method. Serializing the head of a
fan-out the caller expected to run wholly in parallel is a latency regression, and "this fan-out
looks like it can wait" is exactly the inference ADR-017 refused to make about batch — there, to
avoid putting a waiting user behind a 24-hour queue; here, to avoid doubling a page's time to first
byte to save a fraction of a cent.

### 4. Warm up only when a shared prefix exists and clears the floor

The check is real, measured, and provider-agnostic: count the leading messages identical across
every request, count their tokens, and compare against `MIN_PREFIX_TOKENS`. Below it, dispatch
everything concurrently and warm nothing.

This matters more than it looks. Below the provider's floor the cache is not populated at all, so a
warm-up call there buys **zero saving for a full round trip of added latency** — a pure loss, and
silent. It is the same failure that had one measurement script reporting zero cache reads in both
arms because its prompt sat under Haiku 4.5's 4,096-token floor.

`MIN_PREFIX_TOKENS` is the wrong constant in a known way — the real floor spans 512 to 4,096 across
models and this is one number. It is reused deliberately rather than a second guess being invented
next to it: one wrong constant that gets fixed once is better than two that drift.

### 5. The warm-up call is one of the caller's own requests, not a probe

A synthetic prefix-only request would bill for tokens nobody asked for and produce an answer nobody
wanted. `requests[0]` goes first, for real, and its response is returned like any other.

### 6. It claims no saving of its own

There is no stage, so there is no `StageResult` and no number to report. The saving appears where it
actually happens: `provider_cached_tokens` rises and `provider_written_tokens` falls, both already
tracked per response. Nothing here estimates a discount the provider might grant — this package has
twice published an invented number that a live run then corrected.

### Amendment, 2026-07-30: measured, and the arithmetic held

`scripts/measure_fan_out_warm_up.py`, three arms cold/warmed/cold on `claude-haiku-4-5`, five
branches over a 4,223-token shared system prefix, each arm given its own nonce so it starts cold:

| arm | input | cache reads | cache writes | cost |
|---|---|---|---|---|
| cold | 21,194 | 0 | 21,115 | $0.02847 |
| warmed | 21,194 | **16,892** | **4,223** | **$0.00905** |
| cold-2 | 21,194 | 0 | 21,115 | $0.02847 |

**−68.2%** total cost, and **0.0% spread between the two cold arms** — they came out byte-identical,
which is the tightest noise floor any measurement in this project has had. Five writes became one
write and four reads, exactly as the table above this amendment predicted.

Isolating the shared prefix, which is what the 74% claim was about: cold pays
`21,115 × 1.25 = 26,394` rate-units, warmed pays `4,223 × 1.25 + 16,892 × 0.1 = 6,968` — **73.6%
off**, against a predicted 74%. The gap between that and the −68.2% headline is output tokens and the
five distinct user turns, neither of which is cacheable and neither of which this technique claims.

**This is the first modelled number in this package that survived contact with a provider.** The
prior four did not: a simulated 36.3% prefix-cache saving measured −1.8%, a 53.7% figure was 50.1%, a
2,048-token cacheable floor was 4,096, and `reasoning_budget`'s entire safety argument dissolved when
its own live run showed the ceiling was never reached. Worth stating plainly because the reason it
held is structural rather than lucky: nothing here estimates provider behaviour. The saving is a
consequence of dispatch order and published rate cards, and the only empirical question was whether
the provider populates its cache when told to — which it does.

## Consequences

- Fail-open is unchanged and absolute (ADR-004). A failure anywhere in the ordering logic dispatches
  every request concurrently, which is what the caller would have done themselves.
- The added latency is the caller's to accept and must be documented at the call site, not buried in
  an ADR. A fan-out of two is the worst case: one round trip of delay to halve one prefix.
- Anthropic caches nothing without a `cache_control` breakpoint, so on a Claude model with
  `prefix_cache` off this method pays the latency and receives nothing. That is a configuration
  error rather than a code path, and it earns a warning rather than provider-family inference — the
  package holds a model string, and guessing a provider from it is the kind of proxy `route_models`
  already got caught doing badly.
- Short-circuited requests (an exact-cache hit) are answered without dispatch and excluded from the
  ordering entirely. A hit that counted toward "is there a shared prefix worth warming" would let a
  cached answer justify a real call's latency.
- `prepare`/`complete` now carry three callers — batch, streaming, and this. That is the third time
  ADR-017's split has paid for itself, and it is the argument for the seam being public.

## Alternatives considered

**A `warm_up_fan_out=True` config flag on the pipeline.** Rejected. Config flags in this package
turn stages on and off; this is not a stage, and a flag would have to reach across the pipeline's
one-request-at-a-time contract to find requests it never sees together.

**Warm up with a synthetic prefix-only call.** Rejected per decision 5: it bills for a request
nobody made, and on Anthropic it needs the breakpoint anyway, so it saves nothing a real first call
does not.

**Infer the provider from the model string and skip warm-up when it cannot pay.** Rejected for now.
The check that matters — is there a shared prefix above the floor — is measurable without knowing the
provider, and a model-prefix table is a maintenance liability that would be wrong the week a vendor
renames something.

**Make it automatic inside `acall` by detecting concurrent calls with a shared prefix.** Rejected as
the most tempting and worst option. It would mean holding requests back to see whether a sibling
arrives, which is a latency cost imposed on every caller to benefit some of them, decided by a
heuristic none of them can see.
