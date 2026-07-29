# ADR-017 — Batch dispatch is a second surface, not a stage

**Status:** Accepted
**Date:** 2026-07-29
**Related:** ADR-012 (public API), ADR-013 (the package exists), ADR-016 (what belongs here)

## Context

Every major provider sells asynchronous batch processing at roughly **50% off**, with a turnaround
measured in hours. It is the largest unconditional discount in the entire field, it requires no
quality trade whatsoever — the same model returns the same answer — and this package does not
offer it.

ADR-016 put it in scope and flagged it as the one item there that changes the package's shape
rather than extending it. This ADR settles that shape.

The reason it cannot be a `Stage` is structural rather than stylistic. Every stage answers *what
should this request look like*, and the pipeline's contract is that a request goes in, a response
comes back, synchronously, in the same call. Batch answers a different question — *when should
this be sent, and by whom* — and its answer is "in a few hours, by someone who is no longer on
this stack frame". No amount of `StageResult` fits that: there is no response to return and no
error to fail open into, because nothing has gone wrong.

The second difficulty is that "50% off" is only true if the work tolerates the latency, and
nothing in a request says whether it does. That is a fact about the caller's product, not about
the prompt.

## Decision

**Batch dispatch is a separate public class, `BatchOptimizer`, alongside `Optimizer`.** It is
asynchronous by nature and does not pretend otherwise.

1. **The caller declares latency tolerance; the library never infers it.** There is no heuristic
   for "this looks like it can wait". Submitting a user-facing request to a 24-hour queue to save
   half a cent is a product failure this package must be structurally incapable of causing, so the
   decision is a method call, not a config flag on a shared path.

2. **The optimization pipeline runs first, unchanged.** A batched request is still worth trimming,
   deduplicating and minifying, and the discounts compose — the ~50% batch discount applies to
   whatever tokens survive the stages. `BatchOptimizer` therefore *owns* an `Optimizer` rather
   than reimplementing it, and every stage keeps exactly one implementation.

3. **The exact cache is checked before submission, and populated on retrieval.** A request already
   answered should not enter a queue at all, and an answer that comes back hours later is as
   cacheable as one that comes back immediately. This is the one place the two surfaces share
   mutable state, and it is shared deliberately.

4. **Failure is explicit, not fail-open.** ADR-013 rule 1 says a stage failure means "pass the
   original request through", which works because a synchronous call has somewhere to fall back
   to. A batch submission that fails has not degraded to a slower path; it has not happened. So
   `BatchOptimizer` raises, and says which requests were and were not accepted. Silently
   converting a failed batch into 10,000 synchronous calls would be a fail-open that costs
   twice the money it was asked to save.

5. **Polling is the caller's, with a helper.** The library exposes `submit()` returning a handle
   and `results()` to fetch, plus an `await_results()` convenience with an explicit timeout. It
   does not own a background thread: a library that spawns one inside somebody's web worker is a
   dependency that behaves differently in every deployment.

### What this does not include

No queue, no persistence, no retry scheduler. A handle is a provider-issued batch id and the
caller is expected to store it — anything more is infrastructure the caller must operate, which
ADR-016's second test excludes.

## Alternatives

**A `batch=True` flag on `Optimizer.call`.** Rejected: it makes a synchronous method sometimes not
return a response, which is the kind of API that reads fine and cannot be typed honestly.

**Infer batchability from the request.** Rejected as the failure mode described above. There is no
signal in a prompt that distinguishes "a nightly enrichment job" from "a user waiting".

**A stage that short-circuits with a placeholder response.** Rejected: it satisfies the type and
lies. A caller would receive something that looks like an answer and is not.

## Consequences

The package gains a second public entry point, so ADR-012's "the public API is the top level" rule
now covers two classes rather than one. That is a real widening and is the cost of the discount.

Live measurement of batch is bounded by the turnaround, which can be hours. The benchmark harness
is built around a synchronous `compare()` and cannot express that, so batch savings are *arithmetic*
(the provider's published discount applied to measured token counts) rather than measured
end-to-end the way every other claim in `docs/optimize-benchmarks.md` is. That difference must be
stated wherever the number appears — it is a weaker class of evidence than this project otherwise
ships, and the reason is the clock rather than a shortcut.

Two provider adapters gain batch submission, and their file formats differ (JSONL of request
envelopes on OpenAI, a typed array on Anthropic), so the translation that `providers.py` already
does for tool schemas now happens for whole requests too.
