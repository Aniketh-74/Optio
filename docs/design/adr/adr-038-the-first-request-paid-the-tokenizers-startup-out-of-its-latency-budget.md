# ADR-038 — The first request paid the tokenizer's startup out of its latency budget

**Status:** Accepted
**Date:** 2026-08-01
**Related:** ADR-013 rule 1 (never cause a cost increase), ADR-030 (a stage whose cost nobody
accounted for), ADR-037 (the change that exposed this)

## Context

`Pipeline._run_stages` sets a deadline of `latency_budget_ms` — 100 ms by default — and skips every
remaining stage once it passes. That is the right shape for bounding *per-request* work.

`default_counter()` returns `MemoizingCounter(TiktokenCounter())`, and `tiktoken` loads its BPE
vocabulary **lazily, on first use**. Measured on this machine:

| call | elapsed |
|---|---|
| `default_counter()` | 15.23 ms |
| **first `count_text`** | **395.46 ms** |
| second `count_text` | 0.07 ms |
| first `count_text` for a different model | 0.06 ms |

So the first request through any freshly-built optimizer pays ~395 ms inside a 100 ms budget, at
whichever stage happens to count first. Every stage after that one is skipped.

Measured through the real pipeline on a long conversation, default config, first request:

```
  unstable_prefix            0.02 ms
  exact_cache                1.71 ms
  adaptive_max_tokens        0.00 ms
  trim_history             388.03 ms   <- pays the vocabulary load
  budget 100.0 ms; stages that ran: 4 of 9
```

**Five of nine stages never ran** — `deduplicate`, `prune_retrieval`, `cap_tool_results`,
`minify_tools` and `prefix_cache`. That last one is the largest lossless saving this package has, and
Anthropic's only cache mechanism.

The second request through the same process runs all nine, because the vocabulary is loaded. So the
defect is invisible in any test or benchmark that makes more than one call and looks only at
aggregates, and it is worst in exactly the deployments that make one call per process: a serverless
handler, a CLI invocation, a scheduled job.

It is also silent. Exceeding the budget logs at `DEBUG`, and the savings report simply shows the
stages that ran. A user sees a smaller number and no reason for it.

This was found while adding `WindowPressureStage` (ADR-037). That stage counts the request, so it
inherited the cost and moved it one position earlier — which pushed `trim_history` past the deadline
too and failed a test asserting a streamed request gets trimmed. The stage did not create the bug; it
relocated an existing one far enough to become visible. Without it, the same run silently lost five
stages and no test noticed.

## Decision

### 1. The tokenizer is warmed when the pipeline is built, not on the first request

`Pipeline.__init__` calls `count_text` once on a short constant string. The vocabulary load moves out
of the request path entirely and into construction, where it belongs: it is a one-time cost of
*having* an optimizer, not a cost of using one.

`default_counter()` is `lru_cache`d, so this is paid once per process no matter how many pipelines
are built.

### 2. Warming failures are swallowed

The warm-up is an optimization of *when* a cost is paid, never a new way to fail. A counter that
raises here leaves the pipeline exactly as it was — the same fail-open posture every stage gets, and
for the same reason (ADR-013 rule 1).

### 3. Construction is allowed to be slow, and says so

`Optimizer()` now takes roughly 400 ms the first time in a process. That is a real cost and it is
stated in the class docstring rather than hidden, because a caller building an optimizer inside a
request handler needs to know to hoist it out. The alternative — the current behaviour — is not
cheaper, only quieter: the same 400 ms is paid either way, and paying it inside the budget also
discards most of the pipeline.

### 4. The invariant gets a name and a test

**The first request must not be optimized less than the second.** Stated as a test against a
deliberately slow counter, so it holds for any future counter with a lazy initializer rather than for
`tiktoken` specifically.

## Consequences

- **The first request through a process now gets the whole pipeline**, including `prefix_cache`.
  Every one-call-per-process deployment was losing most of this package's value and could not have
  known.
- **`Optimizer()` construction costs ~400 ms once per process.** Callers who build one per request
  will now see that cost where it is attributable, instead of losing optimizations they never knew
  were skipped.
- The latency budget goes back to meaning what it says: a bound on per-request work.
- A whole class of future defect is closed by the invariant in decision 4 — any counter, any lazy
  initializer, same test.
- **This says nothing about the budget being right at 100 ms.** It says the budget was measuring the
  wrong thing on request one. Whether 100 ms is the correct bound for steady-state work is a separate
  question this ADR does not answer.

## Alternatives considered

**Exempt the first request from the deadline.** Rejected: it makes the budget conditional on
something the caller cannot see, and a genuinely slow first request would then blow the latency
expectation instead. Warming at construction removes the cost rather than hiding it.

**Start the deadline after the first stage that counts.** Rejected as the same trick with more
machinery, and it would need every stage to declare whether it counts.

**Raise `latency_budget_ms` above 400 ms.** Rejected, and it is the tempting wrong answer: it would
let one-time initialization set the steady-state bound for every request forever, and 400 ms is far
outside what this package promises per call.

**Warm in a background thread.** Rejected on ADR-016's second test — this package does not own a
thread pool — and it would only narrow the race, not close it.

**Drop `tiktoken` for the heuristic counter.** Rejected: the heuristic is inexact, and
`fits_in_window` and every budget decision are more correct with an exact count. The problem was
never the tokenizer's accuracy, only when it loaded.
