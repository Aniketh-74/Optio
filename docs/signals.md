# Signals — the integration contract

> **This document is authoritative for signal names.** It is mirrored by constants in
> [`src/agentmeter/semconv.py`](../src/agentmeter/semconv.py) and asserted by the contract
> test suite (`tests/contract/`). Downstream OPA / Cedar / AGT policies are written against
> these exact strings, so treat this file with the care of a public API.
>
> Adding or renaming a signal requires: this document updated, a contract test updated, and —
> if the change is breaking — an ADR plus a version bump (IMPLEMENTATION.md §16 rules 5, 8, 12).

**Pinned OTel GenAI semconv version:** `1.37.0` (see R-TECH-2 — upstream attributes carry a
*Development* stability badge and can rename without a major bump, which is why the version is
pinned in code and checked in CI).

---

## Emitted signals

All names live under the `gen_ai.` namespace. Monetary values are USD doubles.

| Signal | Type | Lane | Cardinality | Meaning |
|---|---|---|---|---|
| `gen_ai.run.actual_cost` | double | cost | per run span | Cumulative **reconciled** cost — actual token spend, not estimate. |
| `gen_ai.run.projected_cost` | double | cost | per step + run span | Worst-case projected total cost for the run. Emitted **before** the step's tokens burn, which is what makes pre-spend gating possible. |
| `gen_ai.run.budget_remaining` | double | cost | per step + run span | `budget − reserved`. Absent when no budget policy was supplied. |
| `gen_ai.run.cost_per_successful_task` | double | cost × quality | per run span | `actual_cost / success`. Requires a success signal; absent when the quality lane is off and no heuristic success is available. |
| `gen_ai.run.loop_state` | string (enum) | behavior | per step + run span | One of `healthy`, `repeating`, `looping`, `retry_storm`. |
| `gen_ai.run.repeat_count` | int | behavior | per step + run span | Highest repeated-signature count within the current window. |
| `gen_ai.run.quality.groundedness` | double `[0,1]` | quality | per run span | Judge score. Opt-in; absent unless the quality lane is enabled *and* the run was sampled. |
| `gen_ai.run.quality.task_success` | double `[0,1]` | quality | per run span | Judge score. Same opt-in conditions. |
| `gen_ai.run.success` | bool | quality | per run span | Success flag, from the inline heuristic or the judge. Denominator for `cost_per_successful_task`. |

### `gen_ai.run.loop_state` values

| Value | Meaning |
|---|---|
| `healthy` | No pathology detected. **The fail-open default** — on ambiguity the detector must never fabricate a pathology (ADR-004). |
| `repeating` | Same tool called with equivalent arguments more than the repeat threshold. |
| `looping` | A repeating cycle with no state progress across the window. |
| `retry_storm` | Error-driven retries dominate the recent window. |

See [`behavior.md`](behavior.md) for the thresholds behind each state, the measured
false-positive rate, and guidance on which states are safe to gate on. In short:
gate on `looping` and `retry_storm`; alert on `repeating`, which normal agents produce.

### Absence is meaningful

A signal is **omitted** rather than emitted as zero or null when it cannot be computed —
unknown model price, quality lane off, run not sampled, or a lane failing and being caught by
the fail-open guard. Policies must therefore treat a missing attribute as *unknown*, not as
*zero*. Defaulting a missing `projected_cost` to `0` would silently allow the exact runs the
signal exists to catch.

---

## Attributes consumed (not emitted)

Read off framework-emitted spans as lane inputs:

`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.tool.name`,
`gen_ai.tool.call.id`, `gen_ai.response.finish_reasons`.

`gen_ai.response.finish_reasons` is array-valued upstream (one entry per choice), but several
instrumentations flatten it to a bare string; both forms are read. The quality lane uses it to
catch truncated generations, which read as complete text but are failed tasks.

---

## Run lifecycle

Where each signal appears in a run:

```
run start        RunContext created (run_id, optional budget, sampling decision)
  ├─ pre-step    cost lane RESERVES worst-case  → projected_cost, budget_remaining
  ├─ span        framework emits the LLM/tool span; span tap consumes it
  ├─            behavior lane updates window     → loop_state, repeat_count
  └─ post-step   cost lane RECONCILES to actual  → actual_cost (cumulative)
run end          quality lane (if sampled)       → quality.*, success,
                                                    cost_per_successful_task
export           attributes + metrics leave via OTLP
```

**Invariant (R-TECH-1):** reserve always precedes the step; reconcile replaces that reservation
exactly once. A reserve without a reconcile leaks budget; a double reconcile under-counts. Both
failures are silent in production, which is why they are enforced by property tests rather than
by review.

---

## Self-observability

agentmeter's own health is emitted under a **separate** namespace so it can never be confused
with — or gated on by — a consumer policy:

| Instrument | Meaning |
|---|---|
| `agentmeter.internal.signals_emitted` | Count of signals written. |
| `agentmeter.internal.lane_errors` | Fail-open activations, by lane. A rising value means a lane bug — the agent is still safe. |
| `agentmeter.internal.overhead` | Per-step overhead histogram (SC-5 budget: < 5 ms p99). |
| `agentmeter.internal.sampling_rate` | Effective quality-lane sampling rate. |

---

## Privacy

Signals are numeric, boolean, and enum only. **No prompt or completion content is ever emitted
or logged** (§10). The opt-in LLM-judge is the single component that reads trace content; it
runs on the user's own model credentials and honors the redaction/sampling config. Every new
signal must be reviewed for content-freedom before it is added to this table (R-SEC-1).

---

## Implementation status

| Lane | Milestone | Status |
|---|---|---|
| Cost | M2 | **Implemented.** `actual_cost`, `projected_cost`, `budget_remaining` emit today. |
| Behavior | M3 | **Implemented.** `loop_state`, `repeat_count` emit today. |
| Quality | M5 | **Implemented, off by default** (ADR-003). Emits only when enabled. |

Every signal in this document is now implemented. Names were contract-frozen in M0 so policy
packs and adapters could be written ahead of the code; **no name has changed since**.

Two caveats that matter more than the status column:

`cost_per_successful_task` requires the quality lane, because its denominator is a success count.
With the lane off it is absent — correct for an unknown value, and the reason a policy must never
read absence as zero.

`gen_ai.run.success` is absent on most scored runs too. The inline heuristic reports failure on
evidence and abstains otherwise; it never claims success, because a fluent wrong answer is
indistinguishable from a fluent right one without reading it. See [`quality.md`](quality.md).
