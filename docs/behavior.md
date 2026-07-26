# Behavior signals

The behavior lane answers one question: **is this agent making progress, or is it stuck?**

It emits two signals, both defined in [`signals.md`](signals.md):

| Signal | Type | Meaning |
|---|---|---|
| `gen_ai.run.loop_state` | enum string | `healthy`, `repeating`, `looping`, or `retry_storm` |
| `gen_ai.run.repeat_count` | int | Highest repeated-call count in the current window |

Both are written to the run span after every step, and again at run end.

---

## The bias: never fabricate a pathology

Section 6.4 of `IMPLEMENTATION.md` sets the rule this lane is built around:
**ambiguity defaults to `healthy`.**

The asymmetry is deliberate, and it is not caution for its own sake. A downstream
policy may kill a run on `looping`. So:

- A **false positive** breaks a working agent. Our detection error becomes the user's outage.
- A **false negative** means a stuck run costs money — which the cost lane is already reporting.

These are not comparable, so every threshold below is set against the first one.
This is the same fail-open reasoning as [ADR-004](design/adr/), applied to
classification rather than to exceptions.

---

## States

### `healthy`
No pathology detected, **or not enough evidence to say**. The default in every
ambiguous case, including any run shorter than 5 steps.

### `repeating`
The same call recurs 4+ times in the window. This is a **weak signal** —
real agents legitimately repeat calls (polling a job, paging results, retrying a
flaky endpoint). It is worth surfacing, not worth killing a run over. Treat it as
something to alert on, not to gate on.

### `looping`
A small set of calls dominates the window *and* the run shows no progress:
≥60% of recent steps come from ≤2 distinct calls. This is the "agent is stuck"
state, and the one most likely to be enforced on, so it carries the strictest bar.

Dominance is measured over the **whole recurring set**, not the single most
frequent call. A perfect two-call cycle — `read, think, read, think…` — is the
textbook stuck agent, yet each call holds only half the window. Scoring the top
call alone would put that below any useful threshold, making cycles of length ≥2
structurally undetectable.

### `retry_storm`
Errors dominate: ≥4 errored steps and ≥50% of the window. Distinguished from
`looping` because the **remedy differs** — a retry storm usually means a broken
dependency, not a confused agent. An agent correctly retrying a failing service
is not stuck; it is blocked.

### Precedence

A window can satisfy several conditions at once:

```
retry_storm  >  looping  >  repeating  >  healthy
```

The error-driven diagnosis wins because it names a *cause* rather than a symptom.

---

## Thresholds

| Constant | Value | Why |
|---|---|---|
| `MIN_STEPS_FOR_VERDICT` | 5 | Below this, a loop is indistinguishable from an agent that started with two similar calls. An early false positive can kill a run before it has done any work. |
| `REPEAT_THRESHOLD` | 4 | Three identical calls is normal retry behaviour; four starts to look intentional. |
| `LOOP_DOMINANCE` | 0.6 | The agent is doing the same thing more often than everything else combined. |
| `LOOP_MAX_DISTINCT` | 2 | An agent alternating between two calls forever is as stuck as one repeating a single call. |
| `RETRY_STORM_ERROR_RATE` | 0.5 | More than half the recent work is failing. |
| `RETRY_STORM_MIN_ERRORS` | 4 | Stops a 3-step window with 2 errors from being called a storm. |

**These are heuristics.** They were chosen to be conservative against the labeled
fixtures in `tests/unit/test_detectors.py`, not derived from production traces.
Every one is pinned by a boundary test, so changing a value is a deliberate act
with a visible diff.

---

## Measured false-positive rate

Section 6.4 requires the FP rate to be a published metric. It is measured by
`tests/unit/test_false_positive_rate.py`, which runs in CI and **fails if the
rate regresses** — the number below is generated, not asserted by hand.

```
varied work        0/200    paged retrieval  0/200
polling            0/200    bounded retries  0/200
fan-out            0/200    mixed            0/200

false-positive rate:  0/1200  =  0.000%  healthy runs flagged
detection rate:     600/600   =  100.0%  pathological runs flagged
```

### What this number is and is not

It is a measurement over **1200 synthetic healthy runs** built from the patterns
most likely to be misread as pathology — polling, paged retrieval, bounded
retries, fan-out, and mixtures of all of them. Those shapes are where a false
positive would actually come from, which is what makes the measurement useful.

It is **not a population estimate**. We do not have a corpus of production agent
traces, and this document will not pretend otherwise. A real deployment will
contain shapes this corpus does not model. If you see a false positive in
practice, that is a bug worth reporting — it means the corpus is missing a
pattern.

The detection rate is published alongside it for a specific reason: an FP rate of
zero is trivially achievable by never detecting anything, so the two numbers are
only meaningful together.

---

## Privacy

Tool arguments routinely contain user prompts, retrieved documents, and PII.
Section 10 makes content privacy the primary security control, so:

- Arguments are reduced to a **truncated hash at the boundary**. The raw value is
  never stored on a signature.
- The digest is 8 bytes of `blake2b`, chosen for speed. It is a fingerprint, not a
  security primitive — a collision makes two calls look alike, nothing more. It
  does not defend against an agent deliberately hiding a loop.
- `BehaviorWindow.__repr__` reports only counts, so a log line or crash dump
  cannot leak fingerprinted call shapes.

What actually feeds the signature is the *shape* of the call: its non-content
`gen_ai.*` attributes plus the span name. Volatile attributes are excluded —
token counts, tool call ids, and agentmeter's own emitted signals — because any
of them would make every signature unique and defeat detection entirely.

This is coarser than true argument equality, so it **under-detects rather than
over-detects**. That is the safe direction.

---

## Memory

Each run holds one `deque` bounded by `behavior_window_size` (default 50). The
bound is structural rather than a policy someone must remember to enforce.

Windows are **evicted at run end**. Agent processes are long-lived, and per-run
state that is never released is an unbounded leak — a bug this project already
paid for once in the cost ledger, where every unit test missed it because a
single-run test cannot observe state accumulating across runs.

Unlike the ledger, this lane needs no closed-run memory. Re-adding steps to an
evicted run starts a fresh window, which at worst under-reports; it cannot invent
a pathology or corrupt an already-published total ([ADR-010](design/adr/)).

`repeat_count` is bounded by the window, not by run length. A count of 5000 from
a 50-step window would be a fabricated number.

---

## Overhead

`classify` builds a `Counter` over the window on every step, so its cost is
O(window) — bounded by config, never by run length. Measured
(`tests/bench/test_overhead.py`, cost + behavior lanes together):

```
overhead per step:  mean 73µs   p95 102µs   p99 126µs    (SC-5 budget: 5ms p99)
behavior step cost: 78µs early -> 62µs after 10,000 steps
```

Roughly 40× inside the budget, and flat in run length. The flatness is pinned by
its own benchmark: an accidental change to O(total steps) would make long runs
quadratic and would show up to a user as an agent that mysteriously degrades over
hours, rather than as any test failure.

---

## Using these signals in a policy

`loop_state` is an enum, so policies match on exact strings:

```rego
# OPA/Rego — stop a stuck agent, but only on the strong signal.
deny contains msg if {
    input.attributes["gen_ai.run.loop_state"] == "looping"
    msg := "agent appears stuck in a loop"
}
```

Two things to get right:

**Gate on `looping` and `retry_storm`; alert on `repeating`.** `repeating` is the
weak signal by design and normal agents produce it.

**A missing attribute means *unknown*, not *healthy*.** The signal is omitted
when the lane is disabled, the run had no steps, or the fail-open guard caught an
internal error. Treating absence as healthy is usually the safe default here —
but make it a decision, not an accident.
