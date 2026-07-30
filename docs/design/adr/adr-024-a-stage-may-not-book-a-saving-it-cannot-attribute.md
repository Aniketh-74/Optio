# ADR-024 — A stage may not book a saving it cannot attribute

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (rule 1: never cause a cost increase), ADR-015 (evidence for `ALTERED`/`SHAPED`),
ADR-020 (a technique that claims no saving of its own), `savings.py` ("only count what was avoided")

## Context

The first end-to-end run of `scripts/real_agent_run.py` across all four scenarios, both arms, live on
`gpt-4o-mini`, found the package making requests **more expensive** while reporting a double-digit
saving. Two independent runs agreed to within 0.1 points.

| scenario | baseline input | optimized input | measured cost change | the report claimed |
|---|---|---|---|---|
| support | 7,717 | 3,816 | **+47.1%** | 50.8% saved |
| parallel | 635 | 661 | **−3.0%** | 10.0% saved |
| empty_result | 497 | 523 | **−4.3%** | 13.2% saved |
| long_loop | 24,688 | 8,954 | **+62.3%** | 63.7% saved |

The large wins are real. The two small scenarios are a cost *increase* reported as a saving, which is
the direction this project treats as the serious one — the same asymmetry that turned a published
53.7% into 50.1% and that ADR-021 landed a whole accounting change to prevent.

Isolated with one stage disabled and everything else identical:

| scenario | `structured_output` on | off |
|---|---|---|
| parallel | −3.0% | **+0.4%** |
| empty_result | −4.3% | **0.0%** |
| support | +47.1% | **+47.9%** |
| long_loop | +62.3% | **+62.1%** |

Input returns to *exactly* the unoptimized baseline when the stage is off — 661→635 and 523→497 — so
it accounts for the entire regression. On the two large scenarios it is marginally **worse** than not
running at all. It did not pay for itself in any of the four.

### The root cause is a guard that does not match its own docstring

`StructuredOutputStage`'s docstring says:

> Only acts when a schema is already present. Inventing one would change the contract between the
> caller and their model, which is not this library's call to make.

Its guard says otherwise:

```python
if request.response_format is None and not request.tools:
    return self.declines(request)
```

With `tools` present and **no schema at all**, the stage fires. Every one of the four agent scenarios
is a tool-using loop and none sets `response_format` — so on every call it appended
*"Respond only with the requested structure. No preamble or explanation."* to a request whose reply is
a **tool call**, not prose-wrapped JSON. There was no preamble to suppress, and the measured output
confirms it: 132→**137** (support), 94→**95** (parallel), 28→**28** (empty_result), 165→148
(long_loop). Output rose in two scenarios, was unchanged in a third, and fell only in the one where
trimming dominates everything anyway.

### The accounting compounds it

The stage books:

```python
net = max(0, 40 - instruction_cost)
return StageResult(..., saved_output_tokens=net, ...)
```

40 is a *hypothesis* about a preamble that may not exist, and nothing ever checks whether it did. That
is the one rule `savings.py` states in its own opening: **"Only count what was avoided, never what was
hoped for. No stage reports a saving it did not cause."** This is the stage that does.

The instruction's input cost is netted against an *output* figure, which also mixes units — output
bills at four to six times input on every model in `PRICING`, so subtracting one from the other is not
a conversion, it is a category error that happens to look conservative.

Worse, because `baseline = actual + saved` and `actual_input` comes from the provider, the tokens the
stage *adds* silently inflate the baseline. The reported baseline of 635 for `empty_result` is 110
tokens above the measured truth of 525.

## Decision

### 1. `structured_output` ships off by default

ADR-013 rule 1 is that this package never causes a cost increase, and a default-on stage that raised
cost on two of four real scenarios and helped none of them breaks it. It remains available by name for
a caller who wants terser JSON and has measured that it pays on their traffic.

This is the same bargain `concision` already took, and for the same reason: `concision`'s docstring
already says "Nothing in this suite measures the case this stage exists for." Neither did this one —
the difference is that `concision` was honest about it and off, while this was on.

### 2. The guard matches the documented intent: a schema, not tools

The stage acts only when `response_format` is present. This is a bug fix rather than a new policy —
the docstring already describes this behaviour, the code simply did not implement it, and that
divergence is what made the stage fire on every agent request in the package's whole test corpus.

A tool-using request without a schema is not a structured-output request. The instruction it was
receiving describes a JSON reply it was never going to produce.

### 3. No stage may report a saving it cannot attribute

`saved_output_tokens` from a hypothesised preamble is removed. The stage reports **zero** claimed
output savings. If suppressing the preamble really works, the effect appears where it can be trusted —
in a lower `actual_output_tokens`, measured by the provider — exactly as ADR-020 established for
fan-out warm-up, which "claims no saving of its own" because its effect lands in numbers that are
measured rather than estimated.

### 4. A stage that *adds* tokens reports that as a negative saving

The instruction costs real input tokens on every call. Reporting `saved_input_tokens = -cost` makes
`baseline = actual + saved` produce the **true** unoptimized baseline instead of one inflated by the
stage's own addition — for `empty_result`, 523 + (−26) = 497, which is precisely what the control arm
billed.

`StageSaving`'s fields are plain `int` with no non-negative invariant anywhere, so this needs no type
change. It does mean `reduction_ratio` can go negative, which is correct: a negative saving is
information, and rounding it up to zero is how the current report came to describe a loss as a 13.2%
gain.

`ConcisionStage` carries the same estimated-output-saving pattern and gets the same treatment. It is
already off by default, so this is an accounting fix there rather than a behaviour change.

## Consequences

- **Reported savings will drop on tool-using workloads, and the new numbers are the honest ones.** The
  four scenarios go from claiming 50.8 / 10.0 / 13.2 / 63.7 percent to reporting figures that track
  what the provider actually billed. Two of them will report roughly zero, because roughly zero is
  what the package saved there.
- **A report can now show a negative saving.** That is a feature: it is the only way a stage that
  costs more than it saves can be seen at all, and the whole finding behind this ADR is that such a
  stage was invisible for the package's entire history.
- **`structured_output` loses its only default-on evidence base and has none left.** Nothing in this
  repo has ever measured it doing what it claims, on a request that actually carries a schema. Turning
  it back on by default needs a live run on schema-carrying traffic showing a net win — the ADR-015
  bar, which it has never cleared.
- The stage keeps its `SHAPED` fidelity. Nothing about that was wrong; it does change the reply.

## Alternatives considered

**Keep it on and fix only the accounting.** Rejected: honest numbers would have shown a measured cost
increase on small requests continuing to happen by default, and ADR-013 rule 1 forbids the increase,
not merely misreporting it.

**Keep the `tools` branch and shorten the instruction.** Rejected. It treats the symptom — the
instruction was not too expensive, it was aimed at a reply shape that request could not produce. A
cheaper wrong instruction is still wrong.

**Let the stage measure its own saving by comparing against observed output lengths.** Rejected as the
same class of error one level up: a running average of past replies is not a counterfactual for *this*
reply, and ADR-018 already recorded what happens when this package infers a saving from an observed
distribution rather than measuring it.

**Fix the guard and leave the default on.** Rejected for now, and this is the closest call. With the
guard fixed the stage would no longer fire on tool-only requests, so the measured regression would go
away by itself. But that leaves a `SHAPED`, default-on stage whose benefit has still never been
measured on any workload in this repo — ADR-015's bar is evidence before promotion, not the absence of
a known harm.
