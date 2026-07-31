# ADR-028 — A cost delta is only a measurement when the arms differ

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (never cause a cost increase), ADR-015 (isolated evidence), ADR-024 (a stage may
not book a saving it cannot attribute), ADR-027 (per-model prefix floor)

## Context

The live Anthropic run of 2026-07-31 reported a cost delta for all twelve workloads. Five of those
twelve send **byte-identical requests in both arms and make the same number of provider calls**:

| workload | reported cost delta | what the optimizer actually did |
|---|---|---|
| `timestamped_agent` | **−1.6%** | nothing |
| `sampled_creative` | **−4.7%** | nothing |
| `unique_questions` | **+2.8%** | nothing |
| `multi_turn_chat` | 0.0% | nothing |
| `tool_calling_chat` | 0.0% | nothing |

Verified offline, per workload, by capturing every request each arm hands the provider and comparing
them field by field: `19,200 → 19,200` tokens on `timestamped_agent`, `32,390 → 32,390` on
`tool_calling_chat`, every stage booking exactly zero.

Those five percentages are the provider's own output nondeterminism. They are not measurements of
this library, and three of them are large enough to read as findings:

- **−1.6% and −4.7% look like ADR-013 rule 1 violations** — a cost increase caused by the optimizer,
  which this package treats as the one unacceptable outcome. The previous iteration of the
  measurement loop opened by planning to isolate which stage caused the −1.6%. No stage caused it.
  There was nothing to isolate, and an isolation run would have spent live money to discover that.
- **+2.8% is the flattering direction**, and it landed on `unique_questions` — the workload whose
  stated purpose is *"Included so the suite reports its own limits."* It reported a saving where the
  library did nothing at all.

The suite already knew this class of error existed and had applied the lesson to exactly one number.
`--control` exists to measure provider nondeterminism, and its own help text records gpt-4o-mini
diverging on 4–5 of 12 identical prompts at temperature 0, concluding that "a workload reporting
'1 of 12 diverged' has measured nothing." `QualityResult.is_interpretable` enforces that for the
quality line. The identical nondeterminism moves output length, and therefore cost, and nothing
carried the reasoning across to the cost line.

ADR-024 settled this rule one level down: a stage may not book a saving it cannot attribute. This is
that rule for the harness that grades the stages.

## Decision

### 1. Each arm records a digest of what it sent

`ArmResult.sent_digest` is a running `blake2b` over every request the arm handed the provider. A
digest, never the prompt — §10's content rule binds the benchmark as it binds everything else, and
ADR-022 already settled the shape for images.

### 2. The fingerprint includes `max_tokens`, which `request_key` deliberately omits

`cache.request_key` leaves `max_tokens` out on purpose: a cached completion can be reused across
differing limits by checking `finish_reason`. That reasoning is about *reuse* and does not transfer
to *change detection*. `adaptive_max_tokens` is enabled by default and changes that field and no
other, so a fingerprint borrowed from `request_key` would classify a run where only that stage fired
as a no-op and discard a genuine saving as noise. The two functions answer different questions and do
not share a definition.

### 3. A delta is attributable only when the arms differ

`ABResult.cost_is_attributable` is False when both arms' digests match **and** their provider-call
counts match. Cache hits change the call count; every other stage changes the digest. Nothing else
this package does can alter spend.

### 4. Unattributable deltas are named, not printed as percentages

The two dollar figures stay — money really was spent, and both numbers are true observations of the
run. What is removed is the percentage, which invites reading a noise figure as a result:

```
cost                $0.00312 -> $0.00317   NOT ATTRIBUTABLE (identical requests, same call count)
```

## Consequences

- **Three apparent findings disappear, and that is the point.** `timestamped_agent`'s −1.6% and
  `sampled_creative`'s −4.7% stop reading as ADR-013 violations, because they never were. The loop
  stops spending live money chasing them.
- **`unique_questions` stops claiming 2.8%.** The workload that exists to report the suite's limits
  now reports them.
- **Every remaining percentage in the report is one the library caused.** That is a smaller set of
  numbers and a much stronger claim about each of them.
- **Five of twelve workloads currently measure nothing about this library.** The flag does not fix
  that; it makes it visible. `multi_turn_chat` and `tool_calling_chat` fell to no-ops when ADR-026's
  priced gate correctly stopped `trim_history` from firing on short conversations, and
  `timestamped_agent` is a no-op for the same reason. Whether the shipped defaults *should* leave
  those workloads untouched is a separate question this ADR does not answer — but it can now be asked
  against an honest report.
- A workload that is a no-op today may stop being one when a default changes, and the flag follows
  automatically because it observes rather than declares.

## Alternatives considered

**Return `None` from `cost_reduction` when unattributable.** Rejected. The dollars were really spent
and the ratio is a true observation of the run; what is false is reading it as a saving. Suppressing
the number would also hide a real signal — a large delta on a no-op run means the provider is much
noisier than assumed, which is worth seeing.

**Infer the no-op from `input_tokens` and the stage ledger both being unchanged.** Rejected as
indirect. It would miss `adaptive_max_tokens`, which changes cost without moving an input token, and
it reasons from what the stages *claimed* rather than from what was *sent* — the exact substitution
ADR-024 removed.

**Run every workload twice with the optimizer off and subtract the noise floor.** Rejected for now:
it doubles the live cost of every run to produce a per-workload noise estimate from a single pair,
which is too few samples to subtract with. `--control` already offers this deliberately, as an
opt-in, and that remains the right place for it.
