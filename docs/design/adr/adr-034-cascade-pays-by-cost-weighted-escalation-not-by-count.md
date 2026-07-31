# ADR-034 — Cascade pays by cost-weighted escalation, not by count

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (rule 1), ADR-015 (isolated live evidence), ADR-023 (cascade),
ADR-028 (report only what you can attribute), ADR-033 (found by the same run)

## Context

Cascade's first live run, `claude-haiku-4-5` → `claude-sonnet-4-5`, eight requests:

```
attempted 8   accepted cheap 4   escalated 4   skipped 0
escalation rate: 50.0%

cheap spend        $0.002378
escalation spend   $0.006483
total spend        $0.008861
all-expensive      $0.007074
net saving        -$0.001787      -25.3%
```

**The reported statistic says the run was comfortably profitable and the money says it lost 25%.**

The break-even is exact and comes straight off the rate card. An accepted-cheap request pays `C`
instead of `E`, saving `E − C`. An escalated one pays `C + E` instead of `E`, losing `C`. Setting
those equal:

```
(1 − r)(E − C) = r·C        =>        r = 1 − C/E
```

| pairing | break-even escalation rate |
|---|---|
| Haiku 4.5 → Sonnet 4.5 | **66.7%** |
| Haiku 4.5 → Opus 5 | 80.0% |
| Sonnet 5 → Opus 5 | 60.0% |

At 50% escalation against a 66.7% break-even, this run should have saved money. It did not, because
**`escalation_rate` counts requests and the bill weights them**:

```
escalated requests were $0.006483 of a $0.007074 all-expensive baseline
=> cost-weighted escalation rate 91.6%     (break-even 66.7%)
```

The four requests that escalated were 92% of the baseline spend. The four that passed were 8%.

**That correlation is structural, not an artefact of this traffic mix.** A request is more likely to
fail a verifier when it is longer, carries tools, or demands a schema — and every one of those
properties also makes it more expensive. Difficulty and cost move together, so the requests cascade
fails on are systematically the ones that cost the most. A count-weighted rate will therefore
*flatter* cascade on essentially any real workload, and it is the number the gate prints.

## Decision

### 1. `CascadeCost` reports a cost-weighted escalation rate

Escalated spend over the all-expensive baseline, both already measured. This is the number to read
against break-even.

### 2. The break-even rate is computed from the rate card, not documented in prose

`break_even_escalation_rate(expensive, cheap)` returns `1 − C/E` from `PRICING`, or `None` when
either model is unpriced — the same posture as every other pricing consumer here. A threshold a user
has to derive by hand is one they will not derive.

### 3. `escalation_rate` stays, and says what it is

It is still the right number for "how often is the verifier rejecting", which is what tuning a
verifier needs. It is renamed in documentation, not in code, to make clear it answers a different
question from "is this paying". Both are reported together so neither can be read alone.

### 4. Nothing is turned on or off by this

Cascade remains opt-in and `ALTERED`. This ADR adds the instrument, not a policy. A future decision
could have the router disable itself when the observed cost-weighted rate exceeds break-even, but
that needs evidence about how quickly the rate stabilises, and this run is eight requests.

## Consequences

- **A user can tell whether cascade is paying**, against a threshold the library computes for their
  model pair rather than one they infer.
- **The first live evidence for ADR-023's technique is negative on this mix, and stays recorded that
  way.** Eight requests, deliberately half-adversarial, is not a verdict on cascade — it is a verdict
  on the instrument, which reported 50% while losing money.
- The honest guidance that falls out: cascade pays when the *expensive* requests are the ones the
  cheap model handles well. If the hard requests are also the big ones — the usual case — the margin
  is much thinner than the count suggests, and the wider the price gap the more room there is
  (80% break-even for Haiku → Opus 5 against 60% for Sonnet 5 → Opus 5).
- One more number in the report. Justified: the alternative is a number that is wrong in the
  flattering direction, which this project treats as the serious direction.

## Alternatives considered

**Replace `escalation_rate` with the cost-weighted one.** Rejected: they answer different questions,
and someone tuning a verifier's strictness genuinely wants the count. Reporting both, adjacent, is
what stops either being read as the other.

**Weight by tokens rather than by cost.** Rejected. Input and output bill at different rates and the
ratio differs per model, so a token-weighted rate is a third number that approximates the one that
matters. The cost is already measured.

**Have the router auto-disable above break-even.** Deferred under decision 4 — the right shape is
probably the one ADR-030's amendment used for `prefix_cache` (observe the outcome, stop paying), but
that guard had a clean per-request signal and this one needs a rate estimated over a window.
