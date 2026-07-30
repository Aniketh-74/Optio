# ADR-026 — Trimming must price the output it buys

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (rule 1: never cause a cost increase), ADR-015 (isolated live evidence),
ADR-018 (a proxy that fails on the cases it exists for), ADR-024 (the report cannot see what a stage
costs)

## Context

`trim_history` is on by default and is the package's flagship history stage. The first full live
Anthropic benchmark showed it **increasing** total cost.

Isolated — three arms, one workload, one wall-clock window, only the stage differing:

| arm | input | output | cost |
|---|---|---|---|
| A optimizer off | 20,610 | 379 | $0.02251 |
| B default | 19,494 | **1,096** | **−11.0%** |
| C default minus `trim_history` | 20,610 | **376** | +0.1% |

C returns output to baseline exactly. The stage saved 1,116 input tokens and bought 717 output
tokens, and output bills at **5× input** on Haiku — so a 5.4% input saving became an 11% cost
increase. With it disabled the rest of the default stack is cost-neutral here.

The mechanism is that dropping old turns also drops the model's own prior short replies, and with them
the pattern it was matching, so it reverts to its default verbosity.

### The effect is provider-dependent, which is what makes this hard

The stage's own docstring already recorded the *opposite* result on OpenAI, and re-measuring
confirmed it still holds:

| workload | `gpt-4o-mini` output | cost | `claude-haiku-4-5` output | cost |
|---|---|---|---|---|
| multi_turn_chat | 147 → **91** (−38%) | −0.4% | 379 → **1,096** (+189%) | **−11.0%** |
| multi_turn_chat_long | 672 → **560** (−17%) | **+19.9%** | 997 → **5,693** (+471%) | +10.9% |

Trimming makes GPT-4o-mini *terser* and Haiku *far more verbose*, on identical workloads through
identical code. So there is no constant "trimming inflates output by N tokens" to encode — a figure
fitted to Haiku would forfeit a real 19.9% win on OpenAI, and a figure fitted to OpenAI reproduces the
loss this ADR exists to stop. This is ADR-018's lesson in a new place: a proxy calibrated on one model
fails on exactly the cases it was meant to cover.

Note also that the long workload wins on **both** providers and the short one wins on neither. The
size of the saving, not the vendor, is what separates the four measurements.

### Why nothing caught it

`trim_history` reports the input tokens it removed, and those numbers are correct. The output they
bought is attributed to nobody. This is the third instance of the blindness ADR-024 named — with the
distinction that ADR-024's stages *mis-claimed* a saving, and this one claims a real one while causing
an unattributed cost.

## Decision

### 1. Trimming is gated on the price-weighted saving, not the token saving

The stage trims only when the input tokens it would remove are worth more than the output tokens it
risks buying, priced with the **model's own rates** from `PRICING`:

```
saved_input * input_rate  >  RISK_OUTPUT_TOKENS * output_rate
```

Rearranged, the stage needs a per-request input saving above `RISK_OUTPUT_TOKENS * (output_rate /
input_rate)`. The ratio is 5 on Haiku and 4 on `gpt-4o-mini`, so the same rule produces a different
threshold per model without a per-model constant.

When the model is not in `PRICING` the ratio falls back to a documented default rather than assuming
parity: output is more expensive than input on every model in the table, and treating them as equal
is the flattering direction.

### 2. `RISK_OUTPUT_TOKENS` is the worst *measured* inflation, and it is a bootstrap only

**100 tokens per trimmed request**, and the units cost a wrong answer to get right.

The first version of this decision said 60, computed as `(1096 − 379) / 12` — the inflation divided by
every request in the workload. But only the **nine** requests that actually trimmed caused it; the
first three are inside the window and are never touched. The real figure is `(1096 − 379) / 9 ≈ 80`.

That error was not academic. A threshold built on 60 blocked the early trims and still let the tail of
the conversation through, and the live re-run came back at **−2.3%** — better than −11.0% and still a
cost increase, which is still the rule 1 violation this ADR exists to remove. Rounded up to 100 rather
than to 80, because this figure only decides behaviour before any observation exists and erring toward
not-trimming forfeits a saving while erring the other way spends someone's money.

**Verified live on both providers after the change:**

| workload | provider | before | after |
|---|---|---|---|
| multi_turn_chat | claude-haiku-4-5 | **−11.0%** | **0.0%** |
| multi_turn_chat | gpt-4o-mini | −0.4% | +4.2% |
| multi_turn_chat_long | claude-haiku-4-5 | +10.9% | **+13.8%** |
| multi_turn_chat_long | gpt-4o-mini | +19.9% | +17.6% |

Both losses are gone and both wins survive. The Haiku long case *improved* by three points, which is
the mechanism working rather than luck: the gate declines the early trims, whose small savings could
not cover the output they bought, and allows the later ones, whose savings can. Trimming less made it
worth more.

### 3. The constant is superseded by observation as soon as there is any

The stage records output length for the requests it trimmed and for the requests it declined, and once
it has `MIN_OBSERVATIONS` of each it uses the **observed** difference in place of
`RISK_OUTPUT_TOKENS`. A long-running agent therefore stops relying on the bootstrap figure quickly,
and a model whose behaviour differs from both providers measured here is handled without anyone
editing a constant.

Two groups rather than one, because a single running mean cannot separate "replies got longer because
we trimmed" from "replies got longer because the questions got harder". The declined requests are the
control the production path otherwise lacks.

This is the difference between this and the proxy ADR-018 rejected: nothing here predicts inflation
from prompt shape. It measures it, and falls back to a stated worst case only until it can.

### 4. The stage reports why it declined, and does not invent a negative saving

When the gate blocks a trim the note says so and names the threshold. It does **not** book a negative
output saving for the trim it avoided — that would be counting a hypothetical, which is precisely what
ADR-024 removed from two other stages. The cost this stage causes when it does trim remains
unattributable without a control arm, and saying so is more honest than modelling it.

## Consequences

- **Short conversations stop being trimmed on every provider.** That gives up a small real win on
  OpenAI (−0.4% on `multi_turn_chat`, i.e. roughly nothing) to remove an 11% loss on Anthropic.
  Given rule 1's asymmetry — forgoing a saving is not the same kind of error as causing a charge —
  this is the right side to err on.
- **The large wins are untouched**: 19.9% and 10.9% on the long workloads, which is where trimming was
  always earning its place.
- **The bootstrap constant is a modelled number, and this project's record with those is poor.** It is
  bounded in a way the previous ones were not: it only decides behaviour before any observation
  exists, it errs toward inaction, and it is checked against four live measurements above rather than
  reasoned from first principles. It still needs re-measuring on a third model, and that is recorded
  rather than assumed away.
- `recent_turns` remains the caller's knob. Someone who knows their traffic can still force the old
  behaviour by raising it, and the gate applies to whatever window they choose.
- **`summarize_history` and `compress_prompt` are untested against this effect.** Both replace context
  the same way trimming removes it, so both plausibly move output length too; neither has been
  measured for it. Recorded as an open item, not fixed here.
- **The prompt is no longer bounded by turn count, only by value.** `trim_history` used to guarantee
  that a growing conversation stopped growing; now a conversation of very small turns is never trimmed
  at all. That is the right economic answer — such a conversation costs almost nothing — but it is a
  behaviour change for anyone relying on the stage as a context-window guard.
- **Nothing was relying on it for that, which is itself a finding.** `context_limit` is a documented
  `OptimizeConfig` field that is validated at construction and then **read by no stage**, and
  `fits_in_window` has **no production callers** at all. So this package has no context-window
  enforcement today, and the gate cannot have broken any. Both are recorded as open items; the second
  also means ADR-022 overstated the impact of image under-counting on window safety, and that ADR is
  corrected.
- Several test fixtures had to grow. Conversations of `"question 0"` / `"answer 0"` are three tokens a
  turn, so a ten-turn fixture held less text than one real exchange and the stage now correctly
  declines to trim it. The mechanics under test are unchanged; the fixtures now contain enough for the
  question "is this worth trimming?" to have the answer those tests assume. The recall audit's filler
  needed the same treatment for a sharper reason: with filler that thin the trim arm declined to trim,
  and an audit measuring recall loss *from trimming* was measuring nothing.

## Alternatives considered

**Turn `trim_history` off by default.** Rejected, though it was the safe answer. It would forfeit
19.9% and 10.9% on exactly the long-conversation workloads the stage exists for, on both providers, to
avoid a loss that only appears on short ones — where a threshold removes it precisely.

**A per-provider constant.** Rejected. The two measurements differ by sign, not degree, so this is a
lookup table that will be wrong for the third model added to it, and the package has no way to know
when it goes stale.

**Only trim when the conversation exceeds N turns.** Rejected as the same threshold with worse units:
turn count does not price anything, so a workload with enormous turns and one with tiny turns get the
same answer despite completely different economics.

**Report an estimated negative output saving.** Rejected on ADR-024's rule, which is one commit old
and was written for exactly this temptation: without a control arm the figure is a hypothesis, and
booking it would re-introduce the defect that ADR removed.
