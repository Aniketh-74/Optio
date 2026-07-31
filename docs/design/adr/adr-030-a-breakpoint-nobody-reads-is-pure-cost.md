# ADR-030 — A breakpoint nobody reads is pure cost

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (rule 1: never cause a cost increase), ADR-020/021 (cache economics),
ADR-027 (per-model floor), ADR-028 (attributable deltas)

## Context

The full live suite on `claude-sonnet-4-5`, 2026-07-31, produced ten workloads between 75% and 97%
cost reduction — and one number in the other direction:

```
### timestamped_agent
    cost            $0.06496 -> $0.08021   -23.5%
    provider cache  reads 0   writes 20,333   (baseline reads 0)
    output quality  100.0% identical
```

**20,333 prompt tokens written into the provider's cache at the 1.25× premium, and read back zero
times.** That is not noise and it is not unattributable: the arms differ, the effect is one-directional,
and it reproduces. It is a straight ADR-013 rule 1 violation — the one outcome this package treats as
unacceptable — caused by the stage this package calls its largest lossless win.

The mechanism is exactly the one `reads 0 / writes N` was added to make visible (ADR-027 decision 4).
`timestamped_agent` puts a regenerated timestamp *above* the stable instructions, so every turn's
prefix differs from the last. `prefix_cache` places a breakpoint each turn; the provider dutifully
writes a fresh entry each turn, at 1.25× base rate; nothing ever matches it, so nothing is ever read
back at 0.1×. The library pays the premium twelve times and collects the discount zero times.

**This library already detects the condition and does not act on it.** `detect_unstable_prefix` is on
by default and fired on this very run, before the first workload:

> `optio_optimize: the system prompt differed on 100% of the last 10 requests, so no provider prefix
> cache can hit. Something varying -- a timestamp, request id, or per-user detail -- is above the
> stable instructions.`

"No provider prefix cache can hit" is precisely the fact that makes writing one a pure loss. The
detector knew. The writer never asked.

### The same stage is also declining breakpoints that would pay

Measured on the new `large_system_agent` workload, whose prefix is a realistic operating manual plus
38 MCP tool schemas:

```
prefix is ~1715 tokens, below claude-haiku-4-5's 4096-token cacheable minimum
```

The stable prefix is **5,186 tokens**. The stage counted 1,715, because `_stable_prefix_length` sums
`request.messages` and ignores `request.tools` — 3,471 tokens of schema that Anthropic caches *ahead
of* the system prompt. Anthropic's cached prefix runs tools → system → messages, so a breakpoint in
the system block caches every tool schema before it.

The live `mcp_agent` run is independent evidence: 28,393 reads over ten requests, ~2,839 per request,
against a stable *message* prefix of ~1,387 tokens. The provider cached roughly twice what the stage
believed the prefix to be.

So one stage is wrong in both directions at once — writing where it can never read, and declining
where it would have paid — and the second error lands hardest on tool-carrying agents, which are the
traffic this library most exists for.

## Decision

### 1. Do not place a breakpoint into a prefix known to be unstable

`prefix_cache` consults the same observation `detect_unstable_prefix` publishes. When the prefix has
been observed changing on effectively every recent request, the stage declines and says so. A cache
write is only ever an investment against a future read; where the detector has established there will
be no read, the investment is a loss with no upside.

Deliberately asymmetric in what it requires:

- **To decline, the evidence must be strong** — a prefix that differs on essentially every request
  across a full window. A prefix that changes occasionally still pays: one read recovers several
  writes at these rates.
- **Silence is not evidence.** Before the window fills, the stage behaves exactly as it does today.
  The default is to cache; declining is the exception that must be earned.

### 2. Tool schemas count toward the prefix

`prefix_cache` measures the prefix as tools plus the stable message head, because that is what the
provider caches. Counting only messages understates every tool-carrying request and declines
breakpoints on the workloads with the most to gain.

This is the mirror of ADR-027's correction and carries the same risk profile inverted: ADR-027 stopped
markers being placed below the floor, and this stops them being withheld above it.

### 3. The decline reason keeps naming the number it used

ADR-027 made "no cache reads" diagnosable. Both changes here alter the number in that message, so the
message keeps reporting the figure actually compared against the floor — now including tools.

## Amendment, 2026-07-31 (same day): ask the provider, not the digests

Decision 1 shipped and the live re-run was still negative:

| `timestamped_agent` | before | digest guard | + this amendment |
|---|---|---|---|
| cost reduction | **−23.5%** | −17.1% | **−6.0%** |
| cache writes | 20,333 | 14,795 | **4,637** |

Correct behaviour arriving too late. `unstable_prefix` requires `MIN_OBSERVATIONS = 10` before it
will say anything, and that threshold is right for *reporting a finding to a person* — "three
requests with three distinct system prompts is equally consistent with a bug and with an agent that
has only just started". On a twelve-request workload it means ten requests pay the premium first.

**The stronger signal was there all along and is not a proxy: the provider reports what it actually
served.** `cache_read_input_tokens` comes back on every response, so a marked request that produced
a write and no read is the outcome itself. It converges in three requests rather than ten, and it
covers a case the digests cannot see — a byte-stable prefix that always expires before reuse writes
at 1.25× forever while every digest looks perfect.

**`MAX_UNREWARDED_WRITES = 3`, from the economics.** A wasted write costs 0.25× base per prefix token
(1.25× paid where 1.0× would do); a *missed* cache costs 0.9× (1.0× paid where 0.1× would do).
Declining wrongly is ~3.6× worse per request than writing wrongly, so the stage waits for repeated
unambiguous evidence and any single read resets the run. Three also survives the parallel case:
concurrent requests can each write before any sees another's entry, which is `fan_out`'s shape
exactly, and a threshold of two would risk disabling the stage there.

**What remains is a bounded startup cost, not a rate.** The residual is three writes — $0.00348 of
premium at Sonnet 4.5 rates on this prefix — and it does not grow:

```
  12 requests   overhead 5.35%      100 requests   overhead 0.64%
  25 requests   overhead 2.57%      200 requests   overhead 0.32%
  50 requests   overhead 1.28%      500 requests   overhead 0.13%
```

The 12-request workload is close to the worst case a benchmark can construct. Driving it to zero
would require predicting instability before observing it, which ADR-021 already rejected for the TTL
question and rejects here for the same reason: guessing wrong raises the bill.

`multi_turn_chat` re-measured at **74.5%** on the same run, unchanged, confirming the guard costs
nothing where caching works.

## Consequences

- **`timestamped_agent` stops costing 23.5% extra.** It cannot be made to *save* anything: its prefix
  genuinely cannot be cached, and the honest outcome is that the library does nothing there. The
  workload's stated purpose — pricing the most common caching bug in production — is unchanged, and
  the detector still reports the finding.
- **Any production prompt with a timestamp, request id, session id or per-user detail above the
  stable instructions** is protected by the same change. That shape is common, it is the exact thing
  the detector was written to catch, and until now this library made it worse rather than better.
- **Tool-carrying agents gain caching they were being denied.** `large_system_agent` moves from
  declined to marked on the 4,096-token tier.
- The two changes interact safely: an unstable prefix is declined before the size test runs, so a
  larger measured prefix cannot resurrect a write that can never be read.
- One more coupling between two stages that were independent. Accepted deliberately: they are two
  halves of one question — "is this prefix worth caching" — and answering it in two places with two
  answers is what produced the −23.5%.

## Alternatives considered

**Leave it; the workload is adversarial by construction.** Rejected. `timestamped_agent` is a
*replica* of a production mistake, not a synthetic edge case, and ADR-013 rule 1 does not have an
exemption for prompts the user shaped badly. A library that makes a common mistake 23.5% more
expensive is not plug-and-play.

**Have the user fix their prompt instead.** The detector already tells them, and that stays. But
ADR-001's "emit signals, never enforce" governs the *core*; `optio_optimize` is opt-in and its
contract is rule 1. Emitting a warning while continuing to bill the user extra is not a defensible
reading of either.

**Write with a one-hour TTL so a later read amortises it.** Rejected outright: a one-hour write costs
2× base instead of 1.25×, so on a prefix that never matches this makes the loss 60% larger.

**Infer instability inside `prefix_cache` with its own window.** Rejected as a second copy of an
existing measurement, free to disagree with the first. The detector's window is the observation; this
stage consumes it.
