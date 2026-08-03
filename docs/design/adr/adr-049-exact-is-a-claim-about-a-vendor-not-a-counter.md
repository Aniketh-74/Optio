# ADR-049 — "Exact" is a claim about a vendor, not about a counter

**Status:** Accepted
**Date:** 2026-08-03
**Related:** ADR-015 (measure, do not assume), ADR-036 (calibration is per-vendor),
ADR-037 (the window binds the prompt), ADR-039 (evidence carries its date),
ADR-048 (the exact counter is an instrument, not a request path)

## Context

ADR-048 built the instrument; this ADR records what the first run of it found, and the one
place that number was load-bearing.

`scripts/measure_anthropic_tokenizer_gap.py`, run 2026-08-03 against `messages.count_tokens`
on `claude-haiku-4-5`:

| sample           | tiktoken | anthropic | real/est |
|------------------|---------:|----------:|---------:|
| prose            |      469 |       524 |    1.117 |
| chat turn        |      373 |       392 |    1.051 |
| json tool result |      456 |       475 |    1.042 |
| code             |      720 |       918 |    1.275 |

**tiktoken undercounts Anthropic on every shape**, worst on code at 1.275. The spread
(0.233) is too wide for one correcting constant, which the script was built to detect and
did.

The load-bearing consequence: `TiktokenCounter.is_exact` is `True`, and
`fits_in_window` trusts an exact count to the edge — no margin. Both were written as
though exactness were a property of the counter. It is a property of a **pairing**:
tiktoken is exact for the vendor whose vocabulary it ships and an estimator for every
other, and its estimate errs in the one direction a limit decision must not — a
code-heavy Claude prompt read as fitting could be 27% over the window, past even the
1.15 margin reserved for counts that admit to being estimates.

## Decision

`fits_in_window` now takes the model, and a limit decision applies a **per-vendor
undercount margin** on top of the exactness rule: `TEXT_UNDERCOUNT_BY_MODEL` holds
`{"claude": 1.28}` — the worst measured shape, rounded up and never down, because a limit
guards the worst case. An unmeasured vendor gets `1.0`: no measurement, no margin
(ADR-015).

The two margins **compound** for an inexact counter on a Claude model (1.15 × 1.28 =
1.472). Both error sources are worst on dense text and can co-occur, so multiplication
is the worst-case composition of two measured numbers, not a third guess.

The margin assumes the exact tier on a request path is tiktoken. That is ADR-048's own
rule doing the work: the genuinely exact `AnthropicCounter` stays off the request path,
so an exact counter counting a Claude model there is tiktoken wearing another vendor's
vocabulary. Handing a vendor-exact counter in anyway over-trims — the safe direction.

## Alternatives

**A per-shape correcting constant** is what the measurement itself recommends against a
single number, and it is still rejected here: the request path cannot tell the shapes
apart. The existing dense/prose classifier is two-way and the measured spread cuts
across it — json at 1.042 and code at 1.275 land in the same "dense" class. A
correction applied through a classifier that coarse would be precise about the wrong
thing.

**Correcting savings figures** was considered and declined. Savings are ratios of two
counts from the same counter, where a uniform bias largely cancels; the absolute
Anthropic figures are understated by 4–27% depending on shape, and the module docstring
now says so rather than a multiplier pretending otherwise.

**Making `is_exact` per-model** (`is_exact_for(model)`) is the principled interface and
a breaking change to a published `Protocol` for one call site. Deferred until a second
consumer needs it.

## Consequences

2,226 tests pass; 4 of 4 mutations caught (weakened constant, severed wiring, broken
prefix match, dropped stage pass-through).

Claude limit decisions got stricter: the near-window warning fires ~13% earlier on
exact counts and ~22% earlier on heuristic ones. That re-tuned one test that had been
sized to the 1.15 margin, and the cost is stated in `fits_in_window`'s own terms — a
warning that was not strictly needed, against a rejection the user sees as a crash.

One report is now known to understate: `SavingsReport(exact=...)` claims exactness from
the counter alone, so Anthropic savings totals are labeled exact while carrying the
uncorrected 4–27% shape bias. Left as recorded debt — the label needs per-request model
awareness the report does not have, and correcting the number without it would trade a
stated bias for an unstated one.

The measurement is one vendor, one model, one date. ADR-036 found tool-schema ratios
identical across three Claude models, so one model is likely representative; "likely"
is not measured, and re-running the script on another model costs one command and no
money. The table is keyed by vendor prefix so a second vendor's measurement is one row,
not a redesign.
