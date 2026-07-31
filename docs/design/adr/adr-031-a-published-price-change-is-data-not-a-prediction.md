# ADR-031 — A published price change is data, not a prediction

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-004 (unknown models produce no signal), ADR-018 amendment (this package has twice
published an invented number a live run corrected), ADR-021 (cache write bands),
ADR-029 (a price may not be inferred across model generations)

## Context

ADR-029 left seven currently-served models deliberately unpriced, because nobody here had read their
rates off the vendor's page. The published table has now been supplied, and it settles all seven —
plus two models the API's `models.list` does not return (`Claude Mythos 5`, limited availability) and
the retired rows.

Three things in it matter beyond the numbers themselves.

### The cache multipliers are universal, and were previously assumed

Every one of the sixteen published rows has the identical structure:

```
5-minute write = 1.25x base      1-hour write = 2.00x base      cache hit = 0.10x base
```

`optio_optimize`'s table derived its cache columns from exactly those multipliers. That derivation was
correct and it was an assumption; it is now sourced. Worth recording, because ADR-021 landed a whole
accounting change after a *missing* write rate understated a measurement by 3.6 points, and the same
class of error would follow from a wrong multiplier.

### The four rows already present are confirmed

`claude-opus-4-5` (5/25), `claude-opus-4-1` (15/75), `claude-sonnet-4-5` (3/15) and
`claude-haiku-4-5` (1/5) all match the published table exactly, including their cache columns. The
3× Opus overstatement ADR-029 removed is confirmed as a real error rather than a suspected one:
Opus 4.5 is $5/$25 and was being priced from Opus 4's $15/$75 row.

### One model has two prices and a date between them

```
Claude Sonnet 5 (through Aug 31, 2026)    2 / 10
Claude Sonnet 5 (from Sep 1, 2026)        3 / 15
```

A `dict[str, ModelPrice]` cannot hold this. Whichever single number is written is **wrong on one side
of 2026-09-01, and wrong by 50%** — which is larger than most of the savings this library reports.

The obvious objection is that encoding a future price is prediction, and this package has a standing
rule against publishing numbers it has not verified. It does not apply here. A *prediction* is a
guess about what a vendor will do; this is a **published, dated commitment on the vendor's own
pricing page**, exactly as auditable as the row above it. Refusing to record it would not be caution —
it would be knowingly shipping a number that becomes wrong on a date we already know.

## Decision

### 1. Every published row is added, both tables

Including `claude-mythos-5`, which `models.list` does not return. A price for a model the caller
cannot reach costs nothing; a missing price for one they can reach costs the cost signal.

### 2. A scheduled change is a dated row, resolved at lookup

A small parallel map holds changes with a known effective date. The base table carries the price in
force *before* the first scheduled date; each entry replaces it from its date onward. Resolved
against today's date at lookup time rather than at import, so a process running across the boundary
does not keep serving the stale rate.

The date source is injectable, because a table whose correctness depends on the calendar needs tests
that can stand on both sides of the boundary without waiting five weeks.

### 3. Cache columns stay explicit, not computed from the multipliers

Even though all sixteen rows follow 1.25× / 2.00× / 0.10× exactly, each rate is written out. A
multiplier that holds across every model today is a fact about today's price list, not a law; the
first model that breaks the pattern would be silently mispriced by a formula and visibly wrong in a
table. The same reasoning as ADR-029's, one level down: an inferred number is indistinguishable from
a looked-up one at the point of use.

### 4. `CHEAP_COUNTERPART` is rebuilt from real rates

It can now name a genuine cheap counterpart for every priced Anthropic family, and the invariant that
routing must move to something strictly cheaper is checkable against the table rather than asserted.

## Consequences

- **Every model Anthropic currently serves is priced**, so a caller on Opus 5 or Sonnet 5 gets dollar
  figures instead of `None`. That was the gap ADR-029 opened deliberately and this closes with
  sourced data.
- **Sonnet 5 is priced at $2/$10 today and $3/$15 from 2026-09-01**, without anyone having to
  remember to edit a file that morning.
- The scheduled map is expected to be near-empty most of the time. That is the point: it holds the
  rare published exception rather than becoming a general-purpose price history.
- The staleness caveat is unchanged and now sharper. This is a snapshot of a vendor page on
  2026-07-31; `PricingProvider` remains the supported way to override it, and the ADR-029 warning
  still fires for anything the table does not carry.
- Two tables still hold these numbers separately, and they now agree on sixteen models rather than
  one. A test asserts the overlap matches, so a future edit to one is caught rather than diverging
  quietly.

## Alternatives considered

**Record only today's Sonnet 5 price and revisit in September.** Rejected. It relies on a human
remembering a date, and the failure is silent and 50% large.

**Record only the September price, as the conservative one.** Rejected as the wrong kind of
conservative: it overstates a caller's spend by 50% for five weeks, and a budget policy cannot tell
an overstatement from a real cost any more than it can tell an invented one.

**Compute cache rates from the base rate.** Rejected under decision 3.

**Fetch prices from the vendor at runtime.** Already rejected in this module's opening docstring: a
pricing API call adds a network round trip to every LLM step and a failure mode to the hot path. A
stale price produces a slightly wrong number; a hanging HTTP call produces a slow agent.
