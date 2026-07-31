# ADR-029 — A price may not be inferred across model generations

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-004 (fail open), ADR-018 amendment (this package has twice published an invented
number a live run then corrected), ADR-024 / ADR-028 (do not report what you cannot attribute)

## Context

`optio`'s reason to exist is emitting a cost signal. That signal comes from one table lookup, and the
lookup infers prices it does not have.

`StaticPricingProvider.price_for` documents itself as "matched exactly first and then by longest
prefix", for "vendor-prefixed ids and dated snapshots". The implementation is `if name in
normalised` — **substring containment, not prefix** — over a table whose Anthropic rows are
`claude-opus-4`, `claude-sonnet-4` and `claude-haiku-4`. Measured against the eleven ids
`models.list` returned on 2026-07-31:

```
claude-opus-4-5-20251101      -> claude-opus-4    $15/$75
claude-opus-4-6               -> claude-opus-4    $15/$75
claude-opus-4-7               -> claude-opus-4    $15/$75
claude-opus-4-8               -> claude-opus-4    $15/$75
claude-opus-4-1-20250805      -> claude-opus-4    $15/$75
claude-sonnet-4-5-20250929    -> claude-sonnet-4   $3/$15
claude-sonnet-4-6             -> claude-sonnet-4   $3/$15
```

**Five distinct Opus generations collapse onto one row**, and Anthropic cut Opus list pricing at
4.5. On a million input and two hundred thousand output tokens the table reports **$30.00 against a
$10.00 bill — 3× over, silently.** Nothing marks the number as inferred, because from the caller's
side it is indistinguishable from a row that was actually looked up.

Two further consequences of containment rather than prefix:

```
some-vendor/claude-opus-4-distilled-mini  ->  $15/$75
not-really-gpt-4o-at-all-v2               ->  $2.50/$10
```

Any string containing a known model name is priced as that model.

The three Anthropic keys are also **not model ids at all**: `claude-opus-4`, `claude-sonnet-4` and
`claude-haiku-4` each return `404 not_found_error`. They were already found to be fictional when
`AnthropicProvider.DEFAULT_MODEL` was one of them and no live Anthropic benchmark had ever completed
a call. They survived here as prefix keys, where being fictional is invisible — a key that names no
real model still matches five real ones.

`optio_optimize.PRICING` has the opposite failure and it is the mild one: exact `.get()` only, so
**ten of the eleven served models return `None`** and produce no cost figures at all. Absence, which
this project prefers to a wrong number.

## Decision

### 1. A version-bumped suffix is a different model

Matching is by prefix, on a segment boundary, and the remainder decides:

- **empty** — the id is the key. Match.
- **a date** (`-20251101`, `-2024-11-20`; a leading run of four or more digits) — a dated snapshot of
  the same model, which is the documented intent. Match.
- **anything beginning with a short numeric segment** (`-5`, `-4-1`) — a version bump. **No match.**
- anything else (`-mini`, `-distilled`) — a different model. **No match.**

`gpt-4o-mini` was already protected by exact-match-first; it is now protected by the rule as well.

### 2. Vendor prefixes are stripped, not searched for

`openai/gpt-4o` and `anthropic.claude-3-haiku-v1` resolve by removing a leading vendor segment and
matching the remainder, rather than by scanning for a known name anywhere in the string. The
documented behaviour is preserved; `not-really-gpt-4o-at-all-v2` stops being priced as `gpt-4o`.

### 3. Only rows we can source are added

Four rows are added to both tables — `claude-opus-4-1`, `claude-opus-4-5`, `claude-sonnet-4-5`,
`claude-haiku-4-5` — being the models whose published rates are established. **Seven currently served
models are deliberately left unpriced**: `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5`,
`claude-opus-4-6`, `claude-opus-4-7`, `claude-opus-4-8`, `claude-sonnet-4-6`. Their rates have not
been read off the vendor's page by anyone here, and this package's standing rule — earned twice — is
that an invented number survives until a live run contradicts it. `None` is the correct output for a
model whose price we do not know.

### 4. An unpriced model says so, once

A `None` today is silent: no cost attribute is emitted and nothing explains the gap, so the seven
models above would look like a broken cost lane rather than a missing table row. The lane logs once
per unknown model, naming it and pointing at `PricingProvider`. Once per model, not per call —
ADR-004's fail-open discipline means a pricing gap may never become a log flood.

### 5. `CHEAP_COUNTERPART` stops targeting models that do not exist

It mapped `claude-opus-4` and `claude-sonnet-4` to `claude-haiku-4`, all three of which 404. It is
benchmark-only — production routing reads `config.cheap_model`, which the caller sets — so this is a
smaller blast radius than it appears, but a suite that cannot route is a suite that cannot measure
routing.

## Consequences

- **The 3× Opus overstatement is gone**, and so are the four other generations that were being priced
  from a predecessor's row.
- **Seven served models now report no cost rather than a guess.** That is a visible regression in
  coverage and the honest state of our knowledge. Decision 4 makes it legible and decision 3's list
  is the shopping list for closing it — four numbers per model, read off the vendor's page.
- **`claude-sonnet-4-5` keeps its price by gaining a row**, not by inheriting one. The number does not
  change; what changes is that the table now asserts it rather than inferring it.
- Callers pricing a fine-tune or a proxied id by substring luck will lose that. It was never a
  documented behaviour and it priced `not-really-gpt-4o-at-all-v2` as `gpt-4o`.
- The two tables still exist separately — core prices for signals, `optio_optimize` prices with cache
  rates for savings. Unifying them is a larger change than this one and ADR-013's boundary argues
  against the optimizer's richer table leaking into core.

## Alternatives considered

**Fill the seven missing rows with estimates from the nearest known generation.** Rejected. It is
precisely the inference this ADR removes, moved from the matcher into the table where it would be
harder to see.

**Keep containment and add the missing rows.** Rejected: it fixes today's ids and leaves the
mechanism that will mis-price tomorrow's. Opus 4.9 would inherit Opus 4's row on the day it ships.

**Return the nearest match with a `confidence` field.** Rejected as the shape ADR-024 removed
elsewhere — it keeps the guess and asks every consumer to remember to check a flag. A cost signal is
consumed by dashboards and alerts that will not check it.
