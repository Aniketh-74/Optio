# ADR-021 — Cache TTL selection, and why its accounting has to land first

**Status:** Accepted
**Date:** 2026-07-30
**Related:** ADR-013 (fail-open is rule 1; never cause a cost increase), ADR-016 (the in-scope test),
ADR-020 (the last cache-economics lever), §10 (never log prompt content), §11 (bounded memory)

## Context

Anthropic sells two cache lifetimes. A 5-minute write costs **1.25x** the base input rate, a
one-hour write costs **2.0x**, and a read costs **0.1x** from either. This package has only ever
asked for 5 minutes — `wire.EPHEMERAL_CACHE_CONTROL` is a bare `{"type": "ephemeral"}` — and
`ModelPricing`'s own docstring already notes the 2x rate as the one "it does not [request]".

That is the wrong default for an agent loop. A step that takes more than five minutes — a tool call
waiting on a slow API, a human approval, a queue — lets the entry expire, so the next step pays a
fresh 1.25x write on a prefix it just wrote. Over a ten-step loop that is ten writes where one write
and nine reads would have done.

The break-even is low. Deciding the TTL for *this* write, with `m` further uses of the same prefix
inside the hour, each after a gap long enough to expire a 5-minute entry:

- one hour: `2.0 + 0.1m`
- five minutes: `1.25(m + 1)`

One hour wins when `0.75 < 1.15m`, i.e. **`m >= 1`**. A single further use inside the hour pays for
the upgrade.

### What the SDK and the API actually do, measured before this was written

ADR-018's first draft assumed a shape and was wrong, so this one was probed first:

- `CacheControlEphemeralParam` carries `ttl: Literal["5m", "1h"]` on the **standard** type.
- **`ttl: "1h"` works with no beta header.** A probe sent one without
  `extended-cache-ttl-2025-04-11` and 4,218 tokens landed in `ephemeral_1h_input_tokens`. Sending the
  header changed nothing. The header is still in `anthropic_beta_param.py`, which is what made this
  worth checking rather than assuming — had it been required and absent, the API's likely behaviour
  is to ignore the field silently, and this package would have paid 1.25x forever while reporting
  that it had bought an hour.
- **The provider reports writes split by band**: `usage.cache_creation.ephemeral_1h_input_tokens`
  and `ephemeral_5m_input_tokens`, alongside the combined `cache_creation_input_tokens`.

## Decision

### 1. The accounting lands before the lever, not after

`savings._cost` prices every write at `cache_write_usd_per_m`, which is the 1.25x rate. The moment a
stage asks for `ttl: "1h"`, that number under-bills the most expensive band in the request by
**37.5%** — and it does so in the direction that inflates this package's headline saving.

This project has already paid for that exact asymmetry once: cache writes were omitted from the
prompt total entirely, one turn reported 200 tokens against a true 4,805, and a published 53.7%
figure was really 50.1%. Shipping the lever before the accounting would reproduce it knowingly.

So `LLMResponse` gains a 1-hour write count, `wire` reads the split the provider already sends,
`ModelPricing` gains a 1-hour write rate, and `_cost` grows a fourth band — **and none of that
depends on the stage existing.** It is worth doing on its own: a caller who sets their own `ttl:
"1h"` breakpoint today is already mis-priced by this package, which is a defect independent of
anything decided below.

### 2. TTL selection belongs inside `PrefixCacheStage`, not in a new stage

The marker and its lifetime are one decision. A separate stage would have to run after prefix
marking — which the ordering rules already pin last — and rewrite a decision the previous stage just
made, which is how two stages come to disagree about the same field.

### 3. The choice is driven by observed re-use, never by a guess

The stage records, per prefix, when it last saw that prefix. It asks for one hour only when the same
prefix has already been seen again **after a gap longer than the 5-minute window** — that is, when
expiry has been observed rather than predicted.

By the arithmetic above, one further use inside the hour makes that call correct. If none comes, the
cost is `0.75` rate-units on one write — about $0.003 on a 4,000-token prefix at Haiku's rate. The
upside when it is right is `1.15` rate-units per avoided write. Acting only on observed expiry is
what keeps this from being the "weak proxy" ADR-018 rejected: nothing here predicts a future gap, it
reacts to one that already happened.

### 4. A prefix is identified by a hash, and never by its text

Prefix identity goes through the same rule as `cache.request_key`: §10's content ban applies, these
values reach logs and metrics, and a prompt must not be reconstructible from them. The map is
bounded by an LRU with an entry ceiling (§11) — it lives for the process lifetime in a long-running
agent, which is exactly the leak the core's soak test exists to catch.

### 5. It ships off by default, pending its own live measurement

Fidelity is `IDENTICAL` — no request content changes and no answer can differ — so the usual reason
for shipping off does not apply. A different one does: this is the first lever in the package that
can *raise* a bill. ADR-013's rule 1 forbids a cost-reduction library causing a cost increase, and
while the expected value after observed expiry is strongly positive, "strongly positive in
expectation" is a model, and every model this package has shipped without measuring has been wrong
except one.

`concision` took the same bargain and is still off because the measurement said so. The flag flips
when a live run with real five-minute gaps says it should, and not before.

## Consequences

- Measuring this live is unusually slow: proving the 5-minute entry expires means *waiting out five
  minutes*, twice, so the run is ~12 minutes of wall clock for a handful of calls. That is the only
  honest way to observe the behaviour the lever exists for, and a shorter run would measure the case
  where 5 minutes is already sufficient.
- A caller's own `cache_control` is still never overwritten (enforced in the Anthropic adapter since
  2026-07-30). A caller who chose `ttl: "1h"` has better information than this package's inference,
  and after decision 1 they will finally be billed correctly for it.
- `Message` gains a public field. The alternative — widening `cacheable` from `bool` to a union —
  would break every existing reader of it for no gain.
- OpenAI is unaffected: it caches automatically with no TTL control and no write premium, so the new
  price band is `None` there and the stage's choice reaches nothing. The field is Anthropic-shaped
  because the capability is.

## Alternatives considered

**Always request one hour.** Rejected. It is a 60% premium on every first write, paid on every
prefix that is never re-used — a guaranteed cost increase for a conditional saving.

**Choose the TTL from a configured hint (`expected_gap_seconds`).** Rejected as the whole answer for
ADR-018's reason: it moves the decision to the caller for something they cannot easily know, and this
package's own observation is better information than their estimate. Not rejected as a future
override.

**Predict the gap from step latency.** Rejected. It is the same weak proxy as inferring a reasoning
budget from prompt complexity, applied to a cost decision, and observed expiry is available for free.
