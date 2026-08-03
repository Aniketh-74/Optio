# ADR-048 — The exact counter is an instrument, not a request path

**Status:** Accepted
**Date:** 2026-08-03
**Related:** ADR-013 (a stage may never break a request), ADR-015 (measure, do not assume),
ADR-036 (calibration is per-vendor), ADR-039 (evidence carries its date),
ADR-042 (the extension point existed and nothing could reach it)

## Context

ADR-042 made `Optimizer` accept a `counter` and shipped no implementation of one. An extension point
with no example is a design for an extension point.

It also left the underlying problem exactly where it was. Every token count this package makes on an
Anthropic model goes through `tiktoken`, whose `encoding_for_model` does not know Anthropic and falls
back to `o200k_base`. That fallback is reasonable — much closer than refusing — but it is OpenAI's
tokenizer applied to another vendor, and ADR-036 already measured that the two are not
interchangeable: Anthropic bills **1.29×** the raw-JSON count for tool schemas where OpenAI bills
**0.65×**. Opposite directions, not merely different magnitudes.

`messages.count_tokens` returns the number Anthropic will bill and **bills nothing to say so**. It is
the only free source of ground truth this package has, and it is what made ADR-036's calibration a
measurement rather than an estimate.

## The constraint that shapes the design

`count_request` calls `count_text` once per message and once per tool. Backed by a network endpoint,
a forty-turn conversation with twenty tools is **sixty round trips** against a 100 ms latency budget.

So an exact counter cannot be a request-path counter, however tempting the accuracy is. This is not a
performance note to be optimised away later — it is a property of the endpoint being remote and the
counting interface being per-item.

## Decision

`AnthropicCounter` implements `TokenCounter` and is documented, named and tested as a **measurement
instrument**: for scripts, benchmarks and calibration runs, never for a live request path.

The docstring says so, and a test says so in the way that survives someone not reading it — it runs
one `count_request` over a forty-turn conversation and asserts the round-trip count. A warning in
prose is advice; a number is evidence.

**Failures are loud here**, which inverts ADR-013's rule on purpose. A stage must never break a
request, so stage failures are swallowed. An instrument that silently substituted an estimate would
return a number indistinguishable from an exact one, and being exact is the entire reason to reach
for it.

**Memoized, unbounded.** A miss is a round trip rather than a microsecond, and the same system prompt
is counted on every request. `MemoizingCounter` bounds its cache because eviction there costs
microseconds; here it would cost a network call, and a measurement run is short-lived.

**The model is part of the cache key.** The endpoint takes a model, so caching across models would
answer a question nobody asked, and a future model whose tokenizer differs would be silently
mis-counted from an entry measured on another.

`default_counter()` stays offline, asserted by a test. A default that reached the network would turn
importing this package into a service dependency.

## What it is for

`scripts/measure_anthropic_tokenizer_gap.py` compares `tiktoken`'s estimate against the exact count
across four text shapes — prose, chat turns, JSON tool results, and code — because `tokens.py`
already distinguishes prose from dense text, and a single ratio measured on one shape would be
calibrated for one stage and wrong for the rest.

It prints the ratios and a suggested constant. **It does not write one.** If the spread across shapes
is wide, it says so and recommends calibrating per shape rather than averaging, the way
`TOOL_SCHEMA_CALIBRATION_BY_MODEL` is keyed per vendor. A constant recorded without a date and a
method beside it is the shape of a number nobody can re-check (ADR-039).

## Consequences

2,220 tests pass; 6 of 6 mutations caught.

**No calibration constant ships with this.** The instrument and the script exist; the measurement has
not been run, and inventing the number it would produce is exactly what ADR-015 forbids. Running it
costs an API key and a few seconds — the endpoint is free — and until someone does, every Anthropic
prose figure in this package remains an OpenAI estimate. That is now a stated gap with a one-command
remedy rather than an unexamined default.

The same shape generalises: an exact counter for any vendor is an instrument, and the request path
wants a fast offline estimate calibrated *against* one. This ADR is the first half of that pattern.
The second half is a constant with a date on it.
