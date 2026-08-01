# ADR-040 — A field with a default is a field every old call site keeps compiling around

**Status:** Accepted
**Date:** 2026-08-01
**Related:** ADR-021 (the two cache-write bands), ADR-023 (cascade verification), ADR-027 (decline
reasons), ADR-030 (unrewarded writes), ADR-036 (per-vendor calibration), ADR-038 (counter warm-up)

## Context

A static review of `main...HEAD` — 127 files, ~23k insertions, no tests run, `pytest` not installed
in the reviewing environment — reported eight findings. **Seven were real.** The suite was green at
2,110 tests and `mypy --strict` clean while carrying every one of them.

That is the fact worth recording. None of the seven is an exotic bug. Each survived because it lives
in a *combination* the suite exercised only separately.

### Four are the same defect

ADR-021 added `cache_write_tokens` and `cache_write_1h_tokens` to `LLMResponse`. Every site that
**reads** provider usage was updated. Every site that **copies, zeroes or re-prices** a response was
not — and because both fields carry a `0` default, all of them kept compiling and kept type-checking.

| site | effect |
|---|---|
| `served_from_cache` | zeroed three fields, copied both write bands forward — every `exact_cache` and `semantic_cache` hit re-billed the original call's premium tokens |
| `bench.providers._actual_cost` | priced writes at base rate, so `SpendGuard` undercounted against a live `--cap` |
| `anthropic_streaming._Accumulator` | never read `cache_creation.ephemeral_1h_input_tokens`, so every streamed reply reported `cache_write_1h_tokens=0` and priced 2x tokens at 1.25x |
| `wire.response_from_anthropic_message` | correct — it was the one updated, which is how the divergence stayed invisible |

`ABResult.cost_usd` learned the bands in the same branch that added them. `_actual_cost` computes the
same thing in a second place and did not. Two call sites for one calculation, one updated.

### Three are independent

**`count_tools` applied OpenAI's ratio to everyone.** ADR-036 measured Anthropic at 1.29 against
`gpt-4o-mini`'s 0.65 — different in *direction*, since Anthropic re-renders schemas and bills more
than the raw JSON tokenizes to — and applied it in `minify_tools` only. `count_tools` kept the global
constant, a ~2x undercount feeding `PrefixCacheStage`'s per-model floor. A tool-heavy Anthropic
prefix genuinely over 4,096 measured about 2,000 and was declined: **half of the failure ADR-036 was
written to fix, still live in the other half of the codebase.**

**`_warm_counter` warmed one encoding.** ADR-038 moved `tiktoken`'s 395 ms vocabulary load out of the
first request's latency budget by calling `count_text("warm", "")`. `encoding_for_model("")` raises
`KeyError` and falls back to `o200k_base`, so `cl100k_base` stayed cold and `gpt-4` and
`gpt-3.5-turbo` still paid the full load inside a 100 ms deadline on request one. The bug ADR-038
closed, surviving for the families it did not name.

**Anthropic tool calls never reached the cascade verifier.** `response_from_anthropic_message` joins
only `text` blocks and set no `extra`, so a `tool_use` reply arrived as empty content with no
proposal. `default_verifier`'s empty-answer rule is guarded by *"only when no tool call was
proposed"* — and none was visible. Every tool-using step escalated: cheap model **and** expensive
model, on every call. Strictly worse than not routing, and the config docstring promised the
opposite.

### One is wrong

The report said `ReasoningBudgetStage.after` feeds outputs produced under a reduced budget back into
the same `_lengths` the ceiling derives from, ratcheting toward `MIN_THINKING_BUDGET`. The loop is
real; the collapse is not. `REASONING_CEILING_MULTIPLIER` is **2.0**, and `before` lowers only when
`2 × p95 < budget` — a budget can fall only while the model uses less than half of it, and stops the
moment that ceases. Simulated over 60 turns from 32,000:

| model behaviour | settles at |
|---|---|
| needs a fixed 3,000 whatever it is given | 6,000 |
| fills whatever budget it is given | 32,000 |
| uses 90% of whatever it is given | 32,000 |
| uses 40% of whatever it is given | 25,600 |

Converging on twice what the model uses is the stage working.

## Decision

Fix the seven. Reject the eighth **with a test rather than an argument** — three tests pin the
convergence behaviour, and a mutation setting the multiplier to 1.0 fails them, because that is what
would actually cause the reported failure.

Two changes beyond the report:

`_identity_boundary` is new. `cache_ttl_selection` identified a prefix by `messages[:boundary]` where
`boundary = len(messages) - 2`, so in any appending conversation the digest changed every turn,
`_last_seen` never matched, `_expires` was never populated, and the one-hour TTL could not be emitted
for the agent loop the feature exists for. The existing tests missed it because they vary only the
final message of a fixed four-message request, holding `messages[:2]` constant. **Real conversations
append.** Identity now covers the system block alone — the part that does not grow, and the part
whose expiry Anthropic's incremental caching lets you observe.

`last_decline_reason` is cleared on a successful mark. Correcting `count_tools` made
`PrefixCacheStage` start marking prefixes it had been declining, and a *marked* request still
reported the previous one's "below the cacheable minimum". ADR-027 added that attribute to make "no
cache reads" diagnosable; a stale reason answers the question about the wrong request.

## Consequences

2,133 tests pass. 13 of 13 mutations caught, two of which were gaps in the new tests themselves —
one asserted `unknown <= anthropic` where the over-claiming direction also satisfies `<=`, and one
used a counter that failed for every model, so an early `return` in the warm-up loop passed anyway.

**The general rule this leaves behind:** adding a field with a default to a shared dataclass is a
silent change to every site that constructs, copies or reduces it. `mypy --strict` cannot see it —
that is what the default is *for*. The fix is not more types; it is a test that enumerates the
fields, of the shape now in `test_every_billable_field_is_zeroed`, so the next band added fails on
the day it appears rather than the day someone reads a report and doubts it.

**And on reviews:** seven real findings against a green suite, from a reviewer that could not run it.
Reading for the combinations tests do not cover is worth more than another test of a path already
covered.
