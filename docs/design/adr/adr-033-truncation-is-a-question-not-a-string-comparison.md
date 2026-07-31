# ADR-033 — Truncation is a question, not a string comparison

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-004 (fail open), ADR-013 (rule 1), ADR-022 / ADR-032 (one reading of a wire shape,
shared), ADR-023 (cascade), ADR-025 (the neutral shape needs a translation per provider)

## Context

The first live run of cascade (ADR-023's technique had never touched a real provider) carried a
request with `max_tokens=16` and a prompt that cannot be answered in sixteen tokens. It was there as
the *guaranteed* escalation: `default_verifier`'s very first check rejects a truncated answer, so
this request had to escalate or the verifier was broken.

It did not escalate. The chopped-off answer was accepted and returned as final.

```
raw stop_reason      : 'max_tokens'
our finish_reason    : 'max_tokens'
verifier checks      : finish_reason == 'length'
=> fires on Anthropic? False
```

**Anthropic reports `max_tokens`. OpenAI reports `length`.** Every truncation check in this package
compares against `"length"` only, so on Anthropic all of them are dead code. Two of them matter:

**1. `default_verifier` (cascade).** Its first and most basic check — "did the model run out of
room" — never fires against Anthropic. A cheap model's truncated answer is accepted and handed back
as though it were complete. ADR-023 sells this verifier as catching "the failures a cheap
deterministic check can be *sure* of"; on the vendor this package integrates most deeply with, it is
sure of one fewer than it claims.

**2. `ExactCacheStage` — and this is the serious one.** The stage refuses to serve a stored response
that was truncated, and `cache.py` states exactly why:

> A truncated reply is not a complete answer. Serving one to a caller who allowed more output would
> silently cap them at whatever ceiling happened to apply the first time — and `max_tokens` is
> excluded from the key precisely so those calls share an entry.

That reasoning is sound and the guard implementing it does not run on Anthropic. So: one request with
`max_tokens=16` stores a truncated answer; a later byte-identical request allowing 4,096 tokens gets
that truncated answer served from cache. `ExactCacheStage` is `Fidelity.IDENTICAL`, **lossless and on
by default** — it promises the response is what the provider would have returned. Here it is not.

The two facts compound. `request_key` deliberately omits `max_tokens` *because* the truncation guard
compensates. Remove the guard and the omission becomes unsound, which is precisely the state
Anthropic callers have been in.

This is the same defect shape as ADR-025 and ADR-032: logic written against one provider's wire
vocabulary, silently inert on the other, with no failure anywhere to point at it.

## Decision

### 1. `LLMResponse.was_truncated` replaces every string comparison

A derived property that knows the vocabularies:

```python
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
```

Named for the question being asked rather than the string being compared. Both call sites — the
verifier and the cache — ask `response.was_truncated`, and neither knows a provider's spelling. A
third provider is one entry in a frozenset, in the one place that already holds this knowledge, and
adding it fixes every caller at once.

`max_output_tokens` is included because Gemini spells it that way and the cost of carrying an unused
member is nothing, while the cost of omitting one is this ADR.

### 2. The raw reason is preserved, not normalized away

`finish_reason` keeps whatever the provider said. Rewriting Anthropic's `max_tokens` to `length` at
the wire boundary would fix both call sites too, and it would lie to anyone reading the field —
including callers, who see `LLMResponse` — about what the provider reported. The derived property
answers the question without discarding the evidence.

### 3. A cached response records how it ended

`served_from_cache` zeroes token counts because they describe the original call. `finish_reason`
must survive, because it describes the *answer*, which is the thing being reused. It already does;
this is recorded so it stays that way, since zeroing it would silently re-open the hole above.

## Consequences

- **Anthropic callers stop being served truncated answers from cache.** That is a correctness fix in
  a lossless default-on stage, not an optimization.
- **Cascade escalates on truncation, as ADR-023 always said it did.** Measured: the live run's
  `truncated` request went from silently accepted to escalated.
- No behaviour changes for OpenAI callers; `"length"` remains in the set.
- One more piece of provider vocabulary concentrated in one place. The pattern is now explicit
  enough to state as a rule: **when this package compares a provider's string to a literal, that is a
  bug waiting for the second provider.** ADR-025 (tool roles), ADR-032 (tool-result blocks) and this
  one are the same mistake three times.
- The gap was invisible to the entire test suite because every fixture that exercises truncation was
  written with OpenAI's spelling. Tests now cover both vocabularies at both call sites.

## Alternatives considered

**Normalize `stop_reason` to OpenAI's vocabulary in `wire`.** Rejected under decision 2: it fixes the
call sites by making the reported field wrong, and `finish_reason` is public on `LLMResponse`.

**Compare against a set inline at each call site.** Rejected. It is the same literal duplicated,
which is how one of the two call sites gets missed the next time — and there are exactly two today
because the first version of this fix only found the verifier.

**Treat an unknown `finish_reason` as truncated.** Rejected as failing the wrong way: it would make
`ExactCacheStage` refuse to serve anything from a provider whose vocabulary is not listed, turning a
missing entry into a silently disabled cache rather than a visible one.
