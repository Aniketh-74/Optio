# ADR-037 — The window binds the prompt, and a separate cap binds the reply

**Status:** Accepted
**Date:** 2026-08-01
**Related:** ADR-001 (emit signals, never enforce), ADR-013 rule 1 (never cause a cost increase),
ADR-015 (isolated live evidence), ADR-029 / ADR-031 (a table states what is published, never what is
inferred), ADR-030 (a diagnostic publishing an observation another stage reads), ADR-036 (measure the
provider, do not transcribe a doc page)

## Context

`OptimizeConfig.context_limit` has been accepted, validated and documented since the package's first
release, and **read by nothing**. Setting it changes no behaviour. Its own docstring says so, because
recording that was preferable to leaving a field that looks functional — a configuration field is a
claim, and this one promised a behaviour the package did not have.

`tokens.fits_in_window` is in the same position: written, tested, and never called from production
code.

Giving them a consumer requires knowing what actually fails when a request does not fit. That
question had never been asked here, and the design that seemed obvious turns out to rest on a false
premise. `scripts/measure_window_pressure.py` asked the provider directly, on `claude-haiku-4-5`:

| probe | prompt | `max_tokens` | result |
|---|---|---|---|
| output cap | 10 | 1,000,000 | **400** — `max_tokens: 1000000 > 64000, which is the maximum allowed number of output tokens for claude-haiku-4-5-20251001` |
| sum over window | 158,965 | 21,000 | **accepted** — generated normally |
| prompt over window | 217,554 | 1,024 | **400** — `prompt is too long: 217570 tokens > 200000 maximum` |

Three findings, and the middle row is the one that matters most:

**1. `prompt + max_tokens > context_window` is not an error.** 179,965 against a 200,000 window went
through. The provider does not add the two and reject the sum; it generates until the window fills.
A guard built to prevent that rejection would have been guarding against a failure that does not
happen — the same shape as ADR-033's truncation check, which compared against a string the provider
never sends, and ADR-019's streaming gate, which had never once passed.

**2. There is a second limit this package has never modelled: a per-model maximum on output tokens.**
It is *not* the context window and is not derivable from it. Read from the provider's own error
messages across the priced Anthropic models:

| model | context window | max output |
|---|---|---|
| claude-opus-4-5 | 200,000 | 64,000 |
| claude-opus-4-1 | 200,000 | **32,000** |
| claude-sonnet-4-5 | 200,000 | 64,000 |
| claude-haiku-4-5 | 200,000 | 64,000 |
| claude-fable-5 | > 217,554 | **128,000** |
| claude-opus-5 | > 217,554 | 128,000 |
| claude-opus-4-8 / 4-7 / 4-6 | > 217,554 | 128,000 |
| claude-sonnet-5 | > 217,554 | 128,000 |
| claude-sonnet-4-6 | > 217,554 | 128,000 |

The cap ranges over a factor of four across models this package already prices, and exceeding it is a
hard 400 before any generation. `AdaptiveMaxTokensStage` **can produce that 400 from a request that
had no `max_tokens` at all**: its ceiling is `max(FLOOR_TOKENS, p95 × 2)`, raised further to
`thinking_budget + ANSWER_HEADROOM_TOKENS` when a reasoning budget is set. On `claude-opus-4-1`,
whose cap is 32,000, an observed p95 of 16,001 yields 32,002 — over the cap, rejected, and fail-open
then re-sends the request unoptimized at full price. The stage that exists to lower cost would be
raising it, which is precisely what ADR-013 rule 1 forbids.

**3. Seven windows are known only as a lower bound, and that cost $7.60 to learn.** The first version
of the probe assumed a long-enough prompt guaranteed rejection, and sent `max_tokens=16` alongside
it. For models whose window exceeds the probe the request is simply *accepted* — 217,554 input tokens
billed, seven times, on the most expensive models in the table — while the script printed "Spend:
nothing" and recorded those models' windows as unknown. It paid full price for the one outcome it
could not read.

That is this project's recurring defect in the tooling rather than the library: an assumption that
flatters the measurement, unverified. It is recorded here rather than quietly fixed because the
package's own standing rule is that an unverified assumption in the cost direction is the serious
kind. The fix makes the failure structurally impossible: the window probe now sends
`max_tokens = cap + 1`, so the request is invalid whatever the prompt turns out to be, and validation
precedes generation.

## Decision

### 1. Two tables, and both record only what a provider stated

`CONTEXT_WINDOW` and `MAX_OUTPUT_TOKENS`, keyed like `PRICING` and resolved by the same
`_SAME_MODEL_SUFFIX` rule, so a dated id resolves to its alias.

A model whose window is known only to exceed 217,554 is **absent from `CONTEXT_WINDOW`**, not
recorded as 1,000,000. "Larger than we probed" is not a number, and ADR-029 exists because inferring
one across a generation boundary reported a $10 bill as $30. The `MAX_OUTPUT_TOKENS` row for those
models *is* present, because the provider stated it exactly.

### 2. `WindowPressureStage` observes and reports, and changes nothing

Built in the shape of `UnstablePrefixStage` (ADR-030): it reads each request, compares the prompt
against the window, and reports. Two findings:

`prompt_exceeds_context_window`
: The prompt alone is over the limit. This request **will** be rejected, and no stage in this package
  can rescue it — trimming enough history to fit would be an `ALTERED` change to the caller's
  conversation that ADR-015 has no evidence for. Reporting it is the honest action.

`prompt_near_context_window`
: The prompt is within `PRESSURE_RATIO` of the limit. Predicts the failure above while the caller can
  still act on it.

`Fidelity.IDENTICAL` and reports zero savings, exactly as the other diagnostic does. This is the
first production caller of both `config.context_limit` and `tokens.fits_in_window`.

**It runs last, which is the opposite of where the other diagnostic runs, and the first draft had it
wrong.** `detect_unstable_prefix` goes first because it diagnoses how the *caller* assembles their
prompt, and only the untouched request shows that. This one answers a different question — *will the
provider reject what we are about to send* — which is a fact about the final request. Warning about a
prompt `trim_history` has already cut back is warning about a rejection that will not happen.

Placing it first was also actively harmful, and measurably so. On an 81-turn, 2.7 MB conversation,
counting the untrimmed request cost **152 ms against the 100 ms budget**: the diagnostic consumed the
entire allowance, eight stages were skipped, and the trim that would have fixed the very request it
was complaining about never ran.

| | stages run | pipeline total | result |
|---|---|---|---|
| diagnostic first | 2 of 10 | 152 ms | 81 messages, untrimmed |
| diagnostic last | 10 of 10 | 22 ms | 8 messages, trimmed |

A stage that saves nothing must not be able to do that. Beyond the ordering it also carries a cheap
guard: a character count is a strict upper bound on a token count, so a prompt with fewer characters
than the threshold cannot reach it and no tokenizer runs. That is exact rather than heuristic — it
skips work, never a finding — and costs 0.01 ms on the request that costs 152 ms to tokenize.

### 3. The guard is on the output cap, because that is the limit that is enforced

`AdaptiveMaxTokensStage` clamps its ceiling to `max_output_tokens_for(model)`. This is not the
"substitute our guess for the caller's instruction" the stage rightly refuses elsewhere: it applies
only to a ceiling **this package chose**, and only downward, to a hard provider limit. The stage
still declines outright when the caller set `max_tokens` themselves.

**No stage lowers `max_tokens` on account of window pressure.** Finding 1 says there is no rejection
to prevent, and a lower ceiling would truncate a reply the provider was willing to generate — paying
a real correctness cost for an imaginary error.

### 4. The caller's `context_limit` wins, and an unknown model gets no opinion

`config.context_limit` overrides the table; the table answers when it is unset; when neither can, the
stage declines silently. Absence is never zero and never a guess — the rule ADR-027 and ADR-036 both
land on.

## Consequences

- **`context_limit` becomes functional**, and its docstring stops disclaiming itself.
- **A live 400 that `adaptive_max_tokens` could cause is closed**, on any model whose cap is under
  twice the observed p95 — `claude-opus-4-1` at 32,000 is the reachable case in today's table.
- **The package models a provider limit it did not know existed.** Nothing here had a concept of a
  maximum reply length distinct from the context window.
- **Seven context windows are recorded as unknown**, so `WindowPressureStage` says nothing at all on
  the newest Anthropic models until someone measures them. That is the correct behaviour and it is
  visibly incomplete, which is better than being invisibly wrong.
- The measurement script is free to re-run **only now that the probe is unbillable**; it was not
  before, and the ADR above says what that cost.

## Alternatives considered

**Guard `prompt + max_tokens` against the window.** Rejected on the measurement: the provider accepts
that request. This was the design before the probe ran, and it is the reason the probe ran first.

**Populate the missing windows from the vendor's documentation page.** Rejected. Every other number
in this package's tables is either published price data or a provider-stated measurement, and the
four windows we do have came out of the API's own error text. Mixing a transcribed figure into a
measured table removes the reader's ability to tell which is which.

**Let `trim_history` cut harder under window pressure.** Deferred, not rejected. It drops content, so
it is `Fidelity.ALTERED` and needs its own live evidence under ADR-015. The diagnostic is the
prerequisite for that evidence, not a substitute for it.

**Infer the output cap as a fraction of the window.** Rejected: the two are unrelated. Three models
share a 200,000 window with caps of 32,000 and 64,000, and the 128,000-cap models have a window that
is larger still and unmeasured.
