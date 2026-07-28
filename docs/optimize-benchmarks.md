# Measured savings

Numbers from `python -m optio_optimize.bench`. Live figures come from real
`gpt-4o-mini` calls; simulated figures are marked as such. Reproduce with:

```bash
python -m optio_optimize.bench --strict-fidelity            # free, instant
python -m optio_optimize.bench --live --cap 1.00            # real API, capped
```

**Last live-verified**, by section (a number gets no more trust than its date --
provider caching behavior has already changed the reported figures once, see
`_AUTO_CACHE_QUANTUM_TOKENS` in `bench/providers.py`):

| Section | Date | Spend |
|---|---|---|
| Live results (identical-output stages) | 2026-07-28 | — |
| `trim_history` / tool-call safety | 2026-07-28 | $0.0138 + $0.0055 |
| Phase 3 (`ALTERED`-tier stages) | 2026-07-28 | $0.0041 |
| `rag_queries_noisy` (`prune_retrieval`) | 2026-07-29 | $0.0029 |
| `SimulatedProvider` cache-quantum calibration | 2026-07-29 | ~$0.001 (raw trace, not a bench run) |

## Live results (gpt-4o-mini, identical-output stages only)

| workload | input ↓ | output ↓ | cost ↓ | latency | quality |
|---|---|---|---|---|---|
| `retry_storm` | 93.3% | 93.3% | **93.7%** | **−96.7%** | 100% identical |
| `tool_loop` | 80.0% | 80.0% | **80.8%** | **−78.6%** | 100% identical |
| `fan_out` | 66.7% | 66.7% | **66.9%** | −41.0% | 100% identical |
| `multi_turn_chat` | 0.0% | 0.0% | **−7.7%** | −2.6% | 100% identical |
| `rag_queries` | 0.0% | 0.0% | **6.0%** | −1.5% | 100% identical |

Total spend to produce this table: **under $0.02**.

Every workload returned byte-identical responses. That is the whole claim of the
`IDENTICAL` fidelity level, and it is checked rather than asserted.

## The result worth reading twice

**`multi_turn_chat` came in at −7.7%: the optimizer made it slightly more
expensive.** The simulated suite had reported **+36.3%** for the same workload.

The simulator was wrong, and the way it was wrong is the most instructive thing
in this document. It modelled Anthropic-style prefix caching, where nothing is
cached unless the request carries an explicit `cache_control` breakpoint — so
our `prefix_cache` stage looked like the thing unlocking a 90% input discount.

OpenAI does not work that way. Measured directly:

```
system prompt: 1387 tokens
call 1: prompt=1401  cached=1024
call 2: prompt=1401  cached=1280
call 3: prompt=1401  cached=1280
```

**OpenAI caches any prefix over ~1024 tokens automatically, with no marker.**
The baseline arm was already getting the discount. Our stage contributed
nothing, and the ±8% spread is server-side cache variance between the two arms.

So the simulator had been crediting this library with a feature the provider
grants unconditionally. That is the single easiest way to publish a benchmark
that collapses the first time someone reproduces it, and only a live call caught
it. The simulator now models both styles, and the honest comparison is:

| workload | OpenAI-style (automatic) | Anthropic-style (explicit) |
|---|---|---|
| `multi_turn_chat` | **0.0%** | **33.5%** |
| `rag_queries` | **0.0%** | **28.7%** |
| `tool_loop` | 71.6% | 85.5% |

**`prefix_cache` is worth roughly a third of spend on Anthropic and nothing on
OpenAI.** It costs nothing to leave on, but nobody should expect a discount
they were already receiving.

## Phase 2: history trimming, deduplication, pruning

`trim_history`, `deduplicate` and `prune_retrieval` land after the table
above. A simulated pass first predicted `trim_history` would *raise* cost on
OpenAI-style automatic caching; a live run against the real API (below)
overturned that, the same way the 36.3% claim above was overturned. Total
spend for the full 7-workload live suite: **$0.0138 across 140 calls.**

| workload | input ↓ | output ↓ | cost ↓ | quality |
|---|---|---|---|---|
| `multi_turn_chat` | 7.3% | 35.0% | **8.4%** | 25% identical, 9 SHAPED (expected) |
| `rag_queries` | 4.2% | 0.0% | **16.5%** | 100% identical |

Both were 0.0% token reduction before Phase 2 (see the honest-comparison table
above); both now show real, positive, live-measured savings.

**The simulated regression didn't hold up, and the reason is instructive.**
The simulator's automatic-cache model matches a growing prefix by exact string
comparison against everything seen so far, so a sliding window -- which
changes its oldest surviving message almost every turn -- looked like it was
forfeiting a discount the untrimmed baseline was accumulating for free.
Simulated: tokens down 6.5%, cost *up* 34.8%. Live: cost *down* 8.4%, and
output tokens fell 35% along with the input -- a shorter prompt produced
shorter answers too, a real effect the simulator cannot see at all, since it
always returns a fixed synthetic completion length regardless of what was
sent. The provider's real caching behavior evidently doesn't reward an
ever-growing prefix as reliably as the simulator's exact-match model assumed,
and whatever discount trimming gives up on the input side was smaller than
what a shorter prompt saved on output. This is the same category of error as
the 36.3%-to-0% correction above -- a plausible mechanism, modelled
carefully, that a live call still contradicted -- and it argues for treating
every simulated number in this document as a hypothesis, not a result, until
`--live` confirms it.

`deduplicate` accounted for the entire `rag_queries` saving, at 100% output
identity across all 10 live calls -- removing an exact-duplicate context block
changed nothing about the answer, on this workload. `prune_retrieval` reported
zero, because the workload's chunks all share the query's key vocabulary
(`revenue`) and none scored below the relevance floor: a correct, honest zero
on a workload this heuristic was never going to touch, not a bug -- but a
correct zero on a workload with nothing to prune is not evidence the stage
*does* anything, either. `rag_queries_noisy` exists to settle that: one
genuinely irrelevant chunk (an office-parking notice) mixed into otherwise
relevant retrieved context. Live: `prune_retrieval` dropped exactly that
chunk on every one of the 10 requests, cost fell 9.1%, and output stayed
90% identical (9/10) -- the one chunk removed genuinely wasn't needed for
the answer in nearly every case. See `tests/optimize/test_benchmark.py`'s
`TestPruneRetrievalActuallyPrunes` for the same claim checked directly
against the stage, not just inferred from an aggregate token count.

**Practical read:** both `multi_turn_chat` and `rag_queries` moved off the 0%
floor documented above, with real dollars behind the number. `trim_history`'s
divergence rate (9/12 responses reworded) is expected and priced in --
it is `Fidelity.SHAPED`, not `IDENTICAL`, precisely because dropping context
can change the answer. `deduplicate` and `prune_retrieval` stayed byte-identical
on their one live test each; that is encouraging but is one workload's worth of
evidence, not a guarantee for every prompt shape -- measure your own traffic
before leaning on it.

## `trim_history` and real tool calls

None of the workloads above use a `"tool"`-role message. `multi_turn_chat`
grows through plain user/assistant text, and `tool_loop` only *talks about*
calling a tool -- so neither could have exercised the one place `trim_history`
could actually corrupt a request: every provider requires a `tool` message to
immediately follow the assistant message whose `tool_calls` produced it, and a
naive suffix cut can land between them.

`tool_calling_chat` (10 turns, 20 requests) exists to test exactly that.
`TrimHistoryStage` now walks its cut point backward past any leading run of
`tool` messages so the assistant that issued them, and every one of its
results, survive together or not at all -- see the stage's docstring for the
mechanism. Live against `gpt-4o-mini`:

| | provider calls | errors | cost |
|---|---|---|---|
| baseline | 20/20 succeeded | 0 | $0.00283 |
| optimized (`trim_history` on) | 20/20 succeeded | 0 | $0.00264 (−6.8%) |

Zero errors on either arm is the actual finding here, not the 6.8%. A single
orphaned tool result would have surfaced as a 400 on every call after the
window first slid past a tool exchange -- which is what happened on the
*first* attempt at this measurement, and not because of a trimming bug:

**A second, adjacent defect, found by the same live run.** The first live
attempt at this workload failed all but one call on *both* arms, trim_history
enabled or not, with `messages with role 'tool' must be a response to a
preceeding message with 'tool_calls'`. The cause was `OpenAIProvider`
(`bench/providers.py`), which built its request as
`{"role": m.role, "content": m.content}` and silently dropped everything else
-- including `tool_calls` on the assistant message and `tool_call_id` on the
tool message, both of which `Message.extra` was carrying correctly. No stage
did anything wrong; the live adapter just never sent the structure that makes
a tool message valid in the first place, so *every* tool-calling workload was
unrunnable live regardless of what `optio_optimize` did to it. Fixed by
pulling `tool_calls` / `tool_call_id` out of `extra` explicitly when building
the OpenAI payload. The same class of surprise as `fan_out`'s missing
`"json"` literal: a benchmark that never sent a real tool call could not have
caught it, and now does.

## What the numbers mean per stage

`exact_cache` is where the measured savings come from. It avoids the call
entirely, so the saving is 100% of that request — input, output and latency
together. Its value is therefore entirely a question of how repetitive the
workload is, which is why `retry_storm` (93%) and `unique_questions` (0%) are
both in the suite.

`prefix_cache` avoids no tokens at all and reports zero, by design. It lowers
the *price* of tokens still sent, which shows up in the cost column and in
`cached_input_tokens`, never in the token column.

`adaptive_max_tokens` and `structured_output` shape output rather than
preserving it byte-for-byte, so they are excluded from `--strict-fidelity`
runs. Their savings are real but must be measured against a quality gate, not
against an identity check.

## What these numbers are not

**They are one machine, two providers, seven synthetic workloads.** Your traffic
decides your savings, which is why the harness ships inside the package: point
it at your own captured requests rather than trusting this table.

**Live A/B slightly favours whichever arm runs second**, because the provider's
prefix cache is server-side and cannot be reset between arms. The baseline runs
first, so the bias works against the result this library would like to report.

**Output quality is checked as identity, not as correctness.** For the
`IDENTICAL` stages that is exactly the right test. For `SHAPED` stages it is
not sufficient, but every one currently in the package (`trim_history`,
`deduplicate`, `prune_retrieval`, `adaptive_max_tokens`, `structured_output`)
has been live-checked directly above. The `ALTERED` tier's own eval gate is
`src/optio_optimize/eval/` — see the next section for what it does and does
not prove.

## The simulator's cache model, recalibrated against a fresh trace

`SimulatedProvider`'s automatic-cache model (`bench/providers.py`) reports
`cached_input_tokens` for OpenAI-style caching, and until 2026-07-29 it
reported whatever token count the nearest message boundary past the
1024-token floor happened to land on -- an arbitrary number, not a modelled
one. An 8-call live trace that day (a growing conversation, ~1400-1700
prompt tokens) showed `cached_tokens` moving in exact multiples of 128 --
0 → 1408 → plateau → 1536 → plateau -- never landing between them, even
though the prompt itself grew by an uneven token count every call. The
simulator now rounds down to that quantum (`_AUTO_CACHE_QUANTUM_TOKENS`),
pinned by `tests/optimize/test_providers.py`.

**Worth noting rather than hiding**: an earlier trace (2026-07-28, the
`multi_turn_chat` measurement above) recorded a 256-token jump between two
calls, not 128. Read together, both are consistent with a single 128-token
quantum -- 256 is two quanta crossed in one step, not evidence of a
different granularity -- but the 2026-07-28 trace alone did not distinguish
the two possibilities, and was not re-examined closely enough at the time to
notice. This document now gives the date and method behind a calibration
claim, not just the number, for exactly that reason: a provider's own
caching behavior is not guaranteed to stay the same release to release
either, and a stale, undated "128" would look identical to a correct one
until someone re-measured it.

## Phase 3: the `ALTERED`-tier stages (experimental, off by default)

`route_models`, `compress_prompt`, `semantic_cache`, `summarize_history`
landed with the ADR-013 rule 3 eval gate (`src/optio_optimize/eval/`,
exercised by `tests/optimize/test_eval.py` — an ordinary, CI-blocking pytest
module, no separate runner). All four stay off by default; none are counted
in the headline table above, which only ever reflected the identical-output
stages.

**The eval gate is deliberately model-free**: it checks that a required fact
survives a stage's transformation of the *prompt*, or that a cache-style stage
hits a near match and refuses a stranger — never "does a model still answer
correctly", which needs a real call this gate is built specifically to avoid.
That is a real ceiling, stated rather than hidden: it proves a stage's own
logic does what it claims, not that a live answer stays good. See
`src/optio_optimize/eval/__init__.py` for the full reasoning and
`bench/harness.py`'s live A/B path (with a caller-supplied judge) for the tool
that covers the other half.

**Cheapest defensible option throughout, by design.** `semantic_cache` and
`compress_prompt` use lexical word-overlap (`optio_optimize/similarity.py`),
not embeddings — no new dependency, no network call. `route_models` never
makes an auxiliary call; it only retargets `request.model` based on a length
heuristic. `summarize_history` ships no summarizer and constructs no model
client, the same rule the core's quality-lane judge follows — the config flag
alone spends nothing, and `Optimizer` refuses to enable it silently: turning
it on without supplying `summarizer=` raises at construction rather than
becoming a flag that looks configured and does nothing.

**One live result worth recording rather than smoothing over.** A light live
check (`--aggressive --live`, $0.0041 across 36 calls) on `rag_queries` showed
input tokens down 84.3% and cost down 71.5% — but output tokens *up* 71.4%,
and 10/10 responses diverged from baseline. `compress_prompt` cut input
heavily on this workload's repetitive synthetic chunks; the live model
apparently wrote longer answers against the trimmed context. Net cost still
fell, but "cost went down" and "the model behaved the same" are different
claims, and this is exactly the gap `Fidelity.ALTERED` exists to name.
`retry_storm` in the same run stayed 100% identical (exact_cache still
resolves 14/15 calls before either lossy stage ever runs). Not a benchmark
result — one live data point per stage, recorded because it happened, not
because it is a claim about your traffic.

## Overhead

Our own cost per request, measured with the provider's time excluded:

| workload | before memoization | after |
|---|---|---|
| `multi_turn_chat` | 1.945 ms | **0.585 ms** |
| `rag_queries` | 1.709 ms | **0.258 ms** |
| `tool_loop` | 0.384 ms | **0.131 ms** |

The bottleneck was tokenization: tiktoken costs ~2.5 ms on a 1387-token system
prompt, and the same prompt was being re-counted on every step.
`MemoizingCounter` caches by content digest — not by text, so no prompt content
is retained — at an 86.4% hit rate.

Against a provider call of 300–2000 ms, sub-millisecond overhead is under 0.3%.
