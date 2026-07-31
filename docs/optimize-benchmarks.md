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
| `multi_turn_chat_long` (50-turn `trim_history`) | 2026-07-29 | $0.0163 |
| Real Agents SDK agent (`scripts/real_agent_run.py`, 4 scenarios) | 2026-07-29 | ~$0.013 |
| Batch dispatch (OpenAI, submission + retrieval) | 2026-07-29 | ~$0.0005 |
| Anthropic `prefix_cache`, isolated (`claude-haiku-4-5`) | 2026-07-30 | ~$0.12 |

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

## Does `trim_history`'s win hold at scale? (`multi_turn_chat_long`, 50 turns)

12 turns is short enough that an 8.4% cost win could plausibly be a
small-scale artifact -- ADR-013's own reasoning for why trimming can help or
hurt is a *scale* argument (how much the untrimmed baseline's automatic-cache
discount has grown versus how much smaller the trimmed prompt is), and
IMPLEMENTATION.md's problem statement describes agentic workloads running
5-30x longer than single-shot chat, a regime 12 turns doesn't reach.
`multi_turn_chat_long` reruns the same shape at 50 turns to check the trend,
not to replace the 12-turn figure.

| turns | input ↓ | output ↓ | cost ↓ | quality |
|---|---|---|---|---|
| 12 | 7.3% | 35.0% | **8.4%** | 25% identical |
| 50 | 34.1% | 26.8% | **26.4%** | 48% identical |

**The win compounds, it doesn't plateau or reverse.** Cost reduction more
than tripled going from 12 to 50 turns. This is the opposite of what the
original simulated regression would have predicted extrapolated to more
turns -- consistent with the 12-turn live correction, not a new surprise, but
worth confirming rather than assuming the 12-turn number generalizes.
Mechanically: the untrimmed baseline's cost grows quadratically with
conversation length (`_multi_turn_chat`'s own docstring), while the trimmed
arm's per-call prompt size stays roughly constant once the conversation
exceeds `recent_turns` -- so the *relative* saving widens every additional
turn, on top of the output-length effect already measured at 12 turns. Live
against `gpt-4o-mini`, $0.0163 across 100 calls.

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

> **Resolved.** Both halves of that paragraph turned out to be right about the
> numbers and wrong about the cause. The anomaly is real and reproduces
> exactly with `compress_prompt` isolated — but the longer answers are not the
> model "writing longer answers against the trimmed context", they are short
> refusals being replaced by full sentences, and the guess that it was
> hedging was wrong. See *"`compress_prompt`: the anomaly, chased across six
> workloads"* below.

## `semantic_cache`: the measured false-positive rate (ADR-015)

**The number this section exists for: at the shipped default threshold of
0.97, `semantic_cache` served a wrong answer to 85.7% of the adversarial
probes that reached it.** Not a simulation, not an inference from quality
scores — measured live against `gpt-4o-mini` on 2026-07-29, `$0.0078` across
six threshold runs totalling 247 calls, via
`python -m optio_optimize.bench --semantic-cache-audit --live`.

The workload (`bench/adversarial.py`) is eight near-duplicate pairs sharing a
~100-word support-policy context and differing in exactly one embedded fact,
across the four shapes a word-overlap metric is structurally blind to — a
changed number, a changed entity, a negation, a changed date — plus eight
**controls**: the same questions reworded over an identical context, where a
cache hit is the win rather than the failure. Both halves matter, and running
only one of them is how you get a comfortable answer:

| `semantic_threshold` | false positives (wrong answers served) | benign hits (legitimate reuse) |
|---|---|---|
| 0.90 (`OptimizeConfig`'s floor) | **100.0%** (7/7) | 100.0% (8/8) |
| 0.95 | **100.0%** (7/7) | 75.0% (6/8) |
| **0.97 (shipped default)** | **85.7%** (6/7) | 37.5% (3/8) |
| 0.98 | 14.3% (1/7) | **0.0%** (0/8) |
| 0.99 | 0.0% (0/7) | **0.0%** (0/8) |
| 1.00 | 0.0% (0/7) | **0.0%** (0/8) |

**There is no setting on this workload where the stage is both safe and
useful.** Legitimate reuse reaches zero at 0.98, where wrong answers are still
being served; by the time false positives reach zero the stage has stopped
firing entirely, and the byte-identical case it is then reduced to is already
covered by `exact_cache` — losslessly, on by default, for free. A threshold
sweep against only the adversarial set would have shown 0.99 as a clean fix.
It is not a fix; it is an off switch with extra steps.

One adversarial probe is excluded from every row as **degenerate**: the model
gave the same answer to both halves (both "within four business hours"), so a
hit there is not demonstrably wrong and counting it as one would have inflated
the headline. 6/7 and 7/7 are the honest denominators.

What a false positive actually looked like, verbatim, at the shipped default
(similarity 0.9888 — the highest in the set):

> asked: "Are weekend escalations covered under this plan?" against a context
> reading *"Weekend escalations are **not** covered under this plan at no
> extra cost."*
> served: *"Yes, weekend escalations are covered under this plan at no extra
> cost."*
> correct: *"No, weekend escalations are not covered under this plan at no
> extra cost."*

The negation is the worst case and the clearest one: `not` is a single word in
a ~100-word prompt, so removing it moves Jaccard similarity by almost nothing
while inverting the answer. That is not a tuning problem. A word-set metric
has no representation in which "covered" and "not covered" are far apart, so
no threshold expressible in that metric can separate them — which is why the
resolution recorded in ADR-015 is a documented guardrail rather than a number.

**The single miss at 0.97 shows the mechanism from the other side.** The
`us-east-1` vs `eu-west-1` pair scored 0.9545 and correctly declined — not
because the metric understood that a region changed, but because
`similarity.words()` splits on `[a-z0-9]+`, so those two identifiers
contribute *four* differing tokens (`us`/`east`/`eu`/`west`) instead of one.
It passed for a reason that has nothing to do with meaning, and a workload
using `region A`/`region B` would have collided like all the rest.

**What this does not say.** One workload shape (long retrieved context, one
changed detail), one model, one similarity function. It is the *production*
shape for a RAG pipeline, which is why it was chosen, but a caller whose
prompts are short and lexically diverse will see different numbers — the
audit is checked in and takes `pairs=` so they can measure their own rather
than inherit these. And every number here is a property of the **default
lexical** `similarity_fn`; an embedding-based one is a constructor argument
(`Optimizer(similarity_fn=...)`) and is explicitly out of scope of this
measurement, not implicated by it.

## The divergence floor: what "10/10 diverged" was missing

Every live divergence number this document has ever printed — including the
`compress_prompt` result below and the `--aggressive` one above — was measured
against an assumption nobody had checked: that `temperature=0` makes the
provider deterministic, so any difference between the two arms is the
optimizer's doing. **It does not, and some of it was not.**

`python -m optio_optimize.bench --control --live` runs each workload twice with
the optimizer *off on both arms* and reports how many responses differ anyway.
Measured against `gpt-4o-mini` on 2026-07-29, $0.0179 across 152 calls:

| workload | responses differing with byte-identical prompts |
|---|---|
| `rag_queries` | 0 / 10 |
| `rag_queries_noisy` | 0 / 10 |
| `tool_calling_chat` | 0 / 20 |
| `fan_out` | 0 / 12 |
| `multi_turn_chat` | 1 / 12 |
| `unique_questions` | **4 / 12** |

So the floor is workload-dependent and mostly zero — but not always, and
`unique_questions` reproduced 4/12, 5/12, 4/12 across three runs. Any result
at or under its own workload's floor has measured nothing. This is why the
`compress_prompt` table below carries a floor column: without it,
`multi_turn_chat`'s "1 of 12 diverged" reads as a small quality effect, and it
is not an effect at all.

## `compress_prompt`: the anomaly, chased across six workloads (ADR-015)

The one prior data point — cost −71.5%, **output tokens +71.4%**, 10/10
diverged, on `rag_queries` under the bundled `--aggressive` flag — is
reproduced exactly, now with `semantic_cache` off and every default-on stage
off (`--stage compress_prompt --isolate`). It is real, it is
`compress_prompt`'s alone, and it is **not** general: it belongs to one
workload shape, and the mechanism is not the one ADR-015 guessed.

Live against `gpt-4o-mini`, 2026-07-29, $0.0138 across 181 calls:

| workload | input tokens | cost | output tokens | diverged | floor | judge |
|---|---|---|---|---|---|---|
| `rag_queries` | −84.3% | −65.5% | **+71.4%** | 10/10 | 0/10 | 4 ok, **6 worse** |
| `rag_queries_noisy` | −81.6% | −61.6% | **+71.4%** | 10/10 | 0/10 | 4 ok, **6 worse** |
| `multi_turn_chat` | −73.4% | −51.7% | +5.0% | 1/12 | 1/12 | — (at floor) |
| `tool_calling_chat` | −68.9% | −41.5% | 0.0% | 0/20 | 0/20 | — |
| `fan_out` | −82.9% | −67.8% | 0.0% | 0/12 | 0/12 | — |
| `unique_questions` | 0.0% | −0.3% | +0.4% | 2–5/12 | 4/12 | — (at floor) |

**On four of six workloads `compress_prompt` is close to free money.**
`fan_out` gave up 82.9% of its input tokens and 67.8% of its cost for
byte-identical output on all 12 responses, against a measured floor of zero.
`tool_calling_chat` did the same at 68.9%/41.5% across 20 responses.
`unique_questions` correctly saved nothing at all — the stage declined on every
request, which is the right answer for prompts with no internal repetition, and
is also what made that workload an accidental control.

**On the two RAG workloads it causes a real regression, and the diverged pairs
say what kind.** Read directly (`--show-divergences`), all six judge-WORSE
pairs on each workload have the same shape:

> baseline: `'INSUFFICIENT CONTEXT.'`
> optimized: `'Subscription growth in the enterprise segment drove revenue in Q5.'`

The workload's system prompt says *"Answer using only the context provided. If
the context does not contain the answer, say exactly: INSUFFICIENT CONTEXT.
Never speculate."* The retrieved chunk says revenue was *"driven primarily by
subscription growth in the enterprise segment"* — for *"the period"*, naming no
quarter. The baseline honours the instruction and refuses. The compressed arm
answers, attributing the driver to a specific quarter the context never
establishes, taking the quarter number from the question. That is speculation,
which the prompt explicitly forbids. **The output-token increase is entirely
this**: a 4-token refusal replaced by a 13-token sentence, not the hedging or
over-explaining ADR-015 hypothesised.

**The mechanism, confirmed by inspecting the transformed prompt.** These
workloads' system prompt is `_SYSTEM_PROMPT * 9` — the grounding instruction
is stated nine times. `CompressPromptStage` drops near-duplicate sentences, so
it collapses the system message from 6,408 to 916 characters: `"INSUFFICIENT
CONTEXT"` goes from **9 occurrences to 1**, `"Never speculate"` from 9 to 1.
The user message's 8 near-identical retrieved chunks collapse to 1 at the same
time. No information is *deleted* in either case — every distinct sentence
survives — but the instruction's **salience through repetition** does not, and
that turns out to be what was holding the refusal behaviour in place.

**Which is why the same collapse is harmless on `fan_out` and
`tool_calling_chat`.** They carry the identical 9×-repeated system prompt and
get the identical 9→1 collapse, with zero divergence — because their tasks
never reach the conditional-refusal branch. The answer is present in the
prompt, so no instruction about what to do when it is absent can matter. The
regression is not "compression breaks instructions"; it is narrower and more
specific: **compression can strip the redundancy that a conditional
instruction was relying on for weight, and that only shows up on requests
that actually exercise that condition.**

**What this does not say.** Six workloads, one model, one similarity metric.
The two that regressed are both synthetic RAG shapes built from the same chunk
text, so they are closer to one and a half data points than two — they agree
because they are similar, which is weaker evidence than two independent
workloads agreeing. Costs also shifted between repeat runs of the same
workload (`rag_queries` baseline priced $0.00166 then $0.00137), which is
OpenAI's server-side prefix cache warming between runs; `BenchProvider.reset()`
already documents that a live provider cannot honour a cache reset, and this is
that limit showing up in a price.

## `route_models`: the first live evidence it has ever had (ADR-015)

This stage had **zero** live data before this. It was excluded even from the
old `--aggressive` flag, because the bench CLI had no cheaper model to route
to — and the A/B harness could not have measured it anyway: `ABResult` prices a
whole arm at one flat rate, so a run where *some* requests were downgraded
reports a blended figure reflecting neither model, and a routed request is
never expected to be output-identical, so the identity check the rest of this
suite leans on does not apply at all.

So the instrument is different: `--route-models-audit` asks **both** models the
same twelve short questions and grades them against **known answers**, not a
judge. That is deliberate — a judge is itself a model, and using one to decide
whether a weaker model is good enough puts the capability question inside the
thing being measured.

Live, `gpt-4o` vs `gpt-4o-mini` (16.7× cheaper input), 2026-07-29, $0.0013
across 24 calls, reproduced identically three times:

| probe category | `gpt-4o` | `gpt-4o-mini` |
|---|---|---|
| easy (4 lookups) | 100% | 100% |
| hard (8 short-but-hard) | 100% | **88%** |

**Regression rate: 8.3% (1 of 12).** The single failure is the one that
matters, because it is the exact shape the risk model predicts:

> **"What is 17 times 24, minus 89?"** — eight words, comfortably inside the
> 500-token routing ceiling, no tools, no `response_format`. `gpt-4o` answers
> `319`. `gpt-4o-mini` answers `329`. Fluent, confident, and wrong by ten.

Nothing about that response is distinguishable from a correct one without
knowing the answer. No cost or latency dashboard can see it, and the
`ALTERED` tier exists precisely because no identity check can either.

**The first probe set was too easy, and saying so is part of the result.** It
used four famous reasoning traps — the strawberry letter count, 9.11 vs 9.9,
bat-and-ball, transitive ordering — and `gpt-4o-mini` answered **all four
correctly**, in every run. Those are in everyone's training data by now;
passing them says nothing about a *novel* short-but-hard request. Four
ordinary multi-step problems with no memorable phrasing were added, and one of
those four is what produced the only regression. A probe set of only-easy
requests — or only-famous hard ones — would have cleared this stage at 0%
regression and proved nothing, which is the failure ADR-015 explicitly warned
about for this stage.

**All five decline guards hold live**, checked against real request shapes on
every audit run rather than trusted from the eval gate: tools attached,
`response_format` set, already-cheap, over the token ceiling, and — the one
that would silently disable the stage if it broke — a routable request
actually being routed.

**What this does not say.** Twelve probes on one model pair is a small sample,
and 1/12 is one event: the honest reading of 8.3% is "not zero", not "8.3%".
The probes are also all single-turn and answer-checkable by construction,
which excludes the open-ended requests where a weaker model more plausibly
degrades in ways no string match would catch. This is a floor on the risk, not
a measurement of it.

## `summarize_history`: it works, and it still costs more than doing nothing

This stage also had **zero** live data, by design: `Optimizer` refuses to
enable it without a caller-supplied summarizer, and the bench CLI supplied
none — correctly, because a stub summarizer would have measured the stub.
`--recall-audit` supplies a real one (a live `gpt-4o-mini` call) and runs a
conversation with four load-bearing facts planted in the *first* exchange —
a budget, a date, a decision, a compliance constraint — then eight filler
exchanges to push them out of the `recent_turns=6` window, then asks each fact
back. Three arms on identical requests, live 2026-07-29, $0.0015 across 30
calls, reproduced four times:

| arm | recalled | silently wrong | prompt tok | + summarizer | **total tok** |
|---|---|---|---|---|---|
| `full` (control) | 100% (4/4) | 0% | 466 | 0 | **466** |
| `trim_history` | **0% (0/4)** | 0% | 165 | 0 | **165** |
| `summarize_history` | **100% (4/4)** | 0% | 261 | 361 | **622** |

**The stage does exactly what it claims.** Every fact `trim_history` lost, the
summary preserved — 4 out of 4, verbatim in three cases and correctly
paraphrased in the fourth. And it preserved them *accurately*: the silent-error
column is zero, which is the column that matters most here, because a summary
that misstates a budget figure is worse than one that drops it. `trim_history`'s
four failures were all visible ones — it answered `NOT IN CONTEXT.` every time,
which is the honest failure a caller can react to.

**And it is still the wrong trade on this workload, because of the last
column.** Summarizing spent **622 tokens to reach the same answer that sending
the whole conversation reached for 466**. The prompt genuinely did get smaller
(261 vs 466) — that is real, and it is the number a report showing only prompt
tokens would have stopped at. It is also not the cost. The summarizer call
reads all the dropped history and writes a summary, and that is 361 tokens
nobody was spending before.

**The reason is structural, not a tuning problem: the stage has no summary
cache.** `SummarizeHistoryStage.before()` calls the summarizer unconditionally
on every request — there is no memoization keyed on the dropped history, and
the audit confirms it: 4 probes, 4 summarizer calls. So the summarizer re-reads
the same aged-out turns on every single turn of a conversation. That makes its
token cost scale with conversation length *just like the full history does*,
which means the bounded-prompt advantage can never catch up. A summary computed
once and reused across turns would change this arithmetic completely; the stage
as shipped does not do that.

**What this does not say.** One conversation shape, one length, one summarizer
prompt, one model. In particular the crossover this measurement implies — that
a cached summary would win, and an uncached one cannot — is an argument from
the mechanism, not something measured here, and a longer conversation was not
tried. What *is* measured is that at `recent_turns=6` on an 18-message history,
paying for a summary lost to simply not optimizing at all.

**The tool-call boundary holds**, checked live on every audit run across every
cut point of a synthetic tool-calling conversation: no `tool` result was ever
separated from the assistant message that called it. This is the invariant the
stage shares with `trim_history`, which `tool_calling_chat` confirmed for
trimming — now confirmed for summarizing rather than inherited from it.

## The 2026-07-29 wave: four stages measured, three verdicts against them

Live `gpt-4o-mini`, **$0.0654 across 336 calls** for the main sweep plus ~$0.04 of
probes. Every number below is from a real API call.

### The harness was not sending tool schemas at all

Before any of it meant anything, the live run of `mcp_agent` had to be thrown
away once. `OpenAIProvider` forwarded messages, `max_tokens`, `temperature` and
`response_format` — and not `tools`. So the workload built specifically to
measure tool cost sent **zero tools**, and `minify_tools` reported saving 3,240
tokens while the provider billed byte-identical totals in both arms (76,439
either way). It did not fail; it measured nothing and said so confidently.

This is the same failure the adapter's own comments already describe one field
over: `tool_calls`/`tool_call_id` were dropped by the first version of that
method and found live. A missing field in a provider adapter does not raise —
it quietly changes what you are measuring.

### Our tool-token estimate overstated the provider's by 1.5×

With schemas actually being sent, a second problem surfaced: providers do not
bill the JSON you hand them. They re-render tool schemas into a compact internal
form, so counting serialized JSON overstates. Measured by differencing against a
no-tools call:

| tools | our JSON | OpenAI billed | ratio | our Δ from stripping | real Δ | ratio |
|---|---|---|---|---|---|---|
| 1 | 138 | 103 | 1.34 | 32 | 12 | 0.375 |
| 3 | 415 | 279 | 1.49 | 95 | 35 | 0.368 |
| 5 | 695 | 458 | 1.52 | 160 | 60 | 0.375 |
| 10 | 1395 | 898 | 1.55 | 324 | 121 | 0.373 |

**Two different corrections, and they cannot be merged.** `0.65` calibrates the
*total* nearly exactly (0.65 × 1395 = 907 against 898 billed). But the keys
`minify_tools` strips are unusually punctuation-heavy — `"title": "Record Id"`
is mostly quoting a provider's renderer never emits — so their removal shrinks
the real bill by only `0.37` of the JSON difference.

With both in place the stage claims **1,190** tokens on `mcp_agent` where the
provider stopped billing **1,210**: 1.7% out, and understating. Uncorrected it
claimed 3,240.

### `concision` loses on both workloads, and the literature's 30–50% did not appear

| workload | input | output | cost | identical |
|---|---|---|---|---|
| `multi_turn_chat` | **−1.1%** | +4.5% | **−4.0%** | 11/12 |
| `unique_questions` | **−110.5%** | +0.4% | **−21.3%** | 2/12 |

Output fell 4.5%, not 30–50%, and the one-sentence instruction cost more than
that saved. On short prompts it more than doubles the input. **Stays off**, now
on evidence rather than on caution.

### The timestamp bug, priced against real OpenAI usage numbers

| workload | input | provider-cached |
|---|---|---|
| `multi_turn_chat` | 19,242 | **16,768** (87.1%) |
| `timestamped_agent` | 19,392 | **0** |

One line apart. `detect_unstable_prefix` fired on `timestamped_agent` and stayed
silent on `multi_turn_chat`. This is also the first time the simulator's
automatic-cache model has *agreed* with live data (16,128 simulated vs 16,768
measured).

### append-then-compact loses on cost and wins on fidelity

50 turns, `multi_turn_chat_long`:

| mode | input | output | cost | identical |
|---|---|---|---|---|
| slide every turn | 34.1% | 26.8% | **27.1%** | 24/50 |
| compact @ 2200 tokens | 26.6% | 8.2% | **25.1%** | **42/50** |

**The published guidance did not reproduce.** Sliding every turn is 2 points
cheaper live, not more expensive — and the simulation had said the opposite by a
wide margin, for the third time in this project's history and in the same
direction. What compaction actually buys is fidelity: 42 of 50 responses
unchanged against 24 of 50, because it keeps more context between cuts. That
reframes the option as a cost-versus-fidelity dial rather than a free win, which
is not how it is usually presented.

## Fixing what the measurements found

Two of the three losses above turned out to be mechanical rather than inherent.

### `concision`: the instruction was evicting more than it cost

The stage appended its sentence to the **system prompt** — the exact region a
provider's prefix cache covers. That does not merely add the instruction's
tokens; it shifts everything below the edit out of the provider's 128-token
block alignment, so already-cached content falls back to full rate.

`multi_turn_chat`, 12 requests, full-rate input tokens:

| placement | input | cached | **full-rate** |
|---|---|---|---|
| baseline (stage off) | 19,242 | 16,640 | **2,602** |
| appended to system prompt | 19,446 | 16,384 | **3,062** |
| appended to last message | 19,446 | 17,152 | **2,294** |

The instruction is 204 tokens either way. On the system prompt it moved **460**
tokens into the full-rate column, because 256 tokens of already-cached prompt
were evicted — the eviction cost more than the instruction did. On the tail
message it evicts nothing, because the last message was never cacheable.

**The stage is now roughly cost-neutral where it has nothing to do, which is the
correct behaviour and is not evidence that it helps.** It stays off: no workload
here produces a padded reply, so the case it exists for remains unmeasured.

### `trim_history`: a front cut discards the cheapest and most valuable context

Two earlier measurements point the same way and neither is obvious alone:

* The provider's cached region is `system + oldest turns` — it matches the
  longest common prefix, and the oldest turns are the part that never changes.
  87% of `multi_turn_chat`'s prompt was served from cache before trimming
  touched it.
* The recall audit found load-bearing facts — a budget, a deadline, a decision —
  stated in the *first* exchange and never repeated. `trim_history` recovered
  **0 of 4**.

So a front cut throws away the tokens that were already billing at half rate
*and* the ones most likely to matter, in one move. `anchor_turns` keeps both
ends and takes the middle instead. Live, `multi_turn_chat_long` at 50 turns:

| mode | input ↓ | cost ↓ | identical replies |
|---|---|---|---|
| slide (`anchor_turns=0`) | 34.1% | **26.3%** | 25/50 |
| anchored (`anchor_turns=2`) | 31.7% | 16.8% | **50/50** |

**Anchoring trades about nine points of saving for every reply matching the
unoptimized baseline.** It does not make trimming free — it converts a quality
loss into a smaller, visible cost. Off by default, because changing what every
existing caller sends on one good measurement of one workload is what ADR-016
exists to prevent.

## A real agent, and the bug it found in ninety seconds

Everything above this line is a *workload*: a hand-authored message list driven
straight into `Optimizer.call`. The adapter tests go further — a genuine
`AsyncOpenAI` client with a mocked HTTP transport — and `tests/frameworks/` says
in its own docstring that *"nothing here calls a model."*

None of that is an agent. An agent chooses a tool, receives a payload whose size
it did not choose, appends it to a history it then re-sends, and repeats. That
loop is where these stages either earn their keep or break something, and until
2026-07-29 nothing in this repo had ever run one. `scripts/real_agent_run.py`
does: a real OpenAI Agents SDK agent, four real tools returning real JSON, one
support task needing all four in sequence.

It broke on the first run.

### `trim_history` was dropping the task

In a chat, the first user turn is an old question that has been answered and
superseded. **In an agent loop it is the task**, and every message after it is
the agent's own tool traffic. A sliding window whose oldest entry is the task
therefore removes the only statement of what the model is supposed to do — and
providers accept a conversation with no user message at all, so it fails
silently. Dumped from the fourth call, as sent:

```
system      308 chars
assistant     0 chars  <-- TOOL_CALLS
tool        911 chars
assistant     0 chars  <-- TOOL_CALLS
tool       7151 chars
assistant     0 chars  <-- TOOL_CALLS
tool         55 chars
tool          73 chars
```

No user message anywhere. The model inferred a task from the evidence and
answered a question nobody asked — a markdown table of order fields instead of
the two sentences requested, cut off mid-word.

Isolated one stage at a time (ADR-015 rule 2):

| configuration | input | output | cost | answer |
|---|---|---|---|---|
| unoptimized | 7,717 | 140 | $0.00124 | correct, two sentences |
| defaults | 3,757 | 288 | $0.00074 | **wrong** — field dump, question unanswered |
| `structured_output` off | 3,718 | 290 | $0.00073 | still wrong |
| `trim_history` off | 3,816 | 142 | **$0.00066** | correct, two sentences |

Disabling `structured_output` changed nothing; disabling `trim_history` fixed
it. **The stage that broke the answer also made the run more expensive**, because
a model that has lost the question writes longer: output more than doubled.

The fix is structural rather than a default change. The system prompt was
already exempt from trimming because `PrefixCacheStage` needs it; the first user
turn is now exempt for a stronger reason, as a floor that `anchor_turns=0` does
not lower. After the fix, on the same task: **3,816 in / 131 out / $0.00065,
correct answer, 47.1% cheaper than unoptimized.**

### Two cautions the same runs produced

**Provider cache state makes a live A/B unstable.** Across four runs of the same
command the baseline's cache-served input was 0, then 6,784 of 7,717, then 0
again. In the run where the baseline was 88% cache-served, our measured advantage
fell from 47.1% to **11.4%** — not because the optimizer got worse, but because
rewriting a prompt forfeits a discount the untrimmed baseline was already
earning. `bench/providers.py` already notes that a server-side cache cannot be
reset between arms; this quantifies what that costs a single-run comparison.

**A one-run answer comparison measures the model too.** In the final run the
*unoptimized* arm gave a wrong refund figure (80% rather than 100%), because the
model passed a `reason` string its own tool did not match. So "the optimized
answer differs" is not by itself evidence of a library defect, and the divergence
counts elsewhere in this document are worth reading with that in mind.

### What did not break

Worth recording, because these were the live-fire risks:

- `cap_tool_results` truncated a 7,151-character JSON payload mid-document on
  every run and the agent still completed the task correctly each time. The
  truncation notice did its job; no fixture in this repo had ever handed that
  stage real JSON.
- `minify_tools` handled schemas generated by the SDK's own `@function_tool`
  decorator without incident.
- The adapter round-tripped real `tool_calls`/`tool_call_id` pairs through
  three- and four-step loops.

## `prefix_cache` on Anthropic: the first measurement it has ever had

`PrefixCacheStage` describes itself as the largest lossless saving in the
package, and until 2026-07-30 that rested on a vendor's published discount
rather than a run. On OpenAI the stage is worth nothing — automatic caching
lands on both A/B arms, which is how a simulated 36.3% became a live −1.8%. So
Anthropic is the only place it can pay, and until `wrap_anthropic_client`
shipped there was no way for a user to reach it outside the benchmark module.

Measured live on `claude-haiku-4-5`, six-turn conversation over a stable
system prompt, `prefix_cache` isolated with every other stage off (ADR-015
rule 2), disabled arm first so residual server-side cache favours the baseline:

| arm | input | cache-served | cache writes | output | cost |
|---|---|---|---|---|---|
| `prefix_cache` off | 30,111 | 0 | 0 | 1,623 | $0.03823 |
| `prefix_cache` on | 30,113 | **23,023** (76.5%) | 5,487 | 1,662 | **$0.01907** |

**50.1% cost reduction, on identical token counts.** The two arms sent the same
prompt — 30,111 against 30,113, a two-token difference that is noise — so the
stage avoided no tokens whatsoever and cut the bill in half purely by changing
the price of tokens that were sent anyway. That is the mechanism the stage has
always claimed, measured for the first time.

It also **beats its own docstring**, which said "roughly 30% of total spend on
a long conversation". The claim has been corrected upward and now cites this
run. Worth noting that the correction runs the opposite way to every other one
in this document, all of which took a number down.

### This number was published as 53.7% first, and that was an accounting bug

The token counts above never changed. What changed is what they cost.

Anthropic charges a **premium** to write into its cache — 1.25× the base input
rate for the 5-minute TTL this package requests, 2× for the one-hour it does
not — and `ModelPricing` had no write rate at all. So the 5,487 write tokens
were billed at 1.00× in this table's first version, which put the `on` arm at
$0.01770 and the reduction at 53.7%.

Two distinct defects produced that, and both leaned the same way:

- `wire.response_from_anthropic_message` omitted `cache_creation_input_tokens`
  from the prompt total entirely. Anthropic reports `input_tokens` excluding
  *both* reads and writes; only reads were added back. On this run's first turn
  the library reported 200 prompt tokens against a true 4,805.
- `ModelPricing` had no `cache_write_usd_per_m`, so even once counted, writes
  priced at the base rate.

Neither error could ever make a report look wrong, because both understate the
cost of a *cached* call — that is, both inflate whatever saving `prefix_cache`
appears to deliver. Cache reads are the only prompt-token band this document had
ever modelled as differently-priced, and they are the cheap one. Fixed in both
places, with the arithmetic behind 50.1% now locked by a test.

### The measurement failed once first, and the failure is the interesting part

The first attempt reported **zero cache reads in both arms** and a 0.4% cost
difference — which reads exactly like "the stage does nothing". It was not a
negative result. It was no test at all.

The system prompt was 1,449 tokens, above `MIN_PREFIX_TOKENS = 1024` and below
the model's real floor. The marker was placed, was reported in the savings
ledger, went out on the wire correctly — and the provider ignored it.
Lengthening the prompt produced the table above.

**The floor is per-model, and one constant cannot express it.** As of
2026-07-30 Anthropic's published minimums span a factor of eight:

| model | minimum cacheable prefix |
|---|---|
| Opus 5 | 512 |
| Opus 4.8, Sonnet 5, Sonnet 4.6 / 4.5 | 1,024 |
| Opus 4.7, Haiku 3.5 | 2,048 |
| **Haiku 4.5**, Opus 4.6 / 4.5 | **4,096** |

So `MIN_PREFIX_TOKENS = 1024` is wrong in *both* directions: too high for Opus
5, where it declines a breakpoint that would have worked, and too low for four
models, where it places one the provider discards while the stage's note claims
success. This section originally recorded Haiku 4.5's floor as 2,048 — that is
Haiku *3.5*'s, and the wrong figure shipped in a public docstring for one
commit. The ~4,600-token prompt used above clears 4,096, so the measurement
stands; it cleared the real floor by luck rather than by design.

Left as one constant deliberately: a per-model table changes what every
Anthropic caller sends, and ADR-016 does not let one measurement carry that. The
cost of being wrong is a marker that does nothing, never a wrong answer.

Two consequences worth carrying:

- **A zero-read result must be diagnosed, not believed.** The script now prints
  `cache_creation_input_tokens` separately, because reads-0/writes-0 means the
  breakpoint was ignored while reads-0/writes-positive means the prefix is
  changing between calls. Those are different bugs and the totals cannot tell
  them apart.
- **`MIN_PREFIX_TOKENS` was wrong for Haiku-class models. Fixed 2026-07-31 by
  ADR-027**, which makes the floor a per-model lookup and leaves the old constant
  as the unknown-model fallback. What forced it: `AnthropicProvider`'s default
  became `claude-haiku-4-5` (floor 4,096), and **eleven of twelve workloads have
  prompts below that**, so the full live run reported zero cache reads everywhere
  with nothing explaining why.

  The stage now declines below the floor and names the model and the figure, so
  a zero-read result is diagnosable from the report. Running that against
  `mcp_agent` — the one workload whose *prompt* clears 4,096, at 11,799 tokens —
  produced the finding this table could not have shown:

  > `prefix is ~1387 tokens, below claude-haiku-4-5's 4096-token cacheable minimum`

  **The stable prefix is ~1,400 tokens, not 11,799.** The rest is tool results
  that change every step and were never cacheable. So under the old constant this
  prefix cleared 1,024, got a marker, and had it discarded — "work done for no
  effect" reported as success, on the workload the suite is proudest of.

  The practical consequence, worth stating for anyone choosing a model: this
  package's largest lossless lever needs a *stable* prefix above the model's
  floor, and typical agent traffic carries 1,000–2,000 tokens of stable head. That
  clears Opus 5 (512) comfortably, clears the 1,024 tier often, and **rarely
  clears Haiku 4.5's 4,096** unless the system prompt and tool schemas are large.
  The suite currently cannot demonstrate `prefix_cache` on its own default model —
  recorded as the next open item rather than papered over.

## `prefix_cache` end to end, and the three bugs between us and seeing it

**That open item is closed.** `multi_turn_chat` on `claude-sonnet-4-5` (floor
1,024, prompts 1,406–1,769 tokens), 2026-07-31, both arms live:

| | baseline | optimized | |
|---|---|---|---|
| input tokens | 20,610 | 20,610 | **0.0%** |
| cost | $0.06548 | $0.01672 | **74.5%** |
| provider cache | reads 0, writes 0 | **reads 18,300, writes 1,872** | |
| latency (end-to-end) | — | — | **−26.4%** |
| overhead | — | — | +0.97 ms / request |
| output quality | — | — | 91.7% identical |

**The input token count does not move, and that is the point.** Prefix caching
changes the *rate* those tokens are billed at, not the *volume*. A suite
reporting only token reduction scores this workload at 0.0% — which is why
`cost_reduction` exists as a separate figure and why the provider-cache line is
now printed.

Getting here took three defects off the path, each hiding the next, and all
three ran in the direction that flatters the library.

**`--model` never reached the stages.** `Workload.requests()` called `build()`
with no arguments, so every workload built its requests as `gpt-4o` whatever
`--model` said — the flag reached the provider and the pricing row and never
`LLMRequest.model`. So `min_prefix_tokens_for` read `gpt-4o` on every live
Anthropic run and returned the 1,024 fallback, leaving ADR-027's per-model floor
**inert inside the benchmark written to validate it**. A live Haiku run duly
placed the breakpoint and came back `reads 0 writes 0`: the exact failure ADR-027
exists to prevent, on the exact model it describes it for. It hid because
gpt-4o's fallback equals Sonnet 4.5's real floor — the one model that conceals
the bug is the one on which the stage appeared to work.

**The provider never read `cache_creation_input_tokens`.** The first Sonnet run
reported `reads 18,300 writes 0`, which cannot happen: nothing is read from a
cache that was never written. Written tokens were dropped from `input_tokens`
outright, so the cache-write premium added to `ABResult.cost_usd` was inert —
nothing populated the field it prices.
`wire.response_from_anthropic_message` has been correct since ADR-021 and its
docstring records this same defect being fixed *there*; the streaming adapter
uses it; the benchmark had kept a private copy.

**So the first number was 85.2%, with a 9.1% input-token reduction beside it.**
Both were artefacts: the 9.1% was precisely the 1,872 written tokens missing
from the total. The true figure is 74.5%. Two accounting bugs in the same
direction cost 10.7 percentage points of overstatement — the third time this
document has had to record that pattern, after the 53.7%→50.1% correction above
and the ADR-021 write-rate omission.

## The full suite on Sonnet 4.5, once every number was attributable

2026-07-31, all twelve workloads, both arms live, `$1.61` for 344 calls. The first run where
`--model` reached the stages, cache writes were billed, and unattributable deltas were labelled
rather than printed as percentages.

| workload | calls | cost | quality | what did it |
|---|---|---|---|---|
| `retry_storm` | 15 → 1 | **97.0%** | 6.7% identical | `exact_cache` |
| `tool_loop` | 20 → 4 | **93.0%** | 35.0% identical | `exact_cache` |
| `multi_turn_chat_long` | 50 → 50 | **86.2%** | 38.0% identical | `trim_history` + prefix cache |
| `fan_out` | 12 → 4 | **84.0%** | 66.7% identical | `exact_cache` |
| `mcp_agent` | 10 → 10 | **84.0%** | **100% identical** | trim + cap + minify + prefix |
| `multi_turn_chat` | 12 → 12 | **82.2%** | **100% identical** | prefix cache alone |
| `tool_calling_chat` | 20 → 20 | **81.6%** | **100% identical** | prefix cache alone |
| `rag_queries` | 10 → 10 | **76.4%** | 90.0% identical | `deduplicate` + prefix cache |
| `rag_queries_noisy` | 10 → 10 | **75.9%** | 80.0% identical | `prune_retrieval` + prefix cache |
| `unique_questions` | 12 → 12 | *not attributable* | 75.0% identical | nothing, by design |
| `sampled_creative` | 8 → 8 | *not attributable* | not interpretable | nothing, by design |
| `timestamped_agent` | 12 → 12 | **−23.5%** | 100% identical | **a defect — see below** |

Three things worth reading off this table.

**Prefix caching carries workloads that reduce no tokens at all.** `multi_turn_chat`,
`tool_calling_chat` and `timestamped_agent` all show `input tokens 0.0%`. Two of them still save
above 80%, because the rate changed and not the volume. A benchmark reporting only token reduction
would score all three at zero.

**The two adversarial workloads now say "not attributable" instead of inventing a percentage.**
Before ADR-028 they reported +2.8% and −4.7% from pure output nondeterminism.

**And one workload cost 23.5% more than doing nothing.** `reads 0 writes 20,333`: a breakpoint
placed on every turn of a prompt whose head changes every turn, paying the 1.25× write premium
twelve times and collecting the 0.1× discount zero times. It is an ADR-013 rule 1 violation, on a
workload that replicates the most common prompt-caching mistake in production — and
`detect_unstable_prefix` had already printed the cause at the top of the same run. ADR-030 is the
fix; the measured sequence is −23.5% → −17.1% (digest guard) → **−6.0%** (provider-outcome guard),
with the residual being a bounded three wasted writes that amortize to 0.3% over 200 requests. See
that ADR for why it is not driven to zero.

## Batch dispatch: a weaker class of evidence, stated as such (ADR-017)

Every number above this line is an A/B run: the same workload twice, once with a
stage and once without, both arms billed by a real provider. **The batch
discount is not that**, and it cannot be, because the A/B harness is built
around a synchronous `compare()` and a batched arm returns hours later. So the
saving is *arithmetic* — the provider's published 50% applied to measured token
counts — and `BatchReport.summary_lines` prints that caveat as its last line
rather than leaving a reader to assume parity with the rest of this document.

What *was* measured live, on 2026-07-29 against OpenAI, is the part unit tests
against a fake client cannot reach: whether the real endpoint accepts the
envelope we build, and whether the pipeline halves reconnect across the gap.

Three one-word prompts, one answered synchronously first:

| check | result |
|---|---|
| envelope accepted by `/v1/batches` | yes, `batch_6a69b54c…` |
| the pre-answered request | **not submitted** — served from the shared exact cache |
| requests actually queued | 2 of 3 |
| results | `q-1` → `Tokyo`, `q-2` → `Lima`, 0 errors |
| repeat of `q-1` after retrieval | **0 provider calls** |

The last row is ADR-017 decision 3 working end to end: an answer that arrived
asynchronously ran its stages' `after` hooks on retrieval, populated the exact
cache, and was served to the synchronous path without a call. The two surfaces
share one cache because they were given the same `Optimizer`.

One incidental fact worth recording, since the ADR reasons throughout about a
24-hour window: **this batch completed in about 80 seconds.** The completion
window is a ceiling, not an estimate, and small batches can return in the time a
long synchronous call takes. That is not a basis for treating batch as
low-latency — a queue with no service guarantee is exactly the thing you must
not put a waiting user behind — but it does mean `await_results` is usable in a
smoke test rather than only in a cron job.

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
