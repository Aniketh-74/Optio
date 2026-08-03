# ADR-015 — Evidence bar for promoting an ALTERED-tier stage out of "experimental"

**Status:** Accepted — addendum 2026-08-03: evidence gathered live for all four stages; all four stay off
**Date:** 2026-07-29
**Related:** ADR-013, ADR-014, §9, `src/optio_optimize/eval/`

## Context

ADR-013 rule 3 requires lossy stages to pass a CI-blocking eval suite before shipping, and that
requirement is met: `route_models`, `compress_prompt`, `semantic_cache`, and `summarize_history`
all ship with `src/optio_optimize/eval/` coverage today. But a prior status pass surfaced that
"passes the eval gate" and "safe to recommend turning on" are not the same claim, and the gap
between them is uneven and, in three of four cases, essentially unmeasured:

- `compress_prompt` has exactly **one** live data point, and it is confounded: `--aggressive`
  turns on `compress_prompt` and `semantic_cache` together, so the one live result attributing
  `rag_queries`'s cost drop to `compress_prompt` is a guess, not a measurement. That result also
  showed output tokens *up* 71.4% with 10/10 responses diverging from baseline — a genuinely
  concerning signal that has never been chased.
- `semantic_cache` has **zero** live data of its own. It rode along in the same confounded run
  but never got a chance to fire, because `exact_cache` resolved almost everything first.
- `route_models` has **zero** live data, full stop — it is excluded even from `--aggressive`
  because the bench CLI has no representative `cheap_model` default to route to.
- `summarize_history` has **zero** live data *by design* — `Optimizer` refuses to enable it
  without a caller-supplied summarizer, and the bench CLI supplies none, correctly, because a
  stub summarizer would measure the stub.

The eval gate itself says as much about its own limits (`eval/__init__.py`): it is "deliberately
model-free" and proves a stage's *logic* does what it claims, never that a live answer stays
good. `route_models`'s docstring makes the same point from the stage side: "confirming the
answers stay acceptable needs a live A/B run with a judge... and turning this stage on at all is
the caller's judgment call to make, not this docstring's to settle." Nobody has made that
judgment call with real evidence yet for any of the four stages.

This ADR exists to write down what evidence would actually justify a judgment call, **before**
gathering it — the same reason `docs/optimize-benchmarks.md` insists every simulated number is a
hypothesis until `--live` confirms it, applied one level up: a plan for what confirmation means,
decided before the numbers exist, so the bar cannot quietly move to fit whatever result shows up.

## Decision

**Define, per stage, what evidence would justify changing its default-off status — and commit to
gathering that evidence through isolated live runs before any default changes.** No stage's
default moves based on eval-gate results alone, on a confounded multi-stage run, or on a single
data point. This ADR does not itself change any default; it defines the bar a later, evidence-
backed decision will be measured against, and that later decision is recorded as a resolution to
this ADR, not a silent config edit.

### Cross-stage rules (apply to all four)

1. **Evidence must be live, not simulated.** `SimulatedProvider` cannot see output-length or
   correctness effects at all (`docs/optimize-benchmarks.md` already demonstrates this twice —
   the `prefix_cache` 36.3%→0% correction and the `trim_history` 34.8%-cost-increase→8.4%-decrease
   correction). A stage whose entire risk is "the output might be different or wrong" cannot be
   cleared by a tool that cannot observe output.
2. **Evidence must be isolated.** One stage on at a time. `--aggressive`'s bundling is the defect
   that motivated this ADR and is being retired for exactly that reason (see the companion CLI
   change tracked alongside this document).
3. **Divergence must be characterized, not just counted.** `harness.compare()`'s `Judge` callback
   and the identical/equivalent/divergent breakdown exist for this. A raw "10/10 diverged" number
   answers "did the text change," never "did the answer get worse" — the second question is the
   one that matters and the one that requires either a judge or direct inspection of the diverged
   pairs.
4. **A stage whose failure mode is "silently wrong answer" needs a direct false-positive
   measurement, not an inferred one.** This applies specifically to `semantic_cache`: a dedicated
   adversarial workload (near-duplicate-but-meaningfully-different query pairs) and a live false-
   positive rate, is a first-class deliverable, not an afterthought discovered by reading
   quality-identity numbers after the fact.
5. **Sample sizes and spend are stated as real numbers**, matching every other claim in
   `docs/optimize-benchmarks.md` — "N of M calls, $X spent, on DATE" — not "tested live."
6. **The decision is per-stage.** No lane's evidence implies anything about another's — the same
   independence `stages/__init__.py`'s ordering rules and ADR-013's `Fidelity` split already
   assume. `compress_prompt` clearing its bar says nothing about `semantic_cache`.
7. **"Promoted" does not mean "flipped to `True`."** The available outcomes per stage are: stay
   off by default (evidence doesn't clear the bar, or the risk is inherent to the mechanism and no
   amount of evidence removes it), stay off but with a narrower documented recommendation (e.g.
   "safe to enable if your traffic looks like X"), or a guardrail change that makes the *existing*
   off-by-default stage safer without changing its default (e.g. a stricter threshold, a narrower
   applicability check). A change to the shipped default is the least likely outcome of this
   exercise, not the assumed one.

### Per-stage risk model and acceptance criteria

#### `route_models`

**Mechanism:** retargets `request.model` to `cheap_model` when the request is short (≤500
estimated tokens per `MAX_ROUTABLE_TOKENS`), carries no tools, and has no `response_format`.
Touches only `.model`; never reads or rewrites prompt content.

**What could go wrong in production:** length is a proxy for difficulty, not a measurement of it.
"What is the Riemann hypothesis?" is six words and genuinely hard; "Summarize this 400-word
support ticket" is longer and often easier. A short-but-hard request routed to a weaker model
degrades *silently* — the answer is still fluent and plausible, just wrong or shallower in a way
a monitoring dashboard tracking cost and latency cannot see, and a token-identity check cannot
see either, because the stage is `ALTERED` by design (the whole point is a different answer from
a cheaper model).

**Acceptance criteria for "evidence gathered":**
- A live isolated run (only `route_models` on) against a `cheap_model` default that is actually
  cheaper and actually different-capability from the requested model, not a same-tier alias.
- A workload built to include both genuinely-easy short requests (lookups, simple formatting) and
  genuinely-hard short requests (reasoning, nuanced judgment calls stated tersely) — the failure
  mode only shows up in the second category, so a workload of only-easy requests would clear the
  bar by construction and prove nothing.
- Quality assessed with a judge or direct inspection per routed request, not aggregate identity —
  a routed request is *never* expected to be output-identical, so the identity check this package
  uses elsewhere is not applicable here at all.
- Confirmation, live, that the existing declines (tools present, `response_format` present,
  already-cheap, over the token ceiling) actually hold under real request shapes and none of them
  leak a case they were meant to protect.

**What would justify loosening the guardrail:** a judge-confirmed acceptable-answer rate on
routed requests, on multiple distinct request-shape workloads, high enough that the residual risk
is smaller than what an operator would accept for the money saved — a threshold this ADR
deliberately does not pre-commit to a specific number for, because "acceptable" is a judgment call
that belongs with whoever supplies `cheap_model` and knows their own traffic, not a number this
document can set once for every user. What this ADR *does* commit to: the number must exist and
be reported, not assumed.

#### `compress_prompt`

**Mechanism:** drops sentences whose Jaccard similarity to an earlier-kept sentence in the same
message clears 0.6, per message, never touching a message's final sentence.

**What could go wrong in production:** two sentences can share most of their words while
differing in the one that matters — a date, a negation, a specific figure, a named entity. A
false "this repeats" judgment erases a fact that was never actually restated, and the failure is
invisible in a length or cost metric; it only shows up as a wrong or ungrounded answer. The one
existing live data point ($0.0041, 36 calls, `rag_queries`, confounded with `semantic_cache`)
already shows something worth taking seriously: cost fell 71.5%, but output tokens rose 71.4% and
every response diverged. A model producing *longer* answers against a *shorter* prompt is
consistent with the model being asked a now-underspecified question and hedging or over-
explaining to compensate — but that is a hypothesis from one confounded data point, not a finding.

**Acceptance criteria for "evidence gathered":**
- Isolated live runs (`compress_prompt` only, `semantic_cache` off) across at least three
  structurally different workloads — not just `rag_queries` again, since one workload's chunk
  structure produced the anomaly and a second data point on the same shape would not distinguish
  "this workload" from "this stage."
- The output-length-increase pattern specifically re-checked: does removed context reliably
  produce *longer* answers elsewhere, or was that one workload's synthetic repetition an outlier?
- Diverged response pairs read directly, not just counted — the deliverable is a characterization
  ("harmless rewording" vs. "the model asked a clarifying question it wouldn't have needed to" vs.
  "the model got something wrong it had right in the baseline"), because "10/10 diverged" alone
  cannot distinguish those.
- A judge-equivalence rate alongside the raw divergence rate, so "different text" and "worse
  answer" are reported as the two different numbers they are.

**What would justify loosening the guardrail:** the anomaly resolving into "cosmetic rewording,
net cost win, no correctness regression" across multiple workloads. If instead the pattern
generalizes — shorter input, longer and hedgier output, real answers getting worse — that is
itself a finding worth shipping (the guardrail was right to exist), and the recommendation should
say so plainly rather than quietly re-running until a friendlier workload is found.

#### `semantic_cache` — the deepest risk, and the deepest evidence bar

**Mechanism:** serves a **verbatim stored response** for a new request when lexical (word-overlap)
similarity to a stored prompt clears `semantic_threshold` (default 0.97), restricted to
`temperature == 0` requests matched within the same model.

**What could go wrong in production — this is the one stage where the failure is total, not
partial.** Every other `ALTERED` stage degrades an answer by removing or rewording information
the model still sees something of. `semantic_cache` can serve a **complete, confident, wrong
answer to a different question**, with nothing in the response marking it as such. Two prompts
differing in exactly the clause that changes the correct answer — a number, a negation, an
entity, a date — can share enough vocabulary to clear a lexical-similarity threshold that has no
concept of *which* words matter. This is the stage `caching.py`'s own docstring and
`semantic_cache.py`'s class docstring both flag as categorically worse than the others in the
package, and it is the one this ADR treats with the most weight.

**Acceptance criteria for "evidence gathered" — none of these are optional:**
- A **dedicated adversarial workload** of near-duplicate-but-meaningfully-different query pairs,
  built specifically to probe this failure mode: same topic, same mostly-shared vocabulary,
  different answer. Categories to cover explicitly: a changed number (`"...for 50 users"` vs.
  `"...for 500 users"`), a changed entity (`"...in California"` vs. `"...in Texas"`), a negation
  (`"...is covered"` vs. `"...is not covered"`), and a changed time reference (`"...by Friday"`
  vs. `"...by Monday"`) — the four shapes of "one word changes the answer" that a Jaccard-style
  metric is structurally blind to.
- The **live false-positive rate measured directly**: run the workload live, for every case where
  the stage actually fires (similarity clears the configured threshold), check whether the served
  cached answer is correct for the *new* prompt, not the one it was cached for. This is a number
  this ADR requires to exist and be reported honestly even if it is not zero — especially if it
  is not zero.
- Measured **at the shipped default threshold (0.97)**, since that is the number a caller who
  enables this stage with no further configuration actually gets, plus at least one nearby
  threshold (e.g. 0.90, the config-level floor `OptimizeConfig.__post_init__` already enforces) to
  characterize how much margin the default threshold actually buys versus that floor.
- The eval gate's existing `CacheBehaviorCase` coverage stays in place regardless of outcome —
  this is additive live evidence, not a replacement for the deterministic logic check.

**What would justify loosening the guardrail:** a measured false-positive rate low enough, at the
default threshold, that an operator serving a wrong answer that confidently is a rare enough event
to accept for the savings — and, given the total-failure nature of a hit here, "loosening" more
plausibly means *documenting the measured rate honestly and leaving the default off* than raising
it to on. A guardrail-only outcome (e.g. recommending the threshold never drop below some
measured-safe floor, independent of any default change) is an explicitly acceptable, and arguably
the most likely, resolution for this stage specifically.

#### `summarize_history`

**Mechanism:** replaces history older than `recent_turns` with the output of a caller-supplied
`summarizer: Callable[[str], str]`. Ships no summarizer; with none supplied, the stage always
declines, so `summarize_history=True` alone spends nothing (by design, same rule the core
quality-lane judge follows).

**What could go wrong in production:** a summary omits or misstates a fact a *later* turn depends
on — a budget figure, a decision made three turns ago, a constraint the user stated once and
expects remembered. Unlike `trim_history` (which drops the same aged-out window and is `SHAPED`,
not `ALTERED`), a bad summary is worse than dropping the content outright in one specific way: a
dropped fact is visibly absent (the agent can say "I don't have that context"), while a
*misstated* fact is present and wrong, and nothing about the interaction signals that anything is
off.

**Acceptance criteria for "evidence gathered":**
- Because this stage cannot be evaluated at all without a real summarizer, the **first** piece of
  evidence-gathering work is building one — a minimal, real (live-calling) summarizer, not a stub,
  specifically so the numbers measure the stage, not a placeholder (the exact trap
  `bench/__main__.py`'s existing comment already names for why the CLI has never supplied one).
- A workload with a **deliberately planted, load-bearing fact** early in a long conversation, and
  a later turn that depends on retrieving it correctly (not just "does the conversation still make
  sense" — specifically, "is the load-bearing fact still correct after summarization").
- Live comparison against `trim_history` on the same workload: does summarizing actually preserve
  more than trimming does, for the cost of the extra model call — or does the summary lose the
  same fact trimming would have dropped anyway, in which case the extra call bought nothing.
- The tool-call boundary safety this stage shares with `TrimHistoryStage` re-confirmed live, the
  same way `tool_calling_chat` did for trimming.

**What would justify loosening the guardrail:** live-confirmed fact preservation across turns,
at a cost that is smaller than what the saved tokens are worth, compared honestly against
`trim_history`'s free alternative — if summarizing preserves nothing trimming wouldn't have kept
anyway, the guardrail should stay exactly where it is regardless of how safe the summary text
looks in isolation.

## Alternatives

**Flip all four to on by default now, since the `IDENTICAL`/`SHAPED` tiers turned out fine.**
Rejected. Every `IDENTICAL` and `SHAPED` stage's live verification found real surprises (the
36.3%→0% `prefix_cache` correction, the simulated-regression-that-wasn't for `trim_history`) even
though none of those stages can change the model's *answer* — only its price or its prompt shape
within provably-safe bounds. `ALTERED` stages can change the answer outright, which is a strictly
larger risk surface that the `IDENTICAL`/`SHAPED` track record does not transfer to.

**Set one shared numeric bar (e.g. "95% judge-equivalence") for all four stages.** Rejected. The
four stages have different failure modes — `route_models` degrades gracefully (a worse but still
relevant answer), `semantic_cache` fails totally (a wrong answer to a different question). Forcing
one number either overprotects the low-risk stages or underprotects the highest-risk one.
`semantic_cache` gets the most demanding, most concrete evidence requirements in this document for
exactly that reason.

**Skip the design step and go straight to running isolated live benchmarks.** Considered, since
the mechanism for isolation and the workloads are the actual deliverable a user benefits from.
Rejected because it repeats the exact mistake this ADR exists to prevent: `compress_prompt`
already has one live data point, gathered without first deciding what would count as sufficient
evidence, and it produced a genuinely alarming number (output tokens +71.4%) that nobody has yet
decided how to interpret. Writing the bar down first means a later result cannot be quietly waved
off, or quietly over-weighted, because there was no prior commitment to what would count.

## Consequences

**Gathering this evidence costs real money and real engineering time**, and the honest possible
outcome for any given stage is "stays off, with a documented reason" — this ADR does not
pre-commit to any stage graduating. That is accepted: ADR-013 already chose to ship these stages
lossy and off by default rather than not ship them at all, and confirming that the guardrail was
correctly placed is exactly as valuable an outcome as confirming it can be safely loosened.

**Every future `ALTERED`-tier stage inherits this bar.** A stage that ships without live,
isolated, judge-or-inspection-characterized evidence — and, if its failure mode can be a silently
wrong answer, a direct false-positive measurement — has not met the standard this project now
holds itself to, regardless of what the eval gate alone shows.

**This ADR will be amended, not superseded, once the evidence exists.** Each stage's resolution
(stay off / stay off with guardrail changes / default changes) is recorded as an addendum here —
the same pattern ADR-005 used when its Redis half turned out to be half-implemented — so the
decision and the evidence that produced it live in the same document.

---

## Addendum (2026-08-03) — the evidence, gathered

All four stages measured live against `claude-haiku-4-5` (the vendor whose prefix caching this
library controls, so it measures strictly more), isolated per rule 2, recorded under
`docs/evidence/` so re-checking is free (ADR-039). Total spend for everything below: **$0.85**.
The nondeterminism floor was measured first, because a divergence number without its floor
measures nothing: **1/10 byte-identical prompts diverged** on `rag_queries` ($0.04, 20 calls,
`2026-08-03-control-rag-queries-claude-haiku-4-5.jsonl`).

### `semantic_cache` — stays off; the risk is inherent to the mechanism

`--semantic-cache-audit` at the shipped threshold 0.97: **false-positive rate 87.5%** (7/8 fired
adversarial probes served a wrong answer), benign hit rate 37.5% (3/8). $0.0108, 38 calls,
`2026-08-03-semantic-cache-audit-claude-haiku-4-5.jsonl`.

Every category this ADR named fired wrongly at least once: a changed number (50→500 seats), a
changed entity (Platform→Security team), a negation ("is **not** covered" answered "Yes,
covered"), a changed date (March→September).

The deeper finding is in the similarity distributions, and it forecloses the guardrail-change
outcome this ADR held open: adversarial pairs scored **0.955–0.989** while legitimate rewordings
scored **0.923–0.978**. The populations overlap with the dangerous one *on top*, because changing
one load-bearing word costs less lexical similarity than rephrasing a sentence. No threshold
separates them — derived from the same recorded similarities, the 0.90 config floor fires every
adversarial probe including the one 0.97 correctly declined, and any threshold high enough to be
safe fires on nothing benign either. A lexical metric cannot be tuned out of this; only a metric
that knows *which* words matter could be, and that is a different stage, not a different number.

### `compress_prompt` — stays off; evidence partial, and the first isolated point found a worse failure than the one it chased

`--isolate --stage compress_prompt --judge` on `rag_queries`: input tokens **−84.2%**, cost
**−77.2%** ($0.02004 → $0.00457), end-to-end latency −43.9%. $0.0265, 29 calls,
`2026-08-03-compress-prompt-rag-queries-claude-haiku-4-5.jsonl`.

The +71.4% output anomaly this ADR flagged **did not reproduce** in isolation on this model:
output tokens moved −4.2%. The confounded run's alarm belonged to the bundle, not this stage.

Quality: 1 identical, 7 judged equivalent, **2 divergent against a floor of 1** — and both
divergences read the same way. The baseline correctly answered `INSUFFICIENT CONTEXT` to a
question about a quarter the retrieved documents never confirm; the compressed arm **asserted the
fact anyway** ("revenue in Q7 was driven primarily by subscription growth…"). Compression removed
exactly the hedging context that let the model see the answer was not there: a correct refusal
flipped into a confident unsupported answer, the silent direction rule 4 exists for.

The bar requires three structurally different workloads and this is one, so the evidence is
explicitly partial. Its direction is still a finding: the guardrail was right to exist.

### `route_models` — stays off, now with the number an operator needs

`--route-models-audit`, `claude-sonnet-4-5` routed to `claude-haiku-4-5` (3× cheaper input),
graded against ground truth rather than a judge: easy probes 4/4 for both models; hard probes
sonnet 88% vs haiku 75%; **regression rate 8.3%** — 1 of 12 probes the requested model got right
and the cheap one missed ("4th letter from the end of *extraordinary*"). All **five decline
guards held live** (tools, response_format, already-cheap, token ceiling, routable-is-routed).
$0.006 across two attempts, 24 graded calls,
`2026-08-03-route-models-audit-sonnet-to-haiku.jsonl` with full output beside it.

This ADR committed to the number existing rather than to a threshold: it exists. **One in twelve
short-hard requests regresses** for a 3× input price cut on routed traffic; whether that trade is
acceptable belongs to whoever supplies `cheap_model`, exactly as written above.

### `summarize_history` — stays off; the extra call bought nothing

`--recall-audit` with a real live-calling summarizer (haiku), `recent_turns=6`, all 4 probes
interpretable (the control arm answered every planted fact correctly). $0.0219, 30 calls,
`2026-08-03-recall-audit-claude-haiku-4-5.jsonl`:

| arm | recalled | silently wrong | total tokens (incl. summarizer) |
|---|---|---|---|
| full history | 100% | 0% | 1,154 |
| trim_history | 100% | 0% | **492** |
| summarize_history | **0%** | 0% | 1,364 |

The summarizer lost all four planted facts *and* cost more total tokens than sending the full
history, once its own four calls are counted. It failed honestly — every miss said `NOT IN
CONTEXT`, silently-wrong stayed at zero, and the tool-call boundary held — but this ADR's test
was comparative, and trimming kept everything, free, because `trim_history` anchors the opening
exchange, which is where a conversation's load-bearing facts actually live. One summarizer
prompt, one model, one conversation shape; a better summarizer could raise the recall number, but
it starts 862 tokens behind a free alternative before preserving a single fact trim would have
dropped.

### What generalizes

One vendor, one model, one day. The `semantic_cache` finding is the most portable — the
similarity metric is computed by this library, not by a model, so the overlap of its
distributions travels to every vendor. The other three are one-model measurements and say so.
Re-running any of them costs one command against the recordings' stated caps.
