# Changelog

All notable changes to `optio` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**What versioning means here.** The public API (§8.1) and the emitted signal names (§7.2) are
the compatibility surface. Renaming or removing a `gen_ai.run.*` attribute is a breaking change
even when no Python signature moves, because downstream OPA, Cedar and AGT policies are written
against those exact strings — a policy that silently stops matching is worse than one that
fails to load.

**The public API is exactly what `optio` exports at the top level** — `instrument`, `meter`,
`RunContext`, `Config`, `BudgetPolicy`, `GENAI_SEMCONV_VERSION`, `current_run`. Everything
reachable only through a submodule (`optio.lanes.*`, `optio.runtime.*`, `optio.store.*`,
`optio.adapters.*`) is internal and may change in any release, including a patch, despite being
importable, documented and typed ([ADR-012](docs/design/adr/adr-012-the-public-api-is-the-top-level-package-only.md)).
If you need one of those, please open an issue rather than importing it — a real use case can
be promoted to the top level deliberately.

## [Unreleased]

### Fixed

- **`concision`'s instruction was evicting more cache than it cost.** It appended to the *system
  prompt* — the exact region a provider's prefix cache covers — so it did not merely add its own
  204 tokens, it shifted everything below the edit out of OpenAI's 128-token block alignment and
  knocked 256 tokens of already-cached prompt back to full rate. Measured on `multi_turn_chat`,
  full-rate input: 2,602 baseline → **3,062** with the instruction on the system prompt →
  **2,294** with it on the last message. The last message changes every request anyway, so
  nothing there was cacheable and nothing below it can be displaced. The stage is now roughly
  cost-neutral on a workload where it has nothing to do, which is correct behaviour and is *not*
  evidence that it helps — it stays off, because no workload here produces a padded reply.
- **`trim_history` cut from the front, which discards the cheapest and most valuable context at
  once.** Two earlier measurements point the same way: the provider's cached region is
  `system + oldest turns` (87% of `multi_turn_chat` was cache-served before trimming touched it),
  and the recall audit found load-bearing facts stated in the first exchange and never repeated,
  of which trimming recovered 0 of 4. New `anchor_turns` keeps both ends and takes the middle.
  Live on `multi_turn_chat_long` at 50 turns: sliding saved 26.3% of cost with 25/50 replies
  unchanged; anchoring saved 16.8% with **50/50** unchanged. It converts a quality loss into a
  smaller, visible cost rather than making trimming free. Defaults to `0` — unchanged shipped
  behaviour — because one good measurement on one workload is not grounds for changing what every
  caller sends (ADR-016). An elision marker declares the gap, since a model shown a conversation's
  opening followed by a much later exchange will otherwise try to reconcile the jump.
- **The live benchmark harness was not sending tool schemas at all.** `OpenAIProvider` forwarded
  messages, `max_tokens`, `temperature` and `response_format` — and not `tools`. So `mcp_agent`,
  the workload built specifically to measure tool cost, sent zero tools: `minify_tools` reported
  saving 3,240 tokens while the provider billed byte-identical totals in both arms (76,439
  either way). It did not fail. It measured nothing, confidently. This is the same failure the
  adapter's own comments already record one field over — `tool_calls`/`tool_call_id` were dropped
  by the first version of that method and found the same way. `AnthropicProvider` had the same
  omission and is fixed with it, including the schema-shape translation Anthropic needs.
- **Tool-token estimates overstated what providers actually bill by ~1.5×,** because providers
  re-render tool schemas into a compact internal form rather than billing the JSON handed to them.
  Two measured corrections, which cannot be merged into one: `TOOL_SCHEMA_CALIBRATION = 0.65`
  gets a tool set's *total* nearly exact (0.65 × 1395 = 907 against 898 billed), while
  `ANNOTATION_STRIP_CALIBRATION = 0.37` covers the delta from stripping annotation keys, which
  are unusually punctuation-heavy and so shrink the real bill by far less than they shrink the
  JSON. Both derived from four measured sizes, consistent within 2%. With them, `minify_tools`
  claims **1,190** tokens on `mcp_agent` where the provider stopped billing **1,210** — 1.7% out
  and understating. Without them it claimed 3,240.
- **`semantic_cache` stored entries under text no lookup could ever produce, so its hit rate was
  silently zero whenever any later stage rewrote the prompt.** `before` looked up the request as
  it received it; `after` re-derived the key from the request *as sent* -- every later stage's
  rewrites included. The two only agreed when every message-rewriting stage happened to decline,
  which is why the defect survived a live audit and a dedicated unit-test file: the workloads that
  exercised the cache had no schema for `structured_output`, no surplus history for `trim_history`,
  and no duplicate blocks for `deduplicate`, so nothing between the lookup and the provider ever
  touched the messages. Adding `concision`, which fires on any plain chat request, took the
  adversarial audit's collision count from several to zero in one commit and exposed it.
  `before` now carries the text it looked up through `ctx.scratch`, which is what `exact_cache`
  has always done. Pinned by a pipeline test parameterized over both cache stages, and the
  parameterization is the point -- the one that was correct and the one that was not are now held
  to the same contract. Verified by re-introducing the old `after` in memory: 1 provider call with
  the fix, 2 without.
- **`--strict-fidelity` ran two SHAPED stages inside the arm that exists to prove output is
  byte-identical**, and reported 6 divergent responses on `mcp_agent` rather than failing. The
  flag turned the reshaping stages off through a hand-written list of their names, which was
  complete when written and stopped being complete the moment `minify_tools` and
  `cap_tool_results` landed. It now derives the set from each stage's own declared `Fidelity`
  through a new `Optimizer.stages` accessor, so a stage added tomorrow is excluded without anyone
  remembering to exclude it. Stages named explicitly on the command line stay exempt, so
  `--isolate` can still measure the single `ALTERED` stage it was asked to.

### Added

- **`detect_unstable_prefix`: the first thing here that changes nothing.** Provider prefix caching
  needs only that the head of the prompt be byte-identical between calls, and the field literature
  calls breaking that "the single most common production caching bug" precisely because it leaves
  no trace -- inject a timestamp above the instructions and the hit rate goes to zero while every
  test passes and every response is still correct. This stage reads each request, compares digests
  against what it has seen, and reports two findings with different fixes:
  `unstable_system_prompt` (the first message is never the same twice) and `unstable_tool_order`
  (one unchanging tool set, serialized from an unordered container). It never modifies a request,
  because the fix is always in the caller's prompt assembly and a stage that "helpfully" reordered
  a tool list would be rewriting application logic it does not understand. Findings are readable
  as data through `Optimizer.findings`, not just as a log line, so a smoke test can assert on them
  before an invoice does.
- **A workload pair that prices that bug rather than asserting it.** `timestamped_agent` is
  `multi_turn_chat` with a clock above the system prompt instead of below it -- one line different.
  Against the simulator's automatic-prefix-cache model the clean version has **16,128 of 19,050**
  prompt tokens served from the provider's cache and the broken one has **zero**. The detector
  fires on exactly one of the two, which is what makes the pair a test of the detector and not
  just an illustration.
- **The tool-order check is exact, not statistical, and a test is why.** It was first written as a
  distinct-ratio threshold like the system-prompt check, and could not detect the bug it existed
  for: there are only *n!* orderings of *n* tools, so a genuinely random ordering bug on a small
  tool set repeats constantly and never clears any sensible ratio. It now fires when the sorted
  digest never varied and the unsorted one did -- unambiguous, because unstable ordering is never
  something a caller wants at any frequency.
- **Tool cost, the largest evidenced gap in this package, is now addressable.** Anthropic's
  published figure for deferring tool loading is an 85% token reduction with MCP-evaluation
  accuracy rising 49% -> 74%, and until now nothing here touched `request.tools` at all -- two
  stages read the field only to decline. Three new stages: `minify_tools` (SHAPED, on by default)
  strips `title`/`$schema`/`$id`/`$comment` wherever they appear, which is text the model never
  reads; `cap_tool_results` (SHAPED, on by default) bounds a single tool result at 2000 tokens
  with an explicit truncation notice; `prune_tools` (ALTERED, off by default) drops tools sharing
  no vocabulary with the conversation, never one already called, and never below three.
- **A workload that can measure any of that.** Every existing workload sends `tools=()`, so under
  ADR-016's third test no claim about the three stages above could ship. `mcp_agent` carries ten
  MCP-shaped schemas -- the generated form, with a `title` on every property -- across ten steps,
  with one deliberately oversized tool result at step 3 so the run contains the shape
  `cap_tool_results` exists for. Simulated: `cap_tool_results` 7,831 tokens, `minify_tools` 3,240.
- **`concision`, and it ships off despite being SHAPED.** The stage suppresses the three habits
  the field literature puts at 30-50% of chat output tokens: restating the question, summarizing
  its own reply, offering follow-ups. It also spends input tokens on every request to do it, and
  only a live run can see the other half of that trade. The adversarial `unique_questions`
  workload measured the visible half alone at **-14.8% token reduction** -- a cost *increase*.
  Defaulting it on because a published figure says 30-50% is exactly what ADR-016 forbids.
- **[ADR-016](docs/design/adr/adr-016-the-in-scope-test-for-a-cost-technique.md): the in-scope
  test for a cost technique.** The boundary between what this package implements and what it
  merely recommends had twice been drawn on *effort* -- "that needs a queue", "that needs a team"
  -- which is a reason for a caller to decline a technique and never a reason for a library to,
  since absorbing that work is what a library is for. Effort is retired as a test and replaced by
  three: expressible against the normalized types, requiring no infrastructure the caller must
  operate, and measurable by the bench harness. Includes the full classification of the field's
  techniques, including the seven that stay out and why.
- **`LLMRequest` gains `stop` and `thinking_budget`,** both included in the exact-cache key. A
  stop sequence halts generation mid-answer and cannot be compensated for the way `max_tokens`
  is, since a completion stopped at a delimiter reports `finish_reason="stop"` exactly as a
  finished one does. No stage *sets* `stop`: a correct stop sequence is a fact about the caller's
  output format, and a stage that guessed would silently truncate answers.

- **`summarize_history` has live evidence for the first time: it works, and it still loses.**
  This stage had zero live data *by design* -- `Optimizer` refuses to enable it without a
  caller-supplied summarizer, and the bench CLI supplied none, correctly, because a stub
  summarizer would have measured the stub. New `--recall-audit` supplies a real live-calling one
  and runs the workload ADR-015 specified: four load-bearing facts (a budget, a date, a decision,
  a compliance constraint) planted in the *first* exchange, eight filler exchanges to push them
  out of the `recent_turns=6` window, then each fact asked back. Three arms on identical
  requests, live `gpt-4o-mini` 2026-07-29, $0.0015 across 30 calls, reproduced four times:
  `summarize_history` recalled **4/4** where `trim_history` recalled **0/4**, with **zero silent
  errors** -- no fact was misstated, which is the failure that would matter most, and
  `trim_history`'s four misses were all visible (`NOT IN CONTEXT.`) rather than silent.
- **And the total-token column reverses the verdict.** Summarizing spent **622 tokens to reach
  the same answer sending the whole conversation reached for 466**. The prompt really did shrink
  (261 vs 466) -- that is where a report showing only prompt tokens would have stopped, and it
  would have been wrong. The summarizer call reads all the dropped history and writes a summary:
  361 tokens nobody was spending before. The cause is structural and lives in the stage, not in
  the workload: `SummarizeHistoryStage.before()` calls the summarizer **unconditionally on every
  request**, with no memoization keyed on the dropped history (4 probes produced 4 summarizer
  calls). So the same aged-out turns are re-summarized every turn, and the stage's cost scales
  with conversation length exactly as the full prompt does -- the bounded-prompt advantage can
  never catch up. A summary computed once and reused would change that arithmetic entirely; the
  stage as shipped does not do that, and the class docstring now says so.
- **The tool-call boundary is re-confirmed rather than inherited.** `tool_calling_chat` proved it
  for `trim_history`; `SummarizeHistoryStage` asserts the same invariant in its docstring, and
  the audit now checks it directly across every cut point of a synthetic tool-calling
  conversation -- no `tool` result was ever separated from the assistant message that called it.
- **One more grader bug, same family as the routing one.** `"March 14th"` failed to match
  expected `"march 14"`, because the word-boundary check counts the ordinal suffix as part of the
  token. It excluded the deadline probe as "control arm answered wrong" when the control had
  answered it correctly. Fixed by listing both spellings -- the alternative was teaching the
  matcher English ordinals -- and the exclusion mechanism is worth noting for working as
  intended: it flagged the probe as unusable instead of quietly scoring it against the stage.
  18 tests.

- **`route_models` has live evidence for the first time, and it found a real regression.** This
  stage had *zero* live data: it was excluded even from the old `--aggressive` flag, and the A/B
  harness could not have measured it anyway (`ABResult` prices a whole arm at one flat rate, and
  a routed request is never expected to be output-identical, so the suite's identity check does
  not apply). New `--route-models-audit` asks **both** models the same twelve short questions and
  grades against **known answers rather than a judge** -- a judge is itself a model, and using one
  to decide whether a weaker model is good enough puts the capability question inside the thing
  being measured. Live `gpt-4o` vs `gpt-4o-mini` (16.7x cheaper input), 2026-07-29, $0.0013 across
  24 calls, reproduced identically three times: easy probes 100%/100%, hard probes 100%/**88%**,
  **regression rate 8.3% (1/12)**. The single failure is the exact shape the risk model predicts
  -- *"What is 17 times 24, minus 89?"*, eight words, inside the 500-token ceiling, no tools, no
  schema, so the stage routes it: `gpt-4o` answers 319, `gpt-4o-mini` answers **329**. Nothing
  distinguishes that from a correct answer without knowing the answer. All five decline guards
  (tools, `response_format`, already-cheap, over-ceiling, and routable-is-routed) are re-checked
  live on every audit run rather than trusted from the eval gate.
- **The first probe set was too easy, which is part of the result.** It used four famous
  reasoning traps -- the strawberry letter count, 9.11 vs 9.9, bat-and-ball, transitive ordering
  -- and `gpt-4o-mini` answered **all four correctly** every run. They are in everyone's training
  data by now, so passing them says nothing about a *novel* short-but-hard request. Four ordinary
  multi-step problems with no memorable phrasing were added, and one of those produced the only
  regression. A probe set of only-easy or only-famous-hard requests would have cleared this stage
  at 0% and proved nothing, which is the failure ADR-015 named for this stage specifically. The
  honest reading of 8.3% is "not zero", not "8.3%": twelve probes on one model pair, all
  single-turn and answer-checkable by construction, is a floor on the risk, not a measurement.
- **Two bugs in the audit itself, both of which produced clean-looking wrong numbers.** First,
  the grader's normalization kept every `.` so decimals would survive, which left `"Tokyo."`
  failing a word-boundary check for `"tokyo"`: three correct live answers scored wrong and one was
  reported as a `route_models` REGRESSION that never happened -- a plausible 12.5% regression rate
  that was entirely this bug. Periods are now dropped unless flanked by digits, with tests.
  Second, `run_routing_audit` originally took one provider and varied `request.model`, but
  `OpenAIProvider.__call__` sends `self.model` and ignores `request.model` (deliberately -- it is
  what keeps its pricing honest). Both "arms" would have hit the same model and reported a
  perfectly clean, entirely meaningless 0% regression rate. It now takes two providers and raises
  if they serve the same model. New `--model` flag, since `available_live_provider()` defaults to
  the cheap model and so had nothing to route *down* from. 22 tests.

- **The `compress_prompt` anomaly, chased across six workloads and explained.** The one prior
  live data point -- cost -71.5%, output tokens *+71.4%*, 10/10 diverged, on `rag_queries` under
  the bundled `--aggressive` flag -- reproduces exactly with the stage isolated, so it is real
  and it is `compress_prompt`'s alone. It is also **not general**, and the cause is not the one
  ADR-015 guessed. Live against `gpt-4o-mini` 2026-07-29, $0.0138 across 181 calls: on `fan_out`
  the stage gave up 82.9% of input tokens and 67.8% of cost for **byte-identical output on all
  12 responses**, and `tool_calling_chat` the same at 68.9%/41.5% across 20; `unique_questions`
  correctly saved nothing at all. Only the two synthetic RAG workloads regressed, both by 6 of 10
  judged answers. ADR-015 hypothesised the longer outputs were the model "hedging or
  over-explaining to compensate" for an underspecified prompt. Reading the diverged pairs instead
  of counting them shows something else entirely: the baseline was emitting a 4-token
  `'INSUFFICIENT CONTEXT.'` refusal and the compressed arm a 13-token sentence answering anyway.
  The whole output-token increase is that substitution.
- **The mechanism, confirmed against the transformed prompt rather than inferred.** These
  workloads' system prompt is `_SYSTEM_PROMPT * 9`, so `"If the context does not contain the
  answer, say exactly: INSUFFICIENT CONTEXT. Never speculate."` appears nine times.
  `CompressPromptStage` collapses the system message 6,408 -> 916 characters, taking that
  instruction from **9 occurrences to 1** -- deleting no distinct sentence, and therefore
  information-preserving by the stage's own standard. The model then answered 6 of 10 questions
  it had correctly refused, attributing revenue to a quarter the context never named. So the
  documented risk ("a false near-duplicate judgment erases a fact that was never restated") is
  not what bit; the real one is that **an instruction whose force came from repetition loses that
  force**, and only requests exercising that instruction reveal it -- which is exactly why
  `fan_out` and `tool_calling_chat` took the identical 9->1 collapse with zero divergence. A
  caller whose system prompt leans on repetition for emphasis is the exposed case, and no token
  or cost metric can see it. `stages/compress.py`'s docstring now says so.
- **`--control`: the divergence floor every previous number was missing.** Runs a workload twice
  with the optimizer off on *both* arms, so any difference is the provider's. This project has
  been reporting live divergence counts on the assumption that `temperature=0` is deterministic.
  It is not: `unique_questions` diverged on **4/12, 5/12, 4/12** across three runs with
  byte-identical prompts. The floor is workload-dependent and mostly zero (0/10 `rag_queries`,
  0/20 `tool_calling_chat`, 0/12 `fan_out`, 1/12 `multi_turn_chat`), which is what makes the
  `rag_queries` 10/10 a real finding -- and what makes `multi_turn_chat`'s "1 of 12 diverged" no
  finding at all, since it sits exactly on its own floor. Every divergence figure in
  `docs/optimize-benchmarks.md` now carries its floor beside it.
- **`--isolate`, because `--stage` was still confounded one layer down.** ADR-015 retired
  `--aggressive` for bundling two ALTERED stages, but `--stage compress_prompt` left every
  default-on stage running, and `deduplicate` was contributing 720 tokens to the same delta. The
  first `--stage` run of this session reproduced the original confound in miniature. `--isolate`
  turns off every other stage including the lossless caches -- `exact_cache` resolving a repeat
  before the isolated stage sees it is precisely how the original run credited one stage for
  another's work. Also `--judge` (model-backed equivalence grading) and `--show-divergences N`
  (print the pairs), both required by ADR-015 and neither previously reachable from the CLI even
  though `harness.compare()` has accepted a `Judge` all along.
- **The judge's first prompt was wrong, and its number looked fine.** Asking "is B equivalent to
  A" scored `rag_queries` at 10/10 WORSE -- while the diverged pairs, read directly, showed the
  optimized arm giving *fuller* answers. The judge was treating "B says something A did not" as
  a failure. Re-framed to ask about *regression* specifically, with `BETTER` an allowed verdict,
  the same pair returns BETTER and the workload scores 4 equivalent / 6 worse. `max_tokens` was
  ruled out as the cause first (4 and 16 gave identical verdicts). One limitation is documented
  rather than tuned away: the judge never sees the prompt, so on a pair where the baseline
  declined and the optimized arm answered it cannot tell a recovered answer from a hallucinated
  one -- it votes WORSE, which is the defensible call from what it can see, and those pairs need
  `--show-divergences` plus the workload's ground truth. 24 tests on the bench CLI, up from 13.

- **`semantic_cache`'s live false-positive rate, measured: 85.7% at the shipped default
  threshold.** The first ADR-015 evidence, and it did not go the way the stage's own docstring
  predicted. `bench/adversarial.py` (new) plus `--semantic-cache-audit` run eight
  near-duplicate-but-different-answer prompt pairs -- a changed number, entity, negation, or date
  inside a shared ~100-word context -- and eight same-answer controls, live. Measured against
  `gpt-4o-mini` on 2026-07-29, $0.0078 across 247 calls, sweeping `semantic_threshold` from
  `OptimizeConfig`'s 0.9 floor to 1.0: **wrong answers served 100% / 100% / 85.7% / 14.3% / 0% /
  0%, while legitimate paraphrase reuse ran 100% / 75% / 37.5% / 0% / 0% / 0%.** Reuse hits zero
  at 0.98, one step *before* wrong answers do, so no setting is both safe and useful and the safe
  end is territory `exact_cache` already covers losslessly and by default. The class docstring
  previously claimed the 0.97 default made the stage "conservative to the point of rarely firing
  on anything but near-identical wording" and that "the threshold does the entire job of keeping
  this safe"; both sentences are now quoted in place and corrected, because the mechanism runs
  the other way -- a *longer* shared context makes the one word that changes the answer matter
  *less* to a word-overlap score. Dropping `not` from a ~100-word prompt scored 0.9888. No
  default changed and no threshold moved: this is a metric-level limit, not a tuning one, and the
  recommendation is that the stage is only defensible with a caller-supplied embedding-based
  `similarity_fn`. Full sweep and a verbatim wrong-answer transcript in
  `docs/optimize-benchmarks.md`; resolution in ADR-015.
- **The audit measures three things it could have assumed, each of which would have produced a
  wrong number that looked right.** (1) A hit only counts as a false positive if the model
  demonstrably answers the two halves differently -- so every pair makes a third, cache-disabled
  call, and one probe was excluded as degenerate (the model said "within four business hours" to
  both), making the honest denominator 6/7 rather than a flattering 6/8. (2) The benign control
  set exists because an adversarial-only sweep shows `semantic_threshold=0.99` as a clean fix; it
  is not a fix, it is an off switch, and only the control column reveals that. (3) The reported
  similarity is taken from the stage's own `_similarity_fn` and `_prompt_text` rather than
  re-derived locally, so a future change to how the stage renders a request cannot leave the
  audit reporting a margin the stage never used. 28 tests, including a property asserting the
  control set never outscores the adversarial set -- if a future `similarity_fn` ever separates
  them, that test failing is the signal to re-derive this resolution.

- **`bench/__main__.py`'s `--aggressive` flag, which bundled `semantic_cache` and `compress_prompt`
  together, is replaced by a repeatable `--stage NAME` flag that isolates exactly the named
  `ALTERED`-tier stage(s).** ADR-015 (new: see below) found the old flag had produced the *only*
  live data point `compress_prompt` had, and that result -- cost down 71.5%, output tokens up
  71.4%, 10/10 diverged -- could not honestly be attributed to `compress_prompt` alone, since
  `semantic_cache` was active in the same run. `--stage` also gained `--cheap-model` (with a
  `CHEAP_COUNTERPART`-table default) for isolating `route_models`, and wiring for a real,
  live-calling summarizer for isolating `summarize_history` -- both previously excluded from
  `--aggressive` because the CLI had nothing honest to supply either flag. `harness.compare()`/
  `run_arm()` now thread `summarizer`/`similarity_fn` through to `Optimizer`, including to the
  *baseline* arm's construction: `Optimizer.__init__` validates `summarize_history` regardless of
  `config.enabled`, so the baseline needs a summarizer too even though `enabled=False` guarantees
  `Pipeline.execute` returns before any stage runs it -- a real bug this change's own tests caught
  immediately (`OptimizeConfigError` on the baseline arm) before it shipped. `route_models`
  isolation via this CLI carries a stated caveat: `ArmResult` prices the whole optimized arm at
  one flat rate, so it cannot reflect a per-request model swap actually happening -- `route_models`'s
  real evidence needs a different measurement, tracked in ADR-015. New `tests/optimize/test_bench_cli.py`
  (13 tests) is also this CLI's first test coverage at all; it had none before.
- **ADR-015 — evidence bar for promoting an `ALTERED`-tier stage out of "experimental".** Defines,
  per stage, what "proven safe to recommend" means, a risk model, and what live evidence would
  justify loosening the off-by-default guardrail, written before any of that evidence is gathered.
  `semantic_cache` (silently-wrong-answer failure mode) gets the most demanding bar: a dedicated
  adversarial workload and a directly measured live false-positive rate. See
  `docs/design/adr/adr-015-*.md`.
- **CI now hard-fails if the provider SDKs pinned for testing go missing, and confirmed no test
  can ever spend real money.** Closing the audit's last item: is a real-provider test isolated
  the way `tests/frameworks/` isolates its own optional dependencies? For `openai`/`anthropic`
  the answer needed to be different, not the same mechanism -- these are mocked-SDK tests (no
  network, no key, no spend), so they belong in the main CI job rather than a dedicated
  framework-style matrix job, but they still deserved the same "a gate that can pass by skipping
  is not a gate" treatment the frameworks and policy suites already get. `tests/optimize/conftest.py`
  gained a `pytest_configure` hook that fails the whole session under
  `OPTIO_REQUIRE_PROVIDER_SDKS=1` (now set in the main CI job) if either package is missing --
  one shared check rather than duplicating it per file, and deliberately *not* a per-file
  import-time gate: an inline gate function tried first tripped ruff's `E402` rule, which
  special-cases the literal call `pytest.importorskip(...)` and nothing else (confirmed
  empirically). Also added `tests/optimize/test_ci_isolation.py`, asserting no workflow
  references a real provider API key or invokes the live benchmark CLI, and no test file reads
  a real provider key from the environment to use it (as opposed to setting a fake one for a
  mocked client) -- the actual "confirm" this audit item asked for, now enforced rather than
  simply true today.
- **Mocked-SDK tests for `OpenAIProvider`/`AnthropicProvider`, and `anthropic` joins `openai` as a
  dev dependency.** `bench/providers.py` coverage was 38% -- the real-provider adapters had no
  tests at all beyond the free `SimulatedProvider` and `SpendGuard` cases, because a real API
  call costs money and CI never installed `openai`/`anthropic` for the main test job anyway. Both
  adapters are now driven the same way `test_adapters_openai_agents.py` drives the OpenAI Agents
  adapter: a genuine `openai.OpenAI` / `anthropic.Anthropic` client with only the HTTP transport
  mocked (`httpx.MockTransport`), so the request built and the response parsed are real
  `ChatCompletion`/`Message` shapes validated by each SDK's own pydantic models -- no network, no
  key, no spend. Coverage: 38% -> 95%; the remaining 5% is two `ImportError` branches (untestable
  short of uninstalling a real package mid-suite) and `Protocol` method stubs, which have no body
  to execute. 30 new tests. Installing `anthropic` for real also surfaced 13 real mypy errors in
  `AnthropicProvider.__call__` that `ignore_missing_imports` had been silently hiding since the
  provider was written -- an untyped `list[dict]` passed where the SDK's `TextBlockParam`/
  `MessageParam` TypedDicts were expected, and a `getattr(b, "type", "") == "text"` check mypy
  could not narrow into `.text` being safe to read. Fixed with the same cast-at-the-boundary
  pattern `OpenAIProvider` already uses, plus an `isinstance(b, TextBlock)` check that lets mypy
  actually verify the narrowing rather than trust a string comparison. `anthropic` was never
  installed anywhere in this repo before this change, so this class had gone completely
  unexercised against the real SDK for as long as it has existed. See `tests/optimize/test_providers.py`.
- **OPA and AGT policy packs gain an `optio_optimize` visibility rule.** `optio_optimize`
  (ADR-014) writes `optio_optimize.stage` on the step span when `emit_spans` is on; a source that
  correlates step and run attributes can now surface a warning when a lossy stage
  (`summarize_history`, `route_models`, `semantic_cache`, `compress_prompt`) served a step.
  Warn-only, never deny — enabling a lossy stage is the operator's own ADR-013 choice, not a
  pathology. Cedar is skipped: its permit/forbid model has no non-blocking warn primitive.
  Verified against the real `opa` and `cedarpy` engines (not just structural checks), which
  caught a real test bug: the absence test's input collided with an unrelated pre-existing
  "unpriceable run" warning, unrelated to the new rule, and had to be rewritten to isolate the
  rule actually under test. See `policies/README.md`.
- **`multi_turn_chat_long`: does `trim_history`'s live win hold at scale?** The 12-turn
  `multi_turn_chat` result (cost −8.4%) is short enough that it could plausibly have been a
  small-scale artifact — ADR-013's own reasoning for why trimming can help or hurt is a scale
  argument, and IMPLEMENTATION.md's problem statement describes agentic workloads running 5-30x
  longer than single-shot chat, a regime 12 turns doesn't reach. Reran the same shape at 50 turns:
  cost fell **26.4%**, more than triple the 12-turn figure — the win compounds, it doesn't
  plateau or reverse. Mechanically, the untrimmed baseline's cost grows quadratically with
  conversation length while the trimmed arm's per-call prompt size stays roughly constant past
  `recent_turns`, so the relative saving widens with every additional turn. Live against
  `gpt-4o-mini`, $0.0163 across 100 calls. See `docs/optimize-benchmarks.md`.
- **`SimulatedProvider`'s automatic-cache model, recalibrated against a fresh live trace.**
  It previously reported `cached_input_tokens` as whatever token count the nearest message
  boundary past the 1024-token floor happened to land on — an arbitrary number, not a modelled
  one. An 8-call live trace against `gpt-4o-mini` (2026-07-29) showed OpenAI's real
  `cached_tokens` moving in exact multiples of 128 (`0 → 1408 → plateau → 1536 → plateau`),
  never landing between them even as the prompt grew by an uneven count every call. The
  simulator now rounds down to that quantum (`_AUTO_CACHE_QUANTUM_TOKENS`), pinned by the new
  `tests/optimize/test_providers.py`. An earlier trace (2026-07-28) had recorded a 256-token
  jump, which reads as two quanta crossed in one step now that there's a second data point —
  not a contradiction, but not distinguishable from one at the time either, which is why
  `docs/optimize-benchmarks.md` now dates every calibration claim instead of just stating a
  number: a provider's caching behavior isn't guaranteed to stay fixed release to release, and a
  stale "128" would look identical to a correct one until someone re-measured it. The doc also
  gained a table mapping every live-verified section to its date and spend.
- **`rag_queries_noisy`: a benchmark workload that actually exercises `prune_retrieval`'s
  pruning logic.** `rag_queries` reported 0 tokens saved from `prune_retrieval`, both simulated
  and live — correctly, since every chunk in it shares the query's vocabulary and none should
  score below the relevance floor, but a correct zero on a workload with nothing to prune is not
  evidence the stage does anything. The new workload mixes one genuinely irrelevant chunk (an
  office-parking notice) into otherwise-relevant retrieved context, landing it in the middle so a
  stage that only happened to drop an edge chunk wouldn't look like it works by accident. Live
  against `gpt-4o-mini` ($0.0029/20 calls): the irrelevant chunk was dropped on all 10 requests,
  cost fell 9.1%, output stayed 90% identical. `tests/optimize/test_benchmark.py`'s
  `TestPruneRetrievalActuallyPrunes` checks the same claim directly against the stage — the
  irrelevant chunk is gone and all six relevant ones survive, on every request, not just in
  aggregate token counts.
- **The ADR-013 rule 3 eval gate, and the four `ALTERED`-tier stages it unblocks**:
  `route_models`, `compress_prompt`, `semantic_cache`, `summarize_history`. All four were config
  fields since 0.1.0 with no stage behind them and no gate to ship them under; both land together
  because the gate has no purpose without stages to check and the stages may not ship without it
  (`docs/design/adr/adr-013-optimization-lives-in-a-separate-package.md` rule 3).
  The gate (`src/optio_optimize/eval/`) is deliberately model-free: it checks that required facts
  survive a stage's transformation of the prompt, or that a cache-style stage hits a near match
  and refuses a stranger — not "does a model still answer well", which needs a real call and is
  explicitly out of scope (see the package docstring for why, and what still needs
  `bench/harness.py`'s live judge path instead). It runs as an ordinary, CI-blocking pytest module
  (`tests/optimize/test_eval.py`), the same way property and fail-inject tests already do — no
  separate runner.
  Every stage defaults to the cheapest option that needed no new dependency: `semantic_cache` and
  `compress_prompt` use lexical word-overlap (`optio_optimize/similarity.py`, refactored out of
  `prune_retrieval`'s own helper), not embeddings; `route_models` never makes an auxiliary call,
  only retargets `request.model` by a length heuristic; `summarize_history` ships no summarizer
  and constructs no model client — the same rule the core's quality-lane judge follows — so
  `summarize_history=True` alone spends nothing. `Optimizer` refuses to enable it silently:
  turning it on without `summarizer=` raises at construction rather than becoming a flag that
  looks configured and does nothing forever.
  `summarize_history` and `trim_history` needed an ordering fix: both target the same aged-out
  history window, and if trimming ran first (the pre-existing order) it would already have
  deleted everything there was to summarize. `build_stages()` now runs `summarize_history` first
  when both are enabled, with `trim_history` acting as a backstop on whatever remains.
  `SummarizeHistoryStage` is the one stage that breaks the "stage bodies perform no I/O" premise
  `Pipeline.aexecute` relies on (`pipeline.py`): a real summarizer calls a model. Documented as a
  caveat rather than solved — a blocking summarizer used under `Optimizer.acall` stalls the event
  loop for its duration, same as calling any other blocking function from async code.
  A light live check (`--aggressive --live`, $0.0041/36 calls) found a real, recorded-not-hidden
  result: `rag_queries` cost fell 71.5% but output tokens rose 71.4% and all 10 responses
  diverged — `compress_prompt`'s heavy trimming of this workload's repetitive synthetic chunks
  apparently made the live model write longer answers. See `docs/optimize-benchmarks.md`.
- **Dedicated unit tests for `caching.py` and `output.py`**, both previously covered only
  indirectly through `test_pipeline.py`/the benchmark suite — `output.py` sat at 54% coverage,
  the lowest of any stage module despite output tokens billing at 3-5x the input rate per its
  own module docstring. Now 100% on both. Covers paths only reachable through specific
  `before()`/`after()` sequencing: a truncated (`finish_reason="length"`) stored reply that must
  never be served, `after()` receiving a response that already came from a cache and correctly
  not re-storing it, `AdaptiveMaxTokensStage`'s bounded observation history, and
  `StructuredOutputStage` both appending to an existing system message and inserting a fresh one.
- **`optio_optimize` gained an async entry point (`Optimizer.acall`/`Pipeline.aexecute`)** and its
  first framework adapter, `optio_optimize.adapters.openai_agents.wrap_openai_client`. Building the
  adapter surfaced that the package had no way to support an async provider call at all — every
  realistic target (the OpenAI Agents SDK's `Model.get_response` chief among them) is `async def`
  and abstract, with no synchronous alternative — so `Pipeline`'s stage-running logic was factored
  out into a shared, still-synchronous `_run_stages` that both `execute` and the new `aexecute`
  call; only "call the provider" differs between the two paths.
  The adapter wraps an `AsyncOpenAI` client's `chat.completions.create` (the SDK's own extension
  point for a Chat-Completions-backed model) rather than reimplementing the SDK's Responses-API
  `Model` protocol, which this package's `LLMRequest`/`LLMResponse` were never designed to
  represent. Streaming calls bypass optimization entirely; a request-translation failure falls back
  to the unwrapped client rather than raising, extending Pipeline's fail-open guarantee to code that
  runs outside it.
  Two real bugs were found and fixed while driving the adapter against the actual `openai` and
  `agents` packages (not hand-built stand-ins): a cache hit returned the *original* call's
  `ChatCompletion`, non-zero `usage` included, so a call that cost nothing looked exactly as
  expensive as the one that filled the cache — fixed by gating on `LLMResponse.served_from` rather
  than "is there a native object available." And the Agents SDK represents an unset field
  (temperature, max_tokens) as a falsy `openai.Omit`/`NotGiven` sentinel, not `None`, which every
  `is not None` check in this package's stages was reading as "the caller already set this" —
  found only by calling the real `agents.OpenAIChatCompletionsModel.get_response()` with its
  default `ModelSettings()`, since no hand-written test kwargs would have used a sentinel. See
  `src/optio_optimize/adapters/openai_agents.py` and `tests/optimize/test_adapters_openai_agents.py`.
- **`optio_optimize` now emits spans `optio` already knows how to price (ADR-014).**
  `Optimizer(emit_spans=True)` (default off) makes `Optimizer.call()` emit one
  standard OTel GenAI span per request, using the exact attribute names
  `optio`'s cost and behavior lanes already read off any span. `optio` needed
  zero code changes — its span tap was already built to observe spans from a
  source it knows nothing about, the same mechanism every framework adapter
  uses. `optio_optimize` imports nothing from `optio` to do this, verified by
  `lint-imports` (4/4 contracts kept, including "optio never imports the
  optimizer"). A cache hit's already-zeroed token counts flow straight
  through, so `optio`'s reserve/reconcile ledger prices it at $0 with no
  special-casing on either side — confirmed by a real (non-mocked)
  `TracerProvider` + `CostLane` in `tests/optimize/test_telemetry.py`, not
  asserted by hand. `run_id`, threaded through every `StageContext` since
  0.1.0 and read by nothing, now does something: written as `gen_ai.run.id`
  when given, though correlation to `optio`'s pricing never depended on it —
  that happens through OTel's ambient context, same as every other span
  source. Fail-open: a broken exporter degrades to no span, never an
  exception. Scope is the raw `Optimizer.call()` path only; a framework
  adapter with its own GenAI instrumentation risks double-counting, called
  out as unsolved in ADR-014's Consequences rather than solved implicitly.
  See `docs/design/adr/adr-014-optimize-emits-spans-optio-already-knows-how-to-read.md`.
- **`optio_optimize` Phase 2: `trim_history`, `deduplicate`, `prune_retrieval`.**
  Three bounded-risk stages (`Fidelity.SHAPED`, on by default) that drop context
  rather than invent it: `trim_history` keeps the system prompt plus the most
  recent `recent_turns` messages; `deduplicate` removes exact-repeat
  blank-line-separated context blocks within a message; `prune_retrieval` drops
  blocks that share almost no vocabulary with the message's final block (read as
  the question). All three were config fields already (0.1.0) with no stage
  behind them; `build_stages()` now wires them in between output-shaping and
  prefix marking, per the ordering rule the module docstring already stated.
- **Live-verified: `multi_turn_chat` and `rag_queries` move off the 0% floor.**
  Both showed 0% token reduction on OpenAI before Phase 2 (documented above).
  Live against `gpt-4o-mini`: `multi_turn_chat` cost −8.4% (input −7.3%,
  output −35.0%), `rag_queries` cost −16.5% (input −4.2%, 100% identical
  output across all 10 calls, entirely from `deduplicate`). Total spend to
  measure: $0.0138 across 140 live calls.
- **A simulated finding that a live run overturned**, recorded rather than
  quietly dropped: a simulated pass predicted `trim_history` would *raise*
  cost 34.8% on OpenAI-style automatic prefix caching, because a sliding
  window shares almost no leading text between calls and so appears to defeat
  the provider's own free caching of resent history. The live run showed the
  opposite — cost fell 8.4%, and output tokens fell 35% along with it, an
  effect the simulator cannot model since it always returns a fixed synthetic
  completion length. Same category of error as the 36.3%-to-0% correction
  above, now confirmed a second time: simulated figures in this package are a
  hypothesis, not a result, until `--live` checks them. See
  `docs/optimize-benchmarks.md` and `TrimHistoryStage`'s docstring.
- **`trim_history` no longer risks orphaning a tool result.** A naive suffix
  cut could land between an assistant's `tool_calls` message and its `tool`
  result, which every major provider rejects. The stage now walks the cut
  point backward past any leading run of `tool` messages so the assistant
  that issued them survives with all of them, or the trim is skipped. Added
  `tool_calling_chat`, the first workload in this suite to use `role="tool"`
  messages at all -- no existing workload could have caught this, because none
  of them had a tool result to orphan. Live against `gpt-4o-mini`: 20/20 calls
  succeeded on both arms, zero errors, cost down 6.8%.
- **Fixed a second defect the same live run found**: `OpenAIProvider`
  (`bench/providers.py`) built its request from only `{role, content}`,
  silently dropping `tool_calls` and `tool_call_id` even when `Message.extra`
  carried them correctly — so *any* tool-calling workload failed live with
  `messages with role 'tool' must be a response to a preceeding message with
  'tool_calls'`, regardless of what any stage did. Not a trimming bug; the
  live adapter never sent the structure that makes a tool message valid.
  Fixed by pulling both fields out of `extra` explicitly when building the
  OpenAI payload.

### Changed

- **Behavior classification is O(1) per step instead of O(window).** Call counts are maintained
  as steps arrive rather than recomputed on each one. Per-step cost at the default window falls
  from 53 µs to 37 µs, and stops scaling with `behavior_window_size` entirely — 370 µs to 38 µs
  at a window of 1000. Widening the window to catch longer cycles is now a memory decision
  rather than a latency one. No change to any verdict.
- **`import optio` is about a quarter faster** — 211 ms to 158 ms (median of 12 cold starts).
  `__version__` resolves on first access rather than at import (PEP 562), so `importlib.metadata`
  and the `zipfile`/`email`/`inspect` tree it pulls in are no longer loaded by every program that
  imports optio.
- **A step's cost signals now derive from a single ledger read.** `actual_cost` and
  `budget_remaining` previously came from two separate snapshots, so under concurrency they could
  describe different ledger states — two numbers that were never simultaneously true. A policy
  asserting `remaining == limit - actual` could see a contradiction.

### Documentation

- **[ADR-012](docs/design/adr/adr-012-the-public-api-is-the-top-level-package-only.md) states the
  public API boundary**: the supported surface is exactly what `optio` exports at the top level,
  and everything reachable only through a submodule is internal regardless of being importable,
  documented and typed. Previously unstated, which left every internal signature arguably frozen.
- `GENAI_SEMCONV_VERSION` is exported and is now documented; it was a public promise mentioned
  nowhere.
- Removed a stale README caveat claiming the adapters had not been tested against real
  frameworks. Four isolated CI jobs verify each adapter against genuine LangGraph, CrewAI,
  OpenAI Agents SDK and Claude Agent SDK objects, including the cases each must refuse.

## [0.1.0] — 2026-07-27

First release. The buildable OSS core: cost, behavior, and quality signals emitted as
OpenTelemetry GenAI span attributes, so an existing policy engine can gate on money and outcome
rather than only on permission.

### Added

**Cost lane** — `gen_ai.run.actual_cost`, `projected_cost`, `budget_remaining`,
`cost_per_successful_task`. Reserve/reconcile ledger with a property-tested exactly-once
invariant (R-TECH-1); static pricing table covering 18 models.

**Behavior lane** — `gen_ai.run.loop_state` (`healthy` / `repeating` / `looping` /
`retry_storm`) and `repeat_count`, from a bounded per-run signature window. Measured
false-positive rate **0/1200 (0.000%)** against healthy workloads, detection **600/600**.

**Quality lane** — `gen_ai.run.quality.groundedness`, `quality.task_success`, `success`.
Tiered, sampled, and **off by default** (ADR-003). The judge is a callable you supply: optio
ships no default and constructs no model client, so enabling the lane cannot spend your money
on our initiative.

**Adapters** — LangGraph, OpenAI Agents SDK, CrewAI, Claude Agent SDK. Duck-typed, so no
framework is imported at core import time.

**Policy packs** — OPA/Rego, Cedar, and Microsoft AGT, each with worked rules and tests that
run against the real engines in CI (SC-3).

**Standalone demo** — `docker compose up` in `examples/demo/`: a scripted agent, a real OTel
Collector, no API keys. Catches a retrieval loop at step 23 for $0.36 instead of running 60
steps to $2.18.

**Self-observability** — `optio.internal.signals_emitted`, `lane_errors`, `overhead`,
`sampling_rate` as OTel metrics, deliberately outside the `gen_ai.*` namespace so a consumer
policy cannot gate on optio's own health.

### Guarantees

**Fail-open is absolute** (ADR-004). No internal failure reaches the agent — proven by a
blocking fault-injection suite, with 100% coverage on the guard and the ledger.

**Absence means unknown, never zero.** A signal that cannot be computed is omitted. This is
load-bearing for every policy pack: `budget_remaining` absent on an unpriceable run is the
difference between "unknown spend" and "nothing spent".

**Overhead** — cost + behavior mean 74 µs, p99 107 µs against a 5 ms budget (SC-5). Quality
inline heuristic p99 237 µs against 10 ms. A 200 ms judge costs the run 0.5 ms, because it
never touches the hot path.

### Known limitations

- **`store_backend="redis"` is rejected at construction.** Per-run state is in-process only.
  The distributed path is designed (ADR-005) but unbuilt, and a setting that is accepted and
  then ignored would mean silently wrong cost totals in exactly the deployment where nobody
  would check.
- **Signal names are pinned to OTel GenAI semconv 1.37.0**, which upstream still marks
  Development-stability. A semconv rename is a breaking change here and will be treated as one
  (R-TECH-2).
- **Cost signals are absent for models outside the pricing table.** The table is static and
  hand-maintained, so a model newer than your installed version is unpriceable. All three cost
  signals are omitted rather than reported as zero; supply your own prices to close the gap.
- **The enterprise control plane (M6+) is not implemented** and is out of scope for this line
  of releases (ADR-007).

[Unreleased]: https://github.com/Aniketh-74/Agent-Meter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Aniketh-74/Agent-Meter/releases/tag/v0.1.0
