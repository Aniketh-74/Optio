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

### Added

- **Multi-process runs report one correct cost total: `store_backend="redis"` works**
  ([ADR-050](docs/design/adr/adr-050-the-store-speaks-the-domain.md)).
  Per-run state was process-local, so an agent sharded across workers had each process metering a
  fraction and believing it had the whole — **four workers, a `$0.50` budget, and `$2.00` got
  through**, silently, because every process's arithmetic was internally consistent. A test now
  spawns four real processes against one run and asserts the exact total; a second test measures
  the in-memory backend's limitation rather than describing it.

  ADR-005 left a `StateStore` ABC as the seam for this, and an audit found **nothing had ever
  constructed one** — so its `get`/`set`/`incr`/`delete` shape was an untested guess, and it could
  not express `reconcile` atomically, could not read a run's reservations as a collection, and
  collapsed `is_finalised` into `unknown` under TTL. It is replaced by one Protocol per lane and
  deleted along with `InMemoryStateStore`. `CostLedger` is now a facade over a `LedgerStore`;
  its callers are unchanged and its existing tests were the regression net for the extraction.

  Redis correctness rests on Lua: every compound operation is one script, so `reconcile` cannot
  interleave. The TTL is an idle timeout refreshed on each write (an absolute expiry would drop a
  long run's reservations mid-flight and send `budget_remaining` back to full), and a tombstone
  outlives the payload so a late callback cannot resurrect a closed run. An unreachable store
  emits **no** cost signal rather than a partial one — which needed no new code, because the
  fail-open guard already absorbed it; what was missing was the proof, and that is now a test.

  The gate runs these against a real Redis service container, with `OPTIO_REQUIRE_REDIS` turning
  the usual skip into a failure, because a job whose purpose is running them must not pass by not
  running them.

- **Loop detection works across processes too**
  ([ADR-050](docs/design/adr/adr-050-the-store-speaks-the-domain.md)).
  The same bug as the budget one, failing the opposite way. A sharded budget **over**-spends,
  because four quarters each look affordable. A sharded step window **under**-reports: four
  quarters each fall below `MIN_STEPS_FOR_VERDICT`, and the detector is *required* to answer
  `healthy` when the evidence is thin (§6.4). So the agent is stuck, every worker reports healthy,
  nothing raises and no metric moves. Four spawned processes now classify one run from all of its
  steps; the in-memory counterpart runs the same twelve steps four ways and then one way, which is
  what separates "these steps were not pathological" from "the sharding hid it".

  **What crosses the wire is five numbers.** `classify` never iterates the step signatures — it
  reads the window size, the error count, the distinct-call count and the top two counts, and
  nothing else. The obvious port returns the `Counter`: up to `behavior_window_size` entries per
  step, 1,000 at the documented ceiling, turning an O(1)-in-window-size guarantee the README
  publishes **as measured** into O(window) in bytes without failing a single existing test. So the
  Lua reduces server-side and returns four fields, and a test asserts that against the wire rather
  than against the parsed object.

  The eviction arithmetic now exists in Python and in Lua, which is the highest-risk surface in
  the change: a divergence would not raise, it would give one run two different verdicts depending
  on where it ran. A Hypothesis property test steps both backends together and compares after
  **every** step, since a divergence that a later eviction cancels out would pass a final-state
  check having produced wrong verdicts throughout.

- **Outcome scoring works across processes: all three lanes are now shared-state**
  ([ADR-050](docs/design/adr/adr-050-the-store-speaks-the-domain.md)).
  The third instance of the same bug, and the one where a wrong answer is most likely to be
  believed — a cost total is arithmetic you could check against an invoice, and a loop verdict has
  a step count behind it, but a quality score is a number between 0 and 1 that looks equally
  plausible computed from the whole run or from a quarter of it. Sharded, the judge was told a
  quarter of the run's length, and the heuristic scored whichever step was last *in one worker*, so
  a run that ended in an error could be scored from a healthy step that finished earlier somewhere
  else.

  **The span buffer is gone entirely.** The lane retained up to 64 `ReadableSpan` objects per run;
  deriving what run-end actually reads showed it needs a step count and the final step. So the
  state per run is a counter and one three-field projection, `MAX_RETAINED_SPANS` is deleted, and
  bounded memory stops being a cap someone has to remember and becomes a property of the shape — a
  run costs the same at three steps or thirty thousand. A `ReadableSpan` was also the reason this
  lane could not be shared at all: it is not serializable. A test now drives all three lanes
  through their real code paths and rejects anything reaching a store that could not cross a
  process, checking recursively, because a span nested in a tuple is the shape a leak would take.

  Enabling this lane together with `store_backend="redis"` costs a round trip per step that buys
  nothing until run end, since quality emits no per-step signal. That is measured and published
  rather than smoothed over; the alternative — accumulate locally and merge once at run end — is
  recorded in the ADR as considered and deferred, because choosing the run's genuinely final step
  across processes then needs a comparable clock, and a wrong "last step" is a wrong verdict rather
  than a missing one.

### Changed

- **The overhead table now separates two things it had been conflating.** Classification is flat
  in `behavior_window_size` — that claim holds and is now asserted rather than measured once by
  hand. But the axis it is flat *against* is the number of **distinct calls** a window holds, and
  the two coincide only when a workload's tool diversity is bounded. The published figure's
  workload used 64 tool names, so the distinction never showed. Measured both ways: 12.6 → 13.2 µs
  across windows of 50 to 1000 at fixed diversity, and 8.3 → 35.1 µs as diversity goes from 8 to
  1000 distinct calls. The growth points the safe way — it peaks when every step is different,
  which is the case with no loop to detect — and the worst case is two orders of magnitude inside
  the SC-5 budget. Both axes are now separate benchmark assertions.

### Fixed

- **Your judge was told a 500-step run took 64 steps.** `JudgeRequest.step_count` is documented as
  "how many steps the run took", and [docs/quality.md](docs/quality.md) shows it being passed
  straight into a user's own evaluator as `steps=request.step_count`. It was built from
  `len(spans)`, and that buffer was capped at 64 — so every longer run understated itself, by a
  number plausible enough that nobody would query it, as an input to someone else's scoring logic.

  No test pinned it: the existing judge tests construct a `JudgeRequest` directly and never
  exercise the lane building one. Found by asking what `len(spans)` meant while deriving the
  projection for the shared store — the design spec had asked for that claim to be checked against
  the code rather than trusted, and it was the checking that turned this up.

  **If you calibrated a rubric against the capped value, its input changes.** The documented
  meaning was always the true count.

- **The test fixtures called `flushdb()` on a Redis they did not own.** `OPTIO_TEST_REDIS_URL`
  defaults to port 6379 — the conventional one, which on a developer machine is very often another
  project's server — and the fixtures flushed a database there on every run of the suite,
  unattended. That is correct only while the database number is right, and a database number is a
  weak thing to put between a test run and someone else's data. The reset is now scoped to the
  `optio:` namespace every key this project writes already lives under, so the blast radius is
  structural rather than conditional. Contributors only; no shipped code was affected.

- **A shipped error message pointed at a repository this project no longer owns.** The
  `store_backend='redis'` error still sent users to the pre-rename `Agent-Meter` issue tracker;
  the rename guard only ever scanned five root files and never looked at `src/`. GitHub redirects
  a rename, so it worked — and would have kept working until someone claimed the old name, at
  which point a library would be directing users into a stranger's issue tracker. The guard now
  scans `src/` too, which is also why this entry names the old repository without linking it.

## [0.3.0] — 2026-08-04

**Upgrade if you use the synchronous `OpenAI()` client with `optio_optimize`** — it was silently
broken, and this release fixes it. The plug-and-play wrappers are also public API now, and the four
answer-changing stages have been measured live for the first time.

### Added

- **`AnthropicCounter` — exact token counts, for free
  ([ADR-048](docs/design/adr/adr-048-the-exact-counter-is-an-instrument-not-a-request-path.md)).**
  ADR-042 made the counter pluggable and shipped no implementation of one. This is the first counter
  here that is *exact* for a vendor rather than `tiktoken` applied to everybody:
  `messages.count_tokens` returns the number Anthropic will bill and bills nothing to say so.

  **It is an instrument, not a request-path counter**, and the design follows from that. `count_request`
  calls `count_text` once per message and once per tool, so a forty-turn conversation with twenty
  tools is sixty network round trips against a 100 ms latency budget. A test asserts that round-trip
  count rather than leaving it as a warning in a docstring. `default_counter()` stays offline.

  Failures are loud, inverting ADR-013's rule on purpose: a stage must never break a request, but an
  instrument that quietly substituted an estimate would return a number indistinguishable from an
  exact one.

- **`scripts/measure_anthropic_tokenizer_gap.py`** compares `tiktoken`'s estimate against Anthropic's
  own count across prose, chat turns, JSON tool results and code. It shipped with no constant —
  inventing the number it would produce is what ADR-015 forbids — and was then run on 2026-08-03,
  which produced the entry below.

- **Context-window decisions on Claude models now allow for `tiktoken`'s measured undercount
  ([ADR-049](docs/design/adr/adr-049-exact-is-a-claim-about-a-vendor-not-a-counter.md)).**
  The measurement found `tiktoken` undercounting Anthropic on every text shape — 1.042 (JSON) to
  **1.275 (code)** — which is past even the 1.15 margin reserved for counts that admit to being
  estimates. So a code-heavy Claude prompt that "fit exactly" could be 27% over the window, and the
  provider's rejection is what the user would have seen. `fits_in_window` now takes the model and
  applies `TEXT_UNDERCOUNT_BY_MODEL` (`{"claude": 1.28}`, the worst measured shape) on top of the
  exactness rule; margins compound for an inexact counter, and an unmeasured vendor gets no margin
  rather than an invented one. Savings figures are deliberately uncorrected — they are ratios where
  a uniform bias cancels — and the module docstring now states the 4–27% absolute understatement
  instead of a multiplier pretending otherwise.

- **`wrap_openai_client` now works on the synchronous `OpenAI` client too.** It only spoke
  async — defensible when the target was the Agents SDK, whose `Model.get_response` is
  `async def`, but the plain `openai` SDK's default client is synchronous, and "async only"
  is not plug and play (the Anthropic adapter's own stated standard, which it already meets).
  Sync or async is now detected rather than declared, with the same `inspect.unwrap` guard
  both SDKs need: their shared `@required_args` decorator eats the coroutine marker, and the
  naive check silently routes an async client down the sync branch — a mutation run confirmed
  ten tests catch that on this SDK.

- **ADR-015's evidence bar has been met: all four ALTERED-tier stages measured live, isolated,
  on `claude-haiku-4-5`, for $0.85 total — and all four stay off by default, each now with a
  measured reason** ([addendum](docs/design/adr/adr-015-evidence-bar-for-promoting-an-altered-tier-stage.md)).
  `semantic_cache` served a wrong answer on **7 of 8** adversarial near-duplicates at the shipped
  threshold, and the similarity distributions overlap with the dangerous population on top, so no
  threshold fixes it. `compress_prompt` cut cost **77%** on `rag_queries` but flipped two correct
  `INSUFFICIENT CONTEXT` refusals into confident unsupported answers (floor: 1/10). `route_models`
  regresses **1 in 12** short-hard requests for a 3× input price cut, with all five decline guards
  holding live. `summarize_history` with a real summarizer recalled **0 of 4** planted facts and
  cost more than sending full history, while free `trim_history` recalled 4 of 4. Every run is
  recorded under `docs/evidence/` (ADR-039), so re-checking any number costs nothing.

- **The plug-and-play wrappers are public API now: `wrap_anthropic_client` and
  `wrap_openai_client` are exported from `optio_optimize` directly.** They existed and worked, but
  reaching them meant `from optio_optimize.adapters.anthropic import ...` — a submodule path that
  ADR-012 calls internal and changeable in a patch release. So the easy path had no promise
  attached and the package docstring taught the hard one (hand-translating an SDK call into
  `LLMRequest`). That is ADR-042's shape a second time: the extension point existed and the public
  API did not name it. `optio_optimize` also gains the public-API test `optio` has had since
  0.1.0, including a subprocess check that importing the package still pulls in neither vendor SDK.

- **The README teaches plug-and-play first.** It opened on `Optimizer.call(request, provider_fn)`
  and never mentioned either client wrapper, so the landing page documented the path that requires
  translating your own requests. It now leads with the one-line client wrap for both vendors, and a
  test asserts the centered header uses only HTML that PyPI's sanitizer keeps — `twine check`
  validates that a description *parses*, not that its markup survives, so a hero can render
  centered on GitHub and flat on PyPI with nothing failing.

### Fixed

- **`--record` made `--route-models-audit` impossible to run.** The recording wrapper is applied
  before the audit builds its cheap second arm, and `_same_provider_at` tried to mirror the
  wrapper rather than the provider inside it — so the flag that exists to keep what a run pays
  for prevented the run. Found by the first live routing audit; the second arm is now built from
  the recorded provider's inner client.

## [0.2.0] — 2026-08-02

### Fixed

- **A second tracer provider in the same process could report a full budget for a run it never
  metered ([ADR-044](docs/design/adr/adr-044-a-lane-must-not-report-on-a-run-it-never-saw.md)).**
  Run end is broadcast to *every* registered observer, not only the lane that metered the run. Two
  live providers -- two agents, a test suite, a service that reconfigures tracing -- means two cost
  lanes, and the one that saw nothing answers from an empty ledger. Its zeros are indistinguishable
  from "nothing attempted yet", the one state where a full budget genuinely is available, so it
  emitted `budget_remaining = <the whole limit>` for a run that had been spending throughout.

  **That is the value which guarantees a budget policy never fires** -- the exact failure
  `test_an_unknown_model_reports_no_cost_rather_than_a_free_run` was written to prevent. The
  arithmetic guard was already correct; the hole was in who was allowed to run it. A lane now stays
  silent about runs it never observed.

  Exposed by the ADR-043 fix: before it, foreign taps were mostly never installed, so they were not
  around to answer. Fixing one silent failure revealed the one it had been hiding.


- **A new tracer provider could be left with no tap at all, so nothing was metered
  ([ADR-043](docs/design/adr/adr-043-id-is-an-address-and-addresses-are-recycled.md)).**
  `install_tap` tracked tapped providers in a dict keyed by `id(provider)`. `id()` is a memory
  address and CPython recycles addresses as soon as the object at one is collected -- **189 of 200**
  short-lived `TracerProvider` instances landed on an address a previous one had used. A new provider
  then inherited a dead one's entry, the lookup returned early, and **no span processor was ever
  added**: nothing tapped, nothing priced, and the run span carrying no `gen_ai.run.actual_cost`.

  It surfaced as a `KeyError` in a test about caching, three layers from the cause, and looked like a
  flake because address reuse depends on allocation history -- adding a `print` to the failing test
  made it pass. Entries now carry a weak reference to their provider and are validated on lookup.

- **Run-end observers accumulated without bound.** Every install registered one and nothing ever
  unregistered, so a process gained an observer per provider it had ever created, each firing forever
  against its own empty ledger and reporting runs it never saw as costing nothing. Dead taps are now
  retired on each install -- swept there rather than from a `weakref` callback, which would fire
  while this module's lock is held and deadlock.

### Added

- **`Optimizer` accepts a `counter`, so you can count with your vendor's own tokenizer
  ([ADR-042](docs/design/adr/adr-042-the-extension-point-existed-and-nothing-could-reach-it.md)).**
  Every savings figure here is a token count, and every count goes through `TokenCounter` -- a
  two-method Protocol anything can implement. `Pipeline` has accepted one since ADR-038. `Optimizer`,
  the public entry point, did not -- so the extension point existed, was typed, was documented, and
  **nothing outside the package could reach it.** Every count for every vendor went through
  `tiktoken`, whose `o200k_base` fallback is what Anthropic and Google models resolved to.

  That is not a small approximation: ADR-036 measured Anthropic billing 1.29x what the raw JSON
  tokenizes to for tool schemas against OpenAI's 0.65 -- opposite directions.

  Provider-reported usage still wins over any counter, and that precedence is now asserted rather
  than assumed: a counter is an estimate, the provider's number is the bill, and a counter that could
  override it would make reports *less* accurate on every provider that reports usage. The supplied
  counter is also the one warmed at construction, so ADR-038's fix applies to the tokenizer actually
  in use.


- **Model limits now say where they came from, and the tables are no longer one vendor
  ([ADR-041](docs/design/adr/adr-041-coverage-should-not-depend-on-whose-api-key-is-to-hand.md)).**
  `CONTEXT_WINDOW` and `MAX_OUTPUT_TOKENS` carried 15 Anthropic models and nothing else. Not because
  anything in the code is Anthropic-specific -- `_limit_for` is a string lookup -- but because every
  value had to be *measured*, measurement needs an API key, and **coverage had quietly become a
  function of whose key was to hand.**

  A table entry is now a `Limit`: the number, whether it was measured or is the vendor's published
  figure, a source, and the date the source said so. A bare `int` is no longer a valid entry, so the
  citation is structural rather than a convention. `context_window_for` and `max_output_tokens_for`
  return `int | None` exactly as before; `context_window_provenance` and
  `max_output_tokens_provenance` are new.

  Three states, not two: absent means "no evidence either way" and stays distinct from "the vendor
  states this". Seven Anthropic models remain absent because a probe established only that their
  window exceeds 217,554.

  First non-Anthropic rows either table has ever carried: **`gpt-4o` and `gpt-4o-mini` at 128,000
  context and 16,384 output**, read off OpenAI's pages, costing nothing. That cap is load-bearing --
  16,384 against 128,000 is the widest gap in either table, and `adaptive_max_tokens` now has
  something to clamp to on OpenAI where it previously had nothing.

- **`gemini-2.0-flash` is listed by Google as "Shut down".** Found while looking up its token limits.
  This package prices it, and it is the only Google model priced. The row is kept with a dated note
  rather than deleted -- removing a price silently changes what every historical report meant -- but
  it should not be used for new work. The ADR-029 shape, caught by reading the vendor's page rather
  than by a 404.

### Fixed

- **Seven defects a green suite was hiding
  ([ADR-040](docs/design/adr/adr-040-a-field-with-a-default-is-a-field-every-old-call-site-keeps-compiling-around.md)).**
  A static review reported eight findings against 2,110 passing tests and a clean `mypy --strict`.
  Seven were real.

  Four are one defect wearing four hats. ADR-021 added `cache_write_tokens` and
  `cache_write_1h_tokens` to `LLMResponse`; every site that *reads* provider usage was updated, and
  every site that *copies, zeroes or re-prices* a response was not. Both fields carry a `0` default,
  so all of them kept compiling and kept type-checking:

  - **a cache hit re-billed the original call's premium tokens.** `served_from_cache` zeroed three
    fields and copied both write bands forward, so `exact_cache` and `semantic_cache` reported
    themselves re-spending at the most expensive rate in the table on every hit.
  - **`SpendGuard` undercounted against a live `--cap`**, pricing every Anthropic cache write at base
    rate — the one direction a spend cap must never be wrong in.
  - **streamed replies priced 2x tokens at 1.25x.** The accumulator never read
    `cache_creation.ephemeral_1h_input_tokens`, which the non-streaming path has read since ADR-021.

  Three stand alone:

  - **`count_tools` applied OpenAI's calibration to Anthropic.** ADR-036 measured 1.29 against
    `gpt-4o-mini`'s 0.65 — different in *direction*, since Anthropic re-renders schemas and bills
    more than the raw JSON tokenizes to — and applied it in `minify_tools` only. The ~2x undercount
    fed `PrefixCacheStage`'s floor, so a tool-heavy Anthropic prefix genuinely over 4,096 measured
    about 2,000 and was declined: half of the failure ADR-036 was written to fix, still live.
  - **the counter warm-up covered one encoding.** ADR-038 passed `""` as the model;
    `encoding_for_model` rejects it and falls back to `o200k_base`, so `gpt-4` and `gpt-3.5-turbo`
    still paid the 395 ms vocabulary load inside a 100 ms deadline on request one.
  - **every Anthropic tool call escalated the cascade.** `response_from_anthropic_message` set no
    `extra`, so a `tool_use` reply reached `default_verifier` as empty content with no proposal to
    vet. Cheap model *and* expensive model on every tool-using step — strictly worse than not
    routing, and the opposite of what the config docstring promised.

  Two more surfaced while fixing those: `cache_ttl_selection` identified a prefix by
  `messages[:len-2]`, which grows every turn, so **in any appending conversation the digest never
  matched and the one-hour TTL could never be emitted** for the agent loop it exists for; and
  `last_decline_reason` survived onto a successfully marked request, answering ADR-027's "why did
  this not cache?" about the wrong one.

  Sweeping for the pattern rather than waiting to be told found a **ninth**: `ArmResult` had no
  one-hour field at all, so the benchmark priced every 2x write at 1.25x. The comment directly above
  `ABResult.cost_usd` describes removing that exact asymmetry for the five-minute band; the hour band
  kept it. Same mistake, one layer further out, third occurrence.

  The eighth finding — that `reasoning_budget` ratchets down to its floor — is **wrong**, and is
  rejected with tests rather than an argument. The 2.0 multiplier is a damping term: a budget falls
  only while the model uses under half of it. Simulated over 60 turns it settles at 6,000, 32,000,
  32,000 and 25,600 across four model behaviours, and a mutation setting the multiplier to 1.0 fails
  those tests — because that is what would actually cause the reported failure.

### Added

- **Live measurements are kept now, not just quoted
  ([ADR-039](docs/design/adr/adr-039-a-measurement-that-costs-money-to-recheck-gets-checked-once.md)).**
  Every figure this project has paid for was printed to a terminal, copied into an ADR by hand, and
  the response discarded — forty ADRs and not one recorded exchange anywhere in the tree. The
  numbers were not therefore wrong. The problem is narrower and worse: **re-checking one costs
  money**, so it is checked once and trusted thereafter. That is exactly how `prefix_cache` reported
  a saving for months against a prompt below the provider's cacheable minimum, hitting nothing —
  noticing would have required a paid call.

  `--record PATH` on a live bench run keeps every exchange; `--replay PATH` re-runs it with no key,
  no network and no spend. What a replay proves is narrow and says so: **the library still builds
  the request the provider was measured on.** Not that the provider still answers that way — only a
  fresh call does that, which is why the recording carries its date and prints it in every report.

  A miss raises rather than answering. A replay that fabricated a response would credit this library
  with a saving no provider ever measured — the failure this package is organised against, wearing
  evidence as a disguise. The key covers the `cacheable` markers for the same reason: a key blind to
  them would replay the old cache numbers after `prefix_cache` stopped emitting them, and report the
  saving intact.

- **`detect_window_pressure`, and `context_limit` finally does something
  ([ADR-037](docs/design/adr/adr-037-the-window-binds-the-prompt-and-the-cap-binds-the-reply.md)).**
  `OptimizeConfig.context_limit` had been accepted, validated and read by nothing since the first
  release; so had `tokens.fits_in_window`. A new diagnostic stage reports a prompt approaching the
  limit that will reject it — `prompt_near_context_window` before the failure, and
  `prompt_exceeds_context_window` once it is certain — and is the first production caller of both.
  It observes only: it never modifies a request and reports no saving, like `detect_unstable_prefix`.

  It is **silent on any model whose window this package has not measured**, which today means seven
  Anthropic models. That is deliberate — see below.

- **`CONTEXT_WINDOW` and `MAX_OUTPUT_TOKENS`, read from the provider rather than a doc page.**
  Two per-model tables, both populated from Anthropic's own 400s. They record only what the API
  stated: a model whose window is known merely to *exceed* the probe is absent rather than guessed.

### Changed

- **The real-framework matrix now runs locally, crewai included.** `crewai` declares
  `requires_python = <3.14,>=3.10`, so it cannot be installed into a 3.14 environment: pip resolves
  back to 0.11.2 (a 2024 release) and then fails building an old `numpy`. The compiler error that
  produces is a red herring — the binding constraint is the interpreter, and no toolchain fixes it.
  A Python 3.13 venv (`.venv-frameworks`, gitignored) runs all four adapters with
  `OPTIO_REQUIRE_FRAMEWORKS=1`, so nothing passes by skipping. See the module docstring in
  `tests/frameworks/test_real_frameworks.py` for the commands.

  That venv is also the first check that `requires-python = ">=3.10"` is true below 3.14. The full
  suite passes on 3.13.

### Fixed

- **The first request through any optimizer lost most of the pipeline
  ([ADR-038](docs/design/adr/adr-038-the-first-request-paid-the-tokenizers-startup-out-of-its-latency-budget.md)).**
  `tiktoken` loads its BPE vocabulary lazily, on first use — **395 ms** against a 100 ms
  `latency_budget_ms`. Whichever stage counted first paid that out of the per-request deadline, and
  every stage after it was skipped: measured on a default config, **five of nine never ran**,
  including `prefix_cache`, the largest lossless saving here and Anthropic's only cache mechanism.

  Invisible to every benchmark, because the second request through the same process runs all nine
  and the aggregate looks healthy. Worst in exactly the deployments that make one call per process —
  a serverless handler, a CLI invocation, a scheduled job.

  The counter is now warmed when the pipeline is built. The cost is not avoided, only moved to where
  it is attributable: a one-time cost of *having* an optimizer rather than of using one.
  **`Optimizer()` construction now takes roughly 400 ms the first time in a process** — hoist it out
  of a request handler. Warm-up failures are swallowed; this changes when a cost is paid and must
  never become a new way to fail.

- **`adaptive_max_tokens` could set a ceiling the provider rejects
  ([ADR-037](docs/design/adr/adr-037-the-window-binds-the-prompt-and-the-cap-binds-the-reply.md)).**
  Every model caps completion tokens — a limit distinct from the context window, ranging **32,000 to
  128,000** across the models this package prices — and exceeding it is a hard 400 before any
  generation. The stage sets `max_tokens` on requests that carried none, and its ceiling is
  `max(FLOOR_TOKENS, p95 × 2)`, raised further by a reasoning budget. On `claude-opus-4-1` an
  observed p95 of 16,001 yields 32,002 against a 32,000 cap: rejected, and fail-open then re-sends
  the request unoptimized at full price. The ceiling is now clamped to the model's cap.

  It still never touches a `max_tokens` the caller set, and it declines outright when a reasoning
  budget leaves no legal ceiling at all.

- **`minify_tools` was under-claiming its saving by 71% on Anthropic
  ([ADR-036](docs/design/adr/adr-036-tool-schema-calibration-is-per-vendor.md)).**
  `ANNOTATION_STRIP_CALIBRATION = 0.37` was fitted against `gpt-4o-mini` and applied to every
  vendor. Measured against Anthropic's exact, free `count_tokens` across five tool counts and three
  models, the real ratio is **1.29 — to three decimal places, every time.** The stage claimed 993
  tokens where the provider stopped billing 3,471.

  The vendors differ in **direction**, not just magnitude: OpenAI bills *less* than the raw JSON
  tokenizes to, Anthropic bills *more*, because it re-renders the schema. The calibration is now a
  per-vendor lookup; an unrecognised vendor keeps the *lowest* measured ratio so it is under-claimed
  rather than over-claimed. After the fix, claimed matches real to **0.0%**.

  No cost changes — the same tokens were always being removed. What changes is what the report says
  about it. `scripts/measure_minify_tools.py` re-runs the check against any vendor at zero cost.

- **Benchmark providers serve the request they are given
  ([ADR-035](docs/design/adr/adr-035-a-provider-serves-the-request-it-is-given.md)).** Both providers
  sent `model=self.model` and never read `request.model`. Invisible on an ordinary run — every
  workload is built at the provider's own model — and load-bearing the moment a stage changes it,
  which is exactly what `route_models` and `cascade_routing` do. Routed through these providers
  **both calls would have gone to the same model**, making any reported saving arithmetic over two
  identical calls.

  A test had pinned the old behaviour in place, and its own comment spelled out the consequence:
  *"it means any stage that retargets `request.model` (route_models, cascade) is a **no-op** under
  `bench --live`."* Noticed, understood, and frozen rather than fixed. It is inverted, with a note.

  Per-call cost is now charged against what the API says it served rather than what the provider was
  configured with, and the pre-call estimate uses the requested model — a spend guard tracking the
  wrong one is how a routed run silently overruns its cap. **No existing number moves**, and that is
  asserted rather than assumed.

### Verified

- **Streaming reaches the provider's cache — ADR-019's gate, passed for the first time.** The script
  written to prove it had **never once passed**: its system prompt measured 3,614 tokens against
  Haiku 4.5's 4,096-token floor, so `prefix_cache` correctly declined to mark it and every run
  reported `reads 0 writes 0`. The comment above the prompt claimed "above 4,096"; nobody measured
  it. With an eligible prompt: **6,317 tokens written on call 1 and all 6,317 read back on call 2**,
  and the package saw both *through the stream accumulator*, which is the streaming-specific half of
  the claim. The script now refuses to spend money on an ineligible prefix rather than reporting an
  ambiguous zero.

- **Fault injection against real failures.** The existing suite covers `optio`'s lanes with
  synthetic `RuntimeError`s; `optio_optimize` sits *on* the call path and was untested there. Now
  covered: real `anthropic` exception objects (`RateLimitError`, `InternalServerError`,
  `APIStatusError`, `APIConnectionError`) reach the caller as the *same object*, not merely the same
  type; a failed call writes nothing to the cache; one failure does not poison the optimizer; and a
  live check confirms `NotFoundError` (404) and `BadRequestError` (400) propagate through
  `wrap_anthropic_client` with status intact.

  And the claim `anthropic_streaming` makes most strongly — *"only the terminal event completes a
  request; not exhaustion, not `close()`"* — is now pinned across all five ways a stream dies:
  transport error, caller `break`, early `close()`, early `with`-block exit, and plain exhaustion.
  **None of them caches a partial answer.** That is the streaming half of the hazard ADR-033 found
  live on the unary path. Six mutations, all caught.

- **Framework compatibility** — `langgraph` and `claude-agent-sdk` matrices now run locally
  (skips 9 → 4). `crewai` remains CI-only: it pins a numpy with no cp314 wheel and the source build
  needs `GCC >= 8.4`.

### Fixed

- **Truncation is detected on Anthropic
  ([ADR-033](docs/design/adr/adr-033-truncation-is-a-question-not-a-string-comparison.md)).**
  Anthropic reports `stop_reason: "max_tokens"`; OpenAI reports `finish_reason: "length"`. Every
  truncation check in this package compared against `"length"` only, so **all of them were dead code
  against Anthropic.** Found by the first live cascade run, whose deliberately-truncated request
  (`max_tokens=16`) was supposed to be the guaranteed escalation and was instead accepted and
  returned as final.

  Two guards were affected and the second is the serious one. `default_verifier`'s first and most
  basic check never fired, so a cheap model's cut-off answer was accepted. And **`ExactCacheStage`
  would serve a truncated entry** — `request_key` deliberately omits `max_tokens` *because* that
  guard compensates, so one `max_tokens=16` call could poison the cache entry for every later caller
  who allowed more, in a stage that is `Fidelity.IDENTICAL` and on by default.

  `LLMResponse.was_truncated` now asks the question instead of comparing a string, over a
  `TRUNCATION_REASONS` set covering all three vendors' spellings. The raw `finish_reason` is
  preserved rather than normalised away.

### Added

- **Cascade reports whether it is actually paying
  ([ADR-034](docs/design/adr/adr-034-cascade-pays-by-cost-weighted-escalation-not-by-count.md)).**
  The first live cascade run reported a **50% escalation rate against a 66.7% break-even** — and lost
  **25.3%**. `escalation_rate` counts requests; the bill weights them, and the four requests that
  escalated were 92% of the baseline spend.

  That correlation is structural, not an artefact: a request is likelier to fail a verifier when it
  is long, carries tools, or demands a schema, and each of those also makes it expensive — so a
  count-weighted rate flatters cascade on any realistic workload. `CascadeCost` now reports
  `cost_weighted_escalation_rate`, and `break_even_escalation_rate(expensive, cheap)` computes
  `1 − C/E` from the rate card (66.7% for Haiku 4.5 → Sonnet 4.5, 80% for Haiku 4.5 → Opus 5).

- **`cap_tool_results` now sees the shape Anthropic callers actually send
  ([ADR-032](docs/design/adr/adr-032-cap-tool-results-is-blind-to-the-shape-anthropic-callers-send.md)).**
  The stage saved **7,831 tokens on `mcp_agent`** in the live suite — the second-largest saving
  there — and was a **complete no-op for every `wrap_anthropic_client` user**, the flagship
  integration. Measured on an 8,001-token payload in both shapes:

  ```
  bench shape (role='tool')   content_tokens=8001   saved=5981
  adapter shape               content_tokens=   0   saved=   0
  ```

  Zero for two independent reasons: the stage skipped any message whose `role != "tool"`, and the
  adapter preserves the caller's role, so a tool result arrives as `"user"`. Even without the role
  filter it would have found nothing — a `tool_result` is a non-text block, so `message.content` is
  `""` and the payload sits in `extra[RAW_CONTENT_KEY]`, where the stage never looked. ADR-022
  settled this principle for the cache key (two different images hashed identically); it was never
  carried to the stage whose whole purpose is bounding the largest non-text payload there is.

  Both shapes now cap identically. The wire shape is read through `wire` rather than parsed in the
  stage, and capping rebuilds the raw content instead of editing the caller's dict.

### Added

- **The full published Anthropic price list, both tables
  ([ADR-031](docs/design/adr/adr-031-a-published-price-change-is-data-not-a-prediction.md)).**
  ADR-029 left seven currently-served models deliberately unpriced because nobody here had read
  their rates off the vendor's page. The page has now been supplied and all sixteen rows are
  transcribed, so **every model Anthropic serves reports dollar figures instead of `None`** —
  including Opus 5, Sonnet 5 and Fable 5.

  Three things fell out of the source. The cache multipliers are **universal** — 5-minute write
  1.25×, 1-hour write 2.00×, cache hit 0.10× — holding on all sixteen rows, which confirms a
  derivation `optio_optimize` had been making as an assumption. The four rows ADR-029 added all
  match exactly, so the 3× Opus overstatement it removed is confirmed as a real error rather than a
  suspected one. And the rates are written out per row rather than computed: a multiplier that holds
  across every model today is a fact about today's price list, not a law.

- **Scheduled price changes.** The page lists Sonnet 5 twice — `2 / 10` "through Aug 31, 2026" and
  `3 / 15` "from Sep 1, 2026" — which a `dict[str, ModelPrice]` cannot hold. Whichever single number
  were written would be wrong on one side of the boundary **and wrong by 50%**, larger than most
  savings this library reports. A dated map now resolves it against today's date at lookup time, so
  a process running across the boundary does not keep serving the stale rate and nobody has to
  remember to edit a file that morning. Recording a published, dated commitment is not the
  prediction this project has a rule against.

- **`prefix_cache` no longer pays for a breakpoint nobody reads
  ([ADR-030](docs/design/adr/adr-030-a-breakpoint-nobody-reads-is-pure-cost.md)).** The full live
  Sonnet 4.5 suite returned ten workloads between 75% and 97% cost reduction, and one at **−23.5%**:
  `timestamped_agent` wrote **20,333 prompt tokens at the 1.25× premium and read back zero of them**.
  A prompt whose head changes every turn cannot hit a prefix cache, so every write was a pure loss —
  an ADR-013 rule 1 violation, on a workload that replicates the most common prompt-caching mistake
  in production. `detect_unstable_prefix` had already printed the cause at the top of the same run;
  the writer never asked.

  Two signals now stop it. `unstable_prefix` publishes its verdict for `prefix_cache` to read, and —
  the stronger one — the stage watches what the provider actually served: three consecutive marked
  requests that write and read nothing, and it stops marking. Measured live: **−23.5% → −17.1% →
  −6.0%**, with the residual a bounded three writes that amortize to 0.3% over 200 requests.
  `multi_turn_chat` re-measured unchanged at 74.5%, so the guard costs nothing where caching works.

- **`prefix_cache` counts tool schemas toward the prefix.** It measured `request.messages` alone and
  ignored `request.tools`, which Anthropic caches *ahead of* the system prompt — so it declined
  breakpoints that would have paid, hardest on tool-carrying agents. The new `large_system_agent`
  workload reported "~1715 tokens" for a 5,186-token prefix; the live `mcp_agent` run read ~2,839
  tokens per request against a stable *message* prefix of ~1,387, which is the provider confirming
  what it counts.

### Added

- **`large_system_agent` workload** — a realistic operating manual plus 38 MCP tool schemas, 5,186
  tokens of stable prefix, 1.27× the highest published cacheable floor. The suite's largest stable
  prefix was ~1,400 tokens, leaving it structurally unable to demonstrate `prefix_cache` on Haiku
  4.5, Opus 4.6 or Opus 4.5 at all. Added beside the existing workloads rather than by growing them,
  so every published number stays comparable.

- **`--model` now reaches the stages, not just the provider.** `Workload.requests()` called
  `build()` with no arguments, so every workload built its requests as `gpt-4o` whatever `--model`
  said. Every stage that branches on the model read `gpt-4o` on every live Anthropic run:
  `min_prefix_tokens_for` returned the unknown-model fallback of 1,024, which left
  [ADR-027](docs/design/adr/adr-027-the-cacheable-prefix-floor-is-per-model.md)'s per-model cacheable
  floor **inert inside the benchmark built to validate it** — a live Haiku run placed the breakpoint
  and came back `reads 0 writes 0`, the exact failure ADR-027 exists to prevent, on the exact model
  it describes. It hid because gpt-4o's fallback equals Sonnet 4.5's real floor, so the one model
  that conceals the bug is the one on which `prefix_cache` appeared to work.

- **The benchmark's Anthropic provider reads the cache-write field it bills from.** It took
  `cache_read_input_tokens` and never `cache_creation_input_tokens`, so written tokens were dropped
  from `input_tokens` outright and the cache-write premium added to `ABResult.cost_usd` was inert.
  A live Sonnet 4.5 run reported `reads 18,300 writes 0`, which cannot happen. It now shares
  `wire.response_from_anthropic_message` with the streaming adapter, which has been correct since
  ADR-021. The corrected figure for that run is **74.5%, not the 85.2% the bug reported.**

- **A price is no longer inferred across model generations
  ([ADR-029](docs/design/adr/adr-029-a-price-may-not-be-inferred-across-model-generations.md)).**
  `optio`'s pricing lookup documented itself as matching "exactly first and then by longest prefix"
  and was implemented as `if name in normalised` — substring containment — over a table whose
  Anthropic rows were `claude-opus-4`, `claude-sonnet-4` and `claude-haiku-4`, **none of which is a
  model id the API will serve**.

  Five distinct Opus generations therefore collapsed onto one row. Anthropic cut Opus list pricing
  at 4.5, so a million input and two hundred thousand output tokens on `claude-opus-4-5` reported
  **$30.00 against a $10.00 bill — 3× over, silently.** Containment also priced anything merely
  containing a known name: `not-really-gpt-4o-at-all-v2` came back at gpt-4o's rate. The module
  docstring has always said "not the price of a similar-looking model"; the code now agrees.

  A prefix match requires the remainder to name the *same* model — a release date (`-20251101`) or a
  Bedrock revision tag (`-v1`) — never a version bump. Four rows are added (`claude-opus-4-5`,
  `claude-opus-4-1`, `claude-sonnet-4-5`, `claude-haiku-4-5`); **seven currently served models are
  deliberately left unpriced** because nobody here has read their rates off the vendor's page, and
  the lane now says so once per model rather than returning a silent `None`.

- **The optimizer prices the ids the API reports back.** `optio_optimize.PRICING` was reached by
  exact `.get()` only, so of the eleven ids `models.list` returned, **ten were unpriced** and
  produced no cost figures at all — including dated forms of models the table carried. `pricing_for`
  applies the same rule as core's, duplicated rather than imported to keep ADR-013's boundary.
  `CHEAP_COUNTERPART` also stopped pointing every Anthropic route at `claude-haiku-4`, which 404s.

### Changed

- **The benchmark no longer reports a cost delta it did not cause
  ([ADR-028](docs/design/adr/adr-028-a-cost-delta-is-only-a-measurement-when-the-arms-differ.md)).**
  Five of the twelve live workloads send **byte-identical requests in both arms and make the same
  number of provider calls** — verified by capturing every request each arm hands the provider —
  and the report printed a cost percentage for each of them anyway.

  Those percentages were the provider's own output nondeterminism, and they failed in both
  directions at once. `timestamped_agent` (−1.6%) and `sampled_creative` (−4.7%) read as ADR-013
  rule 1 violations — a cost increase caused by the optimizer, the one outcome this package treats
  as unacceptable — and the previous measurement iteration opened by planning a live isolation run
  to find the stage responsible. No stage was responsible. In the other direction, `unique_questions`
  claimed a **2.8% saving** on the workload whose stated purpose is "Included so the suite reports
  its own limits."

  Each arm now folds a digest of everything it sent, and `cost_is_attributable` is False when the
  digests and call counts both match. The report keeps both dollar figures — money really was spent
  — and replaces the percentage with `NOT ATTRIBUTABLE`. `--control` already measured this
  nondeterminism for the *quality* line; nothing had carried the reasoning across to cost.

- **The cacheable prefix floor is per-model, and the benchmark now prices its own cache writes
  ([ADR-027](docs/design/adr/adr-027-the-cacheable-prefix-floor-is-per-model.md)).** Anthropic's
  minimum cacheable prefix spans a factor of eight — 512 on Opus 5 up to **4,096 on Haiku 4.5** —
  and `MIN_PREFIX_TOKENS = 1024` was wrong in both directions: too high for Opus 5, where it declined
  a breakpoint that would have worked, and too low for four models, where it placed one the provider
  silently discards **while the note reported success**.

  It is now a per-model lookup, longest-prefix matched so dated ids resolve through their alias. An
  unrecognized model keeps 1,024, so nothing outside the table changes behaviour. Below the floor the
  stage **declines and names the model and the figure**, because a zero-cache-read result was
  previously indistinguishable from a broken stage.

  That diagnostic immediately earned itself. On `mcp_agent` — an 11,799-token prompt, the one workload
  clearing 4,096 — it reports `prefix is ~1387 tokens, below claude-haiku-4-5's 4096-token cacheable
  minimum`. **The stable prefix is ~1,400 tokens; the rest is tool results that change every step.**
  Under the old constant that prefix cleared 1,024, got a marker, and had it discarded.

  Worth knowing when choosing a model: this lever needs a *stable* prefix above the floor, and typical
  agent traffic carries 1,000–2,000 tokens of it. That clears Opus 5 comfortably, often clears the
  1,024 tier, and rarely clears Haiku 4.5 unless the system prompt and tool schemas are large.

- **The benchmark under-billed its own cache writes.** `ABResult.cost_usd` priced every prompt token at
  the base or cached rate and never at the **1.25x write premium**, so the arm that places breakpoints
  was billed 1.25x tokens at 1.0x — the identical asymmetry [ADR-021](docs/design/adr/adr-021-cache-ttl-selection-needs-its-accounting-first.md)
  removed from `SavingsReport`, reproduced inside the benchmark that measures it, and flattering the
  optimized arm every time `prefix_cache` wrote. Writes are now tracked on `ArmResult`, priced, and
  reported beside reads — the pair being what tells "the breakpoint was ignored" (reads 0, writes 0)
  apart from "the prefix changed between calls" (reads 0, writes N).


- **`trim_history` now prices the output it buys, and stops trimming when it cannot pay
  ([ADR-026](docs/design/adr/adr-026-trimming-must-price-the-output-it-buys.md)).** The first full
  live Anthropic benchmark caught the package's flagship default-on stage **increasing** total cost.
  Isolated — three arms, one workload, one window — it saved 1,116 input tokens and bought 717 output
  tokens, and output bills at 5x input on Haiku, so a 5.4% input saving became an **11.0% cost
  increase**. Removing the stage returned output to baseline exactly.

  Dropping old turns also drops the model's own prior short replies, and with them the pattern it was
  matching, so it answers at greater length. **And the effect is provider-dependent, which is what
  makes it hard:** on identical workloads through identical code, trimming makes `gpt-4o-mini` *terser*
  (output −38%) and `claude-haiku-4-5` *far more verbose* (+189%). There is no constant to encode —
  a figure fitted to one vendor forfeits a real win on the other.

  So the stage now trims only when the input tokens it removes are worth more than the output it
  risks, priced from the **model's own rates** in `PRICING`. One rule, no per-vendor branch: the
  output/input multiple is 5 on Haiku and 4 on `gpt-4o-mini`, so the same rule produces a different
  threshold per model. The bootstrap risk figure is superseded by **observation** as soon as the stage
  has seen enough trimmed and declined replies to measure the difference itself — two groups, not one
  running mean, because a single average cannot separate "replies got longer because we trimmed" from
  "replies got longer because the questions got harder".

  Verified live on both providers:

  | workload | provider | before | after |
  |---|---|---|---|
  | multi_turn_chat | claude-haiku-4-5 | **−11.0%** | **0.0%** |
  | multi_turn_chat | gpt-4o-mini | −0.4% | +4.2% |
  | multi_turn_chat_long | claude-haiku-4-5 | +10.9% | **+13.8%** |
  | multi_turn_chat_long | gpt-4o-mini | +19.9% | +17.6% |

  Both losses gone, both wins kept — and the Haiku long case *improved by three points*, because
  declining the early unprofitable trims left the profitable ones intact. Trimming less made it worth
  more.

  **Behaviour change worth knowing:** a conversation of very small turns is now never trimmed, so the
  prompt is bounded by value rather than by turn count. Nothing was relying on the old guarantee —
  `context_limit` is a documented config field **no stage reads**, and `fits_in_window` has no
  production callers — but that is a gap in its own right and is recorded as one.


- **`structured_output` is now off by default, and no stage books a saving it cannot attribute
  ([ADR-024](docs/design/adr/adr-024-a-stage-may-not-book-a-saving-it-cannot-attribute.md)).** The
  first end-to-end live agent run — four scenarios, both arms, `gpt-4o-mini`, reproduced twice —
  found this package making two scenarios **more expensive while reporting a saving**: `parallel`
  **−3.0%** against a claimed 10.0%, `empty_result` **−4.3%** against a claimed 13.2%.

  Disabling one stage returned the prompt to *exactly* the control arm's size (661→635, 523→497), so
  `structured_output` owned the entire regression — and on the two large scenarios it was marginally
  *worse* than not running, so it paid for itself in none of the four.

  The cause was a guard that contradicted its own docstring. The class says *"Only acts when a schema
  is already present"*; the code read `response_format is None and not request.tools`, so it also
  fired on any tool-using request with **no schema at all** — which is every call of every agent
  workload. It appended *"Respond only with the requested structure. No preamble or explanation."* to
  requests whose reply is a **tool call**, where no preamble exists to suppress. Measured output:
  132→**137**, 94→**95**, 28→**28**, 165→148. It raised output in two scenarios and changed nothing
  in a third.

  Three changes follow. The guard now requires a schema. The stage no longer claims
  `saved_output_tokens` from a hypothesised 40-token preamble — `savings.py`'s opening rule is *"only
  count what was avoided, never what was hoped for"*, and this was the one stage that did; if the
  suppression works it shows up in provider-measured `actual_output_tokens`, the same place ADR-020
  leaves fan-out's effect. And **a stage that adds tokens now reports that as a negative saving**, so
  `baseline = actual + saved` yields the true baseline instead of one inflated by the stage's own
  instruction: 523 + (−26) = 497, exactly what the control arm billed.

  **A savings report can therefore now show a negative number.** That is deliberate. A stage costing
  more than it saved was structurally invisible for this package's entire history, and rounding the
  loss up to zero is how a 4.3% cost increase came to be reported as a 13.2% saving. `concision`
  carries the identical pattern and gets the identical accounting fix; it was already off by default.

  Verified live after the fix: the regression is gone (0.0% and −0.8%, the latter output-sampling
  noise on byte-identical prompts) and the real wins are intact — **+47.9%** and **+62.2%**.

  **If you relied on `structured_output`,** switch it on by name. Its benefit has never been measured
  on a request that actually carries a schema, which is the evidence it needs before it goes back to
  being a default.

### Added

- **Cascade routing: call cheap, verify, escalate
  ([ADR-023](docs/design/adr/adr-023-cascade-routing-calls-cheap-verifies-and-escalates.md)).**
  `route_models` decides cheap-vs-expensive *before* the call, from prompt length. Its own docstring
  records the guess failing: *"What is 17 times 24, minus 89?"* is eight words with no tools, so it
  routes — and the cheap model answers **329** against 319. Cascade removes the guess. It calls the
  cheap model, runs a verifier over the answer, and escalates to the originally requested model only
  on failure, so the worst case stops being wrong-and-cheap and becomes slow-and-right.

  **Not a stage.** A stage's contract is `before → call → after` with exactly one provider call, and
  this needs two — the same wall ADR-017 hit with batch dispatch. Cascade wraps the provider call
  instead, so the stage contract stays single-call and every existing hook still fires once, against
  whichever request actually goes out last.

  **The verifier is the fidelity claim, not the cascade.** `cascade_verifier` takes a caller-supplied
  `(request, response) -> bool`; the built-in `default_verifier` catches only what a cheap
  deterministic check can — an empty or truncated answer, a `response_format` request whose reply is
  not JSON or drops a required key, a proposed tool call naming an unknown tool or missing a required
  argument. It does **not** catch "329", and says so in its own docstring. Grading with a model would
  spend the very escalation cascade exists to avoid; `ModelJudge` is provided for callers who want
  that trade and reports its own cost so the saving stays net rather than gross.

  **A rejected cheap answer is never cached**, structurally rather than by special case: the cheap
  attempt runs inside the wrapper against the raw provider call, so no cache `after` hook ever sees
  it. Only the accepted final answer reaches the cache. Keying attempts by model instead would have
  reopened the cache key ADR-022 had just finished getting right.

  Live gate: twelve graded probes, gpt-4o against gpt-4o-mini, $0.0013 total. Easy prompts 100% on
  both; hard prompts 100% against 88%, the single regression being the motivating case reproduced
  exactly. With an oracle verifier cascade scores 12/12 at ~14% of all-expensive cost, against static
  routing's 11/12 at 6% — and **that gap is the result**, because with the default verifier cascade
  would score the same 92% static routing does.

  Three eligibility expansions ride along, each off by default: `cascade_structured_output` (the
  requested JSON *is* a verifier), `cascade_max_tokens` (the escalation net turns the length ceiling
  into a cost knob rather than a safety one), and `cascade_tools` (safe only because the provider
  returns a *proposed* call, not an executed one, so it can be vetted before any side effect).
  `CascadeStats` reports attempts, escalation rate, per-phase latency, and a `cost_summary` that
  counts the wasted cheap attempt on escalated requests — spend that wrapping the provider call had
  otherwise made invisible. **Off by default:** the live run clears the mechanism, not the judgment
  ADR-015 reserves for whoever supplies `cheap_model`. Mutually exclusive with `route_models`, which
  raises at construction rather than routing twice.

### Fixed

- **Two different images produced the same cache key, and `exact_cache` is on by default
  ([ADR-022](docs/design/adr/adr-022-an-image-is-content-and-the-cache-key-was-the-urgent-half.md)).**
  `request_key` keyed `[role, content, name]` per message. An image block never reaches
  `Message.content`, so it rode through in `extra["_raw"]`, which `UNKEYED_FIELDS` excluded as
  *"provider transport details, not semantics"* — a classification that does not hold when the image
  **is** the semantics. Two requests with identical prompts and different images hashed identically,
  and `exact_cache` caches at `temperature == 0`, exactly the setting deterministic vision work uses.
  "Describe this image" over two different images returned the first image's description for the
  second. This is a **wrong answer**, not a mis-measurement, and the same defect class as the `stop`
  bug already recorded in that dict.

  Every non-text block is now keyed — `tool_use` arguments, `tool_result` payloads, and block types
  nobody has seen yet — because the bug came from judging a block semantically irrelevant. Text blocks
  stay out, since their text is already keyed through `content`. The digest is appended only when
  non-text content exists, so text-only keys are unchanged. **A digest, never the bytes:** §10 covers
  an image at least as squarely as prose. Expect `exact_cache`'s hit rate on vision requests to drop
  to near zero; those hits were wrong answers.

- **Images counted as zero tokens
  ([ADR-022](docs/design/adr/adr-022-an-image-is-content-and-the-cache-key-was-the-urgent-half.md)).**
  `count_request` returned **8 tokens** for a request billing ~1,535. The reported saving *percentage*
  was never inflated by this — `pipeline` takes the provider's own `input_tokens` on a live call, so
  image tokens sat on both sides of the ratio — but `fits_in_window` applies a 1.15x margin to guard a
  few percent of estimator error while a vision request was off by two orders of magnitude, which is
  precisely the provider-side rejection that function exists to prevent. Short-circuited vision
  requests also under-reported what they avoided, since no provider number exists on that path.

  **Measured, not taken from the documentation.** Anthropic's `messages.count_tokens` is exact and
  free, so the estimator is calibrated against synthetic PNGs with a text-only baseline differenced
  out. That mattered: the published `(w*h)/750` formula is accurate to ±4% from 512x512 through
  1200x958 and then breaks, overstating 1568x1568 by **2.15x** and 400x3000 by 3.6x, because a
  1568-pixel edge cap and an area cap near 1,600 tokens apply that it does not mention — while small
  images cost *more* than it says. Validated afterwards on **thirteen held-out sizes** the constants
  were not fitted to: worst error 5.5%, mean **+1.4%**, ten of thirteen over-estimating, which is the
  safe direction in both places the number is used.

  Dimensions come off PNG, JPEG, GIF and WebP headers with **no new dependency** — Pillow is a large
  native wheel and §4.4 exists to keep it out of a tree for four integers. An image whose dimensions
  cannot be read (a URL source, an unreadable header) counts a documented constant, **never zero**.
  OpenAI uses its published tile formula and says plainly that it is unmeasured here.

  Reducing image cost is deliberately **not** shipped: `detail: "low"` degrades what the model can
  see, making it `ALTERED`, and ADR-015 wants a vision *accuracy* probe first. No flag was added, so
  it cannot be turned on by accident.

### Added

- **One-hour cache entries are priced correctly, and can now be requested
  ([ADR-021](docs/design/adr/adr-021-cache-ttl-selection-needs-its-accounting-first.md)).** Anthropic
  charges **2.0x** base input to populate a one-hour cache entry against 1.25x for a five-minute one.
  `_cost` knew about a single write rate, so any one-hour write was under-billed by **37.5%** — in
  the direction that inflates this package's headline saving. **That half shipped first and stands on
  its own:** a caller who set their own `cache_control: {"ttl": "1h"}` breakpoint was already
  mis-priced by this package, and the provider had been reporting the split in
  `usage.cache_creation.ephemeral_1h_input_tokens` all along with nothing reading it. `LLMResponse`
  gains `cache_write_1h_tokens` (a **subset** of `cache_write_tokens`, matching how the provider
  reports it), `ModelPricing` gains `cache_write_1h_usd_per_m`, and `SavingsReport` carries the band
  through. This project has already paid for the identical asymmetry once, when writes were omitted
  from the prompt total and a published 53.7% figure turned out to be 50.1%.

  `cache_ttl_selection` then makes `PrefixCacheStage` ask for an hour — but only once expiry has been
  **observed**, meaning the same prefix seen again after a gap longer than five minutes. Nothing
  predicts a gap. Prefixes are identified by hash and never by text (§10), in an LRU bounded at 1,024
  entries (§11). A caller's own `cache_control` is still never overwritten.

  Measured live on `claude-haiku-4-5`, four rounds **330 seconds apart** with both arms interleaved
  in one wall-clock window: **−30.9%**, the control re-writing 16,888 tokens against the treated
  arm's 4,222 + 4,222 written and 8,444 read. The 1-hour TTL is honoured and needs no beta header.

  **It ships off by default, and the measurement is why.** The cumulative curve matters more than the
  total: after round 2 the treated arm is **29.9% more expensive**, since the upgrade write costs
  2.0x, and it only crosses over in round 3 — exactly the `m >= 1` break-even. A prefix that expires
  once and is then never used again loses ~30% permanently, and that is a two-turn conversation
  resumed after a break, not an exotic case. Observed expiry is sound backward-looking evidence, but
  the upgrade needs a forward fact — will this prefix be used again inside the hour — that the run
  measured nothing about. ADR-013 rule 1 does not say "reduce cost in expectation". Turn it on for a
  slow agent loop and it pays; it is not something to do to someone unasked.

- **`Optimizer.afan_out`: dispatch order as a cost lever
  ([ADR-020](docs/design/adr/adr-020-fan-out-warm-up-is-an-async-dispatch-order.md)).** N concurrent
  calls over a shared prompt prefix each pay to populate the provider's cache, because none of them
  can see another's write. Sending one first turns that into one write plus N−1 reads.

  Measured on `claude-haiku-4-5`, five branches over a 4,223-token shared prefix, three arms
  cold/warmed/cold with a per-arm nonce so each starts cold: **−68.2%** total cost, with **0.0%
  spread between the two cold arms** — they came out byte-identical. Isolating the shared prefix
  itself gives **73.6% off against a predicted 74%**. This is the first modelled number in this
  package that survived contact with a provider, and the reason is structural: nothing here estimates
  provider behaviour, the saving falls out of dispatch order and published rate cards.

  Not a stage — nothing about any request changes, only the order they go out in, which is ADR-017's
  test. Not a second surface either, since responses come back on the same stack frame from the same
  pipeline and report. **Async only, and not for lack of effort:** a caller looping over five
  synchronous calls already gets warm-up ordering for free, since sequential execution *is* warm-up
  ordering. The problem exists only under real concurrency, and serving a thread-pool caller would
  mean this package owning a thread pool.

  **Opt-in, never inferred.** The cost is one round trip of latency prepended to the batch, and
  doubling a page's time to first byte to save a fraction of a cent is not a trade a library should
  make on anyone's behalf. Warming is skipped automatically when it cannot pay: fewer than two real
  calls, no shared prefix, or a shared prefix below the cacheable floor — below which nothing is
  cached at all, so the warm-up would be pure latency, silently. Short-circuited requests are
  excluded from the decision, so a cache hit can never justify a real call's delay.

- **Streaming is optimized on Anthropic, sync and async
  ([ADR-019](docs/design/adr/adr-019-a-streamed-call-gets-the-request-side-pipeline.md)).** A
  `stream=True` call used to bypass the wrapper entirely, on the stated grounds that a pipeline
  built around one request producing one response "can only buffer a token stream". That holds for
  the few stages which read a reply and not for the majority, which only rewrite the request — so a
  streaming caller was getting **zero of nineteen stages**, including `prefix_cache`, the largest
  lossless saving in the package and the only reason Anthropic caches anything at all. For anything
  user-facing, streaming *is* the production mode; "plug and play except in the mode you ship" was
  the defect.

  Every `before` hook now runs and the transformed request is what goes on the wire. The `after`
  hooks run when the stream finishes, from a proxy that forwards each event unchanged and
  immediately while accumulating alongside — nothing is withheld, so the first token arrives exactly
  as soon as it would without this package, and only the bookkeeping is deferred. A cache hit is
  replayed as a synthesized event sequence built through the SDK's own pydantic types, so a
  streaming caller gets cache hits without needing to know they can happen.

  **An abandoned stream completes nothing.** No cache write, no observation, no report row. A
  half-read reply stored by `exact_cache` would be served confidently and permanently to everyone
  who later asked the same question; the report undercounting an abandoned stream costs a number
  instead.

  Live gate: two streamed calls sharing a 4,217-token system prefix — 4,217 written on the first,
  **4,217 read at 0.1x on the second**, and the savings report's own cached/written figures match
  the provider's exactly, which is what proves the accumulator threaded the usage through rather
  than the discount being granted and unnoticed.

  **OpenAI streaming remains unoptimized and still says so.** `ChatCompletionChunk` is a different
  shape with different rules; shipping "streaming works" while one of two adapters silently did
  nothing would repeat the defect one level up.

- **`reasoning_budget`: the most expensive tokens in a request finally have a lever
  ([ADR-018](docs/design/adr/adr-018-reasoning-budget-is-a-cost-lever-and-an-altered-one.md)).**
  Reasoning tokens bill at the completion rate — 4–5x input on every model in `PRICING` — and on a
  reasoning model the thinking trace routinely runs several times the length of the visible answer.
  They were the only tokens this package could not influence, while nineteen stages aimed at the
  cheaper half of the bill.

  `ReasoningBudgetStage` lowers a caller-set budget toward the p95 of observed output and does
  nothing else: never raises one, never invents one where the caller set none, never goes under
  Anthropic's 1,024 floor, never acts before twenty real observations exist. `ALTERED` and off by
  default. It claims no saving — `output_tokens` bundles thinking with the answer on every provider
  in `PRICING`, so what a lowered budget avoids is unknowable from inside the stage. That also
  settles the `chain_of_draft` overlap more firmly than an ordering could: neither stage credits a
  completion token, so no arrangement of the two can count one twice.

  **The live run found the saving and then demolished the reason it looked safe.** Three arms,
  control-treated-control on `claude-haiku-4-5`, twenty graded tasks: **−21.9%** against the mean of
  two bracketing controls, below both, against a control-to-control noise floor of 11.7%, with
  accuracy unchanged and nothing truncated. But **zero of forty control calls exceeded the ceiling**
  — the longest unconstrained trace was 2,480 tokens against a 4,438-token ceiling. So the reduction
  is not the ceiling truncating anything. `budget_tokens` is a target that shapes how long the model
  thinks, not merely a cap, and the stage's original defence ("it cannot bind on the observed
  distribution") was answering a question nobody had asked. The flag stays off.

  Two limits on that evidence, recorded rather than glossed: every arm scored 100% on both task
  sets, so the accuracy column cannot detect degradation and measures only that Haiku 4.5 finds
  these ten tasks easy; and it is one model, one workload, one treated arm.

### Fixed

- **`adaptive_max_tokens` could turn a working reasoning call into a failed one.** It is on by
  default and derives its ceiling from observed *total* output — 600 tokens on a workload whose
  replies run 300. Anthropic rejects a `max_tokens` at or below `thinking.budget_tokens`, so once
  `thinking_budget` reached the wire, that pairing became a 400 and a fail-open call at full price
  while the report showed a saving. The ceiling is now floored to the budget plus 512 tokens of
  answer headroom, and rule 7 in `stages/__init__.py` puts the budget reduction first so the ceiling
  clears the budget that will actually be sent. Found by a test written before the stage existed.

- **`BatchOptimizer`: a second public entry point, for work that tolerates hours of latency
  ([ADR-017](docs/design/adr/adr-017-batch-dispatch-is-a-second-surface.md)).** Providers sell
  asynchronous processing at roughly half price with no quality trade at all — the same model
  returns the same answer, later — and it is the largest discount this package had never offered.
  It could not be a stage: every stage answers *what should this request look like*, and the
  pipeline's contract is that a response comes back on the same stack frame. Batch answers *when,
  and by whom*, and there is neither a response to return nor an error to fail open into.

  The caller declares latency tolerance and the library never infers it. There is no heuristic for
  "this looks like it can wait", because putting a waiting user behind a 24-hour queue to save half
  a cent is a product failure this package should be structurally incapable of causing — hence a
  separate class rather than a flag on a shared path.

  `BatchOptimizer` *owns* an `Optimizer` rather than reimplementing one, so the stages run first,
  unchanged, and the discounts compose. Pass your synchronous optimizer and the two surfaces share
  an exact cache in both directions: a request already answered never enters a queue, and an answer
  that arrives hours later runs its `after` hooks on retrieval and is served to the synchronous path
  immediately. Verified live against OpenAI — of three requests with one pre-answered, two were
  submitted, and a repeat after retrieval made **0 provider calls**.

  Failure is explicit rather than fail-open. ADR-013's rule 1 works synchronously because there is
  somewhere to fall back to; a failed submission has not degraded to a slower path, it has not
  happened. So `submit()` raises a `BatchSubmissionError` naming which items were and were not
  accepted, and never quietly converts one batch into 10,000 synchronous calls — a fail-open that
  would cost twice what batching was asked to save.

  The savings figure is **arithmetic, not measured**: the provider's published 50% applied to real
  token counts, because the A/B harness cannot express a result that arrives tomorrow.
  `BatchReport.summary_lines` prints that caveat as its own line rather than letting the number sit
  beside the measured ones looking identical.

- **`optio_optimize.wire`: one place that turns a request into a provider payload.** The live
  OpenAI adapter once did not forward `request.tools`. Nothing raised, and the whole `mcp_agent`
  run reported `minify_tools` saving 3,240 tokens while both arms billed byte-identical totals,
  because the field the stage had rewritten was never sent. Batch submission needed the same
  translation as JSON rather than SDK keyword arguments — a second copy, and a second chance at
  exactly that omission — so both call sites now share one. A test walks `LLMRequest`'s own fields
  and fails unless each is demonstrably on the wire or named in `wire.UNSENT_FIELDS` with a reason,
  the same guard `request_key` applies to the cache key.

- **`wrap_anthropic_client`: one line to optimize an Anthropic client, sync or async.** Until now
  the only real-SDK integration was OpenAI, so "plug and play" meant "plug and play if you use one
  vendor" — and Anthropic is the vendor where these stages are worth the most, because its prefix
  caching does nothing at all without the marker `prefix_cache` places. Wraps
  `client.messages.create` in place and returns the same client, so existing call sites are
  untouched.

  Two silent defects surfaced while building it, and both initially looked like success.
  `inspect.iscoroutinefunction` returns `False` for *both* Anthropic clients, because the SDK
  decorates `create` — so `AsyncAnthropic` took the synchronous branch, the un-awaited coroutine was
  handed back, and `await` ran it as an ordinary unoptimized call. **Eight of eleven tests passed
  that way**: an adapter doing nothing while reporting success. Fixed with `inspect.unwrap`.
  Separately, the `cacheable` marker reached the wire through `wire.py` but not through the
  adapter's raw-parameter path: the stage marked the message, the savings ledger recorded the work,
  and the field was never sent — the same shape as the `tools` omission above, one function over.

- **`optio_optimize.invariants`: rules that hold whatever the stages do.** Two entry points for two
  different kinds of rule, and the distinction is the point. `check(request)` enforces what is true
  of *any* request — a tool result must follow the call it answers, something must be answerable.
  `check_transform(original, sent)` enforces what is true of a *rewrite*: the last user message
  still present, the system prompt still present, message order unchanged, no tool invented, no
  called tool removed. The first kind a provider will reject for you. **The second kind nobody
  enforces at all** — a request that quietly lost the user's question is well-formed, accepted, and
  billed, which is exactly how `trim_history`'s defect below survived 1,304 tests.

  A `Violation` carries `(rule, message_index, role)` and never prompt content, because violations
  are printed and reach CI logs (§10).

  Pointing it at real traffic immediately produced **20 violations on a demonstrably clean run** —
  two false positives, both worth recording. Tool calls were read from `extra["tool_calls"]` while
  adapters store them under `extra["_raw"]`; and the system-prompt rule compared identity, so
  `structured_output` editing the system prompt read as dropping it. A checker that cries wolf on
  clean traffic is worse than no checker, because the next real violation gets ignored along with
  the noise.

### Changed

- **`Pipeline.execute` is now `prepare` + `complete`.** Batch needs the two halves of a request
  separately — run the stages now, hand back the provider's answer tomorrow — and a second
  implementation of "run the stages" would be a second place for one to be skipped, with the
  divergence showing up as batch and synchronous calls being optimized differently for reasons
  nobody could see. Behaviour is unchanged; `execute` and `aexecute` are now two calls around the
  provider instead of inline loops.

- **`prefix_cache`'s Anthropic claim is measured, and it was understated.** The docstring called the
  marker worth "roughly 30% of total spend on a long conversation" with no run behind it — the same
  class of unevidenced claim that credited this library with a 36.3% saving the live API measured at
  −1.8%. Measured 2026-07-30, six turns on `claude-haiku-4-5` through `wrap_anthropic_client` with
  the stage isolated (ADR-015 rule 2) and the *disabled* arm run first so residual server-side cache
  favours the baseline: **23,023 of 30,113 input tokens served from cache against 0**, for a **50.1%
  cost reduction on identical token counts** — 30,111 versus 30,113 sent is noise. The stage avoids
  no tokens whatever and halves the bill by changing what they cost. This is the only claim in the
  package a measurement has ever moved *up*.

- **Cache-*write* tokens are counted and priced.** `LLMResponse.cache_write_tokens`,
  `SavingsReport.provider_written_tokens`, and `ModelPricing.cache_write_usd_per_m`, because prompt
  tokens fall into three price bands and this package modelled two. Writes are the band that costs
  **more** than base input — 1.25× on Anthropic for the 5-minute TTL requested here — and they were
  missing twice over: `wire.response_from_anthropic_message` left `cache_creation_input_tokens` out
  of the prompt total altogether (200 reported against a true 4,805 on one measured turn), and even
  once counted there was no rate to price them at.

  Both errors ran in the same direction: they understate what a *cached* call cost, so they inflate
  whatever saving `prefix_cache` appears to deliver. That is why neither was ever caught by a report
  looking wrong — and it is why the figure above reads 50.1% rather than the 53.7% first published.
  The token counts never changed; only their price did. `_cost` now takes `written` and the
  arithmetic behind 50.1% is locked by a test.

- **Pricing rows for `claude-haiku-4-5`,** without which the measurement above could report tokens
  but no cost. Both the alias and the dated id, since callers write one and the API returns the
  other. Every Anthropic row now carries its write rate; OpenAI rows leave it `None`, since OpenAI
  populates its cache for free.

### Known issues

- **`MIN_PREFIX_TOKENS = 1024` cannot express Anthropic's floor, which is per-model.** As of
  2026-07-30 the published minimums span a factor of eight: 512 (Opus 5), 1,024 (Opus 4.8, Sonnet 5,
  Sonnet 4.6/4.5), 2,048 (Opus 4.7, Haiku 3.5), **4,096 (Haiku 4.5, Opus 4.6/4.5)**. So one constant
  is wrong in both directions — too high for Opus 5, where a breakpoint that would work is declined,
  and too low for four models, where one is placed and silently discarded while the stage's note
  claims success. Not fixed here because a model-aware floor changes what every Anthropic caller
  sends, and ADR-016 does not let one measurement carry that. The cost of being wrong is a marker
  that does nothing, never a wrong answer. `docs/optimize-benchmarks.md` carries the table.

- **`cap_tool_results` does nothing on Anthropic.** It selects messages by `role == "tool"`, which
  is OpenAI's protocol; Anthropic returns tool output as `role: "user"` carrying a `tool_result`
  content block, and the text lives inside that block rather than in the message content this
  package models. So the stage skips Anthropic tool traffic entirely — one of the highest-value
  stages for agent loops, silently absent on one of two supported vendors. Fixing it means deciding
  how block-nested text is exposed to stages, which is a design question rather than a patch.

### Fixed

- **`trim_history` dropped the task in agent loops, and did it silently.** In a chat the first user
  turn is an old question that has been answered; in an agent loop it *is the task*, and everything
  after it is the agent's own tool traffic. The sliding window's oldest entry was therefore the one
  statement of what the model was supposed to do — and providers accept a conversation with no user
  message at all, so nothing failed. The model inferred a task from the tool results and answered a
  question nobody asked.

  Found by `scripts/real_agent_run.py`, a new harness that runs a real OpenAI Agents SDK agent with
  four real tools through the optimizer. Nothing in this repo had ever done that: the framework
  tests check adapter recognition and say in their own docstring that "nothing here calls a model",
  and the adapter tests mock the HTTP transport. It broke on the first run.

  Isolated per ADR-015 rule 2: disabling `structured_output` changed nothing, disabling
  `trim_history` fixed it. The stage that broke the answer also cost *more* — output more than
  doubled (140 → 288 tokens), because a model that has lost the question writes longer. Fixed:
  3,757 in / 288 out / $0.00074 and wrong, becomes 3,816 in / 131 out / $0.00065 and correct.

  The first user turn is now a structural floor, the way the system prompt already was, rather than
  a new default — `anchor_turns=0` cannot lower it. There is no workload where discarding the
  question is the cheap option.

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

[Unreleased]: https://github.com/Aniketh-74/Optio/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Aniketh-74/Optio/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Aniketh-74/Optio/releases/tag/v0.2.0
[0.1.0]: https://github.com/Aniketh-74/Optio/commit/840e7f8
