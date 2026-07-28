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
