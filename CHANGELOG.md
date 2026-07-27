# Changelog

All notable changes to `optio` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**What versioning means here.** The public API (§8.1) and the emitted signal names (§7.2) are
the compatibility surface. Renaming or removing a `gen_ai.run.*` attribute is a breaking change
even when no Python signature moves, because downstream OPA, Cedar and AGT policies are written
against those exact strings — a policy that silently stops matching is worse than one that
fails to load.

## [Unreleased]

Nothing yet.

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
