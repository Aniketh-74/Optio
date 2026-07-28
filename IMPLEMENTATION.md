# IMPLEMENTATION.md

> **Single source of truth** for the project. Referenced from first commit through production.
> Read this before writing any code. Do not make architectural decisions during development —
> if a decision is missing, add an ADR (§15) and update this document first.

**Name:** `optio` (resolved Jul 2026, closing OQ-1). Latin *optio* — the officer chosen to
observe and assist, never to command; the fit with ADR-001 ("emit signals, never enforce") is
the reason it was picked. The working name `agentmeter` was retired because the PyPI
distribution was already taken by an unrelated project. Prose below that predates the rename
still says "agentmeter"; it is left as written, because this document records decisions as
they were made and rewriting the record would falsify it.
**Document version:** 0.1 (design-locked for Milestones 0–3; enterprise phases design-level only)
**Last validated against market:** July 2026 (see ADR-000)

---

## Table of contents

1. Project Overview
2. Architecture Summary
3. Repository Structure
4. Development Standards
5. Implementation Order (milestones + atomic tasks)
6. Component Specifications
7. Data Model
8. API Specification
9. Testing Strategy
10. Security
11. Performance Targets
12. Observability
13. Deployment
14. Risk Register
15. Architecture Decision Records
16. Implementation Rules for AI Coding Agents
17. Quality Gates

---

## 1. Project Overview

### 1.1 Vision

Agent governance engines (Microsoft Agent Governance Toolkit, Open Policy Agent, Cedar) can decide whether an agent action is **allowed** and **safe**. None of them can decide whether an agent run is **affordable** or **good**. `agentmeter` supplies the missing signals: real-time **economic cost** and **outcome quality** for agent runs, emitted in the OpenTelemetry GenAI vocabulary so that any policy engine, gateway, or observability backend already in the stack can reason about money and quality — not just permission.

### 1.2 Mission

Make **cost-per-successful-task**, **projected run cost**, **loop/behavioral health**, and **outcome quality** first-class, standard-typed signals that plug into the governance and observability tools teams already run — with a one-line install, zero new required infrastructure, and fail-open safety on the critical path.

### 1.3 Problem statement

Agentic workloads consume 5–30× the tokens of single-shot chat because tool-calling loops re-send accumulated context every step; cost blows up superlinearly with users, and the surprise lands on the invoice. The 2026 governance ecosystem raced to solve *security* (OWASP Agentic Top 10) and *permission* (allow/deny tool calls), driven by regulation. It left two signals unowned:

- **Economic signal** — no engine holds a live per-run cost ledger (reserved vs. actual spend, projected worst-case) that a policy can gate on *before* the tokens burn.
- **Quality signal** — no engine knows the run converged *confidently on a wrong answer*; "well-formed but wrong" is invisible to permission-based governance.

Existing guardrail libraries (LoopGain, AgentBudget) each solve **one** signal and make **their own** stop decision, rather than emitting standard signals other engines can consume. `agentmeter` is the signal layer, not another kernel.

### 1.4 Success criteria

| # | Criterion | Measure |
|---|-----------|---------|
| SC-1 | Zero-infra first value | From `pip install` to first cost-per-successful-task signal visible in an existing OTel backend: **< 5 minutes**, no new services stood up. |
| SC-2 | Standard-native | 100% of emitted signals expressed as OTel GenAI `gen_ai.*` span attributes / metrics; conformance test passes against pinned semconv version. |
| SC-3 | Additive, not competitive | Ships copy-paste policy examples for **≥ 3** engines (AGT, OPA/Rego, Cedar) that consume `agentmeter` signals. |
| SC-4 | Fail-open proven | Fault-injection suite shows agent traffic is **never** blocked or errored by an `agentmeter` internal failure. |
| SC-5 | Overhead bounded | Added wall-clock latency per governed step **< 5 ms p99** for cost+behavior lanes (quality lane excluded; it is opt-in/async). |
| SC-6 | Adoption (north star) | **Integrations**, not signups: count of external repos/policies importing `agentmeter` signals. Target: first external integration within 90 days of public release. |

### 1.5 Core principles

1. **Signals, not decisions.** We emit typed evidence. Enforcement belongs to the engine the user already runs. (ADR-001)
2. **Fail-open, always.** A monitoring/signal layer that can break production is worse than none. Every failure path passes traffic through. (ADR-004)
3. **Standard-native over bespoke.** Express everything in OTel GenAI semconv; never invent a schema where a standard exists. (ADR-002)
4. **Additive integration over replacement.** Incumbents (AGT/OPA/Cedar/LiteLLM/Langfuse) are integration targets and distribution channels, not competitors. (ADR-000)
5. **Correctness on the critical path is sacred.** Stateful invariants (ledger reserve/reconcile ordering, in-place mask semantics) are acceptance criteria, not prose. (§16, R-TECH-1)
6. **Two audiences, two surfaces.** The library (developer use) and the standalone demo (evaluator visibility) are both first-class deliverables. (ADR-006)
7. **Deferred is deferred.** Enterprise control-plane components are designed, not scheduled. No implementation until the OSS core has external adoption. (§5, ADR-007)

### 1.6 Scope (in)

- Python SDK: one-line instrumentation of agent frameworks + raw API.
- **Cost lane:** reserve/reconcile per-run ledger; projected worst-case cost; cost-per-successful-task.
- **Behavior lane:** loop / repeated-tool-call / retry-storm detection as signals.
- **Quality lane:** tiered, sampled, opt-in outcome evaluators (heuristic inline; LLM-judge async/sampled).
- OTel GenAI signal emission (spans, metrics) + OTLP export to any backend.
- Framework adapters: LangGraph (first), then OpenAI Agents SDK, CrewAI, Claude Agent SDK.
- Policy example packs: AGT (YAML), OPA (Rego), Cedar.
- Standalone runnable demo (misbehaving toy agent → signals → engine acts → cost delta).

### 1.7 Scope (out — and why)

| Out-of-scope | Reason |
|--------------|--------|
| Policy DSL / policy engine | Owned by OPA/Cedar/AGT. We emit inputs, not a language. (ADR-001) |
| Enforcement / action layer (block, rollback, reroute) | The engine's job. We surface signals; the kernel decides/acts. (ADR-001) |
| LLM gateway / proxy / routing | LiteLLM/Portkey own this; we consume their cost data, don't replace it. |
| Enterprise control plane (RBAC, SSO, chargeback UI, fleet registry, audit retention) | Designed at intent level only (§5 M6+); **needs a team or sustained commitment post-adoption**. (ADR-007) |
| New observability dashboards | We emit to the user's existing backend. A demo dashboard is the only UI we ship. |
| Non-Python runtimes (TS/Go) | Post-adoption. Python first because the agent framework ecosystem is Python-majority. (OQ-2) |

---

## 2. Architecture Summary

### 2.1 High-level

`agentmeter` is an **in-process Python library** that taps the agent's execution as OTel GenAI spans are produced, computes three lanes of signals, writes them back as standard span attributes / metrics, and exports via OTLP. It holds a small per-run state store (the cost ledger + behavior window). It makes **no enforcement decision** — downstream policy engines read the emitted attributes and decide.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AGENT APPLICATION                                │
│           (LangGraph / OpenAI Agents / CrewAI / custom)                 │
│                                                                         │
│   from agentmeter import instrument                                     │
│   instrument(agent)          # one line; or @meter decorator / ctx mgr  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ emits OTel GenAI spans (gen_ai.*)
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     agentmeter RUNTIME (in-process)                      │
│                                                                         │
│  ┌────────────┐   span in    ┌──────────────────────────────────────┐  │
│  │ SPAN TAP   │─────────────▶│           SIGNAL ENGINE               │  │
│  │(SpanProc-  │              │                                        │  │
│  │ essor)     │              │  ┌──────────┐ ┌─────────┐ ┌────────┐  │  │
│  └────────────┘              │  │ COST     │ │BEHAVIOR │ │QUALITY │  │  │
│                              │  │ LANE     │ │ LANE    │ │ LANE   │  │  │
│         run context          │  │(ledger:  │ │(loop/   │ │(opt-in,│  │  │
│         (run_id, budget) ───▶│  │ reserve/ │ │ tool-   │ │ sampled│  │  │
│                              │  │ reconcile│ │ repeat/ │ │ async) │  │  │
│                              │  │ +project)│ │ retry)  │ │        │  │  │
│                              │  └────┬─────┘ └────┬────┘ └───┬────┘  │  │
│                              │       └────────────┼──────────┘        │  │
│                              │              ┌─────▼──────┐            │  │
│                              │              │ SIGNAL     │            │  │
│                              │              │ WRITER     │            │  │
│                              │              │(sets gen_ai│            │  │
│                              │              │ .* attrs + │            │  │
│                              │              │ metrics)   │            │  │
│                              │              └─────┬──────┘            │  │
│                              └────────────────────┼────────────────────┘  │
│                                          ┌────────▼─────────┐          │
│   ┌───────────────────┐                  │  FAIL-OPEN GUARD │          │
│   │  RUN STATE STORE  │◀─────────────────│ (any lane error  │          │
│   │ (in-mem default;  │                  │  → drop signal,  │          │
│   │  pluggable Redis) │                  │  never raise)    │          │
│   └───────────────────┘                  └────────┬─────────┘          │
└────────────────────────────────────────────────────┼────────────────────┘
                                                     │ OTLP export
                                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │  USER'S EXISTING STACK (unchanged)                          │
        │                                                             │
        │  Observability: Langfuse / Datadog / Grafana / Phoenix      │
        │  Policy engine: AGT / OPA(Rego) / Cedar  ◀── reads gen_ai.* │
        │       run.projected_cost, run.groundedness, run.loop_state  │
        │       → returns allow / deny / require-approval             │
        └────────────────────────────────────────────────────────────┘
```

**Why this shape:** in-process = lowest latency + framework-portable via the OTel standard (no proxy hop). Signal-only = no critical-path enforcement risk and no competition with the kernel. Pluggable state store = in-memory default (zero infra, SC-1) with an optional Redis backend for multi-process/distributed runs (§6.6).

### 2.2 Request (run) lifecycle

```
1. Run starts     → RunContext created (run_id, optional budget policy, sampling decision)
2. Each step:
   a. Pre-step   → COST lane RESERVES projected worst-case cost against ledger
                 → signals written: gen_ai.run.projected_cost, gen_ai.run.budget_remaining
   b. LLM/tool span emitted by framework → SPAN TAP consumes
   c. BEHAVIOR lane updates sliding window (tool-call signature, error trajectory)
                 → signals: gen_ai.run.loop_state, gen_ai.run.repeat_count
   d. Post-step  → COST lane RECONCILES: replace reservation with actual token cost
                 → signals: gen_ai.run.actual_cost (cumulative)
3. Run ends       → QUALITY lane (if sampled) scores outcome (async or inline-heuristic)
                 → signals: gen_ai.run.quality.*, gen_ai.run.success, cost_per_successful_task
4. Export         → all attributes on run/step spans + metrics exported via OTLP
5. (downstream)   → policy engine reads attributes, decides. agentmeter is done.
```

**Critical invariant (R-TECH-1):** reserve *always precedes* the step; reconcile *always* replaces the reservation exactly once. A reserve without a matching reconcile leaks budget; a double-reconcile under-counts. Enforced by ledger unit tests (§9) and the DoD of TASK-M2-*.

### 2.3 Technology stack

| Concern | Choice | Rationale (ADR) |
|---------|--------|-----------------|
| Language | Python ≥ 3.10 | Agent framework ecosystem is Python-majority. (OQ-2) |
| Telemetry | `opentelemetry-sdk`, `opentelemetry-api`, GenAI semconv | Standard-native (ADR-002). Pin semconv version (R-TECH-2). |
| Span tap | Custom `SpanProcessor` | Standard OTel extension point; framework-agnostic. |
| State store | In-memory default; `redis` optional extra | Zero-infra default (SC-1); Redis for distributed (ADR-005). |
| Cost data | Static model-pricing table + optional gateway cost passthrough | Deterministic; no network dependency on the hot path. |
| Quality eval | Heuristic (inline) + pluggable LLM-judge (async) | Tiered to protect latency budget (ADR-003). |
| Packaging | `pyproject.toml`, PEP 621, framework adapters as extras | Clean dependency tree; `pip install agentmeter[langgraph]`. |
| Policy examples | Rego, Cedar, AGT YAML (data files, not code deps) | Copy-paste integration (SC-3). |
| Lint/format | `ruff` (lint+format), `mypy --strict` | Fast, single-tool; strict typing catches the silent-wrong class. |
| Test | `pytest`, `hypothesis` (property tests for ledger), `pytest-benchmark` | Property tests are the defense against stateful bugs (§9). |
| Demo | `docker compose` + a scripted toy agent | One-command evaluator visibility (ADR-006). |

---

## 3. Repository Structure

```
agentmeter/
├── IMPLEMENTATION.md              # this document — source of truth
├── README.md                      # dev quickstart + evaluator demo link
├── pyproject.toml                 # PEP 621; extras: [langgraph],[openai],[crewai],[claude],[redis],[all]
├── LICENSE                        # Apache-2.0 (ADR-008)
├── CONTRIBUTING.md
├── docs/
│   ├── design/                    # phase design records (Phase 0–15 gate docs)
│   │   └── adr/                   # one file per ADR (mirrors §15)
│   ├── signals.md                 # canonical list of emitted gen_ai.* attributes (§7.2) — SOURCE OF TRUTH for signal names
│   └── integrations/              # per-engine integration guides
├── src/
│   └── agentmeter/
│       ├── __init__.py            # public API surface ONLY (instrument, meter, RunContext)
│       ├── api.py                 # instrument(), @meter, context manager
│       ├── runtime/
│       │   ├── span_tap.py        # SpanProcessor implementation
│       │   ├── run_context.py     # RunContext, run_id lifecycle, sampling decision
│       │   ├── signal_writer.py   # writes gen_ai.* attrs + metrics
│       │   └── failopen.py        # the guard wrapper — NOTHING in runtime raises past this
│       ├── lanes/
│       │   ├── base.py            # Lane ABC: process_span(), on_run_end()
│       │   ├── cost/
│       │   │   ├── ledger.py      # reserve/reconcile; the moat component
│       │   │   ├── pricing.py     # model→price table; pluggable provider
│       │   │   └── project.py     # projected worst-case cost estimator
│       │   ├── behavior/
│       │   │   ├── window.py      # sliding window of step signatures
│       │   │   └── detectors.py   # loop / repeat-tool / retry-storm classifiers
│       │   └── quality/
│       │       ├── heuristic.py   # inline cheap checks
│       │       ├── judge.py       # async LLM-judge (sampled, opt-in)
│       │       └── sampling.py    # tier/sample decision
│       ├── store/
│       │   ├── base.py            # StateStore ABC
│       │   ├── memory.py          # default in-process store
│       │   └── redis.py           # optional; extra
│       ├── adapters/
│       │   ├── base.py            # Adapter protocol
│       │   ├── langgraph.py       # M1
│       │   ├── openai_agents.py   # M4
│       │   ├── crewai.py          # M4
│       │   └── claude_agent.py    # M4
│       ├── semconv.py             # pinned GenAI attribute-name constants (R-TECH-2)
│       ├── config.py              # env + programmatic config; feature flags
│       └── errors.py              # internal exception types (never escape failopen)
├── policies/                      # copy-paste integration packs (SC-3) — data, not code
│   ├── opa/                       # *.rego + tests
│   ├── cedar/                     # *.cedar + entities
│   └── agt/                       # *.yaml
├── examples/
│   └── demo/                      # standalone evaluator demo (ADR-006)
│       ├── docker-compose.yml     # toy agent + OTel collector + Langfuse (or console)
│       ├── misbehaving_agent.py   # deliberately overspends + loops
│       └── README.md              # one command → see signals catch it
└── tests/
    ├── unit/
    ├── property/                  # hypothesis suites (ledger invariants)
    ├── integration/               # adapter + real span flow
    ├── contract/                  # semconv conformance (SC-2)
    ├── failinject/                # fail-open proof (SC-4)
    └── bench/                     # overhead benchmarks (SC-5)
```

### 3.1 Layer responsibilities & import boundaries

| Layer | May import | May NOT import | Responsibility |
|-------|-----------|----------------|----------------|
| `api` | `runtime`, `config` | `lanes.*` internals, `store.*` internals | Public surface. The only module users touch. |
| `runtime` | `lanes` (via `base`), `store` (via `base`), `semconv` | concrete lane/adapter modules | Orchestration, span tap, fail-open, signal writing. |
| `lanes/*` | `store.base`, `semconv`, `config` | `runtime`, `adapters`, other lanes | Pure signal computation. **No lane imports another lane.** |
| `store/*` | `config` | `lanes`, `runtime`, `adapters` | State persistence behind an ABC. |
| `adapters/*` | `api`, `semconv` | `lanes` internals, `store` internals | Framework-specific wiring only. |
| `semconv` | (stdlib only) | everything | Frozen attribute-name constants. |

**Dependency rule:** dependencies point *inward* toward `semconv`/`store.base`/`lanes.base`. Concrete implementations are wired at the edges (`api`, `adapters`). No lane knows another lane exists — this keeps the three signal lanes independently testable and independently shippable (M2 cost, M3 behavior, M5 quality can land separately).

---

## 4. Development Standards

### 4.1 Coding standards
- Python ≥ 3.10, `mypy --strict` clean, `ruff` clean (lint + format), no `# type: ignore` without an inline reason.
- Public API fully type-annotated with docstrings (Google style). Internal modules: docstrings only on non-obvious invariant-bearing code (the ledger, the fail-open guard, the mask/window semantics) — mirrors the AQUA convention of documenting only the highest-defense-value pieces.
- No function on the hot path (span processing) may perform network I/O synchronously. Pricing lookups are in-memory; LLM-judge is async/off-path.

### 4.2 Error handling (load-bearing — see R-TECH-1, ADR-004)
- **Nothing inside `runtime/` or `lanes/` may raise past the fail-open guard.** Every lane entry point is wrapped; on any exception the guard logs at WARN, drops the signal for that step, and returns control so the agent proceeds unaffected.
- Internal errors use types in `errors.py`; they are caught at the `failopen.py` boundary, never propagated to user code.
- User-facing config errors (bad model name, unknown adapter) raise **at instrument() time**, not on the hot path — fail loudly at setup, silently (open) at runtime.

### 4.3 Logging & config
- Structured logging via stdlib `logging` under the `agentmeter` namespace; default WARN, never logs prompt/completion content (privacy — §10).
- Config precedence: explicit `instrument(...)` kwargs > env vars (`AGENTMETER_*`) > defaults.
- **Feature flags** (config booleans, default state): `cost_lane` (on), `behavior_lane` (on), `quality_lane` (**off** — opt-in, ADR-003), `store_backend` (`memory`).

### 4.4 Versioning & dependencies
- SemVer. Pre-1.0: minor may break, documented in CHANGELOG. Post-1.0: backward compatibility preserved unless a major bump with an approved ADR (§16).
- **The OTel GenAI semconv version is pinned** in `semconv.py` with the pinned spec version in a module constant; upgrading it is an ADR-worthy change (R-TECH-2), because semconv attributes carry Development-stability badges and can rename without a major bump.
- New runtime dependencies require justification in the PR and, if they touch the hot path, a benchmark showing overhead within SC-5. Adapters' framework deps live behind extras and are never imported at core import time.

---

## 5. Implementation Order

Milestones are ordered so each de-risks the next and each produces something demonstrable. **M0–M5 are the buildable OSS core** (build these with Claude Code). **M6+ are design-level only** — do not implement until the OSS core has external adoption (ADR-007).

| Milestone | Objective | Ships | Gate (must pass §17 before next) |
|-----------|-----------|-------|----------------------------------|
| M0 | Skeleton + standards | Repo, CI, lint/type gates, empty public API, semconv constants | CI green; `import agentmeter` works; contract test scaffold runs |
| M1 | Span tap + one adapter + fail-open | Instrument a LangGraph agent; spans flow through; nothing can break the agent | Fail-inject suite passes (SC-4); LangGraph demo run produces spans |
| M2 | **Cost lane (the moat)** | reserve/reconcile ledger, pricing, projection; cost signals emitted | Ledger property tests pass; cost-per-successful-task visible in backend (SC-1) |
| M3 | Behavior lane | loop/repeat/retry detectors; behavior signals emitted | Detector tests pass on labeled fixtures; overhead within SC-5 |
| M4 | Adapters + policy packs + **demo** | OpenAI Agents/CrewAI/Claude adapters; OPA/Cedar/AGT examples; standalone demo | SC-3 (≥3 policy packs) + ADR-006 demo runs in one command |
| M5 | Quality lane (opt-in) | heuristic + sampled async LLM-judge; quality signals | Quality lane off by default; when on, latency budget still met; eval tests pass |
| **— public v0.1 release gate —** | | Apache-2.0, README, docs/signals.md, ≥1 integration guide | All of §17; SC-1..SC-5 met |
| M6+ (deferred) | Enterprise control plane | fleet policy registry, RBAC/SSO, chargeback, audit retention, pre-deploy simulator | **Not scheduled.** Needs team/sustained commitment post-adoption (ADR-007) |

### 5.1 Atomic tasks

Task format: **[ID] Description | Purpose | Inputs → Outputs | Deps | Acceptance | DoD | Complexity | Tests | Docs**
Complexity: S (≤½ day agent), M (~1 day), L (multi-day / needs human design review).

#### Milestone M0 — Skeleton

- **[M0-1]** Scaffold repo per §3. | Establish structure & boundaries. | — → directory tree, `pyproject.toml`, `LICENSE` (Apache-2.0). | none | Tree matches §3; `pip install -e .` works. | Installs clean in fresh venv. | S | import smoke test. | README quickstart stub.
- **[M0-2]** Wire CI: ruff, mypy --strict, pytest, coverage gate. | Enforce standards from commit 1. | — → `.github/workflows/ci.yml`. | M0-1 | CI runs on PR; fails on lint/type/test error. | Green on empty scaffold. | S | n/a (is the gate). | CONTRIBUTING notes gates.
- **[M0-3]** `semconv.py`: pin GenAI semconv version constant + attribute-name constants used project-wide. | Single source for signal names (R-TECH-2). | spec version → frozen constants. | M0-1 | All later signal writes reference these constants, never string literals. | mypy clean; constants documented in docs/signals.md. | S | contract test asserts constants match pinned spec. | docs/signals.md seeded.
- **[M0-4]** Public API stubs in `api.py` + `__init__.py`: `instrument()`, `@meter`, `RunContext` (no-op). | Freeze the surface users depend on. | — → typed stubs. | M0-1 | `from agentmeter import instrument, meter, RunContext` works; all typed. | Signatures reviewed & locked (breaking them post-M1 = ADR). | S | import + signature tests. | API section of README.

#### Milestone M1 — Span tap + adapter + fail-open

- **[M1-1]** `failopen.py`: guard wrapper that catches all internal exceptions, logs WARN, returns safe default. | The single most important safety component (ADR-004). | callable + fallback → guarded callable. | M0 | No internal exception can propagate through it; verified by injected faults. | Property test: for any raised exception type, guard returns fallback and agent proceeds. | **L** (human-review the guarantee) | failinject/: raise in each lane, assert agent unaffected. | ADR-004 cross-ref.
- **[M1-2]** `run_context.py`: RunContext (run_id gen, start/end, sampling decision, optional budget). | Per-run identity & state anchor. | run start event → RunContext. | M0-4 | Each run gets a stable unique id; end is idempotent. | Reconcile/quality hooks fire exactly once per run end. | M | unit: id uniqueness, idempotent end. | docs/signals.md run lifecycle.
- **[M1-3]** `span_tap.py`: `SpanProcessor` consuming GenAI spans, dispatching to enabled lanes via `lanes/base`. | Framework-agnostic ingestion. | OTel span → lane dispatch. | M1-1, M1-2 | Only `gen_ai.*` spans processed; unknown spans ignored; all dispatch wrapped by fail-open. | Spans from a real LangGraph run reach a stub lane. | M | integration: span flow with stub lane. | integrations/ overview.
- **[M1-4]** `adapters/langgraph.py`: wire `instrument(agent)` for LangGraph. | First real one-line integration. | LangGraph agent → instrumented agent. | M1-3 | One line instruments; spans emitted for LLM + tool steps. | Demo agent runs, spans visible in console exporter. | M | integration: end-to-end LangGraph run. | integrations/langgraph.md.
- **[M1-5]** `signal_writer.py` (write path only; no signals yet): set attributes/metrics on current span using semconv constants. | Central, correct signal emission. | (attr, value) → span attribute/metric. | M1-3 | Writes only via semconv constants; never raises (behind fail-open). | Written attrs appear on exported span. | S | unit + contract (names match spec). | docs/signals.md.

#### Milestone M2 — Cost lane (moat)

- **[M2-1]** `cost/pricing.py`: model→(input,output) price table + pluggable `PricingProvider`. | Deterministic on-path cost, no network. | model id + token counts → cost. | M1 | Known models priced; unknown model → fail-open (no signal, WARN), never raises. | Table versioned; provider swappable. | S | unit: known/unknown models. | docs: pricing source + update process.
- **[M2-2]** `cost/ledger.py`: **reserve/reconcile ledger** with the R-TECH-1 invariant. | The core economic signal; the moat. | reserve(step, projected), reconcile(step, actual) → running totals. | M2-1, M1-2 | reserve precedes reconcile; each reservation reconciled exactly once; no leak, no double-count. | **All property tests pass** (hypothesis: random reserve/reconcile interleavings preserve invariant). | **L** (human-review invariant) | property/: interleavings, missing-reconcile detection, concurrent runs. | ADR + docs/signals.md.
- **[M2-3]** `cost/project.py`: projected worst-case run cost from budget policy + steps-so-far. | Enables *pre-spend* gating by downstream policy. | ledger state + policy → projected_cost. | M2-2 | Projection monotonic w.r.t. remaining steps; documented method. | Estimate within stated bound on fixtures. | M | unit on fixtures. | docs/signals.md method note.
- **[M2-4]** Emit cost signals: `gen_ai.run.actual_cost`, `.projected_cost`, `.budget_remaining`, `.cost_per_successful_task`. | Make cost consumable by any engine. | ledger/project → span attrs + metrics. | M2-2, M2-3, M1-5 | Attributes present & correctly typed on run span; metric instruments registered. | Visible in Langfuse/console within 5 min of install (SC-1). | M | contract + integration. | docs/signals.md (authoritative names). | 

#### Milestone M3 — Behavior lane

- **[M3-1]** `behavior/window.py`: sliding window of step signatures (tool name + arg hash + error magnitude). | Substrate for pathology detection. | step → window update. | M1 | Window bounded; signature stable for identical calls. | Memory bounded under long runs. | M | unit: signature stability, bound. | docs/signals.md.
- **[M3-2]** `behavior/detectors.py`: loop / repeated-tool / retry-storm classifiers → states. | The behavioral signal. | window → {healthy, repeating, looping, retry_storm}. | M3-1 | Labeled fixtures classified correctly; low false-positive on healthy runs. | Fixtures pass; FP rate documented. | **L** (classifier design review) | unit on labeled fixtures (healthy + pathological). | docs/signals.md state semantics.
- **[M3-3]** Emit `gen_ai.run.loop_state`, `.repeat_count`. | Consumable behavior signal. | detector → span attrs. | M3-2, M1-5 | Typed attrs on step/run span. | Overhead within SC-5 (bench). | S | contract + bench. | docs/signals.md.

#### Milestone M4 — Adapters, policy packs, demo

- **[M4-1..3]** Adapters: OpenAI Agents SDK, CrewAI, Claude Agent SDK (one task each, pattern of M1-4). | Portability. | agent → instrumented. | M2, M3 | One-line instrument each; spans + signals flow. | Per-adapter integration test green. | M each | integration per adapter. | integrations/*.md.
- **[M4-4]** Policy packs: OPA/Rego, Cedar, AGT YAML consuming `agentmeter` signals. | SC-3; copy-paste adoption. | signal names → example rules + tests. | M2, M3 | Each pack has a working rule gating on cost/quality/loop + its own test. | Rego/Cedar tests pass in CI. | M | policies/*/ tests. | docs/integrations/*.
- **[M4-5]** Standalone demo (ADR-006): `docker compose up` → misbehaving agent → signals catch overspend+loop → policy acts → cost delta shown. | Evaluator visibility (the job-search lever). | — → one-command demo. | M4-1..4 | Single command; shows before/after cost + caught loop; no manual setup. | Runs clean on a fresh machine. | **L** | smoke test in CI (compose up, assert signals). | examples/demo/README.

#### Milestone M5 — Quality lane (opt-in)

- **[M5-1]** `quality/sampling.py`: tier/sample decision (heuristic-inline vs LLM-judge-async). | Protect latency budget (ADR-003). | run + config → sampling decision. | M2 | Off by default; when on, only sampled runs hit LLM-judge; inline heuristics cheap. | Latency budget held with lane on (bench). | M | unit + bench. | docs: quality tiers.
- **[M5-2]** `quality/heuristic.py`: cheap inline checks (empty/malformed output, obvious non-completion). | Free quality signal. | run output → heuristic score. | M5-1 | Deterministic; near-zero overhead. | Bench within SC-5. | S | unit. | docs/signals.md.
- **[M5-3]** `quality/judge.py`: async, sampled LLM-judge (groundedness/task-success); pluggable. | Deep quality signal, off hot path. | run trace → async score. | M5-1 | Never blocks the run; result attached when ready or on run-end join. | Async; failure = no signal (fail-open). | **L** | integration (mocked judge) + failinject. | docs: judge contract, "who-evals" note.
- **[M5-4]** Emit `gen_ai.run.quality.groundedness`, `.task_success`, `.run.success`. | Consumable quality signal. | scores → attrs. | M5-2, M5-3, M1-5 | Typed attrs; `run.success` drives cost_per_successful_task denominator. | Contract passes. | S | contract + integration. | docs/signals.md.

---

## 6. Component Specifications

### 6.1 Public API (`api.py`)
- **Responsibilities:** the entire user-facing surface. `instrument(target, *, config=None)`, `@meter` decorator, `RunContext` context manager.
- **Interface:** `instrument(agent) -> agent` (returns same object, side-effect wires span tap + adapter). Overloads per adapter resolved by duck-typing / explicit `adapter=`.
- **Failure modes:** invalid config/adapter → raise **at call time** (setup), never at runtime. Unknown framework → clear error listing supported adapters.
- **Performance:** setup-time only; zero hot-path cost.
- **Security:** never accepts or logs credentials; pricing/judge model keys read from env by the user's own SDK, not stored by us.

### 6.2 Fail-open guard (`runtime/failopen.py`)
- **Responsibilities:** wrap every lane/runtime entry point; guarantee no internal error reaches user code. THE safety invariant (ADR-004, SC-4).
- **Interface:** `guard(fn, fallback)` / decorator.
- **Failure modes:** by design, *absorbs* all internal failure modes. If the guard itself is misconfigured, that surfaces at import/setup, not runtime.
- **Performance:** try/except overhead only (<< SC-5).
- **Testing:** failinject suite raises every internal exception type at every lane boundary and asserts the agent completes unaffected.

### 6.3 Cost ledger (`lanes/cost/ledger.py`) — moat
- **Responsibilities:** hold per-run reserved/actual/projected cost; enforce reserve→reconcile invariant (R-TECH-1).
- **Interfaces:** `reserve(run_id, step_id, projected) `, `reconcile(run_id, step_id, actual)`, `snapshot(run_id) -> {reserved, actual, projected, budget_remaining}`.
- **Failure modes:** missing reconcile (leak) → detected by invariant check + WARN; double reconcile → rejected. All non-raising (fail-open) but recorded.
- **Performance:** O(1) per op, in-memory; SC-5.
- **Security:** stores costs/token counts only — never prompt/completion content.

### 6.4 Behavior detectors (`lanes/behavior/detectors.py`)
- **Responsibilities:** classify run health from the step-signature window.
- **Interface:** `classify(window) -> LoopState`.
- **Failure modes:** ambiguous → default `healthy` (fail-open bias: never fabricate a pathology). FP rate is a published metric (SC + R-TECH).
- **Performance:** bounded window scan; SC-5.

### 6.5 Quality lane (`lanes/quality/*`)
- **Responsibilities:** optional outcome scoring; tiered to protect latency (ADR-003).
- **Interface:** `Lane.on_run_end(run) -> QualitySignals | None`.
- **Failure modes:** judge unavailable/slow → no signal, never blocks (fail-open). Off by default.
- **Security/privacy:** if LLM-judge sees trace content, it runs on the user's own model creds and honors the content-sampling/redaction config (§10).

### 6.6 State store (`store/*`)
- **Responsibilities:** persist RunContext + ledger + window behind an ABC.
- **Interface:** `get/set/incr/delete(run_id, ...)`; `memory` (default) and `redis` (extra) impls.
- **Failure modes:** Redis unavailable → fail-open to degraded (drop signal) or fall back to memory per config; never blocks the agent.
- **Rationale:** in-memory keeps SC-1 (zero infra); Redis enables multi-process/distributed runs (ADR-005).

### 6.7 Adapters (`adapters/*`)
- **Responsibilities:** framework-specific wiring of the span tap; nothing else.
- **Interface:** `Adapter.instrument(agent) -> agent`.
- **Failure modes:** framework version mismatch → clear setup-time error; never silent.
- **Maintenance note (R-TECH-3):** adapters are the maintenance tax — each framework release can break them. Pin tested framework versions; CI matrix per adapter.

---

## 7. Data Model

`agentmeter` is **stateless across runs by default** (in-memory per-process). There is no relational database in the OSS core — this is deliberate (SC-1). The "data model" is (a) the transient run-state records and (b) the canonical signal schema.

### 7.1 Transient run state (in `StateStore`, keyed by `run_id`)

| Entity | Fields | Lifetime | Store |
|--------|--------|----------|-------|
| RunContext | run_id, started_at, ended_at, sampling_decision, budget_policy? | one run | memory/redis |
| LedgerEntry | run_id, step_id, reserved, actual, state(reserved/reconciled) | one run | memory/redis |
| BehaviorWindow | run_id, deque[StepSignature], error_trajectory | one run (bounded) | memory/redis |
| QualityResult | run_id, groundedness?, task_success?, success | run-end | memory/redis |

Retention: run state is evicted at run end (or TTL for orphaned runs, default 1h, configurable). No long-term persistence in OSS core; enterprise audit retention is M6+ (deferred).

### 7.2 Canonical signal schema — **authoritative in `docs/signals.md`; mirrored via `semconv.py` constants**

All names live under the pinned GenAI namespace. Names below are the project contract; changing one is an ADR + minor/major bump.

| Signal (attribute/metric) | Type | Lane | Meaning |
|---------------------------|------|------|---------|
| `gen_ai.run.actual_cost` | double (USD) | cost | cumulative reconciled cost |
| `gen_ai.run.projected_cost` | double (USD) | cost | worst-case projected cost |
| `gen_ai.run.budget_remaining` | double (USD) | cost | budget − reserved |
| `gen_ai.run.cost_per_successful_task` | double (USD) | cost×quality | actual_cost / success |
| `gen_ai.run.loop_state` | enum string | behavior | healthy/repeating/looping/retry_storm |
| `gen_ai.run.repeat_count` | int | behavior | max repeated-signature count in window |
| `gen_ai.run.quality.groundedness` | double [0,1] | quality | opt-in judge score |
| `gen_ai.run.quality.task_success` | double [0,1] | quality | opt-in judge score |
| `gen_ai.run.success` | bool | quality | success flag (heuristic or judge) |

> **Rule:** downstream policies (OPA/Cedar/AGT) are written against these exact names. `docs/signals.md` is the contract other people integrate against — treat it with the same care as a public API.

---

## 8. API Specification

The OSS core is a **library**, not a service — its "API" is the Python surface (§6.1) plus the **signal contract** (§7.2) that downstream engines consume. There are no REST/gRPC endpoints in the OSS core. (A control-plane REST API is M6+, deferred; sketched below for continuity only.)

### 8.1 Python SDK API (stable contract)
```python
from agentmeter import instrument, meter, RunContext, Config

# one-line (most users)
instrument(agent)                       # auto-detect adapter
instrument(agent, adapter="langgraph")  # explicit

# decorator
@meter(config=Config(quality_lane=True))
def run_agent(...): ...

# context manager (raw / unsupported frameworks)
with RunContext(budget="$0.50") as run:
    ...  # spans emitted inside are governed
```
Contract guarantees: `instrument` returns the same object; enabling/disabling lanes never changes agent behavior; no method blocks the agent; all raise only at setup.

### 8.2 Signal contract (the real integration surface)
Consumers read the §7.2 attributes off exported OTel spans/metrics. Example (Rego):
```rego
deny if input.attributes["gen_ai.run.projected_cost"] > 0.50
deny if input.attributes["gen_ai.run.loop_state"] == "looping"
deny if input.attributes["gen_ai.run.quality.groundedness"] < 0.7
```

### 8.3 Deferred control-plane REST API (M6+, design-only, NOT to be built now)
`/v1/policies`, `/v1/runs`, `/v1/attribution` — fleet registry, run history, chargeback. Auth via OIDC/SSO. **Do not implement** until ADR-007 deferral is lifted.

---

## 9. Testing Strategy

Because the implementer is an AI coding agent and the reviewer is a single human, **tests are the primary defense against confident-but-wrong code**, not an afterthought. The stateful components (ledger, fail-open, behavior window) get property-based and fault-injection testing, mirroring the AQUA-style discipline where silent stateful bugs are the real risk.

| Test type | Scope | Gate |
|-----------|-------|------|
| Unit | every module; pure logic | per-PR; coverage ≥ 85% core, 100% on `ledger.py`, `failopen.py` |
| Property (`hypothesis`) | ledger reserve/reconcile invariant under random interleavings; window bounds | **blocking** for M2/M3 |
| Fault injection (`failinject/`) | raise every internal exception at every lane boundary; assert agent unaffected | **blocking** for M1+ (SC-4) |
| Contract | emitted attribute names/types == pinned semconv (§7.2) | **blocking** every PR (SC-2) |
| Integration | real span flow per adapter; end-to-end run | per adapter (M1, M4) |
| Benchmark (`pytest-benchmark`) | per-step overhead; publish numbers | **blocking**: cost+behavior < 5 ms p99 (SC-5) |
| Policy-pack | Rego/Cedar/AGT example rules evaluate correctly against sample signals | M4 |
| Demo smoke | `docker compose up` produces expected signals | M4 (ADR-006) |

**Per-feature "done" rule:** a lane/adapter is not complete until its unit + contract + (for stateful lanes) property + fault-injection + benchmark tests are green. No signal is "emitted" until a contract test asserts its name/type against `docs/signals.md`.

**What must be tested before each merge:** invariant preserved (property), agent never broken (failinject), signal names correct (contract), overhead within budget (bench). These four are the CI quality gates.

---

## 10. Security

- **Threat model:** the library runs in the user's process with their agent's privileges. Primary risks: (a) leaking prompt/completion content via signals or logs; (b) becoming a critical-path failure; (c) supply-chain compromise of a dependency on the hot path.
- **Content privacy (primary control):** `agentmeter` **never emits or logs prompt/completion content by default.** Signals are numeric/enum/cost only (§7.2). The LLM-judge (opt-in) is the only component that reads trace content; it runs on the user's own model credentials and honors a redaction/sampling config. Default judge = off.
- **Credentials:** we store no keys. Pricing needs none (static table). Judge uses the user's existing SDK/env creds; we never read or persist them.
- **Fail-open as a security property:** #b above is mitigated by ADR-004 — the layer cannot take down the agent.
- **Supply chain:** minimal core deps; framework deps behind extras (never imported at core import). CI runs `pip-audit`; SBOM generated on release; dependencies pinned. New hot-path deps require justification + benchmark (§4.4).
- **AuthN/AuthZ:** none in OSS core (library, no service). SSO/OIDC/RBAC belong to the deferred control plane (M6+).
- **Compliance:** SOC2/GDPR/ISO obligations attach to the (deferred) hosted control plane, not the OSS library. Note the regulatory backdrop (EU AI Act high-risk obligations Aug 2026; Colorado AI Act Jun 2026) as *why the governance ecosystem we integrate with exists* — but the library itself processes no PII by default.

---

## 11. Performance Targets

| Metric | Target | Test |
|--------|--------|------|
| Added latency/step (cost+behavior) | < 5 ms p99 | bench, blocking (SC-5) |
| Added latency/step (quality lane on, inline heuristic) | < 10 ms p99 | bench |
| Quality LLM-judge | off hot path (async/sampled); 0 ms blocking | integration |
| Ledger op | O(1), < 100 µs | micro-bench |
| Memory/run | bounded (windowed); < a few KB/run | soak test |
| Import time | < 500 ms; extras not imported at core import | test |
| Availability (of the guarantee, not a service) | agent completes 100% of the time regardless of agentmeter state | failinject (SC-4) |

Note: these are **design budgets to be verified by benchmark**, not measured results. Publishing the real numbers on release is itself a deliverable (differentiator + trust), following the pre-registered-methodology precedent set by peers in this category.

---

## 12. Observability

`agentmeter` is itself an observability *producer*; it also observes itself.

- **Self-metrics:** signals-emitted count, lane-error count (fail-open activations), overhead histogram, sampling rate. Emitted under an `agentmeter.internal.*` namespace, separate from `gen_ai.*`.
- **Logging:** structured, `agentmeter` namespace, WARN default; every fail-open activation logs once at WARN with lane + exception type (never content).
- **Tracing:** the library rides the user's existing OTel pipeline; no separate collector required (SC-1).
- **Health:** N/A for a library. The demo's compose stack exposes collector/backend health for evaluators.
- **Runbooks (for adopters):** docs cover "signals not appearing" (check exporter/semconv version), "fail-open activations spiking" (a lane bug — file issue; agent still safe), "overhead above budget" (disable quality lane / check store backend).

---

## 13. Deployment

The OSS core is **published to PyPI**, not deployed as a service.

- **Local dev:** `pip install -e .[all]`; `pre-commit` runs ruff/mypy.
- **Release:** tag → CI builds, runs full gate (§17), generates SBOM, publishes to PyPI (`agentmeter` + extras). SemVer; CHANGELOG mandatory.
- **Demo:** `examples/demo/` ships a `docker-compose.yml` (toy agent + OTel collector + a backend, e.g., Langfuse or console exporter). One command for evaluators (ADR-006). Optionally deploy the demo to a public URL for the "recruiter live toggle."
- **CI/CD:** GitHub Actions — lint/type/test/bench/contract/failinject on PR; release workflow on tag. Per-adapter framework-version matrix (R-TECH-3).
- **Rollback:** library — users pin versions; yanked releases via PyPI if a regression ships. No blue/green/canary in OSS core (those attach to the deferred hosted plane).
- **Deferred (M6+):** Docker images, Helm chart, K8s deploy, multi-region, canary for the hosted control plane. **Not now** (ADR-007).

---

## 14. Risk Register

| ID | Risk | Type | Impact | Likelihood | Mitigation | Review |
|----|------|------|--------|-----------|------------|--------|
| R-MKT-1 | An incumbent (MS AGT / a gateway) adds native cost+quality signals, closing the wedge | Market | High | Medium | Ship fast; be the *reference* signal layer; integrate so tightly we're the default source. Re-validate market each milestone gate (§15 continuous validation). | per milestone |
| R-MKT-2 | Signal layer stays "a library inside other tools," never a business | Market | Medium | High | Accepted by design (goal = contribution + adoption, not revenue). North star = integrations, not signups (SC-6). | quarterly |
| R-TECH-1 | Ledger reserve/reconcile bug → wrong cost signals (silent) | Technical | High | Medium | Property + fault-injection tests are blocking gates; 100% coverage on ledger; human design review of invariant (M2-2 = L). | per PR touching cost |
| R-TECH-2 | GenAI semconv attributes rename (Development-stability) → signals break consumers | Technical | High | Medium | Pin semconv version in `semconv.py`; contract test; upgrades are ADR-gated; document the pinned version prominently. | on OTel releases |
| R-TECH-3 | Adapter breakage on framework releases (maintenance tax) | Technical | Medium | High | Per-adapter CI version matrix; pin tested versions; adapters isolated behind extras so core is unaffected. | per framework release |
| R-TECH-4 | Fail-open guarantee has a gap → library breaks an agent | Technical | Critical | Low | `failopen.py` is L-complexity human-reviewed; failinject suite is a blocking gate at M1+. This is the one thing that must never fail. | per PR touching runtime |
| R-TECH-5 | Quality lane adds latency / "who evals the evaluator" | Technical | Medium | Medium | Off by default; tiered/sampled; async judge off hot path; latency budget enforced by bench (ADR-003). | M5 |
| R-PROD-1 | Nobody adopts because integration still feels heavy | Product | High | Medium | SC-1 (<5 min first value); copy-paste policy packs; one-command demo. Adoption is a gate metric, not a hope. | per milestone |
| R-SEC-1 | Prompt/completion content leaks via a signal or log | Security | High | Low | Content never emitted/logged by default; judge opt-in + redaction; review every new signal for content-freedom. | per new signal |
| R-OPS-1 | Solo maintainer bandwidth; project stalls at enterprise phase | Operational | Medium | High | OSS core is the whole committed scope; M6+ explicitly deferred (ADR-007). Don't start what can't be sustained. | quarterly |

**Open questions (OQ):** ~~OQ-1 final public name/trademark check~~ — **resolved Jul 2026: `optio`** (see the header note; PyPI name verified free on both the JSON API and the simple index, and the import package renamed to match); OQ-2 second-language SDK (TS?) timing; OQ-3 whether to upstream signals as a GenAI semconv proposal (would strengthen R-TECH-2 enormously — high-leverage, investigate post-v0.1).

---

## 15. Architecture Decision Records

> Full ADRs live in `docs/design/adr/`. Summaries here. **Never change an architectural decision without adding/superseding an ADR (§16).**

- **ADR-000 — Product is a signal layer, not a runtime (Option A pivot).** *Context:* re-validation (Jul 2026) found the in-loop guardrail/policy-runtime lane occupied — LoopGain/AgentBudget (guardrails) and, decisively, Microsoft Agent Governance Toolkit (MIT, the `govern(x, policy=...)` primitive) + OPA/Cedar (mature policy engines). *Decision:* do not build a policy runtime/DSL/enforcement layer; build the cost+outcome-quality **signal layer** that those engines lack and integrate with them. *Alternatives:* contest the full runtime (rejected: direct fight with Microsoft's distribution, wrong for solo build). *Consequences:* smaller surface, stronger defensibility via integration, moat = signals+ledger not schema. *This pivot is why every "own the policy object" assumption from earlier design is deprecated.*
- **ADR-001 — Emit signals, never enforce.** Enforcement/action belongs to the downstream engine. Keeps us off the critical-path-decision hook and non-competitive. Consequence: no action layer, no DSL (both §1.7 out-of-scope).
- **ADR-002 — OTel GenAI semconv as the wire format.** Standard-native = portable across every consumer. Consequence: pinned version + contract tests (R-TECH-2); consider upstreaming (OQ-3).
- **ADR-003 — Quality lane is tiered, sampled, opt-in, off by default.** Protects the latency budget and avoids the "who-evals-the-evaluator" cost trap. Consequence: cost+behavior deliver value alone; quality is additive.
- **ADR-004 — Fail-open is absolute.** A signal layer must never break the agent. Consequence: the guard is the most-reviewed component; failinject is a blocking gate; bias detectors toward `healthy` on ambiguity.
- **ADR-005 — Pluggable state store, in-memory default.** Zero-infra first value (SC-1) with a Redis path for distributed runs. Consequence: `StateStore` ABC; store failures fail-open.
- **ADR-006 — Two delivery surfaces: library + standalone demo.** Developer use and evaluator visibility are different artifacts; both first-class. Rationale tied to the user's own diagnosis that visibility, not capability, is the bottleneck. Consequence: the demo is a milestone deliverable with its own smoke test, not documentation.
- **ADR-007 — Enterprise control plane designed but not scheduled.** Solo-with-agent build; the hosted plane (RBAC/SSO/chargeback/audit/simulator) needs a team or sustained post-adoption commitment. Consequence: M6+ is intent-level; building it now would pour concrete on an unadopted foundation.
- **ADR-008 — Apache-2.0.** Permissive (like Portkey/LoopGain/Preloop) and, unlike MIT, carries a patent grant that reassures enterprise adopters; unlike GPL (Alephant), won't be rejected from critical-path use. Open-core split defers commercial surface to M6+.
- **ADR-012 — The public API is the top-level package only.** Everything reachable only through a submodule (`optio.lanes.*`, `optio.runtime.*`, `optio.store.*`, `optio.adapters.*`) is internal and may change in any release, patch included, despite being importable and typed. Consequence: `CHANGELOG.md` treats only the top-level exports as the compatibility surface.
- **ADR-013 — Optimization lives in a separate package (`optio_optimize`).** Acting on cost signals (caching, trimming, routing, compression) conflicts with the core's no-enforcement, no-content-reading, and 5ms-overhead properties; `optio_optimize` is opt-in, depends on `optio` one-way, and accepts its own weaker latency/privacy position honestly. Consequence: two packages, two release cadences; lossy stages gated by an eval suite (rule 3, still unbuilt as of this writing).
- **ADR-014 — `optio_optimize` integrates by emitting spans, not by calling `optio`.** The ledger optio_optimize would need to call is internal (ADR-012) and keyed by span id, not a bare amount — so `Optimizer.call()` emits a standard OTel GenAI span instead, and `optio`'s existing span tap prices and classifies it exactly like any framework adapter's spans, with zero new coupling. Off by default (`emit_spans`). Consequence: the raw-call path gets ADR-013's promised visibility; framework-adapter double-counting is deferred to the adapter work itself.
- **ADR-015 — Evidence bar for promoting an `ALTERED`-tier stage out of "experimental."** ADR-013's eval gate proves a lossy stage's *logic* does what it claims, never that a live answer stays good — and a status pass found `compress_prompt` with one confounded live data point (bundled with `semantic_cache` under `--aggressive`), `semantic_cache` and `route_models` with zero live data, and `summarize_history` structurally untestable without a real summarizer. Decision: define per-stage acceptance criteria and risk models *before* gathering evidence, so a later result cannot quietly move the bar. Consequence: no default changes without isolated live evidence; the ADR gains a per-stage addendum as evidence lands, same pattern as ADR-005's Redis addendum.

---

## 16. Implementation Rules for AI Coding Agents

These are binding constraints for Claude Code (and any human) working in this repo:

1. **Read this document and the relevant ADRs before each task.** If a needed decision is absent, STOP — propose an ADR, get it approved, update this file — then code. Never decide architecture inline.
2. **Never change architecture without an ADR.** Superseding an ADR is allowed; silently diverging is not.
3. **Never introduce a new dependency without justification** in the PR; hot-path deps also require a benchmark within SC-5. Framework deps go behind extras, never core imports.
4. **Fail-open is inviolable.** No code in `runtime/` or `lanes/` may raise past `failopen.py`. Any PR touching these must pass the failinject suite (R-TECH-4).
5. **Signal names come from `semconv.py` / `docs/signals.md` only** — never string literals. Adding/renaming a signal requires updating `docs/signals.md` + a contract test + (if breaking) an ADR and version bump.
6. **The ledger invariant (reserve→reconcile, exactly once) is sacred.** Any change to `cost/` must keep the property tests green; treat these like the AQUA mask/moment invariants — the bug will be silent, so the test is the only guardrail.
7. **Complete ALL required tests before marking a task done** (§9 per-feature rule). "Done" = acceptance criteria met + tests green + docs updated + within performance budget.
8. **Update documentation alongside code** — especially `docs/signals.md` (the integration contract) and integration guides.
9. **Small, reviewable commits**, incremental (not squashed), one logical change each — the human reviewer's capacity is the bottleneck; optimize for reviewability.
10. **No premature optimization** unless a benchmark justifies it. Correctness and fail-open first.
11. **Keep lanes independent** — no lane imports another; respect the §3.1 import boundaries. A lint/import-linter check enforces this.
12. **Preserve backward compatibility** of the public API (§8.1) and signal contract (§7.2) unless a major bump + approved ADR says otherwise.
13. **Do not implement M6+ (deferred) components** (ADR-007) without an explicit decision lifting the deferral.

---

## 17. Quality Gates

A milestone is complete only when **all** of the following pass. This is the checklist run at each milestone gate (§5) and mirrors the design-review-gate discipline.

- [ ] All task acceptance criteria met; every task's Definition of Done satisfied.
- [ ] `ruff` + `mypy --strict` clean; import-boundary check passes (§3.1).
- [ ] Unit + contract tests green; coverage ≥ 85% core / 100% on `ledger.py` + `failopen.py`.
- [ ] Property tests green for any stateful component touched (ledger, window).
- [ ] **Fault-injection suite green — agent is never broken by an internal failure (SC-4).**
- [ ] **Contract tests green — emitted signal names/types match pinned semconv + `docs/signals.md` (SC-2).**
- [ ] Benchmarks within performance budget (§11) — cost+behavior < 5 ms p99 (SC-5).
- [ ] `docs/signals.md` and affected integration guides updated.
- [ ] No unresolved critical (R-TECH-1/-2/-4, R-SEC-1) issues.
- [ ] **Continuous validation:** does this still solve the §1.3 problem? Has an incumbent shipped native cost+quality signals (R-MKT-1)? If yes → revisit ADR-000 before proceeding, don't build on a closed wedge.
- [ ] For the v0.1 release gate additionally: SC-1 (<5 min first value) demonstrated; SC-3 (≥3 policy packs) shipped; ADR-006 demo runs in one command; Apache-2.0 + README + ≥1 integration guide present.

---

*End of IMPLEMENTATION.md v0.1. This document governs Milestones 0–5 (buildable OSS core). Milestones 6+ are design-level (ADR-007) and require an explicit deferral-lift before implementation. Any architectural change requires an ADR update per §16.*
