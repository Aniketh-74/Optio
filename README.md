# optio

**Economic cost and outcome quality signals for agent runs — in the OpenTelemetry GenAI vocabulary.**

> **Status: alpha (0.1.0).** All three lanes work end to end and every signal in the contract is
> implemented, on 99% coverage with 100% on the ledger and the fail-open guard. What "alpha"
> means here, concretely:
>
> - **The adapters have not been tested against the real frameworks.** Matching logic is covered;
>   no adapter has been run against an actual LangGraph, CrewAI, OpenAI Agents SDK, or Claude
>   Agent SDK release ([R-TECH-3](IMPLEMENTATION.md)).
> - **State is in-process only.** `store_backend="redis"` is rejected at setup rather than
>   silently ignored ([ADR-005](docs/design/adr/)).
> - **The signal names may still move.** They are pinned to OTel GenAI semconv 1.37.0, which is
>   itself marked Development-stability upstream ([ADR-002](docs/design/adr/)).
>
> The fail-open guarantee is not provisional: it is a blocking CI gate on every commit.
> See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the full design and milestone plan.

| Lane | Signals | Status |
|---|---|---|
| **Cost** | spend, worst-case projection, budget headroom | Working |
| **Behavior** | loop / repetition / retry-storm state | Working |
| **Quality** | groundedness, task success, cost-per-successful-task | Working — **off by default** (opt-in, [ADR-003](docs/design/adr/)) |

Adapters: LangGraph, OpenAI Agents SDK, CrewAI, Claude Agent SDK.
Policy packs: [OPA/Rego, Cedar, Microsoft AGT](policies/).
Runnable demo: [`examples/demo`](examples/demo/) — one command, no API keys.

Agent governance engines (Microsoft Agent Governance Toolkit, OPA, Cedar) can decide whether an agent action is **allowed** and **safe**. None of them can decide whether an agent run is **affordable** or **good**.

`optio` supplies the missing signals — real-time **cost** and **outcome quality** — emitted as standard `gen_ai.*` span attributes, so the policy engine and observability backend you already run can reason about money and quality, not just permission.

## Principles

- **Signals, not decisions.** We emit typed evidence. Enforcement belongs to the engine you already run. (ADR-001)
- **Fail-open, always.** A monitoring layer that can break production is worse than none. (ADR-004)
- **Standard-native.** Everything is OTel GenAI semconv; we never invent a schema where a standard exists. (ADR-002)
- **Zero new infrastructure.** Per-run state is in-process; nothing to deploy. A distributed
  backend is designed but not built, and `store_backend="redis"` is rejected at setup rather than
  accepted and ignored. (ADR-005)

## Install

```bash
pip install optio                # core
pip install "optio[langgraph]"   # + framework adapter
```

Two runtime dependencies — `opentelemetry-api` and `opentelemetry-sdk`. Nothing else, no
compiled extensions, no service to run.

## Compatibility

Everything in this table is asserted by a CI job, not by hand.

| | Supported | How it's verified |
|---|---|---|
| **Python** | 3.10 – 3.14 | full suite on every version, Linux |
| **OS** | Linux, macOS, Windows | both ends of the Python range on each |
| **OpenTelemetry** | ≥ 1.27.0 | a job pinned to *exactly* 1.27.0 — the floor is tested, not just declared |
| **Install** | wheel + sdist | built, installed into a clean venv outside the repo, then asserted to emit real signals |

**Frameworks.** Each adapter is tested against the real package in its own CI job — a genuine
`CompiledStateGraph`, `agents.Agent`, `ClaudeSDKClient`, and CrewAI `Crew`, not mocks:

| Framework | Instrument | Notes |
|---|---|---|
| LangGraph | the compiled graph | an *uncompiled* `StateGraph` is refused — it has no `invoke()` to meter |
| OpenAI Agents SDK | the `Agent` | |
| Claude Agent SDK | the `ClaudeSDKClient` | `ClaudeAgentOptions` is refused — it's config, not a run |
| CrewAI | the `Crew`, or a single `Agent` | a `Task` is refused |
| **anything else** | `RunContext` / `@meter` | no adapter needed; see below |

**You don't need an adapter.** Adapters only auto-detect the framework. If yours isn't listed —
or you're calling an SDK directly — `RunContext` and `@meter` work with any code that emits OTel
GenAI spans. That's the whole integration surface.

**What optio needs from you:** a configured `TracerProvider` and a framework (or instrumentation
library) that emits `gen_ai.*` spans. With no OTel SDK configured, optio logs a warning at setup
and emits nothing — it does not fail, and it does not guess.

**Not supported yet:** distributed/multi-process runs. State is in-process, and
`store_backend="redis"` is rejected at setup rather than silently ignored (ADR-005).

## Quickstart

```python
from optio import instrument

instrument(agent)  # one line; spans now carry cost + behavior signals
```

Other surfaces:

```python
from optio import meter, RunContext, Config


@meter(budget="$0.50")
def run_agent(prompt: str) -> str: ...


with RunContext(budget="$0.50") as run:
    ...  # spans emitted inside are governed
```

Enabling or disabling lanes never changes agent behavior, nothing blocks the agent, and configuration errors raise at setup — never on the hot path.

## Signals

The emitted attributes are the integration contract. [`docs/signals.md`](docs/signals.md) is authoritative; names are mirrored as constants in `optio.semconv` and asserted by contract tests.

| Signal | Type | Lane |
|---|---|---|
| `gen_ai.run.actual_cost` | double (USD) | cost |
| `gen_ai.run.projected_cost` | double (USD) | cost |
| `gen_ai.run.budget_remaining` | double (USD) | cost |
| `gen_ai.run.cost_per_successful_task` | double (USD) | cost × quality |
| `gen_ai.run.loop_state` | enum string | behavior |
| `gen_ai.run.repeat_count` | int | behavior |
| `gen_ai.run.quality.groundedness` | double [0,1] | quality (opt-in) |
| `gen_ai.run.quality.task_success` | double [0,1] | quality (opt-in) |
| `gen_ai.run.success` | bool | quality |

Downstream policies match on these exact names:

```rego
deny if input.attributes["gen_ai.run.projected_cost"] > 0.50
deny if input.attributes["gen_ai.run.loop_state"] == "looping"
```

## Measured overhead

§11 sets design budgets; publishing the real numbers is itself a deliverable, so these are
generated by the CI benchmark job rather than written by hand:

| Measure | Result | Budget (SC-5) |
|---|---|---|
| Per-step overhead, cost + behavior lanes | mean **68 µs**, p99 **105 µs** | < 5 ms p99 |
| Behavior classification at 10k steps | 52 µs → 54 µs | flat in run length |
| Ledger snapshot, 10k open reservations | 85 µs | O(1)-ish |
| Both lanes disabled | 22 µs p99 | — |

Roughly 50× inside budget. Measured on one developer machine (Windows, CPython 3.13) with an
in-memory store and no exporter attached; treat them as an order of magnitude, not a promise for
your hardware. The p95/p99 figures come from the same modest sample, so they sit within noise of
each other — the honest reading is "~100 µs tail," not a precise percentile split.

Detector accuracy on the synthetic corpus (see [docs/behavior.md](docs/behavior.md) for what that
corpus does and does not model):

| Measure | Result |
|---|---|
| False positives — healthy runs flagged | **0 / 1200** (0.000%) |
| Detection — pathological runs caught | **600 / 600** (100.0%) |

Both are CI-gated. The detection rate is gated alongside the FP rate because a zero false-positive
rate is trivially achievable by never detecting anything.

## Configuration

Precedence: `instrument(...)` kwargs > `OPTIO_*` env vars > defaults.

| Option | Env var | Default |
|---|---|---|
| `cost_lane` | `OPTIO_COST_LANE` | `True` |
| `behavior_lane` | `OPTIO_BEHAVIOR_LANE` | `True` |
| `quality_lane` | `OPTIO_QUALITY_LANE` | `False` (opt-in, ADR-003) |
| `store_backend` | `OPTIO_STORE_BACKEND` | `memory` |

## Development

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest
lint-imports          # §3.1 layer boundaries
```

Contributor rules — including the ones that are non-negotiable (fail-open, the ledger invariant, signal names from `semconv.py` only) — are in [CONTRIBUTING.md](CONTRIBUTING.md) and §16 of IMPLEMENTATION.md.

## License

Apache-2.0 (ADR-008) — permissive, with a patent grant.
