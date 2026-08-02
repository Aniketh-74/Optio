# optio

**Economic cost and outcome quality signals for agent runs — in the OpenTelemetry GenAI vocabulary.**

> **Status: alpha (0.2.0).** All three lanes work end to end and every signal in the contract is
> implemented, on 99% coverage with 100% on the ledger and the fail-open guard. What "alpha"
> means here, concretely:
>
> - **State is in-process only.** `store_backend="redis"` is rejected at setup rather than
>   silently ignored ([ADR-005](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/)).
> - **The signal names may still move.** They are pinned to OTel GenAI semconv 1.37.0, which is
>   itself marked Development-stability upstream ([ADR-002](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/)).
> - **Detector accuracy is measured against synthetic traffic.** The 0/1200 false-positive rate
>   is a regression gate, not a claim about your agent ([docs/testing.md](https://github.com/Aniketh-74/Optio/blob/main/docs/testing.md)).
>
> Each adapter *is* now verified against the real framework — a CI job per framework installs
> LangGraph, CrewAI, the OpenAI Agents SDK and the Claude Agent SDK and runs the adapter against
> genuine objects, including the cases it must refuse ([R-TECH-3](https://github.com/Aniketh-74/Optio/blob/main/IMPLEMENTATION.md)).
>
> The fail-open guarantee is not provisional: it is a blocking CI gate on every commit.
> See [IMPLEMENTATION.md](https://github.com/Aniketh-74/Optio/blob/main/IMPLEMENTATION.md) for the full design and milestone plan.

| Lane | Signals | Status |
|---|---|---|
| **Cost** | spend, worst-case projection, budget headroom | Working |
| **Behavior** | loop / repetition / retry-storm state | Working |
| **Quality** | groundedness, task success, cost-per-successful-task | Working — **off by default** (opt-in, [ADR-003](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/)) |

Adapters: LangGraph, OpenAI Agents SDK, CrewAI, Claude Agent SDK.
Policy packs: [OPA/Rego, Cedar, Microsoft AGT](https://github.com/Aniketh-74/Optio/tree/main/policies/).
Runnable demo: [`examples/demo`](https://github.com/Aniketh-74/Optio/tree/main/examples/demo/) — one command, no API keys.

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
pip install optio                # core: signals only, never touches a request
pip install "optio[langgraph]"   # + framework adapter
pip install "optio[optimize]"    # + optio_optimize: acts on the signals (see below)
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

The emitted attributes are the integration contract. [`docs/signals.md`](https://github.com/Aniketh-74/Optio/blob/main/docs/signals.md) is authoritative; names are mirrored as constants in `optio.semconv` and asserted by contract tests.

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
| Per-step overhead, cost + behavior lanes | mean **~70 µs**, p99 **~130 µs** | < 5 ms p99 |
| Behavior classification at 10k steps | ~40 µs → ~50 µs | flat in run length |
| Behavior classification vs *window size* | 37 µs at 50, 38 µs at 1000 | flat in window size |
| Ledger snapshot, 10k open reservations | ~100 µs | O(1)-ish |
| Both lanes disabled | ~18 µs p99 | — |

Roughly 40× inside budget. Measured on one developer machine (Windows, CPython 3.14) with an
in-memory store and no exporter attached; treat them as an order of magnitude, not a promise for
your hardware. Figures are rounded across repeated runs because run-to-run variance on a desktop
is wider than the differences between them — the honest reading is "tens of microseconds", not a
precise percentile split.

The window-size row is the one that is a structural guarantee rather than a measurement:
classification is O(1) in the window, not O(window), so raising `behavior_window_size` to catch
longer cycles costs memory but not latency. It used to cost both — 370 µs per step at a window of
1000 ([docs/testing.md](https://github.com/Aniketh-74/Optio/blob/main/docs/testing.md)).

`import optio` takes ~158 ms (median of 12 cold starts), against a 500 ms budget.

Detector accuracy on the synthetic corpus (see [docs/behavior.md](https://github.com/Aniketh-74/Optio/blob/main/docs/behavior.md) for what that
corpus does and does not model):

| Measure | Result |
|---|---|
| False positives — healthy runs flagged | **0 / 1200** (0.000%) |
| Detection — pathological runs caught | **600 / 600** (100.0%) |

Both are CI-gated. The detection rate is gated alongside the FP rate because a zero false-positive
rate is trivially achievable by never detecting anything.

## optio_optimize — acting on the signals, not just emitting them

`optio` never changes a request (ADR-001). `optio_optimize` is the separate, opt-in package that
does: caching, history trimming, deduplication, and retrieval pruning, sitting in the request path
rather than beside it ([ADR-013](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-013-optimization-lives-in-a-separate-package.md)).
Installing `optio` alone never pulls it in.

```python
from optio_optimize import Optimizer

optimizer = Optimizer()  # lossless by default: caching, prefix markers, token ceilings
response = optimizer.call(request, provider_fn)  # or `await optimizer.acall(...)` for async
```

**Live-measured, not simulated.** `docs/optimize-benchmarks.md` is generated the same way the
core's overhead numbers are — the file states which figures are live-API-verified and which are
still simulated, and corrects itself in place when a live run contradicts an earlier simulated
claim (it has, twice, over two workloads: multi_turn_chat was measured at +36.3% simulated / −7.7%
live in one round, and simulation vs. live cost direction flipped again for `trim_history` under
OpenAI's automatic prefix caching in the next). Against `gpt-4o-mini`:

| Workload | `gpt-4o-mini` | `claude-sonnet-4-5` |
|---|---|---|
| `retry_storm` | **93.5%** | **97.8%** |
| `tool_loop` | **80.8%** | **93.7%** |
| `multi_turn_chat_long` | — | **86.2%** |
| `mcp_agent` | — | **84.0%** |
| `fan_out` | **68.2%** | **83.7%** |
| `multi_turn_chat` | **8.4%** | **82.2%** |
| `tool_calling_chat` | — | **81.3%** |
| `large_system_agent` | — | **81.0%** |
| `rag_queries` | **16.5%** | **76.2%** |

**The two columns differ by an order of magnitude on `multi_turn_chat`, and the reason is the
product's central fact.** OpenAI populates its prefix cache automatically, so the largest lever here
— placing explicit cache breakpoints — has nothing to do on that vendor and the column shows only
what the other stages manage. Anthropic caches *only* what you mark, which is where a library that
marks correctly is worth 80%+ on ordinary conversation traffic.

Three of the Anthropic figures — `multi_turn_chat`, `tool_calling_chat`, `mcp_agent` — come with
**100% byte-identical output**, and two of them reduce *zero tokens*: prefix caching changes the rate
those tokens are billed at, not the volume. A tool reporting only "tokens saved" scores them at 0.0%.

Two workloads are excluded from the table because the suite reports them as *not attributable* — the
optimizer provably did nothing, so the measured delta is the provider's own nondeterminism rather
than a result. They exist to keep the suite honest about its limits.

**Two adapters: Anthropic and the OpenAI Agents SDK.** Both are opt-in down to their own dependency
— this is the one place in the package that reads prompt content, so neither SDK ships with
`optio[optimize]`.

`optio_optimize.adapters.anthropic.wrap_anthropic_client` wraps a real `Anthropic` client's
`messages.create`, sync and streaming. It is the adapter the numbers above come from, because
Anthropic is where explicit cache breakpoints matter. **Streaming is supported rather than bypassed**
(ADR-019): each event is forwarded the instant it arrives and accounted for alongside, so the caller
sees the first token exactly as soon as they would without this package — measured live at 6,317
tokens written on one streamed call and all 6,317 read back on the next. Only the terminal event
completes a request; a stream that dies mid-generation, or a caller who stops reading, caches
nothing, because serving half an answer from cache would be permanent and confident.

`optio_optimize.adapters.openai_agents.wrap_openai_client` intercepts an `AsyncOpenAI` client's
`chat.completions.create`, the SDK's own extension point for a Chat-Completions-backed model, rather
than reimplementing its Responses-API `Model` protocol. A translation failure falls back to the
unwrapped client rather than raising. Two real bugs were found and fixed building it — a cache hit
that returned the *original* call's non-zero token usage, and the Agents SDK's `Omit` sentinel (not
`None`) for unset fields being read as "the caller already set this" — both caught only by driving
the real SDK, not by hand-written test input; see the adapter's module docstring and
`tests/optimize/test_adapters_openai_agents.py`.

```python
from anthropic import Anthropic
from optio_optimize.adapters.anthropic import wrap_anthropic_client

client = wrap_anthropic_client(Anthropic())  # every call now optimized, lossless by default
```

```python
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from optio_optimize.adapters.openai_agents import wrap_openai_client

client = wrap_openai_client(AsyncOpenAI(), trim_history=True)
model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)
```

**Emits spans `optio` already knows how to price**, opt-in via `Optimizer(emit_spans=True)`
([ADR-014](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-014-optimize-emits-spans-optio-already-knows-how-to-read.md)) — no
code changed on the `optio` side, and `optio_optimize` still imports nothing from `optio` (checked
by `lint-imports`, not just claimed). Don't combine with other GenAI OTel instrumentation on the
same calls, or both will emit `gen_ai.usage.*` for the same request and `optio`'s cost lane will
sum them.

**Four more stages exist, all off by default and experimental**: `route_models`, `compress_prompt`,
`semantic_cache`, `summarize_history`. Each defaults to the cheapest option that needed no new
dependency — lexical word-overlap instead of embeddings, a length heuristic instead of an
auxiliary model call, no bundled summarizer at all (`summarize_history=True` alone spends nothing
and raises at construction if you don't also supply one). Gated by a deliberately model-free eval
suite (`src/optio_optimize/eval/`) rather than a claim of correctness — see
`docs/optimize-benchmarks.md`'s "Phase 3" section for what that gate does and does not prove, and
a live result it does not smooth over: cost fell but output length rose on one workload.

## Configuration

Precedence: `instrument(...)` kwargs > `OPTIO_*` env vars > defaults.

| Option | Env var | Default |
|---|---|---|
| `cost_lane` | `OPTIO_COST_LANE` | `True` |
| `behavior_lane` | `OPTIO_BEHAVIOR_LANE` | `True` |
| `quality_lane` | `OPTIO_QUALITY_LANE` | `False` (opt-in, ADR-003) |
| `store_backend` | `OPTIO_STORE_BACKEND` | `memory` |

## What is public

The supported API is exactly what `optio` exports at the top level:

```python
from optio import instrument, meter, RunContext, Config, BudgetPolicy, current_run
```

…plus `optio.__version__` and `optio.GENAI_SEMCONV_VERSION`, the OTel GenAI semconv release the
signal names are pinned to ([ADR-002](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/)). Read that one if you need to branch on
which vocabulary a given install emits — it changes with a semconv bump, independently of
`__version__`.

Everything reachable only through a submodule — `optio.lanes.*`, `optio.runtime.*`,
`optio.store.*`, `optio.adapters.*` — is **internal**, and may change in any release including a
patch. Those modules are importable, documented and fully typed because contributors read them,
not as a stability promise ([ADR-012](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-012-the-public-api-is-the-top-level-package-only.md)).

The signal names in [docs/signals.md](https://github.com/Aniketh-74/Optio/blob/main/docs/signals.md) are the *other* half of the compatibility
surface, and the stricter one: a Rego or Cedar policy matching `gen_ai.run.projected_cost` stops
matching silently if the name moves, so renaming one is a breaking change even though no Python
signature changed.

If you need something only a submodule exposes, please open an issue — a real use case can be
promoted to the top level deliberately.

## Development

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest
lint-imports          # §3.1 layer boundaries
```

Contributor rules — including the ones that are non-negotiable (fail-open, the ledger invariant, signal names from `semconv.py` only) — are in [CONTRIBUTING.md](https://github.com/Aniketh-74/Optio/blob/main/CONTRIBUTING.md) and §16 of IMPLEMENTATION.md.

## License

Apache-2.0 (ADR-008) — permissive, with a patent grant.
