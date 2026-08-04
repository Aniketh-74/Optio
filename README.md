<h1 align="center">optio</h1>

<p align="center">
  <strong>Know what your agents cost. Then pay less.</strong><br>
  Cost, loop and quality signals in the OpenTelemetry GenAI vocabulary —<br>
  plus an opt-in optimizer that cuts the bill on traffic you already send.
</p>

<p align="center">
  <a href="https://github.com/Aniketh-74/Optio/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Aniketh-74/Optio/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/optio/"><img alt="PyPI" src="https://img.shields.io/pypi/v/optio.svg"></a>
  <a href="https://pypi.org/project/optio/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/optio.svg"></a>
  <a href="https://github.com/Aniketh-74/Optio/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
</p>

---

Your agent framework tells you what happened. `optio` tells you **what it cost, whether it was
going in circles, and whether the answer was any good** — as ordinary OTel span attributes, so
every backend you already run can chart and alert on them.

It ships as two packages, and the split is deliberate:

| | Package | Where it sits | What it does |
|---|---|---|---|
| 👁 | **`optio`** | beside the request | Reads spans, writes signals. **Never touches a request.** |
| ✂️ | **`optio_optimize`** | in the request path | Caches, trims, marks prefixes. Opt-in, installed separately. |

You can adopt either one alone.

## Quickstart

### Cut cost — wrap the client you already have

One line, either vendor, sync or async (the wrapper detects which):

```python
from anthropic import Anthropic
from optio_optimize import wrap_anthropic_client

client = wrap_anthropic_client(Anthropic())

# Everything from here on is optimized. Your call sites do not change.
client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
)
```

```python
from openai import OpenAI
from optio_optimize import wrap_openai_client

client = wrap_openai_client(OpenAI())
client.chat.completions.create(model="gpt-4o-mini", messages=[...])
```

The client is **mutated and returned**, so you keep the object you built with one method replaced.
Async clients (`AsyncAnthropic`, `AsyncOpenAI`) work identically. Every parameter this package does
not model — `seed`, `top_p`, `metadata`, `extra_headers`, tool calls, multimodal blocks — rides
through untouched.

Only lossless stages run by default. Nothing that could change an answer is on unless you turn it
on:

```python
wrap_anthropic_client(Anthropic(), trim_history=False)  # tune individual stages
```

Want the numbers afterwards?

```python
from optio_optimize import Optimizer

optimizer = Optimizer()
client = wrap_anthropic_client(Anthropic(), optimizer)
...
print("\n".join(optimizer.report.summary_lines("claude-haiku-4-5")))
```

### Measure — instrument the agent

```python
from optio import instrument

instrument(agent)  # spans now carry cost + behavior signals
```

That is the whole integration. `optio` reads the spans your framework already emits and writes
numbers back onto them — it never sees or alters a request
([ADR-001](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/)).

Scope a run and give it a budget:

```python
from optio import meter, RunContext


@meter(budget="$0.50")
def run_agent(prompt: str) -> str: ...


with RunContext(budget="$0.50") as run:  # or scope it yourself
    ...  # spans emitted inside are governed
```

No framework? `RunContext` governs any code that emits GenAI spans — a raw SDK loop included.

### Works with

| Framework | `instrument` | Extra | Guide |
|---|---|---|---|
| LangGraph | auto-detected | `optio[langgraph]` | [langgraph.md](https://github.com/Aniketh-74/Optio/blob/main/docs/integrations/langgraph.md) |
| OpenAI Agents SDK | auto-detected | `optio[openai]` | [openai-agents.md](https://github.com/Aniketh-74/Optio/blob/main/docs/integrations/openai-agents.md) |
| CrewAI | auto-detected | `optio[crewai]` | [crewai.md](https://github.com/Aniketh-74/Optio/blob/main/docs/integrations/crewai.md) |
| Claude Agent SDK | auto-detected | `optio[claude]` | [claude-agent-sdk.md](https://github.com/Aniketh-74/Optio/blob/main/docs/integrations/claude-agent-sdk.md) |
| Anything else | `RunContext` | — | [integrations](https://github.com/Aniketh-74/Optio/tree/main/docs/integrations/) |

An adapter **does not** make your framework emit GenAI spans — that is the ecosystem's job (OTel
instrumentation, OpenInference, or your framework's own exporter). optio reads what they produce.

## Install

```bash
pip install optio                # core: signals only, never touches a request
pip install "optio[langgraph]"   # + framework adapter
pip install "optio[optimize]"    # + optio_optimize: acts on the signals
```

Python **3.10 – 3.14**, Linux/macOS/Windows. Installing `optio` alone never pulls the optimizer in,
and an import-linter contract keeps it that way.

## The signals

These attribute names *are* the integration contract — mirrored as constants in `optio.semconv`
and asserted by contract tests.
[`docs/signals.md`](https://github.com/Aniketh-74/Optio/blob/main/docs/signals.md) is authoritative.

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

Policies match the exact names, so they stay portable across backends:

```rego
deny if input.attributes["gen_ai.run.projected_cost"] > 0.50
deny if input.attributes["gen_ai.run.loop_state"] == "looping"
```

## What it saves

Cost reduction, **measured against live APIs**, not simulated:

| Workload | `gpt-4o-mini` | `claude-sonnet-4-5` |
|---|---|---|
| `retry_storm` | **93.5%** | **97.8%** |
| `tool_loop` | **80.8%** | **93.7%** |
| `multi_turn_chat_long` | — | **86.2%** |
| `mcp_agent` | — | **84.0%** |
| `fan_out` | **68.2%** | **83.7%** |
| `multi_turn_chat` | **8.4%** | **82.2%** |
| `rag_queries` | **16.5%** | **76.2%** |

**Those two columns differ by 10× on `multi_turn_chat`, and that gap is the point.** OpenAI
populates its prefix cache automatically, so the biggest lever — placing explicit cache breakpoints
— has nothing to do there. Anthropic caches *only* what you mark, which is where marking correctly
is worth 80%+ on ordinary conversation traffic.

Three Anthropic figures come with **100% byte-identical output**, and two of them reduce *zero
tokens* — prefix caching changes the rate tokens are billed at, not the volume. A tool that reports
only "tokens saved" scores those at 0.0%.

Full method, dates and per-stage detail:
[`docs/optimize-benchmarks.md`](https://github.com/Aniketh-74/Optio/blob/main/docs/optimize-benchmarks.md).

## How it decides what to claim

Every number here is measured, dated and re-runnable — and several exist because a measurement
contradicted an earlier claim and the claim lost:

- **Stages ship off when the evidence says so.** 14 of 24 optimizations are disabled by default,
  including one that measured a **cost increase** of 14.8%.
- **The riskiest stages were measured, then left off.** All four answer-changing stages were run
  live and isolated on 2026-08-03 for $0.85 total, and
  [all four stayed off](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-015-evidence-bar-for-promoting-an-altered-tier-stage.md).
  `semantic_cache` served a **wrong answer to 6 of 7** adversarial near-duplicates on `gpt-4o-mini`
  and **7 of 8** on `claude-haiku-4-5` — two vendors, same verdict.
- **Every run is kept.** Live exchanges are recorded under
  [`docs/evidence/`](https://github.com/Aniketh-74/Optio/tree/main/docs/evidence/), so re-checking
  a published number costs nothing instead of costing money
  ([ADR-039](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-039-a-measurement-that-costs-money-to-recheck-gets-checked-once.md)).
- **Corrections stay in the record.** `optimize-benchmarks.md` rewrites itself in place when a live
  run contradicts a simulated one — it has, twice.

The reasoning behind each decision is in
[51 ADRs](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/).

## Overhead

Published rather than promised — generated by the CI benchmark job, not written by hand.

| Measure | Result | Budget |
|---|---|---|
| Per-step, cost + behavior lanes | mean **~70 µs**, p99 **~130 µs** | < 5 ms p99 |
| Behavior classification at 10k steps | ~40 µs → ~50 µs | flat in run length |
| Classification vs *window size* | ~13 µs at 50, ~13 µs at 1000 | flat in window |
| Classification vs *distinct calls* | 8 µs at 8, 35 µs at 1000 | scales, deliberately |
| Same step over a shared Redis store | ~620 µs, of which ~500 µs is the round trip | < 3 round trips |
| Quality state per step, in process | ~1 µs, flat at 10 steps and 10,000 | — |
| Quality state per step, shared store | ~600 µs, same round trip | < 3 round trips |
| Ledger snapshot, 10k open reservations | ~100 µs | O(1)-ish |
| Both lanes disabled | ~18 µs p99 | — |
| `import optio` | ~158 ms cold | < 500 ms |

Roughly **40× inside budget**. Measured on one developer machine with an in-memory store and no
exporter attached; read them as an order of magnitude, not a promise for your hardware.

Widening `behavior_window_size` to catch longer cycles costs memory, not latency — it used to cost
both, at 370 µs per step. What *does* scale is how many **distinct** calls a window holds, because
finding the top counts means scanning the live ones. That growth points the safe way: it peaks when
every step is different, which is the case with no loop to detect, and it is cheapest on the tight
cycles the lane exists to catch. Both rows are asserted by the benchmark suite, on both backends
([docs/testing.md](https://github.com/Aniketh-74/Optio/blob/main/docs/testing.md)).

The Redis rows are one round trip by construction — a step is a single Lua script. The behaviour
one appends, evicts and reduces server-side, returning four numbers regardless of window size; the
quality one writes four fields. Neither payload grows with the run, which is what keeps the
flatness guarantees true across a network. Both are asserted as a *ratio* against a bare `PING`, so
they mean the same thing on a laptop and on your hardware.

The quality row deserves one caveat rather than a footnote: that lane emits nothing per step, so on
a shared store it pays a round trip that buys nothing until run end. It is off by default, and if
you turn it on together with `store_backend="redis"` that is the price. Stated here because a
number you can see beats a cost you discover.

Detector accuracy on the synthetic corpus, both CI-gated:

| Measure | Result |
|---|---|
| False positives — healthy runs flagged | **0 / 1200** (0.000%) |
| Detection — pathological runs caught | **600 / 600** (100.0%) |

The detection rate is gated *alongside* the false-positive rate, because a zero false-positive rate
is trivially achievable by never detecting anything.

## Status

> **alpha (0.3.0).** All three lanes work end to end and every signal in the contract is
> implemented, on 99% coverage with 100% on the ledger and the fail-open guard.
>
> - **State is in-process by default**, which needs no infrastructure and meters one process. Set
>   `store_backend="redis"` for runs sharded across workers — proved by four processes metering
>   one run to the exact total
>   ([ADR-050](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-050-the-store-speaks-the-domain.md)).
> - **The signal names may still move.** They are pinned to OTel GenAI semconv 1.37.0, which is
>   itself marked Development-stability upstream
>   ([ADR-002](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/)).
>
> `1.0.0` waits for that vocabulary to stabilise — the reasoning and the three conditions are in
> [ADR-046](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-046-1-0-0-waits-for-the-vocabulary-underneath.md).

## Public API

The top-level package, and nothing else
([ADR-012](https://github.com/Aniketh-74/Optio/blob/main/docs/design/adr/adr-012-the-public-api-is-the-top-level-package-only.md)):

```python
from optio import instrument, meter, RunContext, current_run, Config, BudgetPolicy
from optio import __version__, GENAI_SEMCONV_VERSION
```

| | |
|---|---|
| `instrument` | attach to a framework object |
| `meter` | decorator: scope a run, optionally with a budget |
| `RunContext` | the same thing as a context manager |
| `current_run` | the run in scope, or `None` |
| `Config` | everything configurable, validated at setup |
| `BudgetPolicy` | the limits a run is governed by |
| `__version__`, `GENAI_SEMCONV_VERSION` | this release, and the semconv release it tracks |

Plus `optio.semconv`, which holds the attribute names as constants.

From `optio_optimize`, the entry points are `wrap_anthropic_client`, `wrap_openai_client`,
`Optimizer`, `BatchOptimizer` and `OptimizeConfig`.

`optio.runtime`, `optio.lanes` and friends are internal and may change without a major bump — if
you need something from them, please open an issue rather than importing it. A test asserts every
name above appears in this file, so an undocumented export cannot ship.

## Docs

| | |
|---|---|
| [Signals contract](https://github.com/Aniketh-74/Optio/blob/main/docs/signals.md) | every attribute, authoritative |
| [Integrations](https://github.com/Aniketh-74/Optio/tree/main/docs/integrations/) | per-framework setup guides |
| [Behavior detection](https://github.com/Aniketh-74/Optio/blob/main/docs/behavior.md) | what the corpus does and does not model |
| [Optimize benchmarks](https://github.com/Aniketh-74/Optio/blob/main/docs/optimize-benchmarks.md) | per-stage, per-workload, dated |
| [Evidence](https://github.com/Aniketh-74/Optio/tree/main/docs/evidence/) | recorded live runs behind the claims |
| [Testing](https://github.com/Aniketh-74/Optio/blob/main/docs/testing.md) | how the numbers above are produced |
| [Runbooks](https://github.com/Aniketh-74/Optio/blob/main/docs/runbooks.md) | when a signal does not show up |
| [ADRs](https://github.com/Aniketh-74/Optio/tree/main/docs/design/adr/) | why every decision went the way it did |
| [Contributing](https://github.com/Aniketh-74/Optio/blob/main/CONTRIBUTING.md) | the gate you need to pass |
| [Security](https://github.com/Aniketh-74/Optio/blob/main/SECURITY.md) | reporting a vulnerability |
| [Changelog](https://github.com/Aniketh-74/Optio/blob/main/CHANGELOG.md) | what changed, and why |

## Development

```bash
pip install -e ".[dev]"
pytest                       # 2,304 tests
ruff check . && mypy         # lint + types
lint-imports                 # architecture boundaries
```

Contributions are welcome — [CONTRIBUTING.md](https://github.com/Aniketh-74/Optio/blob/main/CONTRIBUTING.md)
describes the gate, which is the same one CI runs.

## License

Apache-2.0 — see [LICENSE](https://github.com/Aniketh-74/Optio/blob/main/LICENSE).
