# agentmeter

**Economic cost and outcome quality signals for agent runs — in the OpenTelemetry GenAI vocabulary.**

> Status: **pre-alpha (M0 skeleton).** The public API surface is frozen; the signal lanes are not yet implemented. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the full design and milestone plan.

Agent governance engines (Microsoft Agent Governance Toolkit, OPA, Cedar) can decide whether an agent action is **allowed** and **safe**. None of them can decide whether an agent run is **affordable** or **good**.

`agentmeter` supplies the missing signals — real-time **cost** and **outcome quality** — emitted as standard `gen_ai.*` span attributes, so the policy engine and observability backend you already run can reason about money and quality, not just permission.

## Principles

- **Signals, not decisions.** We emit typed evidence. Enforcement belongs to the engine you already run. (ADR-001)
- **Fail-open, always.** A monitoring layer that can break production is worse than none. (ADR-004)
- **Standard-native.** Everything is OTel GenAI semconv; we never invent a schema where a standard exists. (ADR-002)
- **Zero new infrastructure.** In-memory by default; Redis only if you need distributed runs. (ADR-005)

## Install

```bash
pip install agentmeter                # core
pip install "agentmeter[langgraph]"   # + framework adapter
```

Python ≥ 3.10.

## Quickstart

```python
from agentmeter import instrument

instrument(agent)  # one line; spans now carry cost + behavior signals
```

Other surfaces:

```python
from agentmeter import meter, RunContext, Config


@meter(budget="$0.50")
def run_agent(prompt: str) -> str: ...


with RunContext(budget="$0.50") as run:
    ...  # spans emitted inside are governed
```

Enabling or disabling lanes never changes agent behavior, nothing blocks the agent, and configuration errors raise at setup — never on the hot path.

## Signals

The emitted attributes are the integration contract. [`docs/signals.md`](docs/signals.md) is authoritative; names are mirrored as constants in `agentmeter.semconv` and asserted by contract tests.

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

## Configuration

Precedence: `instrument(...)` kwargs > `AGENTMETER_*` env vars > defaults.

| Option | Env var | Default |
|---|---|---|
| `cost_lane` | `AGENTMETER_COST_LANE` | `True` |
| `behavior_lane` | `AGENTMETER_BEHAVIOR_LANE` | `True` |
| `quality_lane` | `AGENTMETER_QUALITY_LANE` | `False` (opt-in, ADR-003) |
| `store_backend` | `AGENTMETER_STORE_BACKEND` | `memory` |

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
