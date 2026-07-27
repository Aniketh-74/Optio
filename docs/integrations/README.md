# Integrations

One line instruments an agent:

```python
from agentmeter import instrument

instrument(agent)  # adapter auto-detected
instrument(agent, adapter="crewai")  # or named explicitly
```

| Framework | Adapter name | Extra | Guide |
|---|---|---|---|
| LangGraph | `langgraph` | `agentmeter[langgraph]` | [langgraph.md](langgraph.md) |
| OpenAI Agents SDK | `openai_agents` | `agentmeter[openai]` | [openai-agents.md](openai-agents.md) |
| CrewAI | `crewai` | `agentmeter[crewai]` | [crewai.md](crewai.md) |
| Claude Agent SDK | `claude_agent` | `agentmeter[claude]` | [claude-agent-sdk.md](claude-agent-sdk.md) |
| Anything else | — | — | [Unsupported frameworks](#unsupported-frameworks) |

## What an adapter does — and does not do

An adapter installs the span tap on your tracer provider. That is all it does.

It does **not** make your framework emit GenAI spans. That is the ecosystem's job, handled by
OpenTelemetry instrumentation packages, OpenInference, or the framework's own exporter. Taking it
on would put us on the hook for every framework release (R-TECH-3), and the division is what
keeps a CrewAI upgrade from breaking the cost lane.

**So agentmeter needs two things installed, not one:** the OTel instrumentation that produces
`gen_ai.*` spans, and agentmeter to read them. Each guide names the instrumentation package it
expects. If no tracer provider is configured, `instrument()` logs a warning at setup rather than
failing — you may be wiring things in a different order — but no signals will appear until one is.

## Failure behavior

Adapters fail **loudly**; the runtime fails **open**.

A missing framework, an unknown adapter name, or a target the adapter does not recognise raises
at setup, because silently instrumenting nothing would leave you believing you have coverage you
do not have. Once running, nothing agentmeter does can break or block your agent (ADR-004) — a
lane that fails emits no signal and the run continues.

## Unsupported frameworks

No adapter is required. `RunContext` governs any code that emits GenAI spans:

```python
from agentmeter import RunContext

with RunContext(budget="$0.50"):
    ...  # spans emitted in here are governed
```

This is also the path for a raw SDK loop, a homegrown agent, or a framework we have not adapted
yet.

## Verifying it works

Signals should appear within a few steps. If they do not, work through
[docs/runbooks.md](../runbooks.md) — the usual cause is missing OTel instrumentation, meaning
there are no GenAI spans to tap.
