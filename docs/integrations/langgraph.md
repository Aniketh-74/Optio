# LangGraph

```bash
pip install "agentmeter[langgraph]"
```

```python
from agentmeter import instrument

graph = builder.compile()
instrument(graph)

graph.invoke({"messages": [...]})
```

## Setup

LangGraph builds on LangChain, whose OTel instrumentation comes from third-party packages.
Install one, plus a tracer provider:

```bash
pip install openinference-instrumentation-langchain
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
LangChainInstrumentor().instrument(tracer_provider=provider)

instrument(graph)
```

Start with `ConsoleSpanExporter` to confirm signals appear, then switch to OTLP for your backend.

## What to pass to `instrument()`

Pass the **compiled** graph — the result of `builder.compile()`, which exposes `invoke` and
`stream`. An uncompiled `StateGraph` is rejected, because it cannot run and instrumenting it
would silently cover nothing.

## Cycles and the behavior lane

LangGraph's conditional edges make cycles a first-class construct, which is exactly the shape the
behavior lane watches. A node that keeps routing back to itself with equivalent arguments raises
`repeat_count` and eventually reports `looping`.

Two things worth knowing:

**A cycle is not automatically a pathology.** A retry edge with a bounded counter is healthy and
deliberate. The detector accounts for this — see the false-positive corpus in
[docs/behavior.md](../behavior.md) — but if your graph loops by design over many steps, check
`loop_state` against a few real runs before gating on it.

**Recursion limits and loop detection are different tools.** LangGraph's `recursion_limit` stops
a runaway graph by step count. `loop_state` reports that the run stopped *making progress*, which
fires earlier and carries the reason. Use both; they fail differently.

## Streaming

`graph.stream()` is instrumented the same as `invoke()`. Signals accumulate per step, so a
long-running stream reports rising cost as it goes rather than only at the end — that is what
makes pre-spend gating possible.

## Subgraphs

Subgraph steps join the parent run, so cost is the whole-graph total and the behavior window sees
every step regardless of nesting depth. Use a separate `RunContext` if you need a subgraph
metered on its own budget.

## Troubleshooting

**No signals.** Confirm `gen_ai.*` spans reach your exporter first. LangChain instrumentation
configured *after* the graph runs produces nothing.

**`UnsupportedFrameworkError`.** The target is not a compiled graph. Check for a missing
`.compile()`, or that you are not passing a bare LangChain runnable.

See [docs/runbooks.md](../runbooks.md) for the general checklist.
