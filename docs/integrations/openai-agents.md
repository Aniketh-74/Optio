# OpenAI Agents SDK

```bash
pip install "agentmeter[openai]"
```

```python
from agents import Agent
from agentmeter import instrument

agent = Agent(name="researcher", instructions="...", tools=[...])
instrument(agent)
```

## You need an OTel bridge

**The Agents SDK's built-in tracing is not OpenTelemetry.** It exports to OpenAI's own trace
backend, and agentmeter cannot read it. Without a bridge you will get no signals and no error —
`instrument()` succeeds, the tap is installed, and no `gen_ai.*` spans ever arrive.

Install one of these:

```bash
pip install opentelemetry-instrumentation-openai-agents   # OTel-native
pip install logfire                                       # logfire.instrument_openai_agents()
pip install openinference-instrumentation-openai-agents   # OpenInference
```

Then configure it before the first run, alongside a tracer provider:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# ... your chosen bridge's setup call here ...

instrument(agent)
```

`ConsoleSpanExporter` is the fastest way to confirm signals are flowing; swap it for OTLP once
you can see `gen_ai.run.actual_cost` in the output.

## What to pass to `instrument()`

Pass the `Agent`. Not `Runner`, not `ClaudeAgentOptions`-style config objects.

An `Agent` in this SDK is a passive configuration object — name, instructions, tools, handoffs —
that a separate `Runner` executes. Instrumentation is therefore **process-wide**: the tap goes on
the tracer provider, and the agent object is handed back untouched. Instrumenting one agent
covers every agent run in the process, including handoffs.

`instrument()` returns the same object it was given, so `agent = instrument(agent)` is safe but
unnecessary.

## Handoffs

A handoff produces spans like any other step, so they flow into the same run — cost accumulates
across the whole chain rather than resetting per agent. That is usually what you want for a
budget. If you need per-agent attribution, wrap each in its own `RunContext`.

## Troubleshooting

**No signals, no errors.** Almost always the missing bridge above. Check that `gen_ai.*` spans
reach your exporter *before* suspecting agentmeter — if the console exporter shows no GenAI
spans, there is nothing to tap.

**`UnsupportedFrameworkError: adapter 'openai_agents' cannot instrument ...`.** The object is not
an SDK `Agent`. The adapter requires the module to come from `agents.*` plus SDK-specific fields;
that second check exists because `agents` is a short, generic top-level name a user package could
plausibly claim, and a coincidental match would silently instrument the wrong object.

See [docs/runbooks.md](../runbooks.md) for the general checklist.
