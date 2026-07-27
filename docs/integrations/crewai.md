# CrewAI

```bash
pip install "agentmeter[crewai]"
pip install openinference-instrumentation-crewai   # produces the GenAI spans
```

```python
from crewai import Agent, Crew, Task
from agentmeter import instrument

crew = Crew(agents=[...], tasks=[...])
instrument(crew)

result = crew.kickoff()
```

## Setup

CrewAI does not emit OTel GenAI spans on its own. Install instrumentation and a tracer provider
before the first `kickoff()`:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from openinference.instrumentation.crewai import CrewAIInstrumentor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
CrewAIInstrumentor().instrument(tracer_provider=provider)

instrument(crew)
```

CrewAI also wraps LiteLLM for model calls. If you already run
`openinference-instrumentation-litellm`, you have GenAI spans for the LLM steps even without the
CrewAI instrumentor — but you will miss the task and agent structure, which is what makes the
behavior lane useful.

## Crew or agent

Both work:

```python
instrument(crew)  # usual: the Crew owns kickoff()
instrument(agent)  # fine for a single-agent flow
```

The tap is installed process-wide either way, so there is no behavioral difference — accepting
both just avoids a papercut. A `Task` is rejected: it is not an executable unit and accepting it
would let `instrument()` look like it worked.

## Multi-agent runs and cost attribution

A crew's agents share one run, so `gen_ai.run.actual_cost` is the **crew total**, not per agent.
That is the number a budget policy wants — you care what the workflow cost, not which member
spent it.

For per-agent attribution, wrap each in its own `RunContext`. Note that this also splits the
behavior window, so a loop spanning two agents becomes two shorter, possibly healthy-looking
windows. Split for billing, not for loop detection.

## Hierarchical process

`Process.hierarchical` adds a manager agent that delegates. Delegation calls are tool calls, so
they enter the behavior window like any other step. A manager retrying a failing worker is
exactly the shape `retry_storm` is built to catch.

## Troubleshooting

**No signals.** Confirm `gen_ai.*` spans reach your exporter first. A console exporter showing no
GenAI spans means the instrumentor is missing or was configured after the run started.

**`UnsupportedFrameworkError`.** The object is neither a `Crew` (has `agents` and `tasks`) nor an
`Agent` (has `role` and `goal`). Check you are not passing a `Task` or a `crewai.Flow`.

See [docs/runbooks.md](../runbooks.md) for the general checklist.
