# Claude Agent SDK

```bash
pip install "optio[claude]"
```

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from optio import instrument

client = ClaudeSDKClient(options=ClaudeAgentOptions(...))
instrument(client)

async with client:
    await client.query("...")
    async for message in client.receive_response():
        ...
```

## The out-of-process caveat — read this one

This SDK is shaped differently from the others, and it changes what the signals mean.

The agent loop runs in the **Claude Code process**, not yours. The SDK is a client to it. So the
spans optio taps are the ones *your* OTel instrumentation creates around your SDK calls —
they are not emitted by the agent loop itself.

Two consequences:

**Cost reflects what the SDK reports back.** Token usage arrives in the response and lands on
your spans; optio reconciles from that. If a step fails before returning usage, that step's
cost is missing rather than wrong — absence, not a bad number
([docs/signals.md](../signals.md#absence-is-meaningful)).

**Behavior signals see your calls, not the agent's internal tool use.** Tool calls made *inside*
the Claude Code process are invisible to the span tap unless they surface as messages you
instrument. A tight internal loop may therefore look like one long step from outside. The cost
lane still catches it — spend keeps climbing — but do not expect `loop_state` to have the same
resolution it has under LangGraph or CrewAI.

This is a real limitation of the integration, not a bug to work around. If loop detection matters
more than the SDK's convenience, an in-process framework gives the behavior lane more to work
with.

## Setup

Create spans around your SDK calls and give optio a tracer provider:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

instrument(client)
```

Your spans need the GenAI attributes — at minimum `gen_ai.request.model` and the
`gen_ai.usage.*` token counts — or the cost lane has nothing to price. The names are listed under
[*Attributes consumed*](../signals.md#attributes-consumed-not-emitted).

## What to pass to `instrument()`

Pass the `ClaudeSDKClient`. `ClaudeAgentOptions` is rejected: it is a configuration dataclass, and
accepting it would let `instrument(options)` appear to work while instrumenting nothing.

For the one-shot `query()` function there is no client object to pass, so use `RunContext`:

```python
from optio import RunContext
from claude_agent_sdk import query

with RunContext(budget="$0.50"):
    async for message in query(prompt="..."):
        ...
```

## Package naming

Both `claude_agent_sdk` and the pre-rename `claude_code_sdk` are recognised. If you are on the
old package, the adapter works and there is nothing to change.

## Troubleshooting

**No signals.** Check that your own spans carry `gen_ai.*` attributes. This is the adapter where
that is most often the problem, because you are creating the spans rather than an instrumentation
package doing it for you.

**Cost is lower than the API bill.** Expected if some steps returned no usage data — those are
omitted rather than estimated. Compare step counts against the run.

See [docs/runbooks.md](../runbooks.md) for the general checklist.
