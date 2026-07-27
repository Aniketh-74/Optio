# optio demo

An agent gets stuck in a retrieval loop and keeps paying for it. optio emits the signals;
a policy reads them and stops the run.

```bash
docker compose up --build
```

Or, with no Docker at all:

```bash
pip install -e ../..
python run_demo.py
```

Both take about ten seconds. **No API keys, no network calls** — see [Why the model is fake](#why-the-model-is-fake).

## What you'll see

```
  without optio signals
    steps        60
    cost         $2.1833
    loop_state   looping   (repeat_count 20)

  with optio signals + policy
    steps        23
    cost         $0.3582
    loop_state   looping   (repeat_count 19)
    stopped by   loop_state == looping (repeat_count 19)

  result
    caught the loop after 23 steps instead of 60
    saved $1.8250 of $2.1833 (84% of this run)
```

The same agent, twice. The only difference is that the second run had something watching the
signals.

## The part that matters

**optio did not stop anything.** It emitted `gen_ai.run.loop_state` and the rest; the rules
in [`policy.py`](policy.py) made the call (ADR-001). Edit those thresholds and re-run — that's the
intended way to poke at this.

`policy.py` stands in for OPA, Cedar, or Microsoft AGT so the demo needs no engine installed. The
[shipped packs](../../policies/) are the same rules in each engine's real language.

Note **which** signal caught it. The run was stopped at $0.36 against a $2.00 budget — a
cost-only tool would have let it run four times longer before noticing. The behavioral evidence
arrived while the run still looked affordable, and that gap is the thing optio exists for.

## The compose stack

Two services: a real OTel Collector, and the demo exporting to it over OTLP. The collector's
debug exporter prints every span, so you can watch `gen_ai.run.actual_cost` and
`gen_ai.run.loop_state` land on the run span instead of taking the summary on trust.

Swapping the debug exporter for Langfuse, Honeycomb, or Grafana is a one-line change in
[`otel-collector-config.yaml`](otel-collector-config.yaml). That portability is what ADR-002 buys
by using OTel semconv rather than inventing a schema.

Running `python run_demo.py` directly skips the collector — you get the same summary with no
infrastructure, which is the fastest path to seeing the result.

## Why the model is fake

The "LLM" is a scripted stand-in ([`agent.py`](agent.py)). That's a design constraint, not a
shortcut: ADR-006 makes this an evaluator-facing deliverable that has to run on a fresh machine
in one command, and a demo requiring an API key — and costing real money each run — is a demo
nobody runs.

What is **not** faked is everything being demonstrated. The spans are real OTel GenAI spans with
real semconv attributes, the token counts drive the real pricing table, and the signals come from
the real cost and behavior lanes. Swap `ScriptedModel` for an SDK call and nothing downstream
changes.

## Files

| File | Role |
|---|---|
| [`agent.py`](agent.py) | The misbehaving agent and its scripted model |
| [`policy.py`](policy.py) | The rules that stop the run — edit these |
| [`run_demo.py`](run_demo.py) | Runs both scenarios, prints the comparison, verifies its own claims |
| [`docker-compose.yml`](docker-compose.yml) | Collector + demo |
| [`otel-collector-config.yaml`](otel-collector-config.yaml) | Stock collector config; swap the exporter here |

## A note on the window size

The demo sets `behavior_window_size=20` rather than the default 50, and the reason is worth
knowing before you copy it.

`looping` requires the window to contain almost nothing but the repeated call, so this agent's
four productive opening steps have to age out first. At the default that takes 53 steps — by
which point the run has spent $1.72 and a *cost* rule would have fired first, making the demo
showcase the wrong signal.

20 catches it at step 23 for $0.36. Shorter windows detect faster and tolerate less legitimate
repetition; 20 is a reasonable production value, not a number invented to make the demo look
good. [docs/behavior.md](../../docs/behavior.md) covers the trade-off.

## It tests itself

`run_demo.py` exits non-zero if the governed run didn't actually beat the ungoverned one, so
`docker compose up --exit-code-from demo` works as a smoke test — which is exactly how CI runs
it. A demo that quietly stops demonstrating anything is worse than no demo.
