# Quality lane

Scores whether an agent run actually *worked* — the signal permission-based governance cannot
see. **Off by default** (ADR-003); everything below happens only if you turn it on.

```python
from optio import instrument
from optio.config import Config

instrument(agent, config=Config(quality_lane=True, judge=my_judge))
```

## The honest limits, first

This lane is the one most able to mislead you, so read this before the API.

**It will not tell you an answer is correct.** It can tell you a run errored, produced nothing,
or was truncated. With a judge, it can tell you what *your model* thought of the output. Neither
is ground truth, and the second is a model grading a model — the "who evals the evaluator"
problem (R-TECH-5). Treat these as evidence, not verdicts.

**`gen_ai.run.success` is absent far more often than you might expect.** By design. See below.

## Tiers

| Tier | Runs on | Cost | What it can see |
|---|---|---|---|
| **none** | lane disabled (the default) | zero | nothing |
| **heuristic** | every run when enabled | ~0.1 ms, no network | errors, empty output, truncation |
| **judge** | sampled runs only | one model call, your money | groundedness, task success |

The tier is decided once, at run start, and frozen. Re-rolling would let run start and run end
disagree about whether a run was scored — producing quality signals attached to runs nothing ever
evaluated, which is wrong data rather than missing data.

`quality_sample_rate` (default `0.1`) controls what fraction reaches the judge.

## Why success is usually absent

The heuristic **never reports success**. It reports failure on evidence and abstains otherwise.

That looks like a gap and is the most deliberate decision in the lane. A run that produced fluent
output is indistinguishable, from the outside, from one that produced fluent *wrong* output —
and "well-formed but wrong" is exactly the failure [§1.3](../IMPLEMENTATION.md) says existing
governance misses. A heuristic that emitted `success=true` because nothing looked broken would
manufacture the false assurance this lane exists to replace, on every run.

So:

| Situation | `gen_ai.run.success` | Lands on |
|---|---|---|
| Run errored, produced nothing, or was truncated | `false` | the run span |
| Judge scored `task_success` | `true`/`false` at the 0.5 threshold | the `optio.quality` span |
| Everything else | **absent** | — |

Absent means *unknown*. A policy must not read it as failure — see
[signals.md](signals.md#absence-is-meaningful).

## Where judge scores land

**Judge scores are on their own span, not the run span.** This is the one thing to know before
you build a dashboard on them.

The judge is a model call dispatched when the run ends. It answers hundreds of milliseconds later,
by which time the run span has closed — and an ended OpenTelemetry span cannot be modified. So
there were two options: make the run wait for the judge, or emit the score somewhere else.
Waiting would put model latency on your agent's return path, which is the one thing this library
promises never to do. So the score gets its own span:

```
optio.run.your_agent            ← closes when your agent returns
    gen_ai.run.actual_cost      = 0.0121
    gen_ai.run.loop_state       = "healthy"
    gen_ai.run.success          = false      ← only if the heuristic found failure

optio.quality                   ← emitted when the judge answers, links → the run span
    gen_ai.run.id                       = "8f2c…"
    gen_ai.run.quality.groundedness     = 0.92
    gen_ai.run.quality.task_success     = 0.85
    gen_ai.run.success                  = true
    gen_ai.run.cost_per_successful_task = 0.0121
```

Join the two on `gen_ai.run.id`, or follow the span link. A link rather than a parent-child
relationship, because the run span was already exported — a child of a finished span is rendered
by most backends as a gap, or dropped.

> **Upgrading from 0.3.0 or earlier?** These four attributes used to be documented as living on
> the run span. In practice they were never emitted at all: the lane dispatched the judge and then
> polled it with a zero-second timeout on the next line, so any judge that made a network call
> missed every time. If your dashboards query them on the run span, they have been reading an
> empty set and now need to read `optio.quality`.

### Making sure scores land before your process exits

A short-lived script can finish before the judge answers. `drain()` waits, bounded:

```python
from optio.runtime.installer import install_tap

tap = install_tap(config, provider)
...  # run your agent

for lane in tap.lanes:  # let outstanding judgements land
    if hasattr(lane, "drain"):
        lane.drain(timeout=5.0)  # returns how many were abandoned
```

Long-running services do not need this — the scores arrive while the process keeps running.

## Writing a judge

A judge is a callable you supply. **optio ships no default and constructs no model client**,
because either would mean spending your money and using your credentials on our initiative (§10).

```python
from optio.lanes.quality.judge import JudgeRequest, JudgeScores


def my_judge(request: JudgeRequest) -> JudgeScores:
    # Your SDK, your credentials, your prompt.
    verdict = my_llm.evaluate(run_id=request.run_id, steps=request.step_count)
    return JudgeScores(groundedness=verdict.grounded, task_success=verdict.solved)
```

`request.step_count` is **how many steps the run actually took**. Through 0.3.0 it was the size of
an internal span buffer capped at 64, so any longer run understated itself — a 500-step run was
reported as 64. If you calibrated a rubric against that number, it will now be larger and correct.

Rules the runner enforces so you don't have to:

- **It may block.** It runs on a worker thread, and the run does not wait for it.
- **It may raise.** Any exception becomes a missing signal, never an agent error (ADR-004). Only
  the exception *type* is logged — messages from model clients routinely carry the prompt.
- **Scores outside `[0, 1]` are dropped, not clamped.** A judge returning `7` misread the scale;
  clamping to `1.0` would publish a confident wrong number where absence is honest.
- **Return an empty `JudgeScores()` to decline.** No signal is emitted.
- **At most 64 judgements are in flight at once.** Past that, further runs go unscored and a
  warning is logged once. The pool is two threads wide, so a 200 ms judge sustains roughly ten
  judged runs a second; beyond that, runs end faster than the judge can answer. Dropping the
  extras keeps the queue from growing without bound. If you see that warning, lower
  `quality_sample_rate` or make your judge faster.

The run never waits for it. Scores are emitted on a separate span when they arrive — see
[where judge scores land](#where-judge-scores-land).

### Content and privacy

`JudgeRequest.content` is empty unless you fill it. optio passes no trace text of its own
(§10) — if your judge needs the prompt, close over your own record of it. What the judge sees is
your data going to your model; we neither log nor retain it.

## Cost per successful task

Enabling this lane **with a judge** is what makes `gen_ai.run.cost_per_successful_task` appear. It
is the number that says whether an agent is worth running: the cost lane owns the numerator, the
judge owns the denominator.

It is emitted on the **`optio.quality` span**, for the same reason the scores are: the cost is
final when the run ends and the outcome is not. That span is the only place both are known.

It stays **absent** when a run wasn't judged, or was judged a failure — an unknown denominator
makes the ratio unknowable, and a run that succeeded at nothing has a cost and a failure, both
already reported separately. Publishing a headline unit-economics figure derived from a guess
would be worse than publishing nothing.

The heuristic alone cannot produce this number, because it never reports success. A judge is
required.

## Overhead

Measured by the CI benchmark job, not asserted:

| Path | Result | Budget (§11) |
|---|---|---|
| Inline heuristic, per step | mean 127 µs, p99 243 µs | < 10 ms p99 |
| Run end with a 200 ms judge | **mean 24 µs, p99 243 µs** | judge is off the hot path |
| Per-step state, in process | ~1 µs, flat at 10 steps and 10,000 | — |
| Per-step state, shared store | ~600 µs, of which ~500 µs is the round trip | < 3 round trips |

That second row is the guarantee that makes the judge tier usable: a slow judge does not become a
slow agent. It is ~24× cheaper than the 0.57 ms published through 0.3.0, because run end no longer
polls the judge at all — it dispatches and returns. A test asserts a run with an unanswered judge
completes in under 500 ms, so an implementation that started waiting again would fail rather than
merely get slower.

The last row is worth reading carefully before you enable `store_backend="redis"` with this lane.
Quality emits nothing per step — it only scores at run end — so on a shared store it pays a round
trip that buys nothing until the run finishes. That is a real cost, stated rather than smoothed
over. It is one round trip, not more: a step is a single Lua script writing four fields, and the
benchmark asserts the ratio so a regression shows up as a failure.

## Across processes

If your agent is sharded across workers, set `store_backend="redis"`. Without it, each worker keeps
its own state and the run is scored by whichever worker happens to observe run end — using only the
steps that landed there. Two things go wrong, both quietly:

- **Your judge is told the wrong run length.** Four workers means `step_count` is a quarter of the
  truth, and a rubric that scales with run length scores a long run as if it were short.
- **The heuristic scores the wrong step.** It reads the run's *final* step, which is where the
  answer is. Sharded, each worker has its own last step — so a run that ended in an error can be
  scored from a healthy step that finished earlier somewhere else.

With the shared store, whichever step reaches the server last is the run's last step, and the count
is the whole run. Four processes writing to one `run_id` produce one score, asserted by
`tests/integration/test_multiprocess_quality.py`.

## Configuration

| Option | Env var | Default |
|---|---|---|
| `quality_lane` | `OPTIO_QUALITY_LANE` | `False` |
| `quality_sample_rate` | `OPTIO_QUALITY_SAMPLE_RATE` | `0.1` |
| `judge` | — (code only) | `None` |
| `store_backend` | `OPTIO_STORE_BACKEND` | `"memory"` |

`judge` has no environment variable on purpose: it is a callable, and a config path that could
name an arbitrary import target to invoke with your credentials is not one worth having.

Enabling the lane without a judge is valid and logs a warning at setup — you get the heuristic
tier, and saying so once beats letting you believe you enabled deep scoring.
