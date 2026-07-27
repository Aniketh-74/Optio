# Quality lane

Scores whether an agent run actually *worked* — the signal permission-based governance cannot
see. **Off by default** (ADR-003); everything below happens only if you turn it on.

```python
from agentmeter import instrument
from agentmeter.config import Config

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

| Situation | `gen_ai.run.success` |
|---|---|
| Run errored, produced nothing, or was truncated | `false` |
| Judge scored `task_success` | `true`/`false` at the 0.5 threshold |
| Everything else | **absent** |

Absent means *unknown*. A policy must not read it as failure — see
[signals.md](signals.md#absence-is-meaningful).

## Writing a judge

A judge is a callable you supply. **agentmeter ships no default and constructs no model client**,
because either would mean spending your money and using your credentials on our initiative (§10).

```python
from agentmeter.lanes.quality.judge import JudgeRequest, JudgeScores


def my_judge(request: JudgeRequest) -> JudgeScores:
    # Your SDK, your credentials, your prompt.
    verdict = my_llm.evaluate(run_id=request.run_id, steps=request.step_count)
    return JudgeScores(groundedness=verdict.grounded, task_success=verdict.solved)
```

Rules the runner enforces so you don't have to:

- **It may block.** It runs on a worker thread, and the run does not wait for it.
- **It may raise.** Any exception becomes a missing signal, never an agent error (ADR-004). Only
  the exception *type* is logged — messages from model clients routinely carry the prompt.
- **Scores outside `[0, 1]` are dropped, not clamped.** A judge returning `7` misread the scale;
  clamping to `1.0` would publish a confident wrong number where absence is honest.
- **Return an empty `JudgeScores()` to decline.** No signal is emitted.

If the judge hasn't answered by the time the run ends, **no quality signal is emitted for that
run**. A late score is worth less than a fast run, and the cost signals are already correct
without it.

### Content and privacy

`JudgeRequest.content` is empty unless you fill it. agentmeter passes no trace text of its own
(§10) — if your judge needs the prompt, close over your own record of it. What the judge sees is
your data going to your model; we neither log nor retain it.

## Cost per successful task

Enabling this lane is what makes `gen_ai.run.cost_per_successful_task` appear. It is the number
that says whether an agent is worth running: the cost lane owns the numerator, this lane owns the
denominator.

It stays **absent** when a run wasn't scored — an unknown denominator makes the ratio unknowable,
and publishing a headline unit-economics figure derived from a guess would be worse than
publishing nothing.

## Overhead

Measured by the CI benchmark job, not asserted:

| Path | Result | Budget (§11) |
|---|---|---|
| Inline heuristic, per step | mean 127 µs, p99 243 µs | < 10 ms p99 |
| Run end with a 200 ms judge | **0.57 ms** | judge is off the hot path |

That second row is the guarantee that makes the judge tier usable: a slow judge does not become a
slow agent.

## Configuration

| Option | Env var | Default |
|---|---|---|
| `quality_lane` | `AGENTMETER_QUALITY_LANE` | `False` |
| `quality_sample_rate` | `AGENTMETER_QUALITY_SAMPLE_RATE` | `0.1` |
| `judge` | — (code only) | `None` |

`judge` has no environment variable on purpose: it is a callable, and a config path that could
name an arbitrary import target to invoke with your credentials is not one worth having.

Enabling the lane without a judge is valid and logs a warning at setup — you get the heuristic
tier, and saying so once beats letting you believe you enabled deep scoring.
