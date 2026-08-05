# ADR-051: A late score needs its own span

**Status:** Accepted
**Date:** 2026-08-05
**Supersedes in part:** the emission half of ADR-003 (the quality lane's tiering is unchanged)

## Context

The quality lane's judge tier had never emitted a single score.

`QualityLane.on_run_end` dispatched the judge to a worker pool and then called
`JudgeRunner.collect(run_id)` on the next line, with `DEFAULT_COLLECT_TIMEOUT =
0.0`. For that to return anything, a model call had to complete between two
adjacent statements on the same thread.

Measured against a warm pool:

| Judge | Scores collected |
|---|---|
| instant, in-process | 2 / 200 |
| 5 ms | 0 / 200 |
| 200 ms (realistic) | 0 / 20 |

Every test asserted the opposite, and passed. Each built a *fresh* `QualityLane`,
whose `ThreadPoolExecutor` is created lazily on first submit — and
`Thread.start()` blocks until the new worker is actually running, handing an
instant in-process judge exactly the head start a real deployment never gives it:

| | Scores collected |
|---|---|
| fresh lane per run (what every test did) | 200 / 200 |
| one long-lived lane (what production does) | 2 / 200 |

The blast radius was larger than the two judge scores. `heuristic.score` is
deliberately one-sided — it reports failure or abstains, and *never* claims
success — so the judge was the only source of `gen_ai.run.success = true`
anywhere in the library. With it unreachable, `successes` could only ever be `0`,
which made `gen_ai.run.cost_per_successful_task` unreachable too. The headline
unit-economics signal, named in the README, could not be emitted by any
configuration.

This is the sixth instance of the shape ADR-042 named: the extension point
existed, was documented, was configured, spent the user's money — and nothing
could reach its output.

## Decision

**Judge scores are emitted on their own span, `optio.quality`, linked to the run
span and carrying `gen_ai.run.id`.**

The constraint that forces this is not ours: an ended OpenTelemetry span cannot
be modified. The judge answers hundreds of milliseconds after run end. So a score
can reach the user in exactly two ways — wait for it, or emit it elsewhere.

Waiting is unavailable. `@meter` returns the agent's result *after* run end, so
blocking there puts model latency directly on the agent's return path. That is
the one thing the library promises never to do (ADR-004, SC-5), and it would be a
strange trade: pay 200 ms of user-visible latency per sampled run to move an
attribute onto a span it does not need to be on.

So: elsewhere. Specifically —

- `JudgeRunner.submit(request, on_scores=...)` pushes the validated scores to a
  callback when they land, replacing dispatch-then-poll. Validation is shared
  with `collect` so a garbage judge is rejected identically on both paths.
- `optio.lanes.quality.deferred` emits the span. It is **linked** to the run
  span, not parented: the run span was exported before this one existed, and a
  child of a finished span is rendered by backends as a gap or dropped.
- `cost_per_successful_task` moves to that span too. The numerator is final at
  run end and the denominator is not, so this is the only place both are known.

Three consequences followed and are part of the decision:

**Lane order reversed.** The registry put quality ahead of cost so the cost lane
could read a success count quality wrote. That count can no longer be anything
but zero, so the dependency flipped: the deferred emitter needs `actual_cost`,
which the cost lane now publishes on the run object. Cost runs first. Without
that, an *instant* judge (a test double — never a real one) can answer before the
cost is written, and a signal whose presence depends on a race is worse than one
that is honestly absent.

**The tracer is injected, not resolved.** `trace.get_tracer` reaches for the
global provider. A user who passes their own `TracerProvider` would have had
every quality span recorded where none of their exporters are listening — the
same defect one step further along. The installer now threads its tracer through
the tap to the lane.

**In-flight judgements are drained and bounded.** `shutdown()` waits (bounded)
rather than cancelling, because cancelling a model call does not refund it — it
only discards what it bought. And because delivery now holds a run until its
scores land, at most `MAX_PENDING = 64` may be in flight; beyond that runs go
unscored with one logged warning, since the pool is two threads wide and an
unbounded queue is a leak that ends the process.

## Consequences

**Good.** The judge tier works. Every quality signal is emitted, including
`cost_per_successful_task`. Run end got ~24× cheaper — mean 24 µs against the
0.57 ms published through 0.4.0 — because it dispatches and returns rather than
polling. A test asserts a run with an unanswered judge finishes in under 500 ms,
so an implementation that starts waiting again fails rather than merely slows.

**Bad.** Consumers join two spans instead of reading one. Dashboards written
against the four moved attributes on the run span must be repointed — though
they have been reading an empty set, so nothing that worked stops working.

**Accepted.** A score can arrive after its trace has been exported, so a backend
that assembles traces eagerly may show the quality span separately. That is
honest: the score genuinely was not available when the run ended, and pretending
otherwise is what produced a feature that never worked.

## Alternatives considered

**A configurable blocking timeout.** Smaller change, keeps one span. Rejected as
the primary fix because a default that honours SC-5 is `0`, which leaves the
feature dead until the user finds a second knob — and any value that works adds
model latency to every sampled run's return path. It solves the documentation
problem rather than the defect.

**Hold the run span open until the judge answers.** Ends the span from a worker
thread, and inflates the run's recorded duration by the judge's latency —
corrupting the timing data the span exists to carry.

**Attach the score to the next run's span.** Cheapest, and wrong: it
misattributes an outcome to a different run, which is the silent-wrong class this
project treats as worse than missing.

## References

- ADR-003 — the quality lane is tiered, sampled and off by default
- ADR-004 — fail open; a dropped signal never becomes an agent error
- ADR-042 — the extension point that nothing could reach
- ADR-044 — absence is not zero
- ADR-050 — the store speaks the domain
- §11 — the quality lane's latency budget
