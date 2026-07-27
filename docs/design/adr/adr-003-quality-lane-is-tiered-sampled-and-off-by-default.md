# ADR-003 — The quality lane is tiered, sampled, and off by default

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-004, R-TECH-5, §11, docs/quality.md

## Context

"Was this run any good?" is the signal that makes cost meaningful — $2 well spent and $2 wasted
are different facts, and `cost_per_successful_task` is the number that distinguishes them. It
is also the hardest signal to produce honestly.

Scoring output quality means asking a model to grade a model. That costs money on the user's
account, adds hundreds of milliseconds against a 5 ms per-step budget (SC-5), and produces a
judgement no more trustworthy than the judge — the "who evaluates the evaluator" problem
(R-TECH-5).

A quality lane that ran by default would mean installing optio silently started spending money
and slowing agents down. That is how a monitoring library gets uninstalled.

## Decision

**Three tiers, sampled, off unless explicitly enabled.**

| Tier | Runs on | Cost | Sees |
|---|---|---|---|
| none | the default | zero | nothing |
| heuristic | every run when enabled | ~0.1 ms, no network | errors, empty output, truncation |
| judge | sampled runs only | one model call, user's money | groundedness, task success |

Four constraints make it safe:

**The tier is decided once, at run start, and frozen.** Re-rolling would let run start and run
end disagree about whether a run was scored, producing quality signals attached to runs nothing
evaluated — wrong data rather than missing data.

**The judge never runs on the hot path.** It is dispatched at run end to a worker thread, and
the run does not wait. If it has not answered by the time signals are collected, no quality
signal is emitted. Measured: a 200 ms judge costs the run 0.5 ms.

**optio ships no judge and constructs no model client.** The judge is a callable the user
supplies, running on their SDK and their credentials. A default would mean spending the user's
money the moment they flipped the flag (§10).

**The heuristic never reports success.** It reports failure on evidence and abstains otherwise.
A run that produced fluent output is indistinguishable from one that produced fluent *wrong*
output, and "well-formed but wrong" is exactly the failure §1.3 says existing governance
misses. Emitting `success=true` because nothing looked broken would manufacture the false
assurance this lane exists to replace.

## Alternatives

**On by default.** Rejected. Any default that spends money is a default that should not exist.

**Judge every run.** Rejected on cost and latency. Sampling at 0.1 gives a usable population
estimate at a tenth of the spend, and quality is a distribution question, not a per-run one.

**Ship a default judge using a small cheap model.** Tempting and rejected twice over: it needs
credentials we refuse to handle (§10), and a cheap judge is a bad judge whose scores would
carry the same authority as a good one.

**Block the run until the judge answers.** Rejected outright. It would make a slow judge a slow
agent, violating SC-5 and ADR-004 in one step.

## Consequences

**Good**

- Cost and behavior lanes deliver value with zero configuration and zero spend; quality is
  purely additive.
- No user is ever surprised by a bill or a latency regression from installing this library.
- The heuristic tier gives real failure detection for free, with no network call.

**Costs, accepted deliberately**

- **`gen_ai.run.success` is absent far more often than users expect.** By design, and documented
  prominently, but it is the most common source of "is this broken?" questions.
- **`cost_per_successful_task` — arguably the headline signal — requires opting in.** An unknown
  denominator makes the ratio unknowable, and a unit-economics figure derived from a guess is
  worse than none.
- **Sampled scores need care to interpret.** A judge score describes the sampled tenth, not the
  run in front of you, which is why the sampling rate is published as a self-metric (§12).
- **We do not solve "who evaluates the evaluator".** The user's judge is the user's problem; we
  validate its output range and drop nonsense, and that is the limit of what we can honestly do.
