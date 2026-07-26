# ADR-010 — A closed run is final

**Status:** Accepted
**Date:** 2026-07-27
**Related:** R-TECH-1, ADR-004, M2-2

## Context

The ledger's stated invariant (R-TECH-1) covers ordering *within* a run:
reserve precedes the step, reconcile replaces the reservation exactly once. It
says nothing about what happens **after** a run ends.

The model-based property test found that gap immediately, in two forms:

1. `close_run` recorded `leaked_steps` but never cleared it. Reserve → close →
   reconcile → close reported a leak that no longer existed.
2. Closing a run with no state returned early without marking anything, leaving
   the run re-openable. A later `reserve` succeeded, so cost could accumulate
   against a run whose total had already been reported — and the run would
   appear to have begun after it ended.

Neither raises. Both produce a number that is quietly wrong, which is precisely
the class of bug the ledger's property tests exist to catch and the reason
fail-open does not help here (ADR-004).

The underlying question is real rather than academic. Straggling callbacks are
normal: a framework retry lands late, an async tool call completes after the
graph returns, a run-end hook fires before the last span is exported.

## Decision

**`close_run` is terminal for a run id.** After it:

- `reserve` raises `LedgerInvariantError`.
- `reconcile` raises `LedgerInvariantError`.
- `close_run` again returns the same snapshot, unchanged.
- Closing a run the ledger has never seen still closes it.

`leaked_steps` is *assigned* at close, describing what was open at that moment,
rather than accumulated across calls.

Repeat closes must stay safe because run end can legitimately fire more than
once (M1-2) — the context manager, an adapter's own hook, a framework callback.
Idempotent is not the same as re-openable, and the distinction is the decision.

## Alternatives

**Accept late reconciles and update the total.** Rejected. A downstream policy
may already have read `actual_cost` and acted on it. Changing the number
afterwards means two consumers reading the same run get different answers, with
no way to tell which was current.

**Accept late reconciles into a separate "post-close" bucket.** Rejected as
complexity without a consumer. Nothing in `docs/signals.md` can express
"additional cost discovered after the run was reported", so the bucket would be
recorded and never read.

**Silently ignore operations on a closed run.** Rejected. Silence is how the
original bug hid. A late reconcile means the caller's ordering assumptions are
wrong, and that deserves a fail-open activation the operator can see rising
(`agentmeter.internal.lane_errors`) rather than nothing at all.

## Consequences

**Good**

- A run's reported cost is stable once reported: read it twice, get the same
  answer.
- Late arrivals are visible as guard activations rather than as silent drift.
- Repeat run-end is safe, which M1-2 already required.

**Costs, accepted deliberately**

- **Genuinely late cost is lost.** A step that completes after run end is never
  counted. The run reports its reserved worst case for that step instead, which
  over-reports rather than under-reports — the safe direction, since
  under-reporting is what lets an over-budget run through.
- **`close_run` allocates state for an unknown run id.** Needed to make the
  closure stick. Bounded by `evict`, which the lane calls, but a caller that
  closes arbitrary ids without evicting would grow the dict.
- **The error is raised at the ledger, absorbed at the guard.** A late reconcile
  costs one dropped signal and one activation. That is the intended trade:
  loud enough to diagnose, harmless to the agent.
