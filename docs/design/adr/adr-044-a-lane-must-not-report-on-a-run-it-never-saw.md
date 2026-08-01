# ADR-044 — A lane must not report on a run it never saw

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-010 (closing a run is final), ADR-043 (`id()` is an address), R-TECH-1

## Context

Fixing ADR-043 turned `main` green and turned the **floor job** red:

```
tests/integration/test_cost_signals_end_to_end.py::TestCostReachesTheSpan::
  test_an_unknown_model_reports_no_cost_rather_than_a_free_run
AssertionError: gen_ai.run.budget_remaining was emitted for a run whose every
step was unpriceable; an unknown cost must be absent, never zero or a full budget
assert 'gen_ai.run.budget_remaining' not in mappingproxy(
    {'gen_ai.run.budget_remaining': 10.0, ...})
```

That test carries its own history: *"Previously the run span carried
budget_remaining = 10.0: the full budget, for a run that made three million-token calls. A policy
reading `deny if budget_remaining < 0.50` would never fire, and the runaway agent this library exists
to catch would run unchecked."*

The arithmetic was not wrong. `budget_remaining` already refuses to answer without cost evidence, and
`_has_cost_evidence` already separates the four states carefully — nothing attempted, in flight,
attempted but unpriceable, genuinely free. Reproduced directly, all of it behaves:

| ledger state | evidence | budget_remaining |
|---|---|---|
| 3 steps reserved at 0.0, none reconciled | `False` | `None` |
| same, after `close_run` | `False` | `None` |
| **a run this ledger never saw** | **`True`** | **10.0** |

The hole was never in the arithmetic. It was in **who was allowed to do the arithmetic**.

`on_run_end` is broadcast to *every* registered run-end observer, not only the lane that metered the
run. A second live tracer provider in the same process — two agents, a test suite, a service that
reconfigures tracing — means a second tap with its own `CostLedger`, and that ledger is asked about
runs it never saw. Its zeros are indistinguishable from "nothing attempted yet", the one state where
a full budget genuinely is available. So it reports one.

**ADR-043 is what exposed it.** Before that fix, foreign taps were mostly never installed, because
`id()` reuse made `install_tap` return early. They were not around to answer. Fixing one silent
failure revealed the one it had been hiding — and the CI symptom is exactly diagnostic of it:
`budget_remaining` present, `actual_cost` absent, which no single ledger can produce.

## Decision

`CostLedger.knows(run_id)` — has this ledger ever recorded anything for this run — and `on_run_end`
returns `[]` when the answer is no.

This is the other half of a distinction `is_finalised` already draws. Its docstring says so
outright: *"Distinct from 'unknown': a caller needs to tell this run is over from I have never seen
this run, because an all-zero snapshot looks identical either way."* The hazard was understood and
solved for the finalised case; the unknown case had no equivalent.

`knows` deliberately does **not** also report recently-closed runs. Every caller pairs it with
`is_finalised`, which already returns early for those, so the extra clause cannot change an outcome —
a mutation removing it left every test green, which is how it was found. An untestable branch in a
correctness check is worse than no branch: it looks like protection and cannot be shown to protect.

## Consequences

2,189 tests pass. **5 of 5 mutations caught**, and the two that survived the first round are the
interesting part:

- Removing the `is_finalised` guard broke nothing, because the new `knows` check now covers the
  ordinary second firing — `on_run_end` closes *and* evicts, so the run has left the ledger by then.
  The state `is_finalised` actually guards, closed-but-not-yet-evicted, is only reachable by closing
  without evicting. A test now constructs it directly.
- The redundant clause in `knows` was removed rather than tested, on the principle above.

Both were guards that read as protective and could not be shown to protect. This is the second
consecutive ADR where mutation testing found the *tests* wrong rather than the code — ADR-043 needed
three rounds of it.

**The general rule:** a broadcast is not an address. When a callback is delivered to every registered
observer, "was this addressed to me?" is a question each observer has to answer for itself, and
answering it from an empty state produces a confident number about something never observed. That is
worse than silence, because silence is visibly missing and a full budget looks like good news.
