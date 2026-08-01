# ADR-043 — `id()` is an address, and addresses are recycled

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-010 (closing a run is final), ADR-014 (`optio_optimize` emits spans optio reads),
ADR-039 (evidence carries its date), ADR-040 (replace a lucky case with the general property)

## Context

CI on `main` failed with:

```
tests/optimize/test_telemetry.py::TestOptioPricesASpanThisPackageEmits::
  test_a_cache_hit_is_priced_at_zero_by_optios_own_ledger
KeyError: 'gen_ai.run.actual_cost'
```

The run span carried no cost attribute at all. It passed on the same commit, same Python version, in
a different job, and passed locally in isolation, in the full suite, twice consecutively, with a cold
tokenizer cache, and under CI's exact failing command.

Every obvious explanation was wrong. Not the Python version — 3.12 passed the same test elsewhere in
the same run. Not the latency budget — `record_span` runs in `complete`, outside `_run_stages`. Not a
span-id collision — `_step_id` uses the span id. Not fixture pollution — both fixtures are
function-scoped. Not either recent merge — the test dates from ADR-014 and both merges are in a
different package.

## What it actually was

`install_tap` tracks which providers it has already tapped in `dict[int, OptioSpanTap]`, keyed by
`id(provider)`. **`id()` is a memory address**, and CPython hands an address straight back out once
the object at it is collected. Measured:

```
address reuse across 200 short-lived TracerProvider instances: 189
```

**Ninety-five percent.** So a brand-new provider routinely inherits a dead provider's entry, the
lookup returns early, and **no span processor is ever added to the new provider**. Nothing is tapped,
so `CostLane.process_span` never runs, so no step is reconciled, so `on_run_end` correctly omits the
cost — a run nobody could price reports no cost rather than zero (ADR-010's rule, working exactly as
designed, three layers below where anyone was looking).

A probe confirmed it directly: on the failing attempt there were **no dispatch events at all**, and
`on_run_end` fired seven times, once per leaked tap, each reporting `reconciled_steps == 0`.

That is the second half. Every install also calls `register_run_end_observer(tap.on_run_end)` and
nothing ever unregisters, so a process accumulates one observer per provider it has ever created,
forever, each firing against an empty ledger.

**Why it looked like a race.** Address reuse depends on allocation history, so adding a `print` to
the failing test changed the allocations and made it pass — 0 failures in 12 runs with the print, 4
in 5 without. Observing it changed it, which is the signature of a race and was in fact a lookup
returning the wrong object.

## Decision

Store a **weak reference to the provider** beside each tap, and compare it on lookup. A recycled
address then fails the identity check, the stale entry is retired, and the new provider gets its own
tap. Weak rather than strong so the table cannot keep providers alive.

Retire dead entries on each install rather than from a `weakref` callback. A callback fires whenever
the collector chooses — including while this module's non-reentrant lock is held by the same thread,
which would deadlock. Sweeping at install time does the same work at a point where the lock is known
to be safe.

`installed_tap` gets the same identity check: reporting another provider's tap as this one's would
make `instrument()` look installed when it is not.

## Consequences

The original repro goes from 4 failures in 5 runs to **0 in 8**, and a 200-attempt probe that
previously failed by attempt 3 now completes clean. 2,182 tests pass.

**6 of 6 mutations caught — after three rounds of fixing the tests, not the code.** That process is
the part worth recording:

- Removing the identity check, *the actual fix*, broke nothing at first: the sweep cleared stale
  entries before the lookup ever saw them. The central line of the fix was untested.
- The test that was supposed to prove an observer got retired asserted it about a tap that had never
  been registered, so it passed vacuously.
- Churning providers in a loop cannot isolate the sweep, because ~95% of those addresses are recycled
  and the identity check quietly handles them. Only holding forty providers alive at once — forcing
  distinct addresses — exercises it.

Each gap was a test that passed for the wrong reason, and none would have been found by reading.

**The general rule:** `id()` is not an identity. It is an address that is unique only among *live*
objects, so it is safe as a dictionary key only alongside something that keeps the object alive or
proves it still exists. This one produced a defect where the product's headline number silently
vanished, in a library whose entire purpose is to report that number.
