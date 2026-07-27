# ADR-011 — Lane wiring lives outside the lane ABC

**Status:** Accepted
**Date:** 2026-07-27
**Milestone:** M3
**Relates to:** §3.1 (import boundaries), §16 rule 11, ADR-001

## Context

`enabled_lanes(config)` — the function that decides which concrete lanes to
instantiate — originally lived in `lanes/base.py`, next to the `Lane` ABC. With
only the cost lane implemented (M2), that looked harmless.

Adding the behavior lane in M3 broke the import-independence contract:

```
optio.lanes.cost is not allowed to import optio.lanes.behavior:

- optio.lanes.cost.lane -> optio.lanes.base (l.29, l.42)
  optio.lanes.base -> optio.lanes.behavior.lane (l.136)
```

Every lane imports the ABC. Once the ABC's module also imports every concrete
lane, each lane becomes a transitive importer of every other one — even though
no lane contains a single reference to another.

Two responses were available: relax the contract, or move the wiring.

## Decision

**Lane wiring moves to `lanes/registry.py`. `lanes/base.py` holds only the
abstraction and imports no concrete lane.**

The contract stays exactly as written.

## Rationale

The violation was a genuine coupling change, not a linter artifact, and the
distinction matters because the fix looked optional.

§3.1 makes lanes mutually independent so they can **ship, fail, and be tested
separately** — cost (M2), behavior (M3), and quality (M5) land in different
milestones and the quality lane carries optional dependencies that must not be
imported when it is off. A cycle through the shared base erodes all of that:

* Importing the cost lane would execute the behavior lane's module-level code.
* A syntax error or a missing optional dependency in *any* lane would break
  *every* lane at import time.
* The quality lane's judge dependencies (M5) would become reachable from a
  config with `quality_lane=False`.

None of these are visible in a test suite that imports everything anyway. They
appear in a user's process, at import time, as a failure in a lane they
explicitly disabled.

Relaxing the contract to permit the transitive edge was rejected. The contract's
value is precisely that it fails when coupling grows; a contract amended each
time it fires is a comment. Suppressing this one would also have been a §16
rule 11 violation, which requires an ADR rather than a lint suppression — and
the honest ADR here concludes the code was wrong, not the rule.

Deferring the concrete imports into the function body was also rejected: the
linter correctly reports a function-level import as a dependency, and it is one.
The same failure modes apply the moment the function is called.

## Consequences

* `lanes/base.py` is importable by anything without pulling in a concrete lane.
* `lanes/registry.py` is the single edge that knows the concrete lane set —
  the one place to touch when M5 adds quality.
* `enabled_lanes` moved; `runtime/span_tap.py` and three tests were updated. No
  public API changed: `enabled_lanes` was never exported from `optio`.
* Concrete lanes are still imported *inside* the function, so a lane that fails
  to import cannot take down the library at import time and a disabled lane's
  dependencies are never touched.

## Alternatives considered

**Add an `ignore_imports` exemption for `base -> behavior`.** Rejected: it would
have to be re-granted for every future lane, which is the shape of a contract
being dismantled one exemption at a time.

**Register lanes via an entry-point or decorator.** Rejected as over-engineering
for three known lanes in one repository (§16 rule 10). A plugin mechanism is
worth revisiting only if third-party lanes become a goal.
