# ADR-004 — Fail-open is absolute

**Status:** Accepted
**Date:** 2026-07-26
**Related:** SC-4, R-TECH-4, §6.2, §16 rule 4

## Context

`optio` runs **in-process**, inside the user's agent, on the critical path of every step. That placement is what buys the latency budget and framework portability — and it is also what makes the library dangerous. An uncaught exception in a cost calculation does not produce a missing chart; it produces a broken agent in the user's production system.

The asymmetry is severe and worth stating plainly:

- A dropped signal costs the user one gap in a graph.
- A raised exception costs the user a failed agent run.

There is no observability value that justifies the second outcome. A monitoring layer that can take down the thing it monitors is worse than no monitoring layer, because the user would have been better off never installing it.

This is also an adoption argument, not only a correctness one. The install pitch is "one line, zero risk." That claim is false the first time a lane bug surfaces in someone's agent, and no amount of subsequent reliability recovers the trust.

## Decision

**No internal failure may ever reach user code.**

- Every lane and runtime entry point is wrapped by the guard in `runtime/failopen.py`. On any exception the guard logs once at WARN, drops the signal for that step, and returns control so the agent proceeds unaffected.
- The guard catches `Exception` broadly, not just our own `OptioInternalError` types. A lane raising a plain `KeyError` must not break the agent either — the guarantee is about the agent's safety, not about our exception hygiene.
- Detectors bias toward the benign classification on ambiguity. `loop_state` defaults to `healthy`: a fabricated pathology could cause a downstream policy to kill a healthy run, which converts our false positive into the user's outage.
- Absence is a valid signal state. When a value cannot be computed, the attribute is **omitted** rather than emitted as zero — so consumers can distinguish "unknown" from "zero" (see `docs/signals.md`).

The one deliberate exception is **setup**: configuration errors raise loudly at `instrument()` time. Failing open at runtime is safety; failing open at setup would mean silently shipping a meter that measures nothing.

## Alternatives

**Fail-closed (propagate errors).** Rejected. It optimizes for signal completeness over the user's uptime — exactly the wrong trade for a layer that is optional to their business logic.

**Configurable (`strict=True` opt-in).** Rejected for now. It splits the guarantee into two behaviors, and the strict path would be the less-tested one while carrying the higher blast radius. "The library cannot break your agent" is a stronger and simpler promise than "the library cannot break your agent unless you set a flag."

**Fail-open with error re-raising at run end.** Rejected. It defers the breakage rather than preventing it, and run-end is still the agent's call stack.

## Consequences

**Good**

- The install pitch is honest: adding `optio` cannot break your agent.
- Lane authors write straightforward code instead of scattering defensive `try/except`, because the boundary handles the pathological case.

**Costs, accepted deliberately**

- **Bugs hide.** A broken lane silently emits nothing. This is mitigated, not eliminated, by `optio.internal.lane_errors` — a rising fail-open activation count is the signal that something is wrong. Users are told in the runbooks: activations spiking means a lane bug, and their agent is still safe.
- **Silent wrongness is possible.** A lane that fails *partially* — computing a wrong number rather than raising — is not caught by the guard at all. This is precisely why the ledger gets property tests rather than review (R-TECH-1): fail-open protects against crashes, not against arithmetic that is confidently incorrect.
- **The guard itself is a single point of failure.** It gets 100% coverage, human review, and the fault-injection suite as a blocking CI gate (R-TECH-4). If exactly one component in this repo must be correct, it is this one.
