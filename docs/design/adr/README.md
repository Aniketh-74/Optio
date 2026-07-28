# Architecture Decision Records

Summaries live in [IMPLEMENTATION.md §15](../../../IMPLEMENTATION.md); full records live here.

**No architectural decision changes without adding or superseding an ADR** (§16 rule 2). Superseding is fine; silently diverging is not.

| ADR | Title | Status |
|---|---|---|
| [000](adr-000-signal-layer-not-runtime.md) | Product is a signal layer, not a runtime | Accepted |
| [001](adr-001-emit-signals-never-enforce.md) | Emit signals, never enforce | Accepted |
| [002](adr-002-otel-genai-semconv-as-the-wire-format.md) | OTel GenAI semconv as the wire format | Accepted |
| [003](adr-003-quality-lane-is-tiered-sampled-and-off-by-default.md) | Quality lane is tiered, sampled, opt-in, off by default | Accepted |
| [004](adr-004-fail-open-is-absolute.md) | Fail-open is absolute | Accepted |
| [005](adr-005-pluggable-state-store-in-memory-default.md) | Pluggable state store, in-memory default | Accepted — Redis backend deferred |
| [006](adr-006-two-delivery-surfaces-library-and-demo.md) | Two delivery surfaces: library + standalone demo | Accepted |
| [007](adr-007-enterprise-control-plane-designed-not-scheduled.md) | Enterprise control plane designed but not scheduled | Accepted |
| [008](adr-008-apache-2-0-license.md) | Apache-2.0 | Accepted |
| [009](adr-009-signals-are-written-to-the-run-span.md) | Signals are written to the run span, not the step span | Accepted |
| [010](adr-010-a-closed-run-is-final.md) | A closed run is final | Accepted |
| [011](adr-011-lane-wiring-lives-outside-the-lane-abc.md) | Lane wiring lives outside the lane ABC | Accepted |
| [012](adr-012-the-public-api-is-the-top-level-package-only.md) | The public API is the top-level package only | Accepted |
| [013](adr-013-optimization-lives-in-a-separate-package.md) | Optimization lives in a separate package (`optio_optimize`) | Accepted |
| [014](adr-014-optimize-emits-spans-optio-already-knows-how-to-read.md) | `optio_optimize` integrates by emitting spans, not by calling `optio` | Accepted |

ADRs 000, 001, and 004 were expanded first because M0 code already depended on them: they
determine what the library is allowed to do, and what it must never do.

002, 003, 005, 006, 007 and 008 were written up before the 0.1.0 release. They had existed only
as §15 summaries while 43 references across the code and docs pointed at files that did not
exist — every "see ADR-003" was a dead link, which is the state in which an architectural rule
quietly stops being one. ADR-005 gained an addendum in the process: writing out the state-store
decision is what surfaced that its Redis half had been half-implemented, accepted in config and
ignored at runtime.

ADR-009 is new in M1: it records an architectural change forced by the OTel SDK
(a span processor cannot annotate the span it observes), rather than a choice
made freely.

ADR-010 is new in M2. The ledger's stated invariant (R-TECH-1) covers ordering
*within* a run and is silent on what happens after one ends; the property tests
found that gap on their first run, so the answer is written down.

ADR-011 is new in M3, and is the smallest of these — it records *not* changing
an architectural rule. Adding a second lane made the lane-independence contract
(§3.1) fail, and the fix was to move the wiring rather than relax the boundary.
Written down because the alternative, a one-line lint exemption, would have been
invisible in review and is how such a contract gets dismantled.

## Format

```markdown
# ADR-NNN — Title

**Status:** Proposed | Accepted | Superseded by ADR-MMM
**Date:** YYYY-MM-DD

## Context      — what forced a decision
## Decision     — what was decided
## Alternatives — what was rejected, and why
## Consequences — what this costs us, including the bad parts
```
