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
| [015](adr-015-evidence-bar-for-promoting-an-altered-tier-stage.md) | Evidence bar for promoting an `ALTERED`-tier stage out of "experimental" | Accepted — evidence gathering in progress |
| [016](adr-016-the-in-scope-test-for-a-cost-technique.md) | The in-scope test for a cost technique | Accepted |
| [017](adr-017-batch-dispatch-is-a-second-surface.md) | Batch dispatch is a second surface, not a stage | Accepted — implemented |

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

ADR-015 is a different shape from the rest: it records a decision about how a future decision
will be made, not an architecture change on its own. Written before any evidence-gathering code
existed, specifically so that a later result — `compress_prompt` already produced one alarming
number (output tokens +71.4%) before this document existed — cannot quietly move the bar to fit
itself. It is expected to gain a per-stage addendum as each `ALTERED` stage's live evidence
comes in, the same pattern ADR-005 used for its Redis addendum.

ADR-016 draws the boundary ADR-013 left implicit: which of the field's several dozen published
cost techniques this package should implement at all. It exists because the boundary had twice
been drawn on *effort* — "that needs a queue", "that needs a team" — which is a reason for a
caller to decline a technique and never a reason for a library to, since absorbing that work is
what a library is for. It retires effort as a test and replaces it with three: expressible against
the normalized types, needs no infrastructure the caller must operate, and measurable by the bench
harness. The immediate consequence is that tool-schema cost, the single largest evidenced win in
the published literature, stops being out of scope.

ADR-017 is the one item ADR-016 admitted changes the package's *shape* rather than extending it.
Batch dispatch cannot be a stage — a stage's contract is that a response comes back on the same
stack frame, and this one comes back tomorrow — so it is a second public class, and ADR-012's
"the public API is the top level" rule now covers two entry points instead of one. That widening
is the price of the discount and the ADR says so rather than pretending the surface stayed still.
It is also the only place in the project where a saving is reported from a *published* figure
rather than an A/B run, because the harness cannot express an arm that returns hours later; the
ADR requires that difference to be stated wherever the number appears, and `BatchReport` prints
it as its own line.

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
