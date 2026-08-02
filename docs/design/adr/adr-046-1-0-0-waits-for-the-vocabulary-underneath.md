# ADR-046 — 1.0.0 waits for the vocabulary underneath

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-002 (OTel GenAI semconv as the wire format), ADR-005 (in-process state),
ADR-012 (the public API is the top-level package only), ADR-015 (measure, do not assume)

## Context

The first published release is 0.2.0. The obvious objection is that the work does not look like
`0.x`: 46 ADRs, 2,204 tests, five Python versions on three operating systems, 99% coverage on the
core with 100% on the ledger and the fail-open guard, a signed release pipeline with an SBOM and a
human approval gate. That is stronger than most `1.0.0` releases.

The number is not a verdict on quality. Under SemVer it is a promise about **API stability**, and
that promise is the one thing this project is not in a position to make — for a reason it already
wrote down.

ADR-002 chose the OpenTelemetry GenAI semantic conventions as the wire format, knowing:

> The GenAI conventions are marked **Development** stability upstream. Attributes have been renamed
> between releases and **will be again**. Choosing them means adopting someone else's instability.

and accepted the consequence explicitly:

> **We inherit upstream churn.** A semconv rename is a breaking change here, **needing a major
> version bump.**

The product *is* those signal names. So at `1.0.0`, the first upstream rename owes `2.0.0`, and the
next owes `3.0.0`. The major version becomes a log of OpenTelemetry's churn rather than of any
decision made here — which tells a user nothing about this library while looking like it should.

## Decision

Stay on `0.x` until the vocabulary underneath is stable. Ship `1.0.0` when:

1. **OTel GenAI semconv reaches Stable upstream.** The load-bearing one, and not in this project's
   control. Until then a rename can be forced at any time.
2. **State is no longer in-process only.** `store_backend="redis"` is currently rejected at setup
   rather than silently ignored (ADR-005) — honest, and not something to promise stability around.
3. **The optimizer's default set is settled on evidence.** Fourteen of twenty-four stages ship off
   pending live measurement, and several savings figures are Anthropic-only. A default that is
   still moving is an API that is still moving.

The asymmetry decides the close call: `0.2.0 → 1.0.0` is a milestone anyone can read, and
`1.0.0 → 0.3.0` is not available. When the choice is genuinely uncertain, the reversible one is
correct.

## Consequences

Users see `0.x` and read it as "the interface may move", which is true, and true for a reason that
is upstream rather than a gap here. The README banner already says so concretely rather than
hiding behind the word "alpha": in-process state, and signal names pinned to a semconv release that
is itself marked Development-stability.

**What this deliberately does not do** is understate the work. The README carries the coverage
numbers, the ADR index is public, and the release pipeline is auditable. Someone deciding whether
to depend on this should read those, not the major version — and this ADR exists so that "why is it
still 0.x?" has an answer that is not "nobody got round to it".
