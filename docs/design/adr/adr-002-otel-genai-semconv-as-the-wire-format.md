# ADR-002 — OTel GenAI semconv as the wire format

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-000, ADR-001, R-TECH-2, §7.2, OQ-3

## Context

ADR-001 makes the emitted attribute names the entire product surface. A consumer's policy reads
them; nothing else about this library is visible from outside. So the naming decision is not
cosmetic — it determines whether the signals are portable or captive.

Two options were real. A bespoke schema (`optio.cost.total`, say) would be ours to define and
free to evolve. The OpenTelemetry GenAI semantic conventions already define a vocabulary for
model, tokens, and operation, and every observability backend in this space already parses it.

The complication: the GenAI conventions are marked **Development** stability upstream.
Attributes have been renamed between releases and will be again. Choosing them means adopting
someone else's instability.

## Decision

**Everything rides OTel GenAI semconv, pinned to an exact version.**

`GENAI_SEMCONV_VERSION = "1.37.0"` lives in `semconv.py`, and every emitted name is a constant
in that module. Three rules follow:

1. **No string literals for signal names anywhere else** — enforced by a contract test that
   greps for them. A literal in a lane is how a rename escapes review.
2. **Upgrading the pinned version is an ADR-worthy change**, requiring a review of every
   constant against the new spec.
3. **`semconv.py` is a leaf module** — stdlib imports only, enforced by import-linter. The
   vocabulary cannot come to depend on the implementation.

Signals optio originates rather than consumes (`gen_ai.run.actual_cost` and friends) extend the
namespace rather than inventing a new one, so a consumer's existing `gen_ai.*` tooling sees
them without configuration.

## Alternatives

**A bespoke schema.** Rejected. It would make every consumer write an optio-specific parser,
which is precisely the integration friction ADR-000 says kills adoption for a component that is
not itself a product. The freedom to evolve names is worth little when the names are a contract
with strangers anyway.

**Track semconv `latest` rather than pinning.** Rejected as the worst of both. An upstream
rename would silently change what optio emits, and a downstream policy would stop matching
without erroring — a policy that silently stops firing is the failure mode this project treats
as most dangerous.

**Wait for the conventions to stabilise.** Rejected. Stability is years away, the wedge is open
now (R-MKT-1), and pinning already contains the risk.

## Consequences

**Good**

- Portable by construction: Langfuse, Honeycomb, Grafana, and any OTLP collector understand the
  output with no adapter.
- No schema to document, version, or defend beyond the run-scoped additions.
- Positions the project to upstream its signals as a semconv proposal (OQ-3), which would make
  the naming risk disappear entirely.

**Costs, accepted deliberately**

- **We inherit upstream churn.** A semconv rename is a breaking change here, needing a major
  version bump. The contract test makes it loud rather than silent, which is the most that can
  be done about it (R-TECH-2).
- **The pin will go stale.** Users on a newer semconv get our older names. Deliberate: a
  predictable lag beats attributes that change under a running policy.
- **Some run-scoped names are ours.** `gen_ai.run.*` is not upstream vocabulary, so those
  specific strings carry the schema risk a bespoke design would have carried everywhere.
