# ADR-008 — Apache-2.0

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-000, ADR-007, R-MKT-1, LICENSE

## Context

optio is a library that runs *inside* someone else's agent process, on their critical path. The
licence is therefore not a formality — it decides whether a company's legal review lets the
library into production at all, and legal review is a real gate for exactly the enterprise
users who care most about agent cost.

The comparison set is instructive. Portkey, LoopGain and Preloop ship permissive. Alephant went
GPL. The pattern in this space is that permissive licences get embedded and copyleft ones get
evaluated and declined for critical-path use.

There is also a patent question specific to the domain. Cost projection and loop detection are
the kind of methods that attract patent filings, and a licence with no patent grant leaves an
adopter exposed to a contributor asserting a patent over code they contributed.

## Decision

**Apache-2.0.**

Two properties decided it over the alternatives:

**An explicit patent grant (§3).** Every contributor grants users a licence to any patent claims
their contribution reads on. MIT has no such clause — an MIT project's users rely on the theory
that contributing code implies a patent licence, which is an argument rather than a term. For
an enterprise legal reviewer, that difference is the whole conversation.

**Permissive, so it can be embedded.** No obligation on the user's own code, which is the only
workable position for a library that must sit inside a proprietary agent to function at all.

## Alternatives

**MIT.** The default and rejected on the patent grant alone. Shorter and more familiar, but the
missing clause is precisely the one that matters for a library doing cost computation in a
patent-active field.

**GPL / AGPL.** Rejected. Copyleft on a library that must be imported into the user's process
is effectively a ban on the target use case: no company puts a GPL dependency on their agent's
critical path. Alephant's choice, and a reason it is not embedded.

**BSL or a source-available licence.** Rejected. It would contradict ADR-000 — the strategy is
to be the *reference* signal layer that incumbents adopt, and a licence that restricts
commercial use makes optio a competitor to every potential integrator.

**Open-core split now** (permissive core, commercial extensions). Rejected as premature. ADR-007
defers the commercial surface entirely; splitting the licence before there is anything to put on
the other side would be pure overhead.

## Consequences

**Good**

- Passes enterprise legal review, which is a hard prerequisite for the users who most need cost
  governance.
- The patent grant is a genuine differentiator against MIT-licensed alternatives in a
  patent-active domain.
- Compatible with the ecosystem optio integrates into: OpenTelemetry is Apache-2.0, and
  Microsoft's Agent Governance Toolkit is MIT.

**Costs, accepted deliberately**

- **Anyone may commercialise this**, including a hosted version of the control plane ADR-007
  defers. Accepted: the goal is adoption, and the moat is the signals and ledger rather than the
  licence.
- **Slightly more ceremony than MIT** — a `NOTICE` file convention and a longer header. Trivial
  against the benefit.
- **No copyleft protection**, so improvements need not flow back. Consistent with ADR-000, where
  the aim is contribution rather than extraction.
