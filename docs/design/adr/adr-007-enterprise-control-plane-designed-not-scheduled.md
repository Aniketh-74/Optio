# ADR-007 — The enterprise control plane is designed but not scheduled

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-000, ADR-008, R-OPS-1, §5 (M6+), §16 rule 13

## Context

The obvious commercial shape for a cost-signal library is a hosted control plane: a dashboard
over the signals, RBAC and SSO, chargeback by team, an audit log, a policy simulator. That is
the open-core pattern, and the design work for it is genuinely done — §5 lists M6+ in detail.

The question is not whether it is a good idea. It is whether it should be built *now*, by a
solo maintainer working with an AI agent, before anyone has adopted the library.

Two facts decide it. A hosted plane is not a feature but an ongoing operational commitment:
uptime, security patching, data retention, support, and a bill that arrives monthly whether
anyone uses it or not. And **it would be built on an unadopted foundation** — every assumption
about what teams want to see, slice by, and alert on would be a guess, poured in concrete
before a single real deployment could correct it.

R-OPS-1 names the failure directly: solo maintainer bandwidth, project stalls at the enterprise
phase. The most likely outcome of starting M6+ early is an abandoned half-built service
attached to a library that never got finished.

## Decision

**M0–M5 is the entire committed scope. M6+ is design-level only and must not be implemented
without an explicit decision lifting this deferral.**

This is binding on contributors and on AI coding agents working in this repository (§16 rule
13). "It would be easy to add a small dashboard" is exactly the reasoning this ADR exists to
stop.

The design stays in §5 rather than being deleted, for two reasons: it shows the architecture
has somewhere to go, and it keeps today's decisions from foreclosing tomorrow's — the signal
layer is deliberately shaped so a control plane could consume it later.

**What lifting the deferral would require:** evidence of adoption (SC-6 counts integrations,
not signups), and either a second maintainer or a sustained commitment that survives losing
interest. Not a feature request.

## Alternatives

**Build a minimal dashboard now.** Rejected. There is no such thing as a minimal hosted
service; the moment it holds someone's data it needs auth, backups, and a security contact.

**Build it locally-hosted only, as a container.** Closer to acceptable, and still rejected for
0.1: it doubles the surface under test and answers a question nobody has asked yet. The
existing demo already provides the visual artifact (ADR-006).

**Say nothing and leave the door open.** Rejected. Silence is how scope creeps. An explicit,
citable deferral is what lets a future contributor's PR be declined without relitigating the
whole strategy.

## Consequences

**Good**

- The committed scope is finishable by the people actually available, which is the difference
  between a released library and an abandoned repository.
- No infrastructure cost, no uptime obligation, no data to protect.
- Whatever gets built later will be shaped by real deployments rather than by speculation.

**Costs, accepted deliberately**

- **No revenue path in this line of releases.** Accepted in ADR-000 and R-MKT-2: the goal is
  adoption and contribution, not income.
- **Someone else may build the control plane first.** Genuine risk. The wager is that the signal
  layer is the defensible part — a dashboard over standard OTel attributes is replaceable, the
  ledger and detectors are not.
- **Users wanting a UI will be disappointed**, and told plainly that their existing
  observability backend already renders `gen_ai.*` attributes, which is most of what a dashboard
  would have provided.
