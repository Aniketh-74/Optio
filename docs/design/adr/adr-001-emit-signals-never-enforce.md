# ADR-001 — Emit signals, never enforce

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-000, §1.7, §16 rule 13

## Context

Once a run's projected cost is known to exceed its budget, the obvious next step is to stop the run. Every guardrail library in this space took that step — and it is the step that makes them competitors to the policy engines their users already run.

The 2026 governance stack already has an enforcement layer. Microsoft's Agent Governance Toolkit ships the `govern(x, policy=...)` primitive; OPA and Cedar are mature, audited policy engines with existing deployments. What none of them has is a source of *economic* and *quality* evidence to decide on. Enforcement is solved and crowded; the inputs to enforcement are not.

Building enforcement would also put us on a hook we do not want: the moment `optio` can stop a run, every false positive is our outage, and we inherit the entire "was this the right call" surface — approvals, overrides, audit trails, appeals.

## Decision

**`optio` emits typed evidence and stops there.** The downstream engine reads the attributes and decides.

Concretely, out of scope by construction:

- No policy DSL or language — OPA/Cedar/AGT own that.
- No enforcement or action layer — no blocking, rollback, rerouting, or approval flow.
- No stop decision of any kind, including "advisory" ones.

What we ship instead: exact attribute names, documented semantics, and copy-paste policy packs for three engines so the enforcement someone else owns is trivial to wire up.

## Alternatives

**Contest the full runtime.** Rejected in ADR-000 — a direct fight with Microsoft's distribution, and the wrong shape for a solo build.

**Emit signals *and* offer optional enforcement.** Tempting, and rejected. An optional kernel is still a kernel: it needs the approval and override surface, it competes with the engine the user already runs, and it converts every detector false positive into a production incident we caused. It also weakens the integration pitch, since we would be asking incumbents to adopt a partial competitor.

**Emit a recommendation (`should_stop: true`).** Rejected as enforcement wearing a different hat. A boolean recommendation encodes *our* threshold for *their* risk tolerance. Emitting `projected_cost` lets each consumer apply their own limit; emitting `should_stop` picks one for them and quietly relocates the policy decision back into our library.

## Consequences

**Good**

- Non-competitive with every engine in the ecosystem, which makes them integration targets and distribution channels rather than rivals.
- No critical-path decision risk: we cannot wrongly kill a healthy run, because we cannot kill anything.
- Much smaller surface to build and maintain — appropriate for the actual resourcing.

**Costs, accepted deliberately**

- **We are a component, not a product.** The user needs a policy engine to get value from us. This is the R-MKT-2 risk, accepted by design: the goal is adoption and contribution, not revenue, so the north star is integrations rather than signups.
- **Value is one hop away in the demo.** Signals alone are abstract, which is why the standalone demo (ADR-006) ships a real policy acting on them — the evaluator needs to *see* the loop close.
- **The signal names become a hard contract.** Because policies are written against them, renaming one breaks strangers' deployments. Hence the pinning and contract tests (R-TECH-2).
