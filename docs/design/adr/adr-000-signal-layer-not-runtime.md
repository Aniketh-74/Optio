# ADR-000 — Product is a signal layer, not a runtime

**Status:** Accepted
**Date:** 2026-07-26
**Supersedes:** all earlier "own the policy object" design assumptions
**Related:** ADR-001, R-MKT-1, §17 continuous validation

## Context

Market re-validation in July 2026 found the in-loop guardrail and policy-runtime lane already occupied:

- **Guardrail libraries** — LoopGain, AgentBudget — each solve one signal and make their own stop decision.
- **Microsoft Agent Governance Toolkit** — MIT-licensed, shipping the `govern(x, policy=...)` primitive, carrying Microsoft's distribution.
- **OPA and Cedar** — mature, audited, widely deployed policy engines.

The ecosystem raced to solve *security* (OWASP Agentic Top 10) and *permission* (allow/deny on tool calls), driven by regulation. That race is over and we would be entering it late, from behind, against a vendor whose distribution we cannot match.

But the race left two signals unowned. No engine in the stack holds:

- **An economic signal** — a live per-run cost ledger (reserved vs. actual, projected worst case) that a policy can gate on *before* the tokens burn.
- **A quality signal** — evidence that a run converged confidently on a *wrong* answer. "Well-formed but wrong" is entirely invisible to permission-based governance.

These engines can decide whether an action is **allowed** and **safe**. None can decide whether a run is **affordable** or **good**.

## Decision

**Build the signal layer those engines lack; do not build another engine.**

- No policy runtime, no DSL, no enforcement layer (ADR-001).
- Emit cost and outcome-quality signals in OTel GenAI semconv, so any existing consumer can read them (ADR-002).
- Treat incumbents as integration targets and distribution channels, not competitors — ship copy-paste policy packs for AGT, OPA, and Cedar.

## Alternatives

**Contest the full runtime.** Rejected. A direct fight with Microsoft's distribution, on a problem the market already considers solved, with a solo builder's resources. Losing is the expected outcome and the effort is unrecoverable.

**Build a better guardrail library.** Rejected. Same lane as LoopGain and AgentBudget, same one-signal-plus-a-stop-decision shape, and it competes with the engine the user already runs instead of feeding it.

**Wait for the semconv to standardize cost/quality attributes, then implement.** Rejected as backwards — the way those attributes get standardized is for someone to ship them and demonstrate use. This is the substance of OQ-3: upstreaming our signals as a semconv proposal would substantially de-risk R-TECH-2, and is worth investigating post-v0.1.

## Consequences

**Good**

- Smaller surface, defensible through integration depth rather than feature count.
- The moat is the signals and the ledger, not a schema anyone can copy.
- Every incumbent becomes a distribution channel instead of a threat.

**Costs, accepted deliberately**

- **The wedge can close.** If an incumbent ships native cost and quality signals, the reason to exist goes away (R-MKT-1). Mitigation is speed plus becoming the *reference* implementation — and, structurally, the §17 continuous-validation gate: every milestone re-asks whether this ADR still holds. **Do not keep building on a closed wedge.**
- **We depend on a standard we do not control.** GenAI semconv attributes carry Development-stability badges and can rename without a major bump (R-TECH-2). Pinning plus contract tests convert that from silent breakage into a loud failure.
- **Every "own the policy object" assumption from earlier design is void.** Anything inherited from that framing should be treated as deprecated until re-derived from this ADR.
