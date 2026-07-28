# ADR-013 — Token optimization lives in a separate package

**Status:** Accepted
**Date:** 2026-07-28
**Related:** ADR-001, ADR-004, ADR-012, §10, §11, SC-5

## Context

optio emits cost signals. It does not reduce cost — ADR-001 makes that structural, not incidental:

> No enforcement or action layer — no blocking, rollback, **rerouting**, or approval flow.

That decision is correct for a signals library and it is why the OTel/policy-engine integration
story works. But it leaves a real gap between what users want ("spend less") and what the library
delivers ("know what you spent"). The signal that a run is looping is worth money only if
something acts on it.

Acting on it means intercepting and modifying requests: caching responses, routing to a cheaper
model, trimming history, compressing prompts. Every one of those conflicts with a load-bearing
property of the core:

| Property | Conflict |
|---|---|
| ADR-001, "no rerouting" | Model routing *is* rerouting. |
| §10, "never reads prompt content" | Compression must read *and rewrite* prompt content. |
| SC-5, 5 ms p99 | A semantic-cache lookup embeds the prompt: one network round trip, 10–100 ms. |
| ADR-004, fail-open | A lane that fails drops a signal. A *gateway* that fails breaks the agent. |

The last row is the important one and it is not a naming problem. optio today cannot break the
agent because nothing depends on its output. A component in the request path is on the critical
path by construction, and inherits a different and much harder safety obligation.

## Decision

**Optimization ships as a separate, opt-in package — `optio_optimize` — that depends on `optio`
and is never imported by it.**

```
pip install optio               # observe only: signals, no behaviour change
pip install optio[optimize]     # observe and act
```

The dependency points one way, enforced by an import-linter contract. Someone who wants
observability gets exactly the library ADR-001 describes, with its privacy and latency
properties intact. Someone who wants savings opts in explicitly and knowingly accepts a component
in their request path.

Four rules govern the new package:

**1. Fail-open means "pass the original request through".** The core's fail-open drops a signal;
here it must yield the *unmodified* request or the *real* response. Every stage is wrapped, and a
stage that raises is skipped, not retried and not fatal. A bug in the compressor must degrade to
an uncompressed prompt, never to a failed agent.

**2. Every stage accounts for itself.** A stage reports tokens in, tokens out, and estimated cost
delta. Savings are measured per stage rather than claimed in aggregate, because the whole premise
of this package is a number, and an unmeasured optimization is indistinguishable from a bug.
These are emitted through optio's existing signal path, which is what makes the two packages
worth shipping together.

**3. Lossy stages are gated by evals.** Semantic caching and prompt compression change the output
the model would have produced. They are permitted (that is a deliberate choice — see below), but
they may not ship without a quality suite that runs in CI and fails the build on regression.
"Cheaper" is not a result if nobody measured "correct".

**4. Latency budget is per-stage and configurable, not SC-5.** SC-5's 5 ms exists because optio
is pure overhead. An optimizer that removes 60% of a request's tokens has earned some latency
back. The default ceiling is 100 ms for the whole pipeline, and a stage that exceeds its budget is
skipped for that request rather than allowed to blow the deadline.

## Consequences

**optio's guarantees survive untouched.** No ADR is superseded. §10's content-privacy control
still describes optio, and the new package documents its own, weaker, position honestly: it reads
prompt content because it cannot do its job otherwise.

**Two products, two release cadences.** More packaging work, and a versioning relationship to
maintain. Accepted because the alternative — one package where installing an observability tool
silently puts a request rewriter in your path — is worse.

**We enter a crowded market.** LiteLLM, Portkey, Helicone and GPTCache all do parts of this, all
funded. The differentiation is not the caching; it is that the optimizer is driven by the same
signals the policy engine sees, so a loop detected by the behavior lane can stop a run rather than
merely being reported. That is the one thing the incumbents structurally cannot copy without
building the signals layer first.

**Aggressive mode is the owner's explicit choice.** Semantic caching and lossy compression were
enabled by decision after their risks were stated: both return output the model did not produce.
They are therefore **off by default**, gated behind explicit configuration, and covered by the
eval suite rule above. A user who turns them on is choosing a different point on the
cost/fidelity curve, and the library's job is to make that choice legible rather than to make it
for them.

## Alternatives

**Put it in optio's core.** Rejected: reverses ADR-001, inverts §10, and forces every
observability user to accept a request rewriter.

**optio 2.0, full pivot.** Considered and rejected by the owner. It discards the signals
differentiation to compete head-on with funded incumbents at the exact moment the signals product
has zero users and therefore zero validation.

**Ship only lossless techniques.** Would have removed the eval-suite obligation and most of the
risk. Rejected by the owner in favour of the full set; recorded here so the trade is visible
rather than implied.
