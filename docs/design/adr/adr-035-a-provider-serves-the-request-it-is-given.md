# ADR-035 — A provider serves the request it is given

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-015 (isolated live evidence), ADR-023 (cascade), ADR-028 (attributable deltas),
ADR-034 (cascade's economics)

## Context

`AnthropicProvider.__call__` and `OpenAIProvider.__call__` both send `model=self.model`. Neither
reads `request.model`.

For an ordinary benchmark run that is invisible: since `Workload.requests(model)` was fixed, every
request is built with the provider's own model, so the two agree. It stops being invisible the moment
a stage changes the model — which is precisely what the two model-routing techniques do:

- **`route_models`** rewrites `request.model` to `config.cheap_model`.
- **`cascade_routing`** sends `replace(request, model=cheap_model)` first and re-sends the original on
  rejection.

Routed through these providers, **both calls would go to the same model** and the "saving" would be
arithmetic performed on two identical calls. The benchmark would report a cheap-vs-expensive
comparison having never called the cheap model.

Nothing is currently wrong in a published number: `bench/__main__.py` does not wire cascade at all,
and it works around the limitation for `route_models` by constructing a *second provider* at the
cheap model (`_same_provider_at`). So this is a landmine rather than a live defect — and it is the
same landmine this measurement loop has now found four times, where the harness measures something
other than what it claims and says nothing.

It also already cost something. Cascade's first live run (ADR-034) had to be written as a standalone
script with its own provider closure, because the benchmark could not express "call the model this
request names".

## Decision

### 1. A provider sends `request.model`

The request is the instruction. A provider that ignores the model on it is not serving that request.

### 2. `provider.model` is the model the run was configured for, not an override

It keeps its job: naming the default that workloads are built for and that the report prices the
baseline against. What it stops doing is silently overriding what a stage decided.

The docstring's existing warning — *"the model actually served. Costs must be priced against this.
Not the model the caller asked for"* — is about pricing against reality rather than against
intention, and honouring `request.model` satisfies it more exactly than ignoring it did: what is
served becomes what was asked for.

### 3. Per-call cost is priced against what the provider says it served

`_actual_cost` used `self.model`. It now uses `response.model` — the id the API reports back —
falling back to the request's model when a provider omits it. On a routed call those differ, and the
spend guard tracking the wrong one is how a run silently exceeds its cap.

### 4. `ABResult` keeps one model, and that is a stated limit

The report prices an arm against a single model. On a run where a stage retargets *some* requests,
that figure is no longer exact. Cascade already solves this for itself — `CascadeStats.cost_summary`
prices each side against its own model from measured tokens — and the honest position is that
`ABResult` is a single-model instrument. Making it multi-model is a larger change than this one and
is not needed while cascade reports its own economics.

## Consequences

- **`route_models` and `cascade_routing` become measurable through the benchmark**, which is what
  ADR-015's evidence bar asks for and what neither has had.
- **No existing number moves.** Every current workload builds requests at the provider's own model,
  so `request.model == provider.model` and the change is a no-op for them. That is asserted rather
  than assumed.
- The spend guard becomes correct on routed calls instead of accidentally correct on unrouted ones.
- A provider can now be asked for a model it was not constructed with, including an unpriced or
  nonexistent one. That surfaces as the provider's own error, which is the right failure: the
  alternative is silently serving something else, which is the defect being removed.

## Alternatives considered

**Leave it and keep using `_same_provider_at`.** It works for `route_models` because that stage
picks one alternative model up front. It does not generalise to cascade, which chooses per request,
and it means the bench holds two provider objects whose spend guards must be shared by hand.

**Have the provider raise when `request.model` differs from `self.model`.** Rejected: it makes the
mismatch loud but leaves both routing techniques unmeasurable, trading a silent wrong answer for a
guaranteed failure.

**Give `ABResult` a per-request model.** Deferred under decision 4. It is the right eventual shape
for a routed A/B and it is a larger change than the defect warrants today.
