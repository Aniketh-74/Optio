# ADR-014 — optio_optimize integrates by emitting spans, not by calling optio

**Status:** Accepted
**Date:** 2026-07-28
**Related:** ADR-002, ADR-009, ADR-012, ADR-013, §3.1, §7.2

## Context

ADR-013 shipped `optio_optimize` as a separate, opt-in, request-path package and stated the
differentiator plainly:

> the optimizer is driven by the same signals the policy engine sees, so a loop detected by the
> behavior lane can stop a run rather than merely being reported

Nothing built that. `Optimizer.call()` accepts a `run_id` and threads it through
`Pipeline.execute()` into every `StageContext` (`optio_optimize/pipeline.py`,
`stages/base.py`), and nothing ever reads it. A stage's savings are recorded in
`optio_optimize`'s own `SavingsReport` and go nowhere else. The two packages ship together and
know nothing about each other at runtime.

The first design that comes to mind — have `optio_optimize` call into `optio`'s cost ledger
directly — turns out to be the wrong shape once you look at how `optio` actually works, not just
its public surface:

* **The ledger is not public.** `optio.lanes.cost.ledger.CostLedger` lives under `optio.lanes`,
  which ADR-012 declares internal and free to change on any release, including a patch. Reaching
  into it would import a module the project has explicitly told integrators not to depend on.
* **The ledger is keyed by span id, not by an arbitrary caller-supplied amount.**
  `CostLane._step_id` derives `step_id` from `span.get_span_context().span_id`
  (`optio/lanes/cost/lane.py`). `reserve`/`reconcile` exist to be called *from a span*, not from a
  bare `(run_id, amount)` pair — there is no "just add this dollar figure" entry point, public or
  private, and inventing one would duplicate the reserve/reconcile invariant (R-TECH-1) a second
  time for no reason.
* **optio doesn't compute cost from a direct call either.** `OptioSpanTap` (`optio/runtime/
  span_tap.py`) is an OTel `SpanProcessor`: it watches spans a *tracer* produces and dispatches
  finished ones to the enabled lanes, provided a `RunContext` is active
  (`current_run() is not None`) and the span carries `gen_ai.*` attributes it didn't write itself.
  Every existing integration point — the four framework adapters, the raw `RunContext` context
  manager — works by getting spans in front of this processor. None of them call a lane directly.

That last point is the actual answer: `optio` was already built to observe *any* source of GenAI
spans without knowing what produced them. `optio_optimize` doesn't need a side channel into
`optio` at all. It needs to become one more span source.

## Decision

**When enabled, `Optimizer.call()` emits one standard OTel GenAI span per request/response cycle,
using the exact attribute names `optio`'s cost and behavior lanes already read.** Nothing new is
built on the `optio` side. If a `RunContext` is active and `optio`'s span tap is installed on the
ambient `TracerProvider` — which is already true for anyone using both packages together — the
span is priced and classified automatically, through the pipeline that already exists.

Four points make this concrete:

**1. `optio_optimize` never imports `optio`.** The import-linter contract (`optio` may not import
`optio_optimize`) already enforces one direction; this ADR does not touch it, and does not need
to add the reverse either. `opentelemetry-api` is the only dependency this requires, and it is
already guaranteed present — it is a core dependency of the `optio` distribution both packages
ship inside, per `pyproject.toml`. Correlation to a run happens the same way it does for every
other span source: through OTel's ambient context, not through an explicit call into
`optio.current_run()`.

**2. The span carries only attributes `optio` already consumes, by their pinned names.** Model,
usage tokens, and operation name — `gen_ai.request.model`, `gen_ai.response.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.operation.name` — read directly
off the `LLMRequest`/`LLMResponse` this package already has in hand. These are the OTel GenAI
semantic convention's own names, consumed (not emitted) by `optio.semconv`; `optio_optimize`
defines its own copy of the handful it needs rather than importing `optio.semconv`, keeping the
zero-import property from point 1 exact rather than "except for one module."

**3. A short-circuited response is priced honestly, because it's a real span with real numbers.**
An exact-cache hit already zeroes its `LLMResponse.input_tokens`/`output_tokens`
(`stages/caching.py:served_from_cache`). Emitting that response as a normal span means `optio`'s
cost lane reserves and reconciles it at $0 with no special-casing on either side — the existing
reserve/reconcile invariant just produces the right number, because the input is honest. This is
the concrete mechanism behind ADR-013's "same signals the policy engine sees": a policy watching
`gen_ai.run.actual_cost` sees the avoided cost land in the same total it already watches, with no
new signal name to learn.

**4. `optio_optimize`'s own attributes go on the same span, under their own namespace, and never
through `optio`'s writer.** `optio_optimize.stage`, `optio_optimize.saved_input_tokens`,
`optio_optimize.saved_output_tokens` are set directly via `span.set_attribute` on the span this
package created and owns. They never pass through `optio.runtime.signal_writer.write_signal`,
which would reject them outright — `_is_emittable` raises on any name outside
`semconv.EMITTED_SIGNALS` (§7.2's frozen contract), deliberately, so an unreviewed name never
reaches a consumer. `optio_optimize` is not adding to that contract; it is attaching its own,
separately-versioned data to a span it owns, which is a different and much weaker promise —
consistent with the two packages having separate release cadences (ADR-013).

**Off by default**, behind a new `OptimizeConfig.emit_spans: bool = False`. Not because it's
risky — a span with no configured exporter is close to free, and this ADR adds no behavior when
`optio` isn't installed or no `RunContext` is active — but because it is a new, previously-absent
side effect (span creation) for every existing `Optimizer()` caller, and ADR-013's own pattern is
that new capabilities start opt-in and earn a default once measured. Turning it on is one keyword
argument, `Optimizer(emit_spans=True)`.

**Fail-open, per ADR-013 rule 1.** Span creation happens after the response is already
determined — real call or short-circuit — so a failure here can never turn a successful exchange
into a failed one. It is guarded and logged like every other post-response hook in this package
(`stages/base.py`'s `after` hook already establishes this pattern); a broken exporter or a full
disk degrades to "no span," never to a raised exception reaching the caller.

## Alternatives

**Call into `optio`'s ledger directly, given a public API for it.** Rejected: it would require
either promoting `CostLedger` to the public surface ADR-012 deliberately narrowed, or inventing a
new public entry point (`optio.record_cost(run_id, amount)`) that bypasses the span-based model
every other integration in the project uses, and duplicates the reserve/reconcile invariant a
second way for no benefit — a step's cost already has exactly one correct place to enter the
system, and it is a span.

**Have `optio_optimize` read `optio`'s signals back and change its own behavior** (e.g., cache
more aggressively once the behavior lane reports `looping`). This is the fuller version of
ADR-013's "can stop a run" language, and it is a real, larger feature — a feedback loop, not just
visibility. Deferred: emission has to exist and be trustworthy before anything can safely react to
it, and building both at once makes either one hard to review in isolation (§16 rule 9).

**Default `emit_spans` to `True`.** Considered, since the mechanism is cheap and harmless when
`optio` isn't present. Rejected for this first version in favour of the more conservative,
already-established pattern: new `optio_optimize` capabilities start opt-in (ADR-013's own
`trim_history`/`deduplicate`/`prune_retrieval` did not — they defaulted on immediately, but as
context-dropping stages, not as an integration touching a second package for the first time). Can
be revisited once this has been exercised by more than the raw-call path (see Consequences).

## Consequences

**The visibility ADR-013 promised now exists**, and needed nothing new from `optio` to get there
— which is the strongest evidence the span-tap model was the right foundation to begin with.

**Scope is the raw `Optimizer.call()` path only.** A user who wraps their own provider function
directly (the package's own documented usage) gets this cleanly. A user going through a framework
adapter that has *its own* GenAI instrumentation (LangGraph, CrewAI, OpenAI Agents, Claude Agent
SDK) risks a double-counted span — one from the framework's own instrumentation, one from this
package — if both end up wrapping the same underlying call. Building the first `optio_optimize`
framework adapter (next on the punch list) has to resolve this per-framework, most likely by
having the adapter be the only span source when it's in use. Recorded here rather than solved here
so it isn't quietly reintroduced.

**The feedback loop is still unbuilt.** This ADR makes optimizer activity visible to a policy
engine; it does not make the optimizer *react* to what the behavior lane reports. That is a
legitimate follow-up, not a smaller version of this decision.

**One more small, permanent cost**: `optio_optimize` now defines its own tiny copy of a handful of
GenAI attribute name constants, rather than importing `optio.semconv`'s. Duplication, in exchange
for the zero-import property in point 1 staying literally true rather than true with an exception.
Worth it: a "just this one module" exception is exactly the kind of edit §16 rule 2 warns looks
harmless in review and isn't.
