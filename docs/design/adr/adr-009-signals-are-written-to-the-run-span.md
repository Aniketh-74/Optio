# ADR-009 — Signals are written to the run span, not the step span

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-002, M1-3, M1-5, R-TECH-1

## Context

The M1-3 design assumed the span tap would annotate each GenAI step span with
the signals its lanes computed: read the span, compute, write back, done. That
assumption does not survive contact with the OpenTelemetry SDK.

Probing the SDK (`opentelemetry-sdk`, Python) established the actual contract:

| Hook | `is_recording()` | `set_attribute` present | Writes land? |
|---|---|---|---|
| `on_start` | `True` | yes | yes — but no usage/response attributes exist yet |
| `_on_ending` | `False` | yes | **no** — SDK logs "Setting attribute on ended span" |
| `on_end` | n/a | **no** | **no** — receives a true `ReadableSpan` |

By the time any processor hook fires, the span has already ended. `_on_ending`
accepts the call and silently discards it; `on_end` receives a `ReadableSpan`
that has no `set_attribute` at all. A span processor therefore **cannot**
annotate the span it observes.

This matters beyond mechanics: silently-discarded writes are exactly the failure
mode that produces confident, wrong dashboards. The signals would appear correct
in code and be absent in the backend.

## Decision

**Step spans are input only. Signals are written to the run span.**

- The tap reads finished step spans and feeds them to the lanes.
- Lane output is written to the currently-active span, which during a step is
  the run span enclosing it.
- `@meter` opens a run span (`optio.run.<fn>`) so one always exists.

This is also the semantically correct home. Every signal in `docs/signals.md` is
run-scoped — `actual_cost`, `budget_remaining`, `cost_per_successful_task` are
properties of a run, not of an individual LLM call. Writing them onto step spans
would have forced consumers to reassemble a per-run view by aggregating spans.

A consequence that needed handling: writing `gen_ai.*` signals onto the run span
makes that span match the tap's own "is this a GenAI span" test. When the run
span ends it comes back through the tap, and feeding it to the lanes would
re-ingest a run's own cost as if it were a fresh step — double counting,
silently, in the direction that makes the core number wrong (R-TECH-1). So
`is_genai_span()` ignores attributes that appear in `semconv.EMITTED_SIGNALS`:
our own output is not input.

## Alternatives

**Write in `_on_ending`.** Rejected — it does not work. The SDK has already
ended the span and discards the write with a log line rather than an error,
which is the worst available outcome: it looks like it works.

**Wrap every step span ourselves so we control its lifetime.** Rejected. It
would mean intercepting span creation rather than observing it, making
`optio` a participant in the trace rather than a reader of it. That breaks
the portability argument (any framework emitting GenAI spans is observable
without per-framework work) and puts us on the critical path of span creation,
where a bug is far more dangerous than in a processor.

**Emit signals as separate spans or as metrics only.** Deferred rather than
rejected. Metrics are already planned alongside attributes (M2-4), and a
dedicated signal span is a reasonable future option. For M1 the run span is
simpler and keeps signals joined to the trace consumers already look at.

## Consequences

**Good**

- Signals live where their scope says they belong, so a policy reads one span
  rather than aggregating many.
- The tap stays a pure observer, which preserves framework portability.
- The last-write-wins behaviour on the run span gives running totals for free.

**Costs, accepted deliberately**

- **A run span is required.** Without one, signals have nowhere to go and are
  dropped. `@meter` creates one; users driving `RunContext` directly must open a
  span themselves or see nothing. This needs to be prominent in the docs.
- **Per-step values are not retained.** The run span holds the latest value, not
  a per-step history. Consumers wanting per-step granularity need the metrics
  path (M2-4), not the attributes.
- **The self-ingestion guard is subtle.** `is_genai_span()` now encodes
  "our output is not input", which is not obvious from its name. It is covered
  by explicit tests naming the double-counting risk, because a future edit that
  drops the check would reintroduce a silent cost bug rather than a crash.
