# Runbooks

Operational guidance for teams running `optio` in production (§12).

The theme running through all of these: **`optio` failing never means your
agent is failing.** The library is wrapped end to end by a fail-open guard
(ADR-004), so every problem below degrades signals, not agent traffic.

---

## Fail-open activations are spiking

**Symptom.** A WARN like:

```
optio: cost failed with LedgerInvariantError; signal dropped, agent
unaffected. Further failures from this component are counted but not logged.
```

…or a rising `optio.internal.lane_errors` count.

**What it means.** A lane has a bug. That lane's signals are missing for the
affected steps.

**Is my agent at risk?** No. The guard absorbed the failure and returned control
immediately; your agent ran exactly as if `optio` were not installed. This
is the designed behavior, not a degraded mode.

**What to do.**

1. Note the component name in the WARN line — it identifies the broken lane.
2. Please file an issue with the exception type and the lane name. The
   traceback is attached to the log record at WARN level.
3. As a workaround, disable that lane in config (`cost_lane=False`,
   `behavior_lane=False`, `quality_lane=False`). Your other signals keep
   flowing.

**Why only one log line?** Each component logs its *first* failure, then falls
silent. A lane failing once per span would otherwise flood your logs at exactly
the moment your agent is under stress — turning our bug into your second
incident. The activation counter still records every occurrence.

---

## Signals are not appearing

**Check, in order:**

1. **Is an exporter configured?** `optio` writes to your existing OTel
   pipeline; it does not stand up its own (SC-1). No exporter means no visible
   signals. Try the console exporter first.
2. **Semconv version mismatch.** We pin a specific GenAI semconv version
   (`semconv.GENAI_SEMCONV_VERSION`). If your backend expects different
   attribute names, the signals are arriving under names it does not display.
   The contract test in `tests/contract/` shows the exact names emitted.
3. **Is there a run span?** Signals are written to the enclosing *run* span, not
   to individual step spans (ADR-009). `@meter` opens one for you. If you drive
   `RunContext` directly, you must open a span yourself:

   ```python
   with tracer.start_as_current_span("agent-run"), RunContext():
       agent.invoke(...)
   ```

   Without an enclosing span there is nowhere to put the signals and they are
   dropped.
4. **Is the lane enabled?** The quality lane is **off by default** (ADR-003).
   Cost and behavior are on.
5. **Fail-open activations.** See above — a broken lane emits nothing.

**A missing attribute means "unknown", not "zero".** When a value cannot be
computed the attribute is *omitted* rather than emitted as zero, so a policy can
tell the two apart. Policies must treat absence as unknown; see
[signals.md](signals.md).

---

## A run ended with unreconciled reservations

**Symptom.** A WARN like:

```
optio: run 4f2a... ended with 3 unreconciled reservation(s); cost is
reported as the reserved worst case for those steps.
```

**What it means.** Three steps in that run could not be priced. Almost always
this is a model the pricing table does not carry — see
[pricing.md](pricing.md).

**How the cost is reported.** The reservation is *kept*, not discarded, so the
run's cost includes the reserved worst case for those steps. Discarding them
would make the run look cheaper than the evidence supports, and under-reporting
is the direction that lets an over-budget run through.

If **nothing** in the run could be priced, `actual_cost` is omitted entirely
rather than reported as zero. A policy reading zero would conclude the run was
free; the truth is that its cost is unknown.

**What to do.** Check which model the run used. If it is one we should carry,
please file an issue. If it is a negotiated rate or a self-hosted model, supply
a `PricingProvider` — see [pricing.md](pricing.md).

---

## Cost signals look wrong

**`actual_cost` is missing.** Nothing in the run could be priced. See above.

**`projected_cost` is missing.** It needs a step ceiling. Pass one:

```python
@meter(budget=BudgetPolicy(limit_usd=5.00, max_steps=20))
```

Without `max_steps` there is no finite worst case, so no projection is emitted
rather than an arbitrary one.

**`budget_remaining` is lower than expected.** It subtracts *committed* cost —
reconciled spend plus open reservations — not just what has completed. A step in
flight has already claimed its budget; reporting that money as available would
let a policy authorise spending it twice.

**`cost_per_successful_task` is missing.** It needs a success count, which the
quality lane supplies (M5, off by default per ADR-003). Without it the
denominator is unknown, and assuming one success per run would publish a
headline number derived from a guess.

---

## Overhead is above budget

The design budget is **< 5 ms p99** added per governed step for the cost and
behavior lanes (SC-5).

**What to try:**

1. **Disable the quality lane** if you enabled it. It is opt-in precisely
   because LLM-judge scoring is the expensive path (ADR-003).
2. **Check the store backend.** The in-memory default is O(1) and local. A Redis
   backend adds a network hop per operation; if Redis is slow or distant, that
   cost lands on your step latency.
3. **Report it.** Overhead above budget is a bug on our side.

---

## Redis (or another store) is unreachable

The store fails open like everything else: signals degrade, the agent proceeds.
Depending on configuration, `optio` either drops the signal or falls back
to in-memory state. In-memory fallback means per-process state, so cost totals
for a run spanning multiple processes will be partial rather than wrong-but-
plausible.

---

## Interpreting `loop_state`

`loop_state` defaults to `healthy` when a run is ambiguous. Detectors bias
toward the benign classification deliberately: a fabricated pathology could
cause a downstream policy to kill a healthy run, converting our false positive
into your outage (ADR-004).

Practically: treat a non-`healthy` value as meaningful evidence, and do not
treat `healthy` as a guarantee of health.
