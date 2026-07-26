# Runbooks

Operational guidance for teams running `agentmeter` in production (§12).

The theme running through all of these: **`agentmeter` failing never means your
agent is failing.** The library is wrapped end to end by a fail-open guard
(ADR-004), so every problem below degrades signals, not agent traffic.

---

## Fail-open activations are spiking

**Symptom.** A WARN like:

```
agentmeter: cost failed with LedgerInvariantError; signal dropped, agent
unaffected. Further failures from this component are counted but not logged.
```

…or a rising `agentmeter.internal.lane_errors` count.

**What it means.** A lane has a bug. That lane's signals are missing for the
affected steps.

**Is my agent at risk?** No. The guard absorbed the failure and returned control
immediately; your agent ran exactly as if `agentmeter` were not installed. This
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

1. **Is an exporter configured?** `agentmeter` writes to your existing OTel
   pipeline; it does not stand up its own (SC-1). No exporter means no visible
   signals. Try the console exporter first.
2. **Semconv version mismatch.** We pin a specific GenAI semconv version
   (`semconv.GENAI_SEMCONV_VERSION`). If your backend expects different
   attribute names, the signals are arriving under names it does not display.
   The contract test in `tests/contract/` shows the exact names emitted.
3. **Is the lane enabled?** The quality lane is **off by default** (ADR-003).
   Cost and behavior are on.
4. **Fail-open activations.** See above — a broken lane emits nothing.

**A missing attribute means "unknown", not "zero".** When a value cannot be
computed the attribute is *omitted* rather than emitted as zero, so a policy can
tell the two apart. Policies must treat absence as unknown; see
[signals.md](signals.md).

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
Depending on configuration, `agentmeter` either drops the signal or falls back
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
