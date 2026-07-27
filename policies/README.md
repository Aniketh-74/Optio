# Policy packs

Copy-paste rules that gate agent runs on **cost** and **behavioral health**, for the three
engines teams already run. This is SC-3: optio emits signals and never enforces (ADR-001),
so these packs are where a decision actually gets made — by your engine, not by us.

| Engine | Pack | Tests |
|---|---|---|
| OPA / Rego | [`opa/optio.rego`](opa/optio.rego) | `opa test policies/opa -v` |
| Cedar | [`cedar/optio.cedar`](cedar/optio.cedar) + [schema](cedar/schema.cedarschema) | [`cedar/tests.json`](cedar/tests.json) |
| Microsoft AGT | [`agt/optio-policy.yaml`](agt/optio-policy.yaml) | structural, in `tests/policy/` |

All three are exercised by `pytest -m policy`, which runs in CI.

## The one thing to get right

**A missing signal means _unknown_, never _zero_.**

optio omits any signal it cannot compute — unknown model price, lane disabled, or a lane
that failed and was caught by the fail-open guard — rather than emitting a wrong number
([docs/signals.md](../docs/signals.md#absence-is-meaningful)). So this rule:

```rego
deny if input.attributes["gen_ai.run.projected_cost"] > 0.50   # WRONG
```

is not equivalent to the one in the pack. In Rego an absent key makes the expression undefined,
so the rule silently never fires — and the run you most wanted to catch, the one where the cost
lane broke, sails straight through. Cedar has the mirror-image problem: reading an absent
attribute *raises*, and most authorizers surface that as a deny, turning our internal bug into
your outage.

Each pack handles this explicitly — Rego via `over()`/`under()` helpers, Cedar via `has` guards,
AGT via `on_missing: skip` — and `tests/policy/` fails the build if a guard is ever dropped. If
you write your own rules, carry the pattern across.

## Which states to gate on

Gate on `looping` and `retry_storm`. **Alert** on `repeating`.

Normal agents repeat calls constantly — polling, paging, bounded retries — so denying on
`repeating` produces false positives, and a monitoring layer that kills healthy runs gets
uninstalled. The asymmetry is deliberate: a false positive is our error becoming your outage,
while a false negative just means a stuck run costs money the cost lane is already reporting.
See [docs/behavior.md](../docs/behavior.md) for the thresholds and the measured false-positive
rate.

## Wiring the signals in

The signals arrive as OTel span attributes on the run span (ADR-009). How they reach the policy
engine depends on your stack — a collector export, a gateway hook, an exporter webhook. Both
non-AGT packs expect the attributes under an `attributes` object:

```json
{ "attributes": { "gen_ai.run.projected_cost": 0.42, "gen_ai.run.loop_state": "healthy" } }
```

Cedar needs one extra step, because Cedar identifiers cannot contain dots. Build an
`Optio::Run` entity, dropping the `gen_ai.run.` prefix and replacing dots with underscores:

| Span attribute | Cedar attribute |
|---|---|
| `gen_ai.run.projected_cost` | `projected_cost` |
| `gen_ai.run.loop_state` | `loop_state` |
| `gen_ai.run.quality.groundedness` | `groundedness` |

Omit attributes that were not emitted. Do not fill them with zero — the schema marks every
attribute optional precisely so you don't have to.

## Thresholds

The numbers in these packs (`$0.50` projected cost, `0.7` groundedness) are **placeholders**.
There is no defensible default for what a run should cost; that depends on what the run is worth
to you. Change them before you rely on them.

## Quality signals

`groundedness`, `task_success`, and `cost_per_successful_task` are referenced by all three packs
but are **not emitted yet** — the quality lane is M5, and it is opt-in and off by default even
once it lands (ADR-003). Until then those rules are inert rather than broken, which is the
correct behavior for an unknown value and is covered by a test in each pack.
