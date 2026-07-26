// agentmeter signals -> Cedar policy (M4-4, SC-3)
//
// Written against the attribute names in docs/signals.md. Cedar identifiers
// cannot contain dots, so the `gen_ai.run.*` names are carried as attributes of
// an `AgentMeter::Run` entity with underscored names -- see schema.cedarschema
// for the mapping, and README.md for how to build the entity from a span.
//
// Cedar is deny-by-default and has no "deny unless" -- forbid always wins over
// permit. That fits agentmeter well: one broad permit, then one forbid per
// pathology, each independently readable.
//
// ABSENCE: Cedar's `has` operator is the whole story. A signal agentmeter could
// not compute is *omitted*, and `run has projected_cost` is false for it. Every
// forbid below is guarded by `has`, so an absent signal never triggers a
// denial. Without the guard Cedar raises an evaluation error on the missing
// attribute, which most authorizers surface as a deny -- turning our internal
// failure into the user's outage, exactly what ADR-004 forbids.

// Baseline: runs are permitted. The forbids below carve out the pathologies.
permit (
  principal,
  action == AgentMeter::Action::"execute_step",
  resource
);

// --- cost --------------------------------------------------------------------

// Worst-case projected spend over the ceiling. This is the pre-spend gate: the
// signal is emitted *before* the step's tokens burn, so denying here actually
// saves the money rather than reporting it afterwards.
forbid (
  principal,
  action == AgentMeter::Action::"execute_step",
  resource
) when {
  resource has projected_cost && resource.projected_cost.greaterThan(decimal("0.50"))
};

// Budget headroom exhausted. Absent whenever no budget policy was supplied,
// which is most runs -- hence the `has`.
forbid (
  principal,
  action == AgentMeter::Action::"execute_step",
  resource
) when {
  resource has budget_remaining && resource.budget_remaining.lessThan(decimal("0.05"))
};

// --- behavior ----------------------------------------------------------------

// `looping` and `retry_storm` only. `repeating` is excluded on purpose: healthy
// agents repeat calls (polling, paging, bounded retries), and denying on it is
// the false positive that gets a monitoring layer switched off. Alert on it
// elsewhere; do not gate. See docs/behavior.md for the measured FP rate.
forbid (
  principal,
  action == AgentMeter::Action::"execute_step",
  resource
) when {
  resource has loop_state && ["looping", "retry_storm"].contains(resource.loop_state)
};

// --- quality (opt-in, off by default per ADR-003) -----------------------------

// Fires only when the user enabled the quality lane AND the run was sampled.
// For everyone else the attribute is absent and this is inert.
forbid (
  principal,
  action == AgentMeter::Action::"execute_step",
  resource
) when {
  resource has groundedness && resource.groundedness.lessThan(decimal("0.7"))
};
