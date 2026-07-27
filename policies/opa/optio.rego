# optio signals -> OPA/Rego policy (M4-4, SC-3)
#
# Copy this file, change the thresholds, done. It is written against the exact
# attribute names in docs/signals.md, which are the integration contract.
#
# Input shape: one object per decision, holding the span attributes as exported
# by OTel. Most users feed this from a collector, a gateway, or an exporter
# webhook:
#
#   {"attributes": {"gen_ai.run.projected_cost": 0.42, ...}}
#
# THE ONE RULE THAT MATTERS: a missing attribute means *unknown*, never zero.
# optio omits a signal it cannot compute (unknown model price, lane off,
# lane failed and fail-open caught it) rather than emitting a wrong number. So
# every rule below tests for presence before comparing. Writing
#
#   deny if input.attributes["gen_ai.run.projected_cost"] > 0.50
#
# looks equivalent and is not: in Rego an absent key makes the expression
# undefined, so the rule silently never fires -- and the run you most wanted to
# catch, the one where the cost lane broke, sails through. The helpers make that
# explicit instead of relying on the reader knowing it.

package optio

import rego.v1

# --- thresholds: this is the part you change ---------------------------------

max_projected_cost := 0.50

min_budget_remaining := 0.05

# Loop states that justify stopping a run. `repeating` is deliberately absent:
# healthy agents repeat calls (polling, paging, bounded retries), so gating on
# it produces false positives. Alert on it; do not deny. See docs/behavior.md.
blocking_loop_states := {"looping", "retry_storm"}

min_groundedness := 0.7

# --- helpers -----------------------------------------------------------------

# Read a signal, or nothing at all if it was not emitted.
signal(name) := value if {
	value := input.attributes[name]
}

# True when the signal is present AND breaches. Absence is not a breach.
over(name, limit) if {
	value := signal(name)
	value > limit
}

under(name, limit) if {
	value := signal(name)
	value < limit
}

# --- decisions ---------------------------------------------------------------

default allow := false

allow if count(deny) == 0

deny contains msg if {
	over("gen_ai.run.projected_cost", max_projected_cost)
	msg := sprintf(
		"projected run cost $%.4f exceeds limit $%.2f",
		[signal("gen_ai.run.projected_cost"), max_projected_cost],
	)
}

deny contains msg if {
	under("gen_ai.run.budget_remaining", min_budget_remaining)
	msg := sprintf(
		"budget headroom $%.4f below floor $%.2f",
		[signal("gen_ai.run.budget_remaining"), min_budget_remaining],
	)
}

deny contains msg if {
	state := signal("gen_ai.run.loop_state")
	blocking_loop_states[state]
	msg := sprintf("agent is %s; stopping to avoid burn", [state])
}

# Quality lane is off by default (ADR-003), so this fires only when the user
# enabled it AND the run was sampled. Absent = unknown = not a breach.
deny contains msg if {
	under("gen_ai.run.quality.groundedness", min_groundedness)
	msg := sprintf(
		"groundedness %.2f below %.2f",
		[signal("gen_ai.run.quality.groundedness"), min_groundedness],
	)
}

# --- warnings: worth surfacing, not worth killing a run over -----------------

warn contains msg if {
	state := signal("gen_ai.run.loop_state")
	state == "repeating"
	msg := sprintf("agent is repeating calls (repeat_count %v)", [signal("gen_ai.run.repeat_count")])
}

# A run that emitted cost signals but no loop state means the behavior lane was
# off or failed. Not a denial -- fail-open is absolute (ADR-004) -- but the
# operator should know their coverage has a hole.
warn contains msg if {
	signal("gen_ai.run.actual_cost")
	not signal("gen_ai.run.loop_state")
	msg := "behavior lane emitted no signal; loop detection is not covering this run"
}

# Cost signals absent while the behavior lane reported means optio saw the run
# but could not price a single step of it -- almost always a model newer than
# the installed pricing table.
#
# This is the case that most deserves a warning. The two cost rules above test
# presence before comparing, correctly, so an unpriceable run passes both: the
# budget gate is not merely uninformed, it is *inert*, and nothing else says so.
# Still not a denial, because absence means unknown and denying on unknown would
# make a stale pricing table an outage.
warn contains msg if {
	signal("gen_ai.run.loop_state")
	not signal("gen_ai.run.actual_cost")
	not signal("gen_ai.run.budget_remaining")
	msg := "no step in this run could be priced; cost gating is not in force (check the pricing table)"
}
