# Tests for the agentmeter OPA pack. Run: opa test policies/opa -v
#
# The absence tests are the important ones. Every other case here would pass
# against a naive policy that compares attributes directly; only the absence
# tests distinguish "unknown" from "zero", which is the failure mode that lets a
# broken cost lane look like a free run.

package agentmeter_test

import data.agentmeter
import rego.v1

# --- cost --------------------------------------------------------------------

test_cheap_run_is_allowed if {
	agentmeter.allow with input as {"attributes": {
		"gen_ai.run.projected_cost": 0.10,
		"gen_ai.run.loop_state": "healthy",
	}}
}

test_expensive_run_is_denied if {
	count(agentmeter.deny) == 1 with input as {"attributes": {
		"gen_ai.run.projected_cost": 0.75,
		"gen_ai.run.loop_state": "healthy",
	}}
}

test_cost_exactly_at_the_limit_is_allowed if {
	# The threshold is a ceiling, not a trigger. Stated as a test because
	# off-by-one at the boundary is where reviewers stop reading.
	agentmeter.allow with input as {"attributes": {
		"gen_ai.run.projected_cost": 0.50,
		"gen_ai.run.loop_state": "healthy",
	}}
}

test_exhausted_budget_is_denied if {
	count(agentmeter.deny) == 1 with input as {"attributes": {
		"gen_ai.run.budget_remaining": 0.01,
		"gen_ai.run.loop_state": "healthy",
	}}
}

# --- behavior ----------------------------------------------------------------

test_looping_is_denied if {
	count(agentmeter.deny) == 1 with input as {"attributes": {"gen_ai.run.loop_state": "looping"}}
}

test_retry_storm_is_denied if {
	count(agentmeter.deny) == 1 with input as {"attributes": {"gen_ai.run.loop_state": "retry_storm"}}
}

test_repeating_warns_but_does_not_deny if {
	# Healthy agents repeat: polling, paging, bounded retries. Denying on this
	# is the false positive that gets a monitoring layer uninstalled.
	input_doc := {"attributes": {
		"gen_ai.run.loop_state": "repeating",
		"gen_ai.run.repeat_count": 5,
	}}
	agentmeter.allow with input as input_doc
	count(agentmeter.warn) == 1 with input as input_doc
}

test_healthy_is_allowed if {
	agentmeter.allow with input as {"attributes": {"gen_ai.run.loop_state": "healthy"}}
}

# --- quality (opt-in; usually absent) ----------------------------------------

test_low_groundedness_is_denied if {
	count(agentmeter.deny) == 1 with input as {"attributes": {
		"gen_ai.run.quality.groundedness": 0.4,
		"gen_ai.run.loop_state": "healthy",
	}}
}

test_quality_lane_off_does_not_deny if {
	# The default configuration. A policy that denied here would block every
	# run for every user who never turned the quality lane on.
	agentmeter.allow with input as {"attributes": {
		"gen_ai.run.actual_cost": 0.02,
		"gen_ai.run.loop_state": "healthy",
	}}
}

# --- absence is unknown, not zero -- the load-bearing tests ------------------

test_missing_cost_does_not_deny if {
	agentmeter.allow with input as {"attributes": {"gen_ai.run.loop_state": "healthy"}}
}

test_missing_budget_does_not_deny if {
	# `budget_remaining` is absent whenever no budget policy was supplied,
	# which is most runs. Treating absent as 0 would deny all of them.
	agentmeter.allow with input as {"attributes": {
		"gen_ai.run.actual_cost": 1.50,
		"gen_ai.run.loop_state": "healthy",
	}}
}

test_no_signals_at_all_does_not_deny if {
	# What a fail-open activation looks like from the policy's side: agentmeter
	# caught an internal error and emitted nothing. The agent must not be
	# punished for our bug (ADR-004).
	agentmeter.allow with input as {"attributes": {}}
}

test_cost_without_behavior_warns_about_the_coverage_gap if {
	input_doc := {"attributes": {"gen_ai.run.actual_cost": 0.02}}
	agentmeter.allow with input as input_doc
	count(agentmeter.warn) == 1 with input as input_doc
}

# --- combinations ------------------------------------------------------------

test_multiple_breaches_are_all_reported if {
	# One denial per breach, so the operator sees everything wrong at once
	# rather than fixing one and rediscovering the next.
	count(agentmeter.deny) == 3 with input as {"attributes": {
		"gen_ai.run.projected_cost": 9.99,
		"gen_ai.run.budget_remaining": 0.0,
		"gen_ai.run.loop_state": "looping",
	}}
}
