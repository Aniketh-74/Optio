"""Frozen OpenTelemetry GenAI semantic-convention constants.

This module is the **single source of truth for signal names** (IMPLEMENTATION.md
Section 16 rule 5). No other module may write a signal name as a string literal.

Why this module is frozen and pinned (R-TECH-2):
    GenAI semconv attributes carry a *Development* stability badge upstream, which
    means they can be renamed without a major version bump. Downstream policies
    (OPA/Cedar/AGT) are written against the exact names below. Pinning the spec
    version here, and asserting it in a contract test, turns an upstream rename
    from a silent breakage of every consumer's policy into a loud test failure.

Changing any constant in this module requires: an update to ``docs/signals.md``,
a contract test update, and -- if the change is breaking -- an ADR plus a version
bump (Section 16 rules 5 and 12).

This module imports from the standard library only. It is the innermost layer of
the dependency graph (Section 3.1); nothing in ``optio`` may be imported here.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Pinned specification version (R-TECH-2)
# ---------------------------------------------------------------------------

#: The OpenTelemetry semantic conventions release these constants were validated
#: against. Upgrading this value is an ADR-worthy change: it must be accompanied
#: by a review of every constant below against the new spec.
GENAI_SEMCONV_VERSION: Final = "1.37.0"

#: Namespace root for all standard GenAI signals we emit.
GENAI_NAMESPACE: Final = "gen_ai"

#: Namespace root for optio's *self*-observability (Section 12). Deliberately
#: separate from ``gen_ai.*`` so our own health metrics can never be mistaken for
#: -- or gated on by -- a consumer's policy.
INTERNAL_NAMESPACE: Final = "optio.internal"


# ---------------------------------------------------------------------------
# Upstream GenAI attributes we consume (read off framework-emitted spans)
# ---------------------------------------------------------------------------

GEN_AI_SYSTEM: Final = "gen_ai.system"
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL: Final = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME: Final = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID: Final = "gen_ai.tool.call.id"

#: Why the model stopped generating. Plural and array-valued upstream (one entry
#: per choice), so readers must handle both a bare string and a sequence -- some
#: instrumentations flatten it. Consumed by the quality lane's inline heuristic
#: (M5-2) to catch truncated generations, which read as complete text but are
#: failed tasks.
GEN_AI_RESPONSE_FINISH_REASONS: Final = "gen_ai.response.finish_reasons"


# ---------------------------------------------------------------------------
# Signals optio emits -- the integration contract (Section 7.2)
#
# These exact strings are what downstream OPA/Cedar/AGT policies match on.
# Treat them with the care of a public API.
# ---------------------------------------------------------------------------

# Run identity
RUN_ID: Final = "gen_ai.run.id"

# Cost lane (Section 7.2) -- all monetary values are USD doubles.
RUN_ACTUAL_COST: Final = "gen_ai.run.actual_cost"
RUN_PROJECTED_COST: Final = "gen_ai.run.projected_cost"
RUN_BUDGET_REMAINING: Final = "gen_ai.run.budget_remaining"
RUN_COST_PER_SUCCESSFUL_TASK: Final = "gen_ai.run.cost_per_successful_task"

# Behavior lane
RUN_LOOP_STATE: Final = "gen_ai.run.loop_state"
RUN_REPEAT_COUNT: Final = "gen_ai.run.repeat_count"

# Quality lane (opt-in, ADR-003)
RUN_QUALITY_GROUNDEDNESS: Final = "gen_ai.run.quality.groundedness"
RUN_QUALITY_TASK_SUCCESS: Final = "gen_ai.run.quality.task_success"
RUN_SUCCESS: Final = "gen_ai.run.success"

#: Name of the span carrying judge scores that arrived after the run ended.
#:
#: Part of the integration contract, not an implementation detail: a consumer
#: querying quality has to know where to look. The judge is a model call
#: dispatched at run end, so its answer arrives hundreds of milliseconds after
#: the run span has closed -- and an ended span cannot be modified. Waiting for
#: it would put model latency on the agent's return path, so the score is
#: emitted on its own span instead, linked to the run's, carrying
#: :data:`RUN_ID` to join on.
QUALITY_SPAN_NAME: Final = "optio.quality"


# ---------------------------------------------------------------------------
# Enumerated values
# ---------------------------------------------------------------------------

#: Permitted values of :data:`RUN_LOOP_STATE`. ``healthy`` is the fail-open
#: default: on ambiguity the behavior lane must never fabricate a pathology
#: (ADR-004).
LOOP_STATE_HEALTHY: Final = "healthy"
LOOP_STATE_REPEATING: Final = "repeating"
LOOP_STATE_LOOPING: Final = "looping"
LOOP_STATE_RETRY_STORM: Final = "retry_storm"

LOOP_STATES: Final[frozenset[str]] = frozenset(
    {
        LOOP_STATE_HEALTHY,
        LOOP_STATE_REPEATING,
        LOOP_STATE_LOOPING,
        LOOP_STATE_RETRY_STORM,
    }
)


# ---------------------------------------------------------------------------
# Self-observability instruments (Section 12)
# ---------------------------------------------------------------------------

INTERNAL_SIGNALS_EMITTED: Final = "optio.internal.signals_emitted"
INTERNAL_LANE_ERRORS: Final = "optio.internal.lane_errors"
INTERNAL_OVERHEAD_SECONDS: Final = "optio.internal.overhead"
INTERNAL_SAMPLING_RATE: Final = "optio.internal.sampling_rate"


#: Every signal optio emits, as a set. The contract test asserts this set
#: equals the table in ``docs/signals.md`` -- so a signal cannot be added in code
#: without also being documented for integrators (Section 16 rules 5 and 8).
EMITTED_SIGNALS: Final[frozenset[str]] = frozenset(
    {
        RUN_ACTUAL_COST,
        RUN_PROJECTED_COST,
        RUN_BUDGET_REMAINING,
        RUN_COST_PER_SUCCESSFUL_TASK,
        RUN_LOOP_STATE,
        RUN_REPEAT_COUNT,
        RUN_QUALITY_GROUNDEDNESS,
        RUN_QUALITY_TASK_SUCCESS,
        RUN_SUCCESS,
    }
)
