"""Emits a span ``optio``'s own span tap already knows how to read (ADR-014).

This module is the entire integration with ``optio``, and it contains no
import of it. ``optio.runtime.span_tap.OptioSpanTap`` is an OTel
``SpanProcessor``: it prices and classifies *any* span carrying the GenAI
attribute names below, from whatever produced it, provided a ``RunContext``
is active in the ambient OTel context. Framework adapters use exactly this
mechanism. So does this module -- the only difference is that the span
describes a request this package already optimized, rather than one a
framework emitted untouched.

A short-circuited response (an exact-cache hit) already carries zeroed
``input_tokens``/``output_tokens`` (:func:`~optio_optimize.stages.caching.
served_from_cache`). Emitting it as an ordinary span means ``optio``'s
reserve/reconcile ledger prices it at $0 with no special-casing on either
side -- the existing invariant just produces the right number, because the
input is honest.

The attribute names below are ``optio_optimize``'s own copy of a handful of
OTel GenAI semantic-convention names, not an import of ``optio.semconv``.
Duplication, deliberately: it is what keeps "this package imports nothing
from optio" true without an asterisk. They must stay in step with
``optio.semconv.GENAI_SEMCONV_VERSION`` (1.37.0 as of this writing) --
verified by ``tests/optimize/test_telemetry.py``, which imports both modules
specifically to compare them, so a drift is a test failure rather than a
silent mismatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import TracerProvider

    from optio_optimize.types import LLMRequest, LLMResponse

_log: Final = logging.getLogger("optio_optimize")

# ---------------------------------------------------------------------------
# Upstream GenAI attribute names optio's cost and behavior lanes read off any
# span. Not optio_optimize's to define -- copied to avoid importing optio.
# ---------------------------------------------------------------------------
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL: Final = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"

# ---------------------------------------------------------------------------
# This package's own attributes. Set directly via span.set_attribute, never
# through optio.runtime.signal_writer.write_signal -- which would reject them:
# it raises on any name outside optio.semconv.EMITTED_SIGNALS by design (§7.2's
# frozen contract). These are a separate, weaker, separately-versioned promise
# attached to a span this package owns, not an addition to that contract.
# ---------------------------------------------------------------------------
OPTIMIZE_STAGE: Final = "optio_optimize.stage"
OPTIMIZE_SAVED_INPUT_TOKENS: Final = "optio_optimize.saved_input_tokens"
OPTIMIZE_SAVED_OUTPUT_TOKENS: Final = "optio_optimize.saved_output_tokens"
OPTIMIZE_SHORT_CIRCUITED: Final = "optio_optimize.short_circuited"

#: "Run identity" in optio's own vocabulary (``optio.semconv.RUN_ID``) --
#: declared there but never written by any optio code today. Setting it here
#: is what finally gives ``Optimizer.call(..., run_id=...)`` a real effect:
#: the parameter has existed since this package's first version, threaded
#: through every ``StageContext``, and read by nothing until now.
GEN_AI_RUN_ID: Final = "gen_ai.run.id"

#: Instrumentation-scope name for the tracer this module creates. A dotted
#: package path, matching the convention every other OTel instrumentation
#: uses for ``get_tracer(name)``.
_TRACER_NAME: Final = "optio_optimize"


def record_span(
    request: LLMRequest,
    response: LLMResponse,
    *,
    stages: Sequence[str],
    saved_input_tokens: int,
    saved_output_tokens: int,
    short_circuited: bool,
    run_id: str | None = None,
    tracer_provider: TracerProvider | None = None,
) -> None:
    """Emit one GenAI span for a completed request/response cycle.

    Called after the response is already determined -- real call or
    short-circuit -- so a failure here can never turn a successful exchange
    into a failed one (ADR-013 rule 1). Every exception is caught and logged;
    none propagate. A missing or misconfigured OTel SDK, no active run, or no
    registered span processor are all silently fine: the span either goes
    nowhere or is never created, and the caller's response is unaffected
    either way.

    Args:
        request: The request as actually sent (after every stage transform).
        response: What was returned to the caller.
        stages: Names of the stages that did something to this request, in
            execution order. A stage that declined contributes nothing here,
            even if it ran.
        saved_input_tokens: Total input tokens avoided across every stage.
        saved_output_tokens: Total output tokens avoided across every stage.
        short_circuited: Whether the provider was never called.
        run_id: The caller's own run identifier, if any, written as
            ``gen_ai.run.id``. Correlation to ``optio``'s pricing does not
            depend on this -- that happens through the ambient OTel context,
            same as every other span source -- so an absent or mismatched
            run_id costs nothing but a debugging convenience.
        tracer_provider: Provider to draw the tracer from. Defaults to OTel's
            global provider -- the standard setup, and what any real
            deployment sharing one process-wide provider with ``optio``
            wants. Overridable for the same reason ``optio.api.meter`` and
            ``optio.runtime.installer.install_tap`` accept the same
            parameter: OTel's global provider is write-once per process, so
            tests (and multi-tenant setups running more than one provider)
            need a way in that doesn't touch it.
    """
    try:
        _record_span(
            request,
            response,
            stages=stages,
            saved_input_tokens=saved_input_tokens,
            saved_output_tokens=saved_output_tokens,
            short_circuited=short_circuited,
            run_id=run_id,
            tracer_provider=tracer_provider,
        )
    except Exception as exc:  # noqa: BLE001 - must never reach the caller
        _log.debug("optio_optimize: span emission failed (%s)", type(exc).__name__)


def _record_span(
    request: LLMRequest,
    response: LLMResponse,
    *,
    stages: Sequence[str],
    saved_input_tokens: int,
    saved_output_tokens: int,
    short_circuited: bool,
    run_id: str | None,
    tracer_provider: TracerProvider | None,
) -> None:
    from opentelemetry import trace

    tracer = trace.get_tracer(_TRACER_NAME, tracer_provider=tracer_provider)
    # A model name is required for the span to be worth anything downstream:
    # optio's cost lane cannot price a span with no model attribute (it reads
    # gen_ai.response.model, falling back to gen_ai.request.model). Using the
    # response's own model -- what actually served the request, not what was
    # asked for -- matches the same reasoning bench/providers.py already
    # applies to live pricing.
    model = response.model or request.model
    span_name = f"chat {model}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(GEN_AI_REQUEST_MODEL, request.model)
        span.set_attribute(GEN_AI_RESPONSE_MODEL, model)
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, response.input_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, response.output_tokens)
        if run_id:
            span.set_attribute(GEN_AI_RUN_ID, run_id)

        span.set_attribute(OPTIMIZE_SHORT_CIRCUITED, short_circuited)
        span.set_attribute(OPTIMIZE_SAVED_INPUT_TOKENS, saved_input_tokens)
        span.set_attribute(OPTIMIZE_SAVED_OUTPUT_TOKENS, saved_output_tokens)
        if stages:
            span.set_attribute(OPTIMIZE_STAGE, ",".join(stages))
