"""Emitting a judge score that arrived after the run span closed.

## Why this module exists

The judge is a model call dispatched at run end. By the time it answers, the run
span has ended, and an ended OpenTelemetry span cannot be modified -- so there
are exactly two ways for a score to reach the user: wait for it, or emit it
somewhere else.

Waiting is not available. ``@meter`` returns the agent's result *after* run end,
so blocking there puts model latency on the agent's return path, which is the
one thing this library promises never to do (ADR-004, SC-5). That leaves
somewhere else, and this is it: a short span named
:data:`~optio.semconv.QUALITY_SPAN_NAME`, carrying the scores and a
:data:`~optio.semconv.RUN_ID` to join on, **linked** to the run span it belongs
to.

A link rather than a parent. The run span has already been exported by the time
this fires, and parenting to a finished span would nest a child under a
completed interval -- backends render that as a gap or drop it. A link says
"about that run" without claiming to be part of its timing.

## Everything here is total

This runs on a judge worker thread, off the agent's call stack, which means the
fail-open guard is no longer above it (and cannot be: ``optio.lanes`` sits below
``optio.runtime`` in the layering, so this module could not import it anyway).
An exception escaping into ``concurrent.futures`` would surface as an unhandled
callback error with no useful attribution. So the emitter absorbs its own
failures and logs the exception *type* only -- a tracer's exception can carry
connection strings, and Section 10 forbids putting payloads in logs.

## Privacy

Only numbers cross this boundary: two scores in ``[0, 1]``, a boolean, a cost,
and the run id. No prompt, no completion, no judge rationale (Section 10). The
judge saw the user's content; optio never receives it back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from opentelemetry import trace
from opentelemetry.trace import Link

from optio import semconv

if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext, Tracer

    from optio.lanes.base import RunLike
    from optio.lanes.quality.judge import JudgeScores

_log: Final = logging.getLogger("optio")


def parent_context() -> SpanContext | None:
    """Capture the run span's identity, for linking to later.

    Called at run end, on the run's own thread, while its span is still the
    current one -- the worker thread that finally emits has no OpenTelemetry
    context of its own, and asking it for one would link the quality span to
    nothing.

    Returns:
        The active span's context, or ``None`` when nothing is recording. A
        user running a bare ``RunContext`` with no enclosing span is the normal
        way to get ``None``, and an unlinked quality span still carries the run
        id, so the score is reachable either way.
    """
    context = trace.get_current_span().get_span_context()
    return context if context.is_valid else None


def emit_scores(
    tracer: Tracer,
    run: RunLike,
    parent: SpanContext | None,
    scores: JudgeScores,
    *,
    success_threshold: float,
) -> None:
    """Emit the quality span for a judgement that arrived late.

    Args:
        tracer: Tracer from the provider the tap was installed on. Passed in
            rather than resolved here, because ``trace.get_tracer`` reaches for
            the *global* provider -- so a user who configures their own would
            have every quality span recorded somewhere they are not listening,
            and see the feature as still broken.
        run: The run that was judged. Read for its id and, if the cost lane
            recorded one, its final cost.
        parent: The run span's context, from :func:`parent_context`.
        scores: Validated judge scores. Already checked for range and type by
            :class:`~optio.lanes.quality.judge.JudgeRunner`.
        success_threshold: ``task_success`` at or above this counts as success.
    """
    try:
        _emit(tracer, run, parent, scores, success_threshold)
    except Exception as error:  # noqa: BLE001 - see module docstring: must be total
        # Type only, never the message (Section 10).
        _log.warning(
            "optio: could not emit the deferred quality span (%s); scores dropped",
            type(error).__name__,
        )


def _emit(
    tracer: Tracer,
    run: RunLike,
    parent: SpanContext | None,
    scores: JudgeScores,
    success_threshold: float,
) -> None:
    """Build and end the span. Separated so the guard above has one job.

    Args:
        tracer: Tracer to emit with.
        run: The judged run.
        parent: The run span's context, or ``None``.
        scores: Validated scores.
        success_threshold: The success cutoff.
    """
    links = [Link(parent)] if parent is not None else []

    # `start_span`, not `start_as_current_span`: this thread is a worker in a
    # pool, and making the quality span current there would adopt any span the
    # user's own judge implementation happens to leave open.
    span = tracer.start_span(semconv.QUALITY_SPAN_NAME, links=links)
    try:
        span.set_attribute(semconv.RUN_ID, run.run_id)

        if scores.groundedness is not None:
            span.set_attribute(semconv.RUN_QUALITY_GROUNDEDNESS, scores.groundedness)
        if scores.task_success is None:
            return

        span.set_attribute(semconv.RUN_QUALITY_TASK_SUCCESS, scores.task_success)
        succeeded = scores.task_success >= success_threshold
        span.set_attribute(semconv.RUN_SUCCESS, succeeded)

        per_task = _cost_per_successful_task(run, succeeded=succeeded)
        if per_task is not None:
            span.set_attribute(semconv.RUN_COST_PER_SUCCESSFUL_TASK, per_task)
    finally:
        # In a `finally` so a set_attribute that rejects a value cannot leak an
        # unended span, which would hold a trace open until the process exits.
        span.end()


def _cost_per_successful_task(run: RunLike, *, succeeded: bool) -> float | None:
    """Return the run's cost per success, or ``None`` when it has none.

    The headline unit-economics number, and the reason it is computed here
    rather than on the run span: the numerator is final at run end and the
    denominator is not, so the only place both are known is after the judge has
    spoken. A run is one task, so a successful run's cost per success is simply
    its cost.

    Args:
        run: The judged run. ``actual_cost`` is written by the cost lane at run
            end through the same run object the quality lane writes
            ``successes`` to -- a value on an object both lanes already hold,
            so neither imports the other (Section 3.1).
        succeeded: Whether the judge scored this run a success.

    Returns:
        Cost per successful task in USD, or ``None`` when the run failed, was
        never priced, or the cost lane is switched off. Absence rather than
        infinity: a run that succeeded at nothing has a cost and a failure,
        both already reported separately.
    """
    if not succeeded:
        return None

    cost = getattr(run, "actual_cost", None)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        # Includes the ordinary case of the cost lane being disabled, and the
        # narrow race where an *instant* in-process judge answers before the
        # cost lane's run end has run. A judge that makes a network call never
        # wins that race; a test double can, and omitting the ratio is the
        # correct degradation either way (ADR-044).
        return None
    return float(cost)
