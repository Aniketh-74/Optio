"""Behavior classifiers (M3-2) -- window in, ``LoopState`` out.

Section 6.4 states the rule this module is built around: **ambiguous defaults to
``healthy``. Never fabricate a pathology.** The asymmetry is not caution for its
own sake. A downstream policy may kill a run on ``looping``, so a false positive
converts our detection error into the user's outage -- we break a working agent.
A false negative merely means a pathological run costs money the cost lane is
already reporting. The two errors are not comparable, and every threshold below
is set on that basis.

## The states

``healthy``
    Nothing detected, or not enough evidence to say. The default.

``repeating``
    The same call recurs more than :data:`REPEAT_THRESHOLD` times in the
    window. Real agents legitimately repeat calls -- polling a job, paging
    through results, retrying a flaky endpoint once or twice -- so this is a
    weak signal, and it is deliberately the mildest state.

``looping``
    A repeating call *dominates* the window and the run shows no progress:
    almost every recent step is the same small set of calls. This is the state
    that says "this agent is stuck", and the one most likely to be enforced on,
    so it carries the strictest evidence bar.

``retry_storm``
    Errors dominate the recent window. Distinguished from ``looping`` because
    the remedy differs -- a retry storm is usually a broken dependency, not a
    confused agent -- and because an agent correctly retrying a failing service
    is not stuck, it is blocked.

## Precedence

A window can satisfy several conditions at once. ``retry_storm`` wins over
``looping`` wins over ``repeating``: report the most specific *and* most
actionable diagnosis, and let the error-driven case dominate because it names a
cause rather than a symptom.

## On the thresholds

They are heuristics, and this module says so plainly rather than implying
calibration nobody performed. They were chosen to be conservative against the
labeled fixtures in ``tests/unit/test_detectors.py``, which include the healthy
patterns most likely to be misread -- polling loops, paged retrieval, and
bounded retries. Section 6.4 requires the false-positive rate to be a published
number; ``docs/behavior.md`` carries it, measured by the fixture suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from optio import semconv

if TYPE_CHECKING:
    from optio.lanes.behavior.store import WindowState
    from optio.lanes.behavior.window import BehaviorWindow

#: Minimum steps before any pathology may be reported. Below this there is not
#: enough evidence to distinguish a loop from an agent that has simply started
#: with two similar calls -- and an early false positive is the most damaging
#: kind, since it can kill a run before it has done any work.
MIN_STEPS_FOR_VERDICT: Final = 5

#: Repeats of one call needed for ``repeating``. Three identical calls is within
#: normal retry behaviour; four begins to look intentional.
REPEAT_THRESHOLD: Final = 4

#: Share of the window one call must occupy for ``looping``. At 0.6 the agent is
#: doing the same thing more often than everything else combined.
LOOP_DOMINANCE: Final = 0.6

#: Distinct calls at or below which a window counts as "no progress". An agent
#: alternating between two calls forever is looping just as much as one
#: repeating a single call.
LOOP_MAX_DISTINCT: Final = 2

#: Share of the window that must be errors for ``retry_storm``. Above half the
#: recent work is failing.
RETRY_STORM_ERROR_RATE: Final = 0.5

#: Minimum errored steps for ``retry_storm``, independent of rate. Stops a short
#: window of three steps, two of them errors, from being called a storm.
RETRY_STORM_MIN_ERRORS: Final = 4


@dataclass(frozen=True, slots=True)
class Verdict:
    """The behavior lane's read on a window.

    Attributes:
        state: One of the :data:`optio.semconv.LOOP_STATES` values.
        repeat_count: Highest repeat count for any single call in the window.
            Reported even when the state is ``healthy`` -- it is evidence, and a
            policy may want to alert on a rising count before it crosses our
            threshold.
    """

    state: str
    repeat_count: int


#: The verdict for a window with too little evidence.
HEALTHY: Final = Verdict(state=semconv.LOOP_STATE_HEALTHY, repeat_count=0)


def classify(window: BehaviorWindow) -> Verdict:
    """Classify a run's health from its step-signature window.

    A thin wrapper over :func:`classify_state`, which is where the logic lives
    and what a shared backend feeds. Kept because the detector suite, the
    property tests and the semconv contract test all classify a window they
    have just built, and rewriting those call sites would mean editing the
    regression net for the change that introduced the split.

    Args:
        window: The run's window. Not mutated.

    Returns:
        The verdict.
    """
    return classify_state(window.state(LOOP_MAX_DISTINCT))


def classify_state(state: WindowState) -> Verdict:
    """Classify a run's health from a summary of its recent steps.

    Takes a :class:`~optio.lanes.behavior.store.WindowState` rather than a
    window so the same logic serves a process-local window and a shared backend
    that computed the summary server-side (ADR-050).

    Args:
        state: The window's summary. ``top_counts`` must be descending.

    Returns:
        The verdict. ``healthy`` whenever the evidence is insufficient or
        ambiguous.
    """
    # Every number here was computed where the steps live -- incrementally by
    # `BehaviorWindow.add` in process, or by one Lua script on a shared server.
    # Recomputing them from the raw steps made each step cost O(window); see
    # `BehaviorWindow.add` for the measurement that removed it.
    size = state.size
    # `top_counts` is descending, so the largest is the first -- and it may be
    # absent entirely, which indexing without the guard would raise on. A
    # detector that raises breaks the lane rather than declining.
    repeat_count = state.top_counts[0] if state.top_counts else 0

    if size < MIN_STEPS_FOR_VERDICT:
        # Report the count as evidence, but never a pathology this early.
        return Verdict(state=semconv.LOOP_STATE_HEALTHY, repeat_count=repeat_count)

    errors = state.errors
    if errors >= RETRY_STORM_MIN_ERRORS and errors / size >= RETRY_STORM_ERROR_RATE:
        return Verdict(state=semconv.LOOP_STATE_RETRY_STORM, repeat_count=repeat_count)

    # `looping` needs both dominance and an absence of progress. Dominance
    # alone would flag an agent that calls one cheap tool often while doing
    # varied work around it -- common, and healthy.
    #
    # Dominance is measured over the whole small set of recurring calls, not
    # over the single most frequent one. A perfect two-call cycle -- read,
    # think, read, think, forever -- is the textbook stuck agent, yet each call
    # holds only half the window. Scoring the top call alone would put that
    # below any useful threshold, making a cycle of length >= 2 structurally
    # undetectable: precisely the case Section 6.4 names in the definition of
    # `looping`.
    #
    # `top_counts` is already truncated to LOOP_MAX_DISTINCT by whoever built
    # the state, which is why `distinct_calls` is carried separately: it is the
    # number of *keys*, and summing a truncated list cannot recover it.
    distinct_calls = state.distinct_calls
    cycle_share = sum(state.top_counts) / size
    if cycle_share >= LOOP_DOMINANCE and distinct_calls <= LOOP_MAX_DISTINCT:
        return Verdict(state=semconv.LOOP_STATE_LOOPING, repeat_count=repeat_count)

    if repeat_count >= REPEAT_THRESHOLD:
        return Verdict(state=semconv.LOOP_STATE_REPEATING, repeat_count=repeat_count)

    return Verdict(state=semconv.LOOP_STATE_HEALTHY, repeat_count=repeat_count)
