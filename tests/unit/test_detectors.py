"""Labeled-fixture tests for the behavior detectors (M3-2).

Section 6.4's acceptance criterion is two-sided: pathological runs classified
correctly, *and* a low false-positive rate on healthy ones. The second half is
the one that matters more, because a downstream policy may kill a run on
`looping` -- so a false positive breaks a working agent, while a false negative
only means a stuck run costs money the cost lane is already reporting.

The healthy fixtures are therefore not easy cases. They are the patterns most
likely to be misread as pathology: polling loops, paged retrieval, and bounded
retries -- all of which look repetitive because they *are* repetitive, and all
of which are correct agent behaviour.
"""

from __future__ import annotations

import pytest

from optio import semconv
from optio.lanes.behavior.detectors import (
    LOOP_DOMINANCE,
    MIN_STEPS_FOR_VERDICT,
    REPEAT_THRESHOLD,
    RETRY_STORM_MIN_ERRORS,
    classify,
)
from optio.lanes.behavior.window import BehaviorWindow, StepSignature


def sig(tool: str, args: str = "a", *, errored: bool = False) -> StepSignature:
    """Build a signature directly, bypassing span construction."""
    return StepSignature(tool=tool, args_digest=args, errored=errored)


def window_of(*signatures: StepSignature, maxlen: int = 50) -> BehaviorWindow:
    """Build a window containing the given signatures."""
    window = BehaviorWindow(maxlen)
    for signature in signatures:
        window.add(signature)
    return window


# ---------------------------------------------------------------------------
# Healthy fixtures -- the false-positive suite
# ---------------------------------------------------------------------------


class TestHealthyRunsAreNotFlagged:
    def test_varied_tool_use(self) -> None:
        window = window_of(
            sig("search"),
            sig("read_file", "b"),
            sig("summarise", "c"),
            sig("write_file", "d"),
            sig("search", "e"),
            sig("finish", "f"),
        )
        assert classify(window).state == semconv.LOOP_STATE_HEALTHY

    def test_paged_retrieval_is_not_looping(self) -> None:
        # Same tool eight times, different page each time. Repetitive by
        # design, and completely healthy.
        window = window_of(*[sig("fetch_page", f"page={n}") for n in range(8)])
        assert classify(window).state == semconv.LOOP_STATE_HEALTHY

    def test_a_short_run_is_never_pathological(self) -> None:
        # Four identical calls -- past the repeat threshold, but too little
        # evidence. An early false positive is the most damaging kind: it can
        # kill a run before it has done any work.
        window = window_of(*[sig("poll") for _ in range(4)])
        verdict = classify(window)
        assert verdict.state == semconv.LOOP_STATE_HEALTHY
        assert verdict.repeat_count == 4, "the count is still reported as evidence"

    def test_bounded_retries_are_not_a_storm(self) -> None:
        # Two failures then success, inside otherwise varied work. The common
        # shape of a flaky dependency handled correctly.
        window = window_of(
            sig("search"),
            sig("call_api", "x", errored=True),
            sig("call_api", "x", errored=True),
            sig("call_api", "x"),
            sig("summarise", "c"),
            sig("finish", "d"),
        )
        assert classify(window).state == semconv.LOOP_STATE_HEALTHY

    def test_one_frequent_cheap_tool_amid_varied_work(self) -> None:
        # `check_status` recurs often but the run is clearly progressing.
        # Dominance alone must not be enough to call this looping.
        window = window_of(
            sig("check_status"),
            sig("step_a", "a"),
            sig("check_status"),
            sig("step_b", "b"),
            sig("check_status"),
            sig("step_c", "c"),
            sig("check_status"),
            sig("step_d", "d"),
        )
        verdict = classify(window)
        assert verdict.state != semconv.LOOP_STATE_LOOPING
        assert verdict.repeat_count == 4

    def test_an_empty_window_is_healthy(self) -> None:
        verdict = classify(BehaviorWindow(50))
        assert verdict.state == semconv.LOOP_STATE_HEALTHY
        assert verdict.repeat_count == 0


# ---------------------------------------------------------------------------
# Pathological fixtures
# ---------------------------------------------------------------------------


class TestPathologiesAreDetected:
    def test_identical_calls_forever_is_looping(self) -> None:
        window = window_of(*[sig("search", "same query") for _ in range(10)])
        verdict = classify(window)
        assert verdict.state == semconv.LOOP_STATE_LOOPING
        assert verdict.repeat_count == 10

    def test_a_two_call_cycle_is_looping(self) -> None:
        # The classic stuck agent: read, think, read, think, forever. No single
        # call dominates 60% here, so this leans on the distinct-call bound.
        window = window_of(*[sig("read" if n % 2 else "think", "x") for n in range(12)])
        assert classify(window).state == semconv.LOOP_STATE_LOOPING

    def test_a_cycle_is_detected_even_when_no_single_call_dominates(self) -> None:
        # The regression that shaped the dominance rule. Scoring only the most
        # frequent call puts each half of a two-call cycle at 50%, below any
        # useful threshold -- which made cycles of length >= 2 structurally
        # undetectable, the exact pattern Section 6.4 defines `looping` as.
        window = window_of(*[sig("a" if n % 2 else "b", "x") for n in range(10)])
        verdict = classify(window)

        # The premise: no single call reaches the dominance threshold.
        assert verdict.repeat_count / len(window) < LOOP_DOMINANCE
        assert verdict.state == semconv.LOOP_STATE_LOOPING

    def test_repetition_amid_progress_is_only_repeating(self) -> None:
        # Six of one call, but four other distinct calls alongside. Worth
        # reporting; not worth killing the run over.
        window = window_of(
            *[sig("retry_thing") for _ in range(6)],
            sig("a", "1"),
            sig("b", "2"),
            sig("c", "3"),
            sig("d", "4"),
        )
        verdict = classify(window)
        assert verdict.state == semconv.LOOP_STATE_REPEATING
        assert verdict.repeat_count == 6

    def test_error_dominated_window_is_a_retry_storm(self) -> None:
        window = window_of(
            *[sig("call_api", "x", errored=True) for _ in range(6)],
            sig("search", "a"),
        )
        assert classify(window).state == semconv.LOOP_STATE_RETRY_STORM

    def test_retry_storm_outranks_looping(self) -> None:
        # Both conditions hold. The error-driven diagnosis names a cause
        # rather than a symptom, so it wins.
        window = window_of(*[sig("call_api", "x", errored=True) for _ in range(8)])
        assert classify(window).state == semconv.LOOP_STATE_RETRY_STORM

    def test_errors_across_varied_calls_still_storm(self) -> None:
        # A broken dependency hit through different tools is still a storm.
        window = window_of(
            sig("a", "1", errored=True),
            sig("b", "2", errored=True),
            sig("c", "3", errored=True),
            sig("d", "4", errored=True),
            sig("e", "5"),
        )
        assert classify(window).state == semconv.LOOP_STATE_RETRY_STORM


# ---------------------------------------------------------------------------
# Threshold boundaries -- pinned so a tweak is a deliberate act
# ---------------------------------------------------------------------------


class TestThresholdBoundaries:
    def test_verdict_requires_minimum_evidence(self) -> None:
        below = window_of(*[sig("x") for _ in range(MIN_STEPS_FOR_VERDICT - 1)])
        at = window_of(*[sig("x") for _ in range(MIN_STEPS_FOR_VERDICT)])

        assert classify(below).state == semconv.LOOP_STATE_HEALTHY
        assert classify(at).state != semconv.LOOP_STATE_HEALTHY

    def test_repeat_threshold_boundary(self) -> None:
        # Padded with distinct calls so the loop rule cannot fire, isolating
        # the repeat rule.
        def run(repeats: int) -> str:
            padding = [sig(f"pad{n}", str(n)) for n in range(6)]
            return classify(window_of(*[sig("r") for _ in range(repeats)], *padding)).state

        assert run(REPEAT_THRESHOLD - 1) == semconv.LOOP_STATE_HEALTHY
        assert run(REPEAT_THRESHOLD) == semconv.LOOP_STATE_REPEATING

    def test_loop_needs_both_dominance_and_no_progress(self) -> None:
        # Dominant but three distinct calls -> not looping.
        dominant_with_progress = window_of(
            *[sig("main") for _ in range(8)], sig("a", "1"), sig("b", "2")
        )
        assert classify(dominant_with_progress).state == semconv.LOOP_STATE_REPEATING

        # An even two-call cycle: neither call dominates alone, but together
        # they are the entire window and nothing else is happening.
        assert pytest.approx(0.6) == LOOP_DOMINANCE
        sparse = window_of(*[sig("a" if n % 2 else "b", "x") for n in range(10)])
        assert classify(sparse).state == semconv.LOOP_STATE_LOOPING

    def test_storm_needs_an_absolute_error_floor(self) -> None:
        # 3 of 5 errored is a majority, but too few errors to call a storm.
        few = window_of(
            *[sig("x", "1", errored=True) for _ in range(RETRY_STORM_MIN_ERRORS - 1)],
            sig("a", "2"),
            sig("b", "3"),
        )
        assert few is not None
        assert classify(few).state != semconv.LOOP_STATE_RETRY_STORM


class TestVerdictContract:
    def test_state_is_always_a_permitted_enum_value(self) -> None:
        # The signal writer rejects an unknown loop_state, so a detector
        # returning something off-contract would silently drop the signal.
        windows = [
            window_of(),
            window_of(*[sig("x") for _ in range(10)]),
            window_of(*[sig("x", errored=True) for _ in range(10)]),
            window_of(*[sig(f"t{n}", str(n)) for n in range(10)]),
        ]
        for window in windows:
            assert classify(window).state in semconv.LOOP_STATES

    def test_repeat_count_is_never_negative(self) -> None:
        assert classify(BehaviorWindow(10)).repeat_count == 0
