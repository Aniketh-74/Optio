"""``classify`` reads five numbers, and five numbers are what cross a process.

Pinned as a test because the tempting Redis port returns the whole counter --
up to ``behavior_window_size`` entries per step, 1,000 at the documented
ceiling. That would turn the O(1)-in-window-size guarantee the README publishes
as measured ("37 us at 50, 38 us at 1000, flat in window") into O(window) in
bytes, and every existing test would still pass, because nothing else measures
the payload.

So the summary is a fixed shape: three scalars plus at most ``k`` counts, where
``k`` is :data:`~optio.lanes.behavior.detectors.LOOP_MAX_DISTINCT` and the
caller supplies it -- the detector keeps its own thresholds rather than the
store learning them.
"""

from __future__ import annotations

from optio import semconv
from optio.lanes.behavior.detectors import LOOP_MAX_DISTINCT, classify, classify_state
from optio.lanes.behavior.store import WindowState
from optio.lanes.behavior.window import BehaviorWindow, StepSignature


def _sig(tool: str, errored: bool = False) -> StepSignature:
    """A signature whose call identity is just the tool name."""
    return StepSignature(tool=tool, args_digest="d", errored=errored)


class TestTheStateCarriesOnlyWhatClassifyUses:
    def test_state_summarises_a_window(self) -> None:
        window = BehaviorWindow(maxlen=50)
        for _ in range(3):
            window.add(_sig("read"))
        window.add(_sig("write", errored=True))

        state = window.state(LOOP_MAX_DISTINCT)

        assert state.size == 4
        assert state.errors == 1
        assert state.distinct_calls == 2
        assert state.top_counts == (3, 1)

    def test_top_counts_is_capped_at_k(self) -> None:
        """The payload must not grow with the window.

        Ten distinct calls still yield two counts, because two is all
        ``cycle_share`` ever sums. This is the assertion a naive Redis port
        fails while every behavioural test still passes.
        """
        window = BehaviorWindow(maxlen=50)
        for n in range(10):
            window.add(_sig(f"call{n}"))

        state = window.state(LOOP_MAX_DISTINCT)

        assert len(state.top_counts) == LOOP_MAX_DISTINCT
        assert state.distinct_calls == 10

    def test_the_payload_is_the_same_size_at_any_window(self) -> None:
        """The guarantee stated directly: widening the window to catch longer
        cycles must cost memory, never per-step work or bytes."""
        narrow = BehaviorWindow(maxlen=50)
        wide = BehaviorWindow(maxlen=1000)
        for n in range(300):
            narrow.add(_sig(f"call{n}"))
            wide.add(_sig(f"call{n}"))

        assert len(narrow.state(LOOP_MAX_DISTINCT).top_counts) == len(
            wide.state(LOOP_MAX_DISTINCT).top_counts
        )

    def test_an_empty_window_has_no_top_counts(self) -> None:
        state = BehaviorWindow(maxlen=50).state(LOOP_MAX_DISTINCT)

        assert state.size == 0
        assert state.errors == 0
        assert state.distinct_calls == 0
        assert state.top_counts == ()


class TestClassifyStateReadsTheSummary:
    def test_a_two_call_cycle_is_looping(self) -> None:
        """The textbook stuck agent: read, think, read, think.

        Neither call holds a majority alone, which is exactly why dominance is
        measured over the top two rather than the single most frequent.
        """
        state = WindowState(size=10, errors=0, distinct_calls=2, top_counts=(5, 5))

        assert classify_state(state).state == semconv.LOOP_STATE_LOOPING

    def test_varied_work_is_healthy(self) -> None:
        state = WindowState(size=10, errors=0, distinct_calls=8, top_counts=(2, 2))

        assert classify_state(state).state == semconv.LOOP_STATE_HEALTHY

    def test_errors_dominating_is_a_retry_storm(self) -> None:
        state = WindowState(size=10, errors=6, distinct_calls=3, top_counts=(4, 3))

        assert classify_state(state).state == semconv.LOOP_STATE_RETRY_STORM

    def test_too_few_steps_is_never_a_pathology(self) -> None:
        state = WindowState(size=3, errors=3, distinct_calls=1, top_counts=(3,))

        assert classify_state(state).state == semconv.LOOP_STATE_HEALTHY

    def test_repeat_count_is_the_largest_count(self) -> None:
        state = WindowState(size=10, errors=0, distinct_calls=4, top_counts=(4, 3))

        assert classify_state(state).repeat_count == 4

    def test_distinct_calls_is_not_inferred_from_the_truncated_counts(self) -> None:
        """The bug this shape invites, and the reason ``distinct_calls`` is
        carried separately.

        ``top_counts`` is truncated to two, so ``len(top_counts)`` is at most
        two and would satisfy ``distinct_calls <= LOOP_MAX_DISTINCT`` for
        *every* window. Four distinct calls whose top two happen to dominate is
        an agent repeating itself while still making progress -- ``repeating``,
        not ``looping`` -- and inferring the count would silently promote it to
        the strictest verdict this lane can emit.
        """
        state = WindowState(size=10, errors=0, distinct_calls=4, top_counts=(4, 3))

        assert sum(state.top_counts) / state.size >= 0.6, "the dominance test must be live"
        assert classify_state(state).state == semconv.LOOP_STATE_REPEATING

    def test_an_empty_state_reports_no_repeats(self) -> None:
        """Indexing ``top_counts[0]`` on an empty window would raise, and a
        detector that raises breaks the lane rather than declining."""
        state = WindowState(size=0, errors=0, distinct_calls=0, top_counts=())

        verdict = classify_state(state)

        assert verdict.state == semconv.LOOP_STATE_HEALTHY
        assert verdict.repeat_count == 0


class TestClassifyStillTakesAWindow:
    """The wrapper exists so 37 existing call sites keep meaning what they meant.

    ``classify_state`` is what the shared store feeds, but the detector suite,
    the property tests and the semconv contract test all classify a window they
    just built. Rewriting those call sites would have edited the regression net
    for this very change.
    """

    def test_it_agrees_with_classify_state(self) -> None:
        window = BehaviorWindow(maxlen=50)
        for _ in range(6):
            window.add(_sig("read"))

        assert classify(window) == classify_state(window.state(LOOP_MAX_DISTINCT))
