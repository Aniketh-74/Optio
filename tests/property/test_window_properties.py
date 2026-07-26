"""Property tests for the behavior window and detectors (M3-1, M3-2).

Section 9 makes these blocking for M3, alongside the ledger's. The shared reason
is that these are stateful components whose failures are *silent*: a window that
grows without bound leaks memory in the user's process for hours before anything
notices, and a detector that fabricates a pathology gets a healthy run killed by
a downstream policy. Neither produces an error anyone would see.

The classifier properties are one-sided on purpose. They assert that the
detector never *invents* a pathology -- never on an empty window, never below
the evidence floor, never on a run with no repetition. They deliberately do not
assert that it always catches one. Detection is a heuristic and false negatives
are acceptable (Section 6.4); false positives are not.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agentmeter import semconv
from agentmeter.lanes.behavior.detectors import MIN_STEPS_FOR_VERDICT, classify
from agentmeter.lanes.behavior.window import BehaviorWindow, StepSignature, digest_args

pytestmark = pytest.mark.property

#: Tool names drawn from a small alphabet so repetition actually occurs.
tool_names = st.sampled_from(["search", "read", "write", "think", "call_api"])

signatures = st.builds(
    StepSignature,
    tool=tool_names,
    args_digest=st.sampled_from(["d0", "d1", "d2", "d3"]),
    errored=st.booleans(),
)


class TestWindowBounds:
    @given(
        maxlen=st.integers(min_value=1, max_value=64),
        steps=st.lists(signatures, max_size=400),
    )
    def test_length_never_exceeds_the_bound(self, maxlen: int, steps: list[StepSignature]) -> None:
        window = BehaviorWindow(maxlen)
        for step in steps:
            window.add(step)
            # Checked every iteration, not just at the end: a bound that is
            # only restored on the next add would still peak unboundedly.
            assert len(window) <= maxlen

        assert len(window) == min(len(steps), maxlen)

    @given(
        maxlen=st.integers(min_value=1, max_value=32),
        steps=st.lists(signatures, min_size=1, max_size=200),
    )
    def test_the_window_holds_the_most_recent_steps(
        self, maxlen: int, steps: list[StepSignature]
    ) -> None:
        window = BehaviorWindow(maxlen)
        for step in steps:
            window.add(step)

        assert list(window) == steps[-maxlen:]

    @given(steps=st.lists(signatures, max_size=200))
    def test_total_steps_is_the_true_count(self, steps: list[StepSignature]) -> None:
        window = BehaviorWindow(8)
        for step in steps:
            window.add(step)

        assert window.total_steps == len(steps)


class TestSignatureStability:
    @given(value=st.text(max_size=200))
    def test_digest_is_deterministic(self, value: str) -> None:
        assert digest_args(value) == digest_args(value)

    @given(
        pairs=st.lists(
            st.tuples(st.text(min_size=1, max_size=20), st.integers()),
            min_size=1,
            max_size=12,
            unique_by=lambda pair: pair[0],
        )
    )
    def test_mapping_digest_ignores_key_order(self, pairs: list[tuple[str, int]]) -> None:
        forward = dict(pairs)
        backward = dict(reversed(pairs))

        assert digest_args(forward) == digest_args(backward)

    @given(value=st.text(max_size=100))
    def test_digest_never_contains_its_input(self, value: str) -> None:
        # Section 10: content must not survive into retained state. A digest
        # that echoed its input would leak prompt text into memory that a crash
        # dump or debugger could read.
        digest = digest_args(value)
        if len(value) > 16:
            assert value not in digest

    @given(value=st.text(max_size=100))
    def test_digest_has_a_fixed_width(self, value: str) -> None:
        # Bounded output is what keeps window memory bounded regardless of how
        # large the arguments were.
        assert len(digest_args(value)) == 16


class TestClassifierNeverFabricates:
    """The one-sided guarantee from Section 6.4."""

    @given(steps=st.lists(signatures, max_size=MIN_STEPS_FOR_VERDICT - 1))
    def test_below_the_evidence_floor_is_always_healthy(self, steps: list[StepSignature]) -> None:
        window = BehaviorWindow(50)
        for step in steps:
            window.add(step)

        assert classify(window).state == semconv.LOOP_STATE_HEALTHY

    @given(
        maxlen=st.integers(min_value=1, max_value=64),
        steps=st.lists(signatures, max_size=300),
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_the_state_is_always_on_contract(self, maxlen: int, steps: list[StepSignature]) -> None:
        # An off-contract value would be rejected by the signal writer, so the
        # behavior signal would vanish silently rather than fail loudly.
        window = BehaviorWindow(maxlen)
        for step in steps:
            window.add(step)

        verdict = classify(window)
        assert verdict.state in semconv.LOOP_STATES
        assert isinstance(verdict.repeat_count, int)
        assert verdict.repeat_count >= 0

    @given(count=st.integers(min_value=0, max_value=200))
    def test_all_distinct_successful_calls_are_never_pathological(self, count: int) -> None:
        # A run where every step is different and nothing fails is the
        # definition of progress. Flagging it would be the worst false
        # positive the detector could produce.
        window = BehaviorWindow(64)
        for n in range(count):
            window.add(StepSignature(f"tool{n}", f"d{n}", errored=False))

        assert classify(window).state == semconv.LOOP_STATE_HEALTHY

    @given(steps=st.lists(signatures, max_size=200))
    def test_no_errors_means_no_retry_storm(self, steps: list[StepSignature]) -> None:
        window = BehaviorWindow(50)
        for step in steps:
            window.add(StepSignature(step.tool, step.args_digest, errored=False))

        assert classify(window).state != semconv.LOOP_STATE_RETRY_STORM

    @given(
        maxlen=st.integers(min_value=1, max_value=64),
        steps=st.lists(signatures, min_size=1, max_size=300),
    )
    def test_repeat_count_never_exceeds_the_window(
        self, maxlen: int, steps: list[StepSignature]
    ) -> None:
        # repeat_count is published as evidence. A count larger than the window
        # it was measured over would be nonsense to a policy reading it.
        window = BehaviorWindow(maxlen)
        for step in steps:
            window.add(step)

        assert classify(window).repeat_count <= len(window)

    @given(steps=st.lists(signatures, max_size=200))
    def test_classification_does_not_mutate_the_window(self, steps: list[StepSignature]) -> None:
        window = BehaviorWindow(32)
        for step in steps:
            window.add(step)

        before = list(window)
        classify(window)

        assert list(window) == before
        assert classify(window) == classify(window), "classification is pure"
