"""Contract tests: emitted signal names match the pinned spec and docs (SC-2).

These are blocking on every PR (Section 9). They exist because the integration
surface is a set of strings: a rename that a type checker cannot see would
silently break every downstream OPA/Cedar/AGT policy written against it.

The core check parses the signal table out of ``docs/signals.md`` and asserts it
equals ``semconv.EMITTED_SIGNALS``. That makes the documentation load-bearing --
a signal cannot be added in code without being documented for integrators, and
vice versa (Section 16 rules 5 and 8).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from optio import semconv

DOCS_SIGNALS = Path(__file__).resolve().parents[2] / "docs" / "signals.md"

pytestmark = pytest.mark.contract


def _documented_signals() -> set[str]:
    """Extract every ``gen_ai.run.*`` name from the docs signal table."""
    text = DOCS_SIGNALS.read_text(encoding="utf-8")
    table = text.split("## Emitted signals", 1)[1].split("### `gen_ai.run.loop_state`", 1)[0]
    return set(re.findall(r"`(gen_ai\.run\.[a-z_.]+)`", table))


def test_docs_signals_file_exists():
    assert DOCS_SIGNALS.is_file(), "docs/signals.md is the authoritative signal contract"


def test_emitted_signals_match_docs():
    assert _documented_signals() == set(semconv.EMITTED_SIGNALS)


def test_semconv_version_is_pinned():
    # A pin that drifts to a range or a wildcard defeats R-TECH-2.
    assert re.fullmatch(r"\d+\.\d+\.\d+", semconv.GENAI_SEMCONV_VERSION)


def test_docs_declare_the_pinned_version():
    text = DOCS_SIGNALS.read_text(encoding="utf-8")
    assert f"`{semconv.GENAI_SEMCONV_VERSION}`" in text, (
        "docs/signals.md must state the semconv version the constants were validated against"
    )


def test_all_emitted_signals_are_namespaced():
    for name in semconv.EMITTED_SIGNALS:
        assert name.startswith(f"{semconv.GENAI_NAMESPACE}."), name


def test_internal_signals_are_not_in_the_genai_namespace():
    # Self-metrics must never be mistakable for a signal a policy can gate on.
    internal = [
        semconv.INTERNAL_SIGNALS_EMITTED,
        semconv.INTERNAL_LANE_ERRORS,
        semconv.INTERNAL_OVERHEAD_SECONDS,
        semconv.INTERNAL_SAMPLING_RATE,
    ]
    for name in internal:
        assert name.startswith(semconv.INTERNAL_NAMESPACE)
        assert not name.startswith(f"{semconv.GENAI_NAMESPACE}.")


def test_loop_states_are_exactly_the_documented_four():
    assert frozenset({"healthy", "repeating", "looping", "retry_storm"}) == semconv.LOOP_STATES


def test_healthy_is_a_loop_state():
    # The fail-open default must exist as a valid value (ADR-004).
    assert semconv.LOOP_STATE_HEALTHY in semconv.LOOP_STATES


def test_signal_constants_are_unique():
    names = [
        semconv.RUN_ACTUAL_COST,
        semconv.RUN_PROJECTED_COST,
        semconv.RUN_BUDGET_REMAINING,
        semconv.RUN_COST_PER_SUCCESSFUL_TASK,
        semconv.RUN_LOOP_STATE,
        semconv.RUN_REPEAT_COUNT,
        semconv.RUN_QUALITY_GROUNDEDNESS,
        semconv.RUN_QUALITY_TASK_SUCCESS,
        semconv.RUN_SUCCESS,
    ]
    assert len(names) == len(set(names))


def test_the_detector_only_produces_documented_states():
    # Binds the classifier to the published contract. semconv could list the
    # four states while the detector returned a fifth: the signal writer would
    # reject it and the behavior signal would vanish silently in production
    # while every unit test stayed green.
    from optio.lanes.behavior.detectors import classify
    from optio.lanes.behavior.window import BehaviorWindow, StepSignature

    produced = set()
    patterns = [
        [],
        [("same", "d", False)] * 12,
        [("api", "d", True)] * 12,
        [(f"t{n}", f"d{n}", False) for n in range(12)],
        [("a", "d", False)] * 6 + [(f"t{n}", f"d{n}", False) for n in range(6)],
    ]
    for pattern in patterns:
        window = BehaviorWindow(50)
        for tool, digest, errored in pattern:
            window.add(StepSignature(tool, digest, errored))
        produced.add(classify(window).state)

    assert produced <= semconv.LOOP_STATES
    # All four are reachable; a state nobody can produce is dead contract.
    assert produced == semconv.LOOP_STATES
