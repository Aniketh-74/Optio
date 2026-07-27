"""The working tree contains committed source, not a leftover mutation.

`scripts/mutate.py` runs mutation testing against a throwaway copy, so the
working tree should never see a mutation. This is the backstop for when someone
runs `cosmic-ray` directly -- it mutates files in place and restores them with a
context manager, which a kill signal skips entirely.

That has happened here. An interrupted run left `project.py` holding
``committed - remaining * estimate`` instead of ``+``: every worst-case cost
projection with its sign flipped, sitting in the working tree, one ``git commit
-a`` away from being real. It was caught by reading ``git status``, which is
luck rather than process.

This test is the process. It asserts the specific arithmetic and boolean
expressions that mutation operators target and that no other test pins directly
-- the ones where a flip produces a plausible wrong number rather than an
error. Coverage cannot catch these: the mutated line still runs.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from optio.lanes.behavior import detectors
from optio.lanes.cost import ledger, project
from optio.runtime import failopen

pytestmark = pytest.mark.contract


def _source(module: object) -> str:
    """Return a module's source with comments and docstrings left intact."""
    return inspect.getsource(module)  # type: ignore[arg-type]


class TestArithmeticSignsAreIntact:
    """Operators whose flip yields a wrong number instead of a crash."""

    def test_projection_adds_remaining_spend(self) -> None:
        # The exact mutation an interrupted run left behind. A minus here makes
        # a long expensive run project as *cheaper* than a short one, which no
        # type checker or coverage gate would notice.
        assert "snapshot.committed + remaining * estimate" in _source(project)
        assert "snapshot.committed - remaining * estimate" not in _source(project)

    def test_budget_remaining_subtracts_committed(self) -> None:
        # A plus here would grow the remaining budget as the run spent money.
        assert "budget.limit_usd - snapshot.committed" in _source(project)

    def test_steps_taken_sums_both_kinds(self) -> None:
        assert "snapshot.reconciled_steps + snapshot.open_steps" in _source(project)

    def test_per_step_estimate_divides(self) -> None:
        source = _source(project)
        assert "snapshot.actual / snapshot.reconciled_steps" in source
        assert "snapshot.reserved / snapshot.open_steps" in source


class TestGuardConditionsAreNotInverted:
    """Booleans whose inversion silences a warning or admits a bad value."""

    def test_the_leak_warning_fires_on_leaks(self) -> None:
        # Inverting this suppressed the only signal that a reported cost is a
        # reserved worst case rather than measured spend. Mutation testing
        # found it surviving the whole suite; three tests in
        # test_ledger_lifecycle.py now assert the behaviour, and this pins the
        # source so the two cannot drift apart.
        assert "if leaked:" in _source(ledger)
        assert "if not leaked:" not in _source(ledger)

    def test_cost_evidence_requires_positive_reserved(self) -> None:
        assert "if snapshot.reserved > 0.0:" in _source(project)

    def test_unpriced_steps_are_not_treated_as_evidence(self) -> None:
        assert "if snapshot.reserved <= 0.0:" in _source(project)

    def test_config_errors_are_never_absorbed(self) -> None:
        # The guard must re-raise OptioConfigError. Absorbing it would turn a
        # setup mistake into silent non-instrumentation (Section 4.2).
        assert "_NEVER_ABSORB" in _source(failopen)
        assert "OptioConfigError" in _source(failopen)


class TestThresholdComparisonsKeepTheirDirection:
    """Detector boundaries: a flipped comparison is a false positive machine."""

    def test_retry_storm_needs_enough_errors(self) -> None:
        source = _source(detectors)
        assert "errors >= RETRY_STORM_MIN_ERRORS" in source
        assert "errors / size >= RETRY_STORM_ERROR_RATE" in source

    def test_looping_needs_dominance_and_few_distinct_calls(self) -> None:
        source = _source(detectors)
        assert "cycle_share >= LOOP_DOMINANCE" in source
        assert "distinct_calls <= LOOP_MAX_DISTINCT" in source

    def test_a_short_window_never_reports_a_pathology(self) -> None:
        assert "size < MIN_STEPS_FOR_VERDICT" in _source(detectors)


class TestNoStrayEditsInShippedSource:
    """A blunt sweep for the debris a killed tool leaves behind."""

    def test_no_source_file_contains_a_mutation_marker(self) -> None:
        # cosmic-ray does not leave markers, but a half-applied patch, a merge
        # conflict, or an editor crash does.
        markers = ("<<<<<<<", ">>>>>>>", "=======\n=======")
        for path in Path("src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for marker in markers:
                assert marker not in text, f"{path} contains {marker!r}"

    def test_every_source_file_still_parses(self) -> None:
        # A truncated write leaves a file that imports fine until the branch
        # that needs it runs. Parsing every file catches that immediately.
        for path in Path("src").rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as error:  # pragma: no cover - failure path
                pytest.fail(f"{path} does not parse: {error}")
