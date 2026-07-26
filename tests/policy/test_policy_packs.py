"""Policy packs evaluate correctly against sample signals (M4-4, SC-3).

Three engines, three packs: OPA/Rego, Cedar, AGT YAML. SC-3 asks for at least
three, and the point of shipping them is copy-paste adoption -- which only works
if they actually run.

**Two tiers of checking, deliberately.**

The engine-level tests run the packs through the real engines. Cedar goes
through ``cedarpy``, which ships in the dev extras and so runs everywhere. OPA
needs the ``opa`` binary, which CI installs and most developers do not have, so
those two tests skip locally -- and ``AGENTMETER_REQUIRE_POLICY_ENGINES=1``
turns that skip into a failure in CI, because a gate that can pass by skipping
is not a gate.

Running the real engines earned its keep immediately: it caught two errors in
the Cedar pack that no amount of structural checking would have found -- the
schema type is spelled ``decimal`` rather than ``Decimal``, and an action
declared inside a namespace must be referenced as ``AgentMeter::Action::"..."``.
Both would have shipped as a pack that simply does not load.

The structural tests then cover what the engines cannot: that every signal name
in every pack exists in ``semconv.py``, that no pack invents a loop state the
detector cannot produce, and that every comparison is guarded against a missing
signal. That last one is the bug this file exists to prevent -- a pack that
reads absence as zero turns a broken cost lane into a free-looking run, and it
would pass every functional test in all three engines.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agentmeter import semconv

pytestmark = pytest.mark.policy

POLICIES = Path(__file__).resolve().parents[2] / "policies"
OPA_DIR = POLICIES / "opa"
CEDAR_DIR = POLICIES / "cedar"
AGT_POLICY = POLICIES / "agt" / "agentmeter-policy.yaml"

#: Set in CI. Turns a missing binary from "skip" into "fail", so the SC-3 gate
#: cannot quietly stop running because a tool install step broke.
REQUIRE_ENGINES = os.environ.get("AGENTMETER_REQUIRE_POLICY_ENGINES") == "1"

#: Signal names that may appear in a pack. Sourced from semconv so a renamed
#: signal breaks the packs here rather than in a user's policy (Section 16
#: rule 5).
KNOWN_SIGNALS = frozenset(semconv.EMITTED_SIGNALS)

#: Every `gen_ai.*` string a pack could reference. The trailing `[a-z_]` matters:
#: without it the pattern also captures the bare prefix `gen_ai.run.` where the
#: comments discuss naming, which is prose rather than a signal reference.
_SIGNAL_PATTERN = re.compile(r"gen_ai(?:\.[a-z_]+)+")


def _rule_lines(text: str) -> list[str]:
    """Return the lines of a policy file that are not comments.

    Signal names and loop states are checked against the *rules*, not the
    commentary. Every pack explains in prose which states it deliberately does
    not gate on, and a checker that read those sentences as rules would flag
    exactly the packs that documented themselves best.

    Args:
        text: Full policy file contents.

    Returns:
        Non-comment, non-blank lines, stripped.
    """
    lines: list[str] = []
    in_yaml_block = False
    block_indent = 0

    for raw in text.splitlines():
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())

        # A YAML block scalar (`description: >-`) makes its continuation lines
        # prose even though they carry no comment marker.
        if in_yaml_block:
            if stripped and indent <= block_indent:
                in_yaml_block = False
            else:
                continue
        if re.search(r":\s*[|>][-+]?\s*$", stripped):
            in_yaml_block = True
            block_indent = indent
            continue

        if not stripped or stripped.startswith(("#", "//")):
            continue
        lines.append(stripped)

    return lines


def _require(tool: str) -> str:
    """Return the path to a policy engine binary, or skip/fail.

    Args:
        tool: Executable name.

    Returns:
        Absolute path to the binary.
    """
    path = shutil.which(tool)
    if path is None:
        if REQUIRE_ENGINES:
            pytest.fail(
                f"{tool!r} is not installed but AGENTMETER_REQUIRE_POLICY_ENGINES=1. "
                f"The SC-3 gate must not pass by skipping."
            )
        pytest.skip(f"{tool} not installed; engine-level check runs in CI")
    return path


def _require_cedarpy() -> Any:
    """Return the ``cedarpy`` module, or skip/fail.

    Ships in the dev extras, so this should always be present; the guard exists
    so a broken install surfaces as a clear message rather than a collection
    error, and so ``REQUIRE_ENGINES`` still turns absence into a failure.

    Returns:
        The imported module.
    """
    try:
        import cedarpy
    except ImportError:  # pragma: no cover - dev extras always provide it
        if REQUIRE_ENGINES:
            pytest.fail(
                "cedarpy is not installed but AGENTMETER_REQUIRE_POLICY_ENGINES=1. "
                "The SC-3 gate must not pass by skipping."
            )
        pytest.skip("cedarpy not installed; install the dev extras")
    return cedarpy


def _pack_files() -> list[Path]:
    """Return every policy file across the three packs."""
    return [
        *sorted(OPA_DIR.glob("*.rego")),
        *sorted(CEDAR_DIR.glob("*.cedar")),
        CEDAR_DIR / "schema.cedarschema",
        CEDAR_DIR / "tests.json",
        AGT_POLICY,
    ]


class TestPacksExist:
    """SC-3: at least three engines, shipped and findable."""

    def test_three_engines_are_covered(self) -> None:
        assert (OPA_DIR / "agentmeter.rego").is_file()
        assert (CEDAR_DIR / "agentmeter.cedar").is_file()
        assert AGT_POLICY.is_file()

    def test_every_pack_file_is_non_empty(self) -> None:
        empty = [p.name for p in _pack_files() if not p.is_file() or not p.read_text().strip()]
        assert not empty, f"empty or missing pack files: {empty}"


class TestSignalNamesAreReal:
    """A pack referencing a signal we do not emit is a silently dead rule."""

    @pytest.mark.parametrize("path", _pack_files(), ids=lambda p: p.name)
    def test_referenced_signals_all_exist(self, path: Path) -> None:
        rules = "\n".join(_rule_lines(path.read_text(encoding="utf-8")))
        referenced = {
            name
            for name in _SIGNAL_PATTERN.findall(rules)
            # Namespace declarations, not signal references.
            if name not in {semconv.GENAI_NAMESPACE, f"{semconv.GENAI_NAMESPACE}.run"}
        }
        unknown = sorted(referenced - KNOWN_SIGNALS)

        assert not unknown, (
            f"{path.name} references signals agentmeter does not emit: {unknown}. "
            f"A rule on a nonexistent attribute never fires, so this fails "
            f"open silently rather than erroring."
        )

    def test_the_packs_cover_the_signals_that_exist_today(self) -> None:
        # Cost and behavior ship (M2, M3). A pack that ignored them would not
        # deliver the copy-paste adoption SC-3 is for.
        rego = (OPA_DIR / "agentmeter.rego").read_text(encoding="utf-8")
        for signal in (
            semconv.RUN_PROJECTED_COST,
            semconv.RUN_BUDGET_REMAINING,
            semconv.RUN_LOOP_STATE,
        ):
            assert signal in rego, f"{signal} is emitted today but no OPA rule uses it"


class TestLoopStatesAreReal:
    """A pack gating on a state the detector cannot produce is a dead rule."""

    @pytest.mark.parametrize("path", _pack_files(), ids=lambda p: p.name)
    def test_no_pack_invents_a_loop_state(self, path: Path) -> None:
        rules = "\n".join(_rule_lines(path.read_text(encoding="utf-8")))
        # Quoted tokens in the shape of a loop state. The signal's own name is
        # excluded -- it is the attribute being read, not one of its values.
        candidates = set(re.findall(r'"(healthy|repeating|looping|\w*(?:storm|loop)\w*)"', rules))
        candidates -= {"loop_state", semconv.RUN_LOOP_STATE}
        unknown = sorted(candidates - set(semconv.LOOP_STATES))

        assert not unknown, f"{path.name} references unknown loop states: {unknown}"

    def test_the_blocking_states_are_exactly_looping_and_retry_storm(self) -> None:
        # docs/behavior.md: gate on `looping` and `retry_storm`; alert on
        # `repeating`, which normal agents produce in bulk (polling, paging,
        # bounded retries). A pack that denied on `repeating` would be the false
        # positive that gets agentmeter uninstalled -- our error becoming the
        # user's outage, which is the asymmetry ADR-004 exists for.
        expected = {semconv.LOOP_STATE_LOOPING, semconv.LOOP_STATE_RETRY_STORM}

        rego = (OPA_DIR / "agentmeter.rego").read_text(encoding="utf-8")
        rego_set = re.search(r"blocking_loop_states\s*:=\s*\{([^}]*)\}", rego)
        assert rego_set is not None, "OPA pack no longer declares blocking_loop_states"
        assert set(re.findall(r'"(\w+)"', rego_set.group(1))) == expected

        cedar = (CEDAR_DIR / "agentmeter.cedar").read_text(encoding="utf-8")
        cedar_list = re.search(r"\[([^\]]*)\]\.contains\(resource\.loop_state\)", cedar)
        assert cedar_list is not None, "Cedar pack no longer gates on loop_state"
        assert set(re.findall(r'"(\w+)"', cedar_list.group(1))) == expected

        agt = AGT_POLICY.read_text(encoding="utf-8")
        agt_rule = agt.split("- name: stuck-agent", 1)[1].split("- name:", 1)[0]
        assert set(re.findall(r'"(\w+)"', agt_rule)) == expected

    @pytest.mark.parametrize(
        ("path", "marker"),
        [
            (OPA_DIR / "agentmeter.rego", "warn contains"),
            (AGT_POLICY, "effect: warn"),
        ],
        ids=["opa", "agt"],
    )
    def test_repeating_is_surfaced_as_a_warning(self, path: Path, marker: str) -> None:
        # Not gated on, but not discarded either: a rising repeat count is the
        # early signal that precedes a real loop.
        text = path.read_text(encoding="utf-8")
        assert marker in text
        assert semconv.LOOP_STATE_REPEATING in text


class TestAbsenceIsNeverTreatedAsZero:
    """The failure mode this whole file exists to prevent.

    agentmeter omits a signal it cannot compute rather than emitting zero
    (docs/signals.md). A pack that compares an attribute without first checking
    presence reads a broken cost lane as a free run -- and passes every
    functional test, because the fixtures all have the attribute.
    """

    def test_rego_guards_every_comparison(self) -> None:
        text = (OPA_DIR / "agentmeter.rego").read_text(encoding="utf-8")
        offenders = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"input\.attributes\[[^\]]+\]\s*[<>]", line)
            and not line.strip().startswith("#")
        ]
        assert not offenders, (
            "Rego rules must compare through the over()/under() helpers, which "
            "test presence first. Direct comparison on an absent key makes the "
            "rule undefined, so it never fires:\n  " + "\n  ".join(offenders)
        )

    def test_cedar_guards_every_attribute_read(self) -> None:
        # Parses `when { ... }` bodies out of the comment-stripped source. An
        # earlier version split the raw text on the words "forbid" and "when",
        # which also matched them inside comments -- it found nine blocks in a
        # four-rule file and inspected none of the real conditions. It passed,
        # and kept passing when the guard was deleted.
        conditions = self._when_blocks((CEDAR_DIR / "agentmeter.cedar").read_text(encoding="utf-8"))
        assert len(conditions) == 4, (
            f"expected 4 forbid conditions, parsed {len(conditions)}; "
            f"the parser and the pack have drifted apart"
        )

        for condition in conditions:
            # `resource.foo.greaterThan(...)` -- take the attribute, not the
            # extension method chained onto it.
            reads = set(re.findall(r"resource\.(\w+)", condition))
            guarded = set(re.findall(r"resource has (\w+)", condition))
            unguarded = sorted(reads - guarded)
            assert not unguarded, (
                f"Cedar condition reads {unguarded} without a `has` guard:\n"
                f"  {condition.strip()}\n"
                f"On an absent attribute Cedar raises, and most authorizers "
                f"surface that as a deny -- inverting fail-open (ADR-004)."
            )

    @staticmethod
    def _when_blocks(text: str) -> list[str]:
        """Return the body of every ``when { ... }`` clause, comments removed.

        Args:
            text: Cedar policy source.

        Returns:
            One string per condition body.
        """
        source = "\n".join(line for line in text.splitlines() if not line.strip().startswith("//"))
        return re.findall(r"when\s*\{([^}]*)\}", source)

    def test_cedar_schema_marks_every_signal_optional(self) -> None:
        text = (CEDAR_DIR / "schema.cedarschema").read_text(encoding="utf-8")
        body = text.split("entity Run {", 1)[1].split("};", 1)[0]
        required = [
            line.strip()
            for line in body.splitlines()
            if ":" in line and "?" not in line.split(":")[0] and not line.strip().startswith("//")
        ]
        assert not required, f"every signal must be optional in the schema; required: {required}"

    def test_agt_sets_on_missing_for_every_rule(self) -> None:
        text = AGT_POLICY.read_text(encoding="utf-8")
        rule_names = re.findall(r"^\s*- name: (\S+)", text, re.MULTILINE)
        on_missing = re.findall(r"^\s*on_missing: (\S+)", text, re.MULTILINE)

        assert len(rule_names) == len(on_missing), (
            f"{len(rule_names)} rules but {len(on_missing)} on_missing settings; "
            f"a rule without one falls back to engine default, which may be zero"
        )
        assert set(on_missing) == {"skip"}, f"on_missing must be 'skip', found: {set(on_missing)}"

    def test_agt_fails_open_on_signal_outage(self) -> None:
        text = AGT_POLICY.read_text(encoding="utf-8")
        assert "onSignalUnavailable: allow" in text, (
            "denying when signals are unavailable turns an agentmeter bug into "
            "the user's outage, which ADR-004 forbids"
        )


class TestCedarCasesAreWellFormed:
    """The Cedar fixtures are data, so they get their own structural check."""

    def test_cases_parse_and_cover_both_decisions(self) -> None:
        data = json.loads((CEDAR_DIR / "tests.json").read_text(encoding="utf-8"))
        cases = data["cases"]
        decisions = {case["decision"] for case in cases}

        assert decisions == {"Allow", "Deny"}
        assert len(cases) >= 10

    def test_absence_cases_are_present(self) -> None:
        # Without these the suite would only prove the easy direction.
        data = json.loads((CEDAR_DIR / "tests.json").read_text(encoding="utf-8"))
        empty_run = [c for c in data["cases"] if c["run"] == {}]

        assert empty_run, "no case covers a run with no signals (fail-open activation)"
        assert empty_run[0]["decision"] == "Allow"


class TestOpaEngine:
    """Real `opa test` run. Blocking in CI; skipped locally without the binary."""

    def test_rego_policy_is_valid(self) -> None:
        opa = _require("opa")
        result = subprocess.run(
            [opa, "check", "--v1-compatible", str(OPA_DIR)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"opa check failed:\n{result.stdout}\n{result.stderr}"

    def test_rego_tests_pass(self) -> None:
        opa = _require("opa")
        result = subprocess.run(
            [opa, "test", str(OPA_DIR), "-v"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"opa test failed:\n{result.stdout}\n{result.stderr}"


class TestCedarEngine:
    """Real Cedar evaluation, via the `cedarpy` bindings.

    Uses the Python bindings rather than the `cedar` CLI: the CLI is a Rust
    build (~2 min in CI, and a toolchain most contributors do not have), while
    `cedarpy` is a wheel that installs with the dev extras. Same engine
    underneath, so these tests run locally too -- which matters, because they
    caught two real errors in the pack that structural checks could not:
    `Decimal` is spelled `decimal` in a schema, and an action declared inside a
    namespace must be referenced as `AgentMeter::Action::"..."`. Both would have
    shipped as a pack that does not load.
    """

    #: Signals typed as `decimal` in the schema. Cedar has no implicit
    #: conversion, so these must be built as extension values.
    DECIMAL_SIGNALS = frozenset(
        {
            "projected_cost",
            "budget_remaining",
            "actual_cost",
            "groundedness",
            "task_success",
            "cost_per_successful_task",
        }
    )

    @staticmethod
    def _policy_and_schema() -> tuple[str, str]:
        return (
            (CEDAR_DIR / "agentmeter.cedar").read_text(encoding="utf-8"),
            (CEDAR_DIR / "schema.cedarschema").read_text(encoding="utf-8"),
        )

    @classmethod
    def _entities(cls, run: dict[str, object]) -> list[dict[str, object]]:
        """Build the entity list for one authorization request."""
        attrs = {
            key: (
                {"__extn": {"fn": "decimal", "arg": value}} if key in cls.DECIMAL_SIGNALS else value
            )
            for key, value in run.items()
        }
        return [
            {"uid": {"type": "AgentMeter::Run", "id": "run-1"}, "attrs": attrs, "parents": []},
            {"uid": {"type": "AgentMeter::Agent", "id": "agent-1"}, "attrs": {}, "parents": []},
        ]

    def test_cedar_policy_validates_against_the_schema(self) -> None:
        cedarpy = _require_cedarpy()
        policy, schema = self._policy_and_schema()
        result = cedarpy.validate_policies(policy, schema)

        assert result.validation_passed, [error.error for error in result.errors]

    def test_every_case_produces_the_expected_decision(self) -> None:
        cedarpy = _require_cedarpy()
        policy, schema = self._policy_and_schema()
        cases = json.loads((CEDAR_DIR / "tests.json").read_text(encoding="utf-8"))["cases"]

        wrong: list[str] = []
        for case in cases:
            response = cedarpy.is_authorized(
                {
                    "principal": 'AgentMeter::Agent::"agent-1"',
                    "action": 'AgentMeter::Action::"execute_step"',
                    "resource": 'AgentMeter::Run::"run-1"',
                    "context": {},
                },
                policy,
                self._entities(case["run"]),
                schema=schema,
            )
            decision = str(response.decision).rsplit(".", maxsplit=1)[-1].title()
            if decision != case["decision"]:
                wrong.append(f"{case['name']}: expected {case['decision']}, got {decision}")

        assert not wrong, "\n".join(wrong)
