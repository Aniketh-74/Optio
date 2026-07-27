"""Enforce Section 16 rule 5: signal names come from ``semconv.py``, never literals.

A hardcoded ``"gen_ai.run.actual_cost"`` somewhere in a lane would work perfectly
until the day the pinned semconv version moves and the constant changes but the
literal does not. That divergence is invisible to mypy and to every unit test
that asserts behavior rather than names -- so it gets its own check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "optio"
SEMCONV = SRC / "semconv.py"

pytestmark = pytest.mark.contract


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_no_module_defines_a_genai_literal_except_semconv():
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path == SEMCONV:
            continue
        for value in _string_constants(path):
            # Docstrings legitimately mention names in prose; only flag values
            # that are exactly an attribute name, which is how they'd be used.
            if value.startswith("gen_ai.") and " " not in value:
                offenders.append(f"{path.relative_to(SRC)}: {value!r}")
    assert not offenders, (
        "signal names must be imported from optio.semconv, not written as literals:\n"
        + "\n".join(offenders)
    )
