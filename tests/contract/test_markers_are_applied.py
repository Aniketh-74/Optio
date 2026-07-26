"""Every test directory with a CI gate carries the matching marker.

CI runs some suites by marker (``pytest -m failinject``), so a test file that
forgets its marker is silently skipped by that gate while still passing the
default run. Nothing else catches that -- the suite looks green either way,
which is precisely the failure mode worth a test.

Caught a real gap: ``tests/property/test_failopen_properties.py`` was invisible
to the ``property`` gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

TESTS = Path(__file__).resolve().parents[1]

#: Directories whose contents CI selects by marker.
MARKED_DIRECTORIES: dict[str, str] = {
    "failinject": "failinject",
    "property": "property",
    "contract": "contract",
    "integration": "integration",
    "bench": "bench",
    "policy": "policy",
}


def _declared_markers(path: Path) -> set[str]:
    """Return marker names applied at module level in a test file.

    Recognises both ``pytestmark = pytest.mark.<name>`` and the list form.

    Args:
        path: The test module to inspect.

    Returns:
        Marker names found.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    markers: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue

        values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        for value in values:
            if isinstance(value, ast.Attribute):
                markers.add(value.attr)

    return markers


def _test_modules(directory: str) -> list[Path]:
    """Return the test modules inside a directory."""
    return sorted((TESTS / directory).glob("test_*.py"))


@pytest.mark.parametrize(("directory", "marker"), sorted(MARKED_DIRECTORIES.items()))
def test_every_module_declares_its_marker(directory: str, marker: str) -> None:
    """Each module in a gated directory declares the gate's marker."""
    missing = [
        path.name for path in _test_modules(directory) if marker not in _declared_markers(path)
    ]

    assert not missing, (
        f"tests/{directory}/ modules missing `pytestmark = pytest.mark.{marker}`: "
        f"{missing}. CI selects this suite with `-m {marker}`, so these tests "
        f"would be silently skipped by that gate."
    )
