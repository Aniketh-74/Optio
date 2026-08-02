"""What ships must look like a library to the tools that consume it (ADR-047).

The wheel is the only artifact most users ever see, and several of its
properties are invisible from inside the repository. `mypy --strict` passing
here says nothing about whether a *consumer's* type checker can see the
annotations; that depends on a file whose absence is silent.

``optio_optimize`` shipped without ``py.typed`` while ``optio`` had one. Both
are declared in ``[tool.hatch.build.targets.wheel]``, the whole tree is
``--strict`` clean, and ``pyproject`` advertises ``Typing :: Typed`` -- so a
consumer running mypy against ``optio_optimize`` would have had every annotation
in 47 modules ignored under PEP 561, with no error anywhere to explain why.

These checks read the built wheel rather than the source tree, because the gap
was precisely between the two.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Packages the wheel is configured to ship, read from pyproject rather than
#: hardcoded -- a third package added to that list should be covered here
#: automatically rather than silently escape.
SHIPPED = [
    pathlib.Path(entry).name
    for entry in re.findall(
        r"packages\s*=\s*\[([^\]]*)\]", (REPO / "pyproject.toml").read_text(encoding="utf-8")
    )[0]
    .replace('"', "")
    .split(",")
    if entry.strip()
]


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Build a wheel and return its entries.

    Built rather than assumed: the question is what hatchling actually puts in
    the artifact, and a file present in ``src/`` is not evidence that it is.
    """
    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    wheels = list(out.glob("*.whl"))
    if not wheels:
        pytest.skip(f"could not build a wheel here: {proc.stderr.strip()[-300:]}")
    return zipfile.ZipFile(wheels[0]).namelist()


def test_there_is_something_to_check() -> None:
    """``SHIPPED`` is parsed out of pyproject; an empty list would make every
    parametrised test below vacuously pass."""
    assert SHIPPED, "no packages parsed from [tool.hatch.build.targets.wheel]"


@pytest.mark.parametrize("package", SHIPPED)
def test_every_shipped_package_declares_its_typing(package: str, wheel_names: list[str]) -> None:
    """PEP 561: without ``py.typed`` in the *installed* package, a consumer's
    type checker ignores every annotation and says nothing about it."""
    assert f"{package}/py.typed" in wheel_names, (
        f"{package} ships without py.typed, so its annotations are invisible to consumers"
    )


@pytest.mark.parametrize("package", SHIPPED)
def test_every_shipped_package_is_importable_from_the_wheel(
    package: str, wheel_names: list[str]
) -> None:
    """A package listed for shipping but absent from the artifact is the
    packaging mistake an editable install cannot show you."""
    assert any(name.startswith(f"{package}/") for name in wheel_names), (
        f"{package} is declared in pyproject but absent from the wheel"
    )


def test_the_wheel_carries_no_tests_or_docs(wheel_names: list[str]) -> None:
    """The sdist ships tests and docs deliberately, so a downstream packager can
    verify what they are shipping. The **wheel** is what lands in a user's
    site-packages and should carry the library and nothing else.
    """
    strays = [
        name
        for name in wheel_names
        if name.split("/")[0] not in {*SHIPPED} and not name.split("/")[0].endswith(".dist-info")
    ]

    assert strays == [], f"the wheel carries non-library entries: {strays[:5]}"
