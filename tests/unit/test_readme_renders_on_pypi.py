"""README.md is the PyPI landing page, and PyPI is not GitHub (ADR-045).

``pyproject.toml`` sets ``readme = "README.md"``, so this file becomes the
project description rendered at ``pypi.org/project/optio``. Two things are true
there that are not true on GitHub, and both were about to ship broken in 0.2.0:

* **Relative links do not resolve.** PyPI renders the markdown as-is against its
  own origin, so ``](docs/testing.md)`` becomes
  ``pypi.org/project/optio/docs/testing.md`` and 404s. Fifteen links in this
  README were relative and none were absolute.
* **The version in the banner is the first line anyone reads**, and it had gone
  a release behind for the second time.

Neither is caught by any other check. `twine check --strict` validates that the
description *parses*; it has no opinion on whether the links in it point at
anything.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "README.md"

#: Every markdown link target in the README, anchors stripped.
LINKS = [
    m.partition("#")[0]
    for m in re.findall(r"\]\(([^)]+)\)", README.read_text(encoding="utf-8"))
    if not m.startswith("#")
]


def test_the_readme_has_no_relative_links() -> None:
    """They render as dead links on PyPI, which is where this file is read."""
    relative = [link for link in LINKS if not link.startswith(("http", "mailto:"))]

    assert relative == [], f"{len(relative)} relative link(s) would 404 on PyPI: {relative[:5]}"


@pytest.mark.parametrize(
    "link", sorted({link for link in LINKS if "github.com/Aniketh-74/Agent-Meter" in link})
)
def test_every_repo_link_points_at_something_that_exists(link: str) -> None:
    """Absolute URLs cannot be checked by the filesystem, so check the path part.

    Converting a relative link to an absolute one moves the failure from
    "obviously broken" to "broken only when someone clicks it". This keeps the
    target verifiable without a network call: the URL is decomposed back to a
    repo path and that path must exist here.
    """
    path = re.sub(r"^https://github\.com/Aniketh-74/Agent-Meter/(?:blob|tree)/main/", "", link)
    assert (REPO / path.rstrip("/")).exists(), f"{link} points at a path not in this repo"


def test_the_status_banner_matches_the_packaged_version() -> None:
    """`pip install optio` shows this banner above everything else.

    It said 0.1.0 while ``pyproject`` said 0.2.0, which would have put a wrong
    version on the first line of the landing page. The two are now checked
    against each other rather than kept in step by hand.
    """
    # Read rather than parsed: `tomllib` is 3.11+ and this suite runs on 3.10,
    # where importing it fails outright. One regex against one well-known line
    # is cheaper than taking a `tomli` dependency for a single lookup.
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    declared_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert declared_match is not None, "pyproject.toml has no top-level version"
    declared = declared_match.group(1)
    banner = re.search(r"Status:\s*\w+\s*\(([^)]+)\)", README.read_text(encoding="utf-8"))

    assert banner is not None, "the README status banner is missing or reworded"
    assert banner.group(1) == declared, (
        f"README banner says {banner.group(1)}, pyproject says {declared}"
    )
