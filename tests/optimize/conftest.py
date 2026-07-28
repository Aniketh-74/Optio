"""Shared fixtures for optio_optimize tests.

``pytest.importorskip("openai")``/``("anthropic")`` in ``test_providers.py``
and ``test_adapters_openai_agents.py`` let a contributor without those
packages installed still run the suite. Both are pinned in the ``dev`` extra
specifically so that skip should never actually happen -- see
``pyproject.toml``. ``pytest_configure`` below is the hard-fail half of that
guarantee: with ``OPTIO_REQUIRE_PROVIDER_SDKS=1`` (set in CI), a missing
package fails the whole session immediately instead of letting every
mocked-SDK test quietly report "skipped", the same "a gate that can pass by
skipping is not a gate" reasoning ``tests/frameworks/`` and ``tests/policy/``
already apply to their own optional dependencies.

A session hook rather than a per-file check: doing this inline in each test
file (calling a custom gate function before the file's other imports) trips
ruff's E402 rule, because ruff's module-import-not-at-top-of-file check
special-cases the exact call ``pytest.importorskip(...)`` and nothing else --
confirmed empirically, not documented. Keeping every file's own gate as a
plain ``importorskip`` call sidesteps that, and puts the one-time hard-fail
check in exactly one place.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

#: Set in CI. Turns "package failed to install" from a silent skip into a
#: hard failure -- a gate that can pass by skipping is not a gate.
REQUIRE_PROVIDER_SDKS = os.environ.get("OPTIO_REQUIRE_PROVIDER_SDKS") == "1"

#: Both pinned in the dev extra (pyproject.toml) purely so the mocked-SDK
#: provider tests below can build real request/response objects; neither is
#: imported by any runtime code path.
_REQUIRED_MODULES = ("openai", "anthropic")


def pytest_configure(config: pytest.Config) -> None:
    """Fail collection outright if a pinned provider SDK did not install."""
    if not REQUIRE_PROVIDER_SDKS:
        return
    missing = [m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise pytest.UsageError(
            f"{', '.join(missing)} not installed but OPTIO_REQUIRE_PROVIDER_SDKS=1. "
            f"Both are pinned in the dev extra precisely so the mocked-SDK provider "
            f"tests run rather than skip; a missing import here means the install "
            f"step broke, not that the dependency is optional."
        )
