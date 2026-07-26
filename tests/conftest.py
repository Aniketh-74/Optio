"""Shared fixtures.

Run state lives in a :class:`~contextvars.ContextVar`, which is process-wide and
survives across tests. A test that starts a run without ending it -- several do,
deliberately, to assert lifecycle states -- would otherwise leave that run
"current" for every test that follows, producing failures far from their cause.

The autouse fixture below pins each test to a clean run context so tests cannot
leak into one another.
"""

from __future__ import annotations

import pytest

from agentmeter.runtime import run_context


@pytest.fixture(autouse=True)
def _isolated_run_context():
    """Reset the current-run ContextVar around every test."""
    token = run_context._current_run.set(None)
    try:
        yield
    finally:
        run_context._current_run.reset(token)


@pytest.fixture(autouse=True)
def _clean_agentmeter_env(monkeypatch):
    """Remove ``AGENTMETER_*`` variables so a developer's shell cannot alter results."""
    for key in list(__import__("os").environ):
        if key.startswith("AGENTMETER_"):
            monkeypatch.delenv(key, raising=False)
