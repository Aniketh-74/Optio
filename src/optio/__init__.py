"""optio -- economic cost and outcome quality signals for agent runs.

Emits per-run cost, behavioral health, and outcome quality as OpenTelemetry
GenAI span attributes and metrics, so the policy engine and observability backend
you already run can reason about money and quality -- not just permission.

optio emits signals; it never enforces (ADR-001) and never breaks the agent
(ADR-004).

Typical use::

    from optio import instrument

    instrument(agent)

This module is the entire supported surface. Importing from ``optio.runtime``,
``optio.lanes``, or ``optio.store`` is unsupported -- those are internal
and change without a major bump.

Framework extras (``optio[langgraph]`` and friends) are deliberately *not*
imported here: core import must stay dependency-light and fast (Section 11).
"""

from __future__ import annotations

import importlib.metadata as _metadata

from optio.api import instrument, meter
from optio.config import BudgetPolicy, Config
from optio.runtime.run_context import RunContext, current_run
from optio.semconv import GENAI_SEMCONV_VERSION

try:
    # Read from installed metadata rather than hardcoding: a literal here is a
    # second source of truth that silently drifts from pyproject.toml the first
    # time someone bumps one and not the other.
    #
    # Imported as a private alias so the names it binds do not become part of
    # the public surface -- `from optio import version` should not resolve.
    __version__ = _metadata.version("optio")
except _metadata.PackageNotFoundError:  # pragma: no cover - only when running
    # from a source tree with no install, where there is no metadata to read.
    __version__ = "0.0.0.dev0"

__all__ = [
    "GENAI_SEMCONV_VERSION",
    "BudgetPolicy",
    "Config",
    "RunContext",
    "__version__",
    "current_run",
    "instrument",
    "meter",
]
