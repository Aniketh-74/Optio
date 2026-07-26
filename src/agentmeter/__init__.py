"""agentmeter -- economic cost and outcome quality signals for agent runs.

Emits per-run cost, behavioral health, and outcome quality as OpenTelemetry
GenAI span attributes and metrics, so the policy engine and observability backend
you already run can reason about money and quality -- not just permission.

agentmeter emits signals; it never enforces (ADR-001) and never breaks the agent
(ADR-004).

Typical use::

    from agentmeter import instrument

    instrument(agent)

This module is the entire supported surface. Importing from ``agentmeter.runtime``,
``agentmeter.lanes``, or ``agentmeter.store`` is unsupported -- those are internal
and change without a major bump.

Framework extras (``agentmeter[langgraph]`` and friends) are deliberately *not*
imported here: core import must stay dependency-light and fast (Section 11).
"""

from __future__ import annotations

from agentmeter.api import instrument, meter
from agentmeter.config import BudgetPolicy, Config
from agentmeter.runtime.run_context import RunContext, current_run
from agentmeter.semconv import GENAI_SEMCONV_VERSION

__version__ = "0.1.0.dev0"

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
