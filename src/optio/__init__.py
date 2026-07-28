"""optio -- economic cost and outcome quality signals for agent runs.

Emits per-run cost, behavioral health, and outcome quality as OpenTelemetry
GenAI span attributes and metrics, so the policy engine and observability backend
you already run can reason about money and quality -- not just permission.

optio emits signals; it never enforces (ADR-001) and never breaks the agent
(ADR-004).

Typical use::

    from optio import instrument

    instrument(agent)

This module is the entire supported surface: the public API is exactly the
names in ``__all__`` below (ADR-012). Everything under ``optio.runtime``,
``optio.lanes``, ``optio.store`` and ``optio.adapters`` is internal and may
change in any release, including a patch -- their docstrings and type
annotations are there for contributors, not as a stability promise. If you need
something that is only reachable through a submodule, please open an issue so
it can be promoted deliberately.

Framework extras (``optio[langgraph]`` and friends) are deliberately *not*
imported here: core import must stay dependency-light and fast (Section 11).
"""

from __future__ import annotations

from optio.api import instrument, meter
from optio.config import BudgetPolicy, Config
from optio.runtime.run_context import RunContext, current_run
from optio.semconv import GENAI_SEMCONV_VERSION

#: Declared for type checkers, which cannot see through `__getattr__`. An
#: annotation without a value binds no attribute at runtime, so this does not
#: put `__version__` in the namespace eagerly -- and importing TYPE_CHECKING to
#: guard it would itself add a public `optio.TYPE_CHECKING`.
__version__: str


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` on first access (PEP 562).

    The version is read from installed metadata rather than hardcoded: a
    literal here is a second source of truth that silently drifts from
    pyproject.toml the first time someone bumps one and not the other.

    Reading it *lazily* is what matters for import cost. ``importlib.metadata``
    pulls in ``zipfile``, ``email.message``, ``pathlib``, and ``inspect``, and
    measured at 88 ms of optio's 211 ms import -- 42% of it, spent on a string
    that most programs never read. Since this is a library, every user pays
    that at startup whether they touch ``__version__`` or not.
    """
    if name == "__version__":
        import importlib.metadata

        try:
            version = importlib.metadata.version("optio")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover -
            # only when running from a source tree with no install, where there
            # is no metadata to read.
            version = "0.0.0.dev0"
        # Cache on the module so repeat access skips the lookup entirely.
        globals()["__version__"] = version
        return version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the module's attributes, including the lazy ones (PEP 562).

    Without this, ``__version__`` is missing from ``dir(optio)`` until someone
    happens to access it -- so tab-completion and introspection would not show
    a name that ``__all__`` advertises.
    """
    return sorted({*globals(), *__all__})


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
