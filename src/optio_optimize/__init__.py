"""optio_optimize -- token and cost reduction for LLM calls.

The acting half of optio. Where :mod:`optio` observes a run and emits signals,
this package sits in the request path and changes what gets sent, so the cost it
saves is real rather than merely visible.

That is a materially different bargain and the package states it plainly:

* **It reads prompt content.** It cannot compress, trim, or cache what it may
  not look at. :mod:`optio`'s §10 rule -- never read prompt content -- describes
  :mod:`optio`, and does not and cannot describe this package (ADR-013).
* **It is on your critical path.** A failure here is absorbed and the original
  request passes through untouched, but the code is in the path either way.
* **Some stages change the answer.** Every one is off by default and named in
  :attr:`~optio_optimize.config.OptimizeConfig.lossy_enabled` when on.

Typical use::

    from optio_optimize import Optimizer

    optimizer = Optimizer()
    response = optimizer.call(request, provider_fn)
    for line in optimizer.report.summary_lines("gpt-4o"):
        print(line)

Installing :mod:`optio` alone never brings this in. Nothing in :mod:`optio`
imports it, and an import-linter contract keeps it that way.
"""

from __future__ import annotations

from optio_optimize.config import OptimizeConfig, config_from_mapping
from optio_optimize.errors import OptimizeConfigError, OptimizeError
from optio_optimize.optimizer import Optimizer
from optio_optimize.savings import SavingsReport, StageSaving
from optio_optimize.types import LLMRequest, LLMResponse, Message

#: Declared for type checkers; resolved lazily by ``__getattr__`` (ADR-012's
#: pattern, and the same 88 ms import saving applies here).
__version__: str


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` on first access (PEP 562)."""
    if name == "__version__":
        import importlib.metadata

        try:
            version = importlib.metadata.version("optio")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            version = "0.0.0.dev0"
        globals()["__version__"] = version
        return version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List attributes including the lazy ones (PEP 562)."""
    return sorted({*globals(), *__all__})


__all__ = [
    "LLMRequest",
    "LLMResponse",
    "Message",
    "OptimizeConfig",
    "OptimizeConfigError",
    "OptimizeError",
    "Optimizer",
    "SavingsReport",
    "StageSaving",
    "__version__",
    "config_from_mapping",
]
