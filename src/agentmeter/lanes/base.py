"""The Lane contract.

A lane is a pure signal computation: spans and run-end events in, signals out. A
lane never enforces (ADR-001), never performs synchronous network I/O on the hot
path (Section 4.1), and -- critically -- **never imports another lane**
(Section 3.1, rule 11). That independence is what lets cost (M2), behavior (M3),
and quality (M5) ship, fail, and be tested separately.

Lanes are permitted to raise. Every call site wraps them in the fail-open guard,
which converts any exception into a dropped signal (ADR-004). Lane authors should
therefore write straightforward code and let the guard handle the pathological
case, rather than scattering defensive try/except blocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from agentmeter.config import BudgetPolicy, Config


@runtime_checkable
class RunLike(Protocol):
    """The slice of a run that lanes are allowed to see.

    Lanes sit *below* the runtime in the dependency graph (Section 3.1), so they
    must not import ``runtime.run_context`` -- not even under ``TYPE_CHECKING``,
    since the layering rule is about coupling, not about what survives to
    runtime. This structural protocol lets a lane be typed against a run without
    depending on the concrete class, and keeps lanes testable with a plain stub.
    """

    @property
    def run_id(self) -> str:
        """Stable unique identifier for the run."""
        ...

    @property
    def budget(self) -> BudgetPolicy | None:
        """Optional per-run spend limit."""
        ...


@dataclass(frozen=True, slots=True)
class Signal:
    """One emitted value, named by a ``semconv`` constant.

    Attributes:
        name: The attribute name. Must come from :mod:`agentmeter.semconv`,
            never a string literal (rule 5).
        value: The value. Restricted to OTel attribute types.
    """

    name: str
    value: bool | int | float | str


class Lane(ABC):
    """Base class for the cost, behavior, and quality lanes.

    Attributes:
        config: The active configuration.
    """

    #: Short lane name used in logs and self-metrics.
    name: str = "lane"

    def __init__(self, config: Config) -> None:
        """Store configuration.

        Args:
            config: Active configuration.
        """
        self.config = config

    @abstractmethod
    def process_span(self, span: ReadableSpan, run: RunLike) -> list[Signal]:
        """Compute signals from one span.

        Called on the hot path, once per GenAI span. Must not block.

        Args:
            span: The span the framework just produced.
            run: The run the span belongs to.

        Returns:
            Signals to write. Empty when this span yields nothing.
        """

    def on_run_end(self, run: RunLike) -> list[Signal]:
        """Compute signals available only once the run is complete.

        Args:
            run: The run that just ended.

        Returns:
            Signals to write. Empty by default.
        """
        return []

    def __repr__(self) -> str:
        """Return a debug representation naming the lane."""
        return f"<{type(self).__name__} name={self.name!r}>"


def enabled_lanes(config: Config) -> list[Any]:
    """Return the lane instances enabled by configuration.

    M0 returns nothing -- lanes land in M2, M3, and M5. Centralising the wiring
    here keeps ``runtime`` free of concrete lane imports (Section 3.1).

    Args:
        config: Active configuration.

    Returns:
        Enabled lane instances, in dispatch order.
    """
    return []
