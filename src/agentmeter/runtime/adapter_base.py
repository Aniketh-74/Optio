"""The adapter contract (Section 6.7).

An adapter does one thing: make sure the target framework emits GenAI spans and
that the tap is installed to receive them. It computes no signals and holds no
lane logic -- that separation is what keeps a framework release from breaking
anything except its own adapter (R-TECH-3).

Adapters fail **loudly**, unlike the runtime. A missing framework or a version
mismatch is a setup error, and silently instrumenting nothing would leave the
user believing they have coverage they do not have (Section 4.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from agentmeter.config import Config


class Adapter(ABC):
    """Framework-specific wiring for the span tap.

    Attributes:
        name: Short adapter name, matching the ``adapter=`` argument users pass
            to :func:`~agentmeter.instrument`.
    """

    name: str = "adapter"

    @abstractmethod
    def matches(self, target: object) -> bool:
        """Whether this adapter handles the given object.

        Used for duck-typed resolution when the user does not name an adapter.
        Must not import the framework: a probe that imports every candidate
        framework would make resolution cost proportional to the number of
        adapters we ship, on a machine that has only one of them installed.

        Args:
            target: The object passed to ``instrument()``.

        Returns:
            ``True`` if this adapter can instrument the target.
        """

    @abstractmethod
    def instrument(
        self, target: Any, config: Config, provider: TracerProvider | None = None
    ) -> Any:
        """Wire the target so its GenAI spans reach agentmeter.

        Args:
            target: The agent object to instrument.
            config: Active configuration.
            provider: Tracer provider to install the tap on. Defaults to the
                global one; injectable because OTel's global provider is
                write-once per process.

        Returns:
            The same object that was passed in.

        Raises:
            AgentMeterConfigError: If the framework is missing or unsupported.
        """
