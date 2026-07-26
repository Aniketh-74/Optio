"""Internal exception types.

Two families with opposite escape rules (Section 4.2):

* :class:`AgentMeterInternalError` and its subclasses are raised *inside*
  ``runtime/`` and ``lanes/``. They are caught at the ``failopen.py`` boundary and
  **never** reach user code. If one of these escapes to an agent, that is a
  violation of ADR-004 and a critical bug (R-TECH-4).
* :class:`AgentMeterConfigError` is the opposite: it is raised at ``instrument()``
  time, loudly, so misconfiguration surfaces at setup rather than degrading
  silently at runtime. It is *not* caught by the fail-open guard.
"""

from __future__ import annotations


class AgentMeterError(Exception):
    """Base class for every exception this library defines."""


class AgentMeterConfigError(AgentMeterError):
    """Invalid user-supplied configuration.

    Raised at setup time only (``instrument()``, ``Config`` construction). Fail
    loudly here so the user never ships a silently-disabled meter.
    """


class UnsupportedFrameworkError(AgentMeterConfigError):
    """The target object matches no known adapter."""


class AgentMeterInternalError(AgentMeterError):
    """Base for errors that must be absorbed by the fail-open guard.

    Anything raised inside ``runtime/`` or ``lanes/`` should derive from this so
    the guard's intent is explicit. The guard nonetheless catches
    :class:`Exception` broadly -- a lane bug raising a plain ``KeyError`` must not
    break the agent either (ADR-004).
    """


class LedgerInvariantError(AgentMeterInternalError):
    """The reserve/reconcile invariant was violated (R-TECH-1).

    Raised on a double reconcile or a reconcile without a matching reservation.
    Absorbed by the guard and logged at WARN: the cost signal for that step is
    dropped, the agent proceeds.
    """


class StateStoreError(AgentMeterInternalError):
    """The state store was unreachable or returned an unusable value."""


class SignalWriteError(AgentMeterInternalError):
    """A signal could not be written to the active span."""
