"""Exceptions for the optimization package.

Only setup errors are public. There is deliberately no runtime exception type:
ADR-013 rule 1 makes every runtime failure a *skipped stage*, so a runtime error
class would advertise a control flow the pipeline does not offer.
"""

from __future__ import annotations


class OptimizeError(Exception):
    """Base class for this package's errors."""


class OptimizeConfigError(OptimizeError, ValueError):
    """Invalid configuration, raised at construction.

    Subclasses ``ValueError`` so that callers already catching bad arguments
    keep working, matching ``optio.errors.OptioConfigError`` in the core.
    """
