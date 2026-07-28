"""Framework adapters -- optional, one per integration target.

Every adapter here wraps a real, framework-supplied extension point rather
than reimplementing part of the framework: this package intercepts a call
that was always going to happen, it does not become a second way to drive
the framework. See each adapter's module docstring for its specific
interception point and why it was chosen over the alternatives.
"""

from __future__ import annotations
