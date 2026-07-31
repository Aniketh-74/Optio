"""Property tests assert invariants, not wall-clock time.

Hypothesis applies a 200 ms per-example deadline by default and fails the test
when an example exceeds it. That turns "the machine was busy" into "the cost
ledger is broken", which is a false alarm on the most safety-critical invariant
in this codebase -- and it happened: ``test_partial_reconciliation_leaves_the
_rest_reserved`` failed once during a run that shared the machine with a package
build, then passed on replay with its own recorded seed on an idle machine.

Two tests in ``test_ledger_invariant.py`` already carried
``@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])``.
That is the same fix applied twice, per test, as each one happened to flake --
which leaves every property test that has not flaked *yet* exposed. Registering
it once for the directory is the same decision made in one place.

Timing is measured deliberately elsewhere: the ``bench`` marker covers the
overhead budget (SC-5), against a fixed workload rather than against
Hypothesis-generated input whose size varies per example.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "property",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("property")
