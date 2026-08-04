"""A shared store that stops answering must produce silence, not a number.

In memory, degrading meant one process's data -- complete for that process. On
a shared store it means the *other* processes' spend is invisible, so a
``budget_remaining`` computed from what this process happened to see could
report a full budget for a run that is already overspent. That is a wrong
number rather than a missing one, and this project treats those differently.

**No new code implements this**, which is the finding worth recording. The
existing fail-open guard (ADR-004) already absorbs any ``Exception`` from a
lane, returns the empty signal list, warns once per component and counts the
rest for the operator metric -- and ``StoreUnavailableError`` is a
``StateStoreError``, so it lands there like everything else. Adding a second
layer inside ``CostLane`` would have duplicated a guarantee that already
existed.

What did not exist was proof. These tests are that: they go through
``guard_signals``, the same call the span tap makes, because asserting on
``CostLane.on_run_end`` directly would test a path no agent takes.
"""

from __future__ import annotations

import logging

import pytest

from optio.config import default_config
from optio.lanes.cost.lane import CostLane
from optio.lanes.cost.ledger import CostLedger
from optio.lanes.cost.ledger_store import LedgerSnapshot
from optio.runtime import failopen
from optio.runtime.failopen import guard_signals
from optio.store.redis_client import StoreUnavailableError

pytestmark = pytest.mark.failinject


class _StubRun:
    """A minimal RunLike."""

    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.budget = None


class _DeadStore:
    """Every operation fails the way an unreachable Redis does."""

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        raise StoreUnavailableError("redis unavailable")

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        raise StoreUnavailableError("redis unavailable")

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        raise StoreUnavailableError("redis unavailable")

    def close_run(self, run_id: str) -> LedgerSnapshot:
        raise StoreUnavailableError("redis unavailable")

    def is_finalised(self, run_id: str) -> bool:
        raise StoreUnavailableError("redis unavailable")

    def knows(self, run_id: str) -> bool:
        raise StoreUnavailableError("redis unavailable")

    def evict(self, run_id: str) -> None:
        raise StoreUnavailableError("redis unavailable")

    def run_count(self) -> int:
        raise StoreUnavailableError("redis unavailable")


@pytest.fixture(autouse=True)
def _reset_failopen_state():
    """Clear the guard's once-per-component log memory between tests.

    It is process-global by design -- the point is that a lane failing on every
    span logs once, not once per span -- so a test asserting on it has to start
    from a known state or it passes because an earlier test already logged.
    """
    failopen._logged_components.clear()
    failopen._activations.clear()
    yield
    failopen._logged_components.clear()
    failopen._activations.clear()


def _dead_lane() -> CostLane:
    return CostLane(default_config(), ledger=CostLedger(store=_DeadStore()))


class TestAnUnreachableStoreEmitsNothing:
    def test_run_end_produces_no_signal(self) -> None:
        """Not a zero, not a full budget -- nothing.

        ``budget_remaining`` is the dangerous one: the value that would be
        emitted from an empty view is exactly the value that guarantees
        ``deny if budget_remaining < X`` never fires.
        """
        lane = _dead_lane()

        signals = guard_signals(lane.on_run_end, _StubRun(), component=lane.name)

        assert signals == [], f"emitted {[s.name for s in signals]} from a store it could not read"

    def test_the_agent_is_not_broken(self) -> None:
        """The whole guarantee: a store outage costs a gap in a graph, never a
        failed agent run (ADR-004)."""
        lane = _dead_lane()

        # Must not raise.
        guard_signals(lane.on_run_end, _StubRun(), component=lane.name)

    def test_the_failure_is_counted_for_the_operator(self) -> None:
        """A lane that fails on every span is invisible by construction -- the
        agent keeps working -- so the counter is how it becomes visible."""
        lane = _dead_lane()

        for _ in range(3):
            guard_signals(lane.on_run_end, _StubRun(), component=lane.name)

        assert failopen._activations[("cost", "StoreUnavailableError")] == 3


class TestItWarnsOnceRatherThanEveryStep:
    def test_a_repeated_outage_logs_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """Silence must not be mistaken for zero spend -- but a warning in a hot
        loop is one people filter out, so it fires once and then counts."""
        lane = _dead_lane()

        with caplog.at_level(logging.WARNING, logger="optio"):
            for _ in range(5):
                guard_signals(lane.on_run_end, _StubRun(), component=lane.name)

        assert len(caplog.records) == 1

    def test_the_warning_never_carries_the_exception_payload(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Section 10: an exception message can carry a connection string, and
        a connection string can carry a password."""
        lane = CostLane(
            default_config(),
            ledger=CostLedger(store=_DeadStore()),
        )

        with caplog.at_level(logging.WARNING, logger="optio"):
            guard_signals(lane.on_run_end, _StubRun(), component=lane.name)

        message = caplog.records[0].getMessage()
        assert "redis unavailable" not in message
        assert "StoreUnavailableError" in message
