"""End-to-end proof of SC-4: a broken lane cannot break the agent.

The unit-level tests prove ``guard`` returns a fallback. That is necessary but
not sufficient -- the claim users actually rely on is that an agent loop with a
thoroughly broken lane installed still produces correct results. These tests
simulate that loop.

The lanes here fail in the ways real lanes fail: on every call, on some calls,
on the first call only, and at run end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from optio import semconv
from optio.config import default_config
from optio.errors import LedgerInvariantError, StateStoreError
from optio.lanes.base import Lane, Signal
from optio.runtime import failopen

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.failinject


@pytest.fixture(autouse=True)
def _clean_activations() -> Iterator[None]:
    """Isolate activation counters between tests."""
    failopen.reset_activations()
    yield
    failopen.reset_activations()


class _AlwaysBrokenLane(Lane):
    """A lane that raises on every single call."""

    name = "always_broken"

    def process_span(self, span: object, run: object) -> list[Signal]:
        """Raise unconditionally."""
        raise LedgerInvariantError("reserve without reconcile")

    def on_run_end(self, run: object) -> list[Signal]:
        """Raise unconditionally."""
        raise StateStoreError("store vanished at run end")


class _IntermittentLane(Lane):
    """A lane that fails on every third span -- the harder bug to notice."""

    name = "intermittent"

    def __init__(self) -> None:
        """Start the call counter."""
        super().__init__(default_config())
        self.calls = 0

    def process_span(self, span: object, run: object) -> list[Signal]:
        """Raise on every third call."""
        self.calls += 1
        if self.calls % 3 == 0:
            raise ValueError("intermittent lane bug")
        return [Signal(name=semconv.RUN_ACTUAL_COST, value=0.01)]

    def on_run_end(self, run: object) -> list[Signal]:
        """Yield nothing."""
        return []


def _run_agent_loop(lane: Lane, steps: int = 10) -> list[str]:
    """Simulate an agent loop with a lane installed on the critical path.

    Every lane call goes through the guard, exactly as the span tap will do in
    M1-3. The "agent work" is a plain computation whose results are asserted --
    if the guard leaks, this function raises and the results are lost.

    Args:
        lane: The (broken) lane to install.
        steps: How many agent steps to simulate.

    Returns:
        The agent's own results, one per step.
    """
    results: list[str] = []
    for i in range(steps):
        # The agent does its real work.
        results.append(f"step-{i}-result")
        # optio observes, behind the guard.
        failopen.guard_signals(lane.process_span, object(), object(), component=lane.name)
    failopen.guard_signals(lane.on_run_end, object(), component=lane.name)
    return results


def test_agent_completes_with_a_totally_broken_lane() -> None:
    """Ten steps, a lane that raises every time, and the agent is unaffected."""
    results = _run_agent_loop(_AlwaysBrokenLane(default_config()), steps=10)

    assert results == [f"step-{i}-result" for i in range(10)]
    # 10 spans + 1 run end.
    assert failopen.activation_count("always_broken") == 11


def test_agent_completes_with_an_intermittent_lane() -> None:
    """Partial failure degrades signals, never the agent."""
    lane = _IntermittentLane()
    results = _run_agent_loop(lane, steps=9)

    assert results == [f"step-{i}-result" for i in range(9)]
    assert lane.calls == 9
    # Steps 3, 6, 9 failed; the other six produced signals.
    assert failopen.activation_count("intermittent") == 3


def test_broken_lane_yields_no_signals_rather_than_wrong_ones() -> None:
    """A failed lane returns the fallback, not a fabricated value.

    Absence is a valid signal state. Emitting a zero here would be worse than
    emitting nothing, because a downstream policy cannot distinguish a real
    zero cost from a broken lane (ADR-004, docs/signals.md).
    """
    lane = _AlwaysBrokenLane(default_config())
    signals = failopen.guard_signals(lane.process_span, object(), object(), component=lane.name)

    assert signals == []


def test_lane_constructor_failure_is_contained() -> None:
    """A lane that cannot even be built must not break instrumentation."""

    class _UnbuildableLane(Lane):
        name = "unbuildable"

        def __init__(self) -> None:
            raise RuntimeError("lane constructor exploded")

        def process_span(self, span: object, run: object) -> list[Signal]:
            """Never reached."""
            return []

    built: Any = failopen.guard(_UnbuildableLane, None, component="unbuildable")

    assert built is None
    assert failopen.activation_count("unbuildable") == 1
