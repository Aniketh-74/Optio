"""Lane wiring -- the one module that knows which concrete lanes exist.

This is deliberately *not* in :mod:`agentmeter.lanes.base`. Putting the wiring
beside the ABC made every lane a transitive importer of every other lane
(``cost.lane -> lanes.base -> behavior.lane``), which breaks the independence
contract in Section 3.1 even though no lane refers to another.

That is not a linter technicality. The contract exists so the three lanes can
ship, fail, and be tested separately -- and a cycle through the shared base
module is exactly how that property erodes without anyone deciding to give it
up. Keeping the ABC free of concrete imports means a lane depends only on the
abstraction, and this module is the single edge where the concrete set is known.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from agentmeter.config import Config
    from agentmeter.lanes.base import Lane
    from agentmeter.lanes.quality.judge import Judge


def enabled_lanes(config: Config) -> list[Lane]:
    """Return the lane instances enabled by configuration.

    Concrete lanes are imported inside the function so that a lane module which
    fails to import cannot take down the library at import time, and so the
    optional dependencies of a lane are only touched when it is enabled.

    **Order is load-bearing at run end.** The tap dispatches lanes in list
    order, and ``cost_per_successful_task`` is a cost x quality signal: the cost
    lane reads a success count that the quality lane records. With cost first,
    that count is always one run stale -- the signal would be computed from the
    *previous* run's outcome, which is wrong rather than absent, and silently so.
    So quality is placed ahead of cost.

    This is the only inter-lane ordering dependency in the system, and it does
    not breach the independence contract (Section 3.1): neither lane imports the
    other, and each still produces its own signals correctly in isolation. What
    the ordering buys is one *derived* signal. If that ever grows into a second
    case, the honest fix is an explicit two-phase run-end rather than more
    implicit ordering.

    Args:
        config: Active configuration.

    Returns:
        Enabled lane instances, in dispatch order.
    """
    lanes: list[Lane] = []

    # Quality first -- see the ordering note above. Off by default (ADR-003), so
    # for most users this branch does nothing and costs one boolean check.
    if config.quality_lane:
        from agentmeter.lanes.quality.lane import QualityLane

        # `Config.judge` is typed loosely (`Callable[[Any], Any]`) because
        # config sits below lanes in the layering and must not import the
        # concrete `Judge` type. The cast re-narrows it here, at the one place
        # that legitimately knows both sides. A judge with the wrong signature
        # fails inside the runner, which treats it as a judge that declined --
        # a missing signal, never an agent error (ADR-004).
        lanes.append(QualityLane(config, judge=cast("Judge | None", config.judge)))

    if config.cost_lane:
        from agentmeter.lanes.cost.lane import CostLane

        lanes.append(CostLane(config))

    if config.behavior_lane:
        from agentmeter.lanes.behavior.lane import BehaviorLane

        lanes.append(BehaviorLane(config))

    return lanes
