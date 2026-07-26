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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentmeter.config import Config
    from agentmeter.lanes.base import Lane


def enabled_lanes(config: Config) -> list[Lane]:
    """Return the lane instances enabled by configuration.

    Concrete lanes are imported inside the function so that a lane module which
    fails to import cannot take down the library at import time, and so the
    optional dependencies of a lane are only touched when it is enabled.

    Args:
        config: Active configuration.

    Returns:
        Enabled lane instances, in dispatch order. Quality lands in M5.
    """
    lanes: list[Lane] = []

    if config.cost_lane:
        from agentmeter.lanes.cost.lane import CostLane

        lanes.append(CostLane(config))

    if config.behavior_lane:
        from agentmeter.lanes.behavior.lane import BehaviorLane

        lanes.append(BehaviorLane(config))

    return lanes
