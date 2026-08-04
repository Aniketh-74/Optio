"""Lane wiring -- the one module that knows which concrete lanes exist.

This is deliberately *not* in :mod:`optio.lanes.base`. Putting the wiring
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
    from optio.config import Config
    from optio.lanes.base import Lane
    from optio.lanes.behavior.store import BehaviorStore
    from optio.lanes.cost.ledger import CostLedger
    from optio.lanes.quality.judge import Judge


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
        from optio.lanes.quality.lane import QualityLane

        # `Config.judge` is typed loosely (`Callable[[Any], Any]`) because
        # config sits below lanes in the layering and must not import the
        # concrete `Judge` type. The cast re-narrows it here, at the one place
        # that legitimately knows both sides. A judge with the wrong signature
        # fails inside the runner, which treats it as a judge that declined --
        # a missing signal, never an agent error (ADR-004).
        lanes.append(QualityLane(config, judge=cast("Judge | None", config.judge)))

    if config.cost_lane:
        from optio.lanes.cost.lane import CostLane

        lanes.append(CostLane(config, ledger=_ledger(config)))

    if config.behavior_lane:
        from optio.lanes.behavior.lane import BehaviorLane

        lanes.append(BehaviorLane(config, store=_behavior_store(config)))

    return lanes


def _ledger(config: Config) -> CostLedger:
    """Build the ledger over the configured backend.

    An unreachable Redis raises **here**, at setup, rather than on the agent's
    path. A backend that cannot be reached at wiring time is a configuration
    error and Section 4.2 says those fail loudly; a backend that stops
    answering *later* is a runtime condition, and the lane fails open on it.

    Args:
        config: Active configuration.

    Returns:
        A ledger over the in-memory backend, or over Redis when configured.

    Raises:
        StoreUnavailableError: If ``store_backend='redis'`` and the server does
            not answer at setup.
    """
    from optio.lanes.cost.ledger import CostLedger

    if config.store_backend != "redis":
        return CostLedger()

    from optio.lanes.cost.ledger_redis import RedisLedgerStore
    from optio.store.redis_client import RedisClient

    # `redis_url` is guaranteed non-empty: Config rejects the combination at
    # construction, so the fallback here is unreachable rather than a default.
    client = RedisClient(config.redis_url or "", timeout_ms=config.store_timeout_ms)
    client.ping()
    return CostLedger(
        store=RedisLedgerStore(
            client,
            ttl_seconds=config.run_ttl_seconds,
            # Finality has to outlive the state it was derived from, or a late
            # callback finds nothing and starts a second total under a run id
            # that was already reported.
            tombstone_ttl_seconds=config.run_ttl_seconds * 10,
        )
    )


def _behavior_store(config: Config) -> BehaviorStore:
    """Build the window store over the configured backend.

    Mirrors :func:`_ledger`, including where it raises: an unreachable Redis is
    a configuration error and fails **here**, at setup (Section 4.2), rather
    than on the agent's path. A backend that stops answering later is a runtime
    condition and the lane fails open on it.

    Each lane opens its **own** client rather than sharing one. They have
    independent lifecycles by contract (Section 3.1) -- either can be disabled
    without the other noticing -- and a shared connection would make the
    behaviour lane's timeout budget depend on whether the cost lane happened to
    be enabled.

    Args:
        config: Active configuration.

    Returns:
        The process-local backend, or the Redis one when configured.

    Raises:
        StoreUnavailableError: If ``store_backend='redis'`` and the server does
            not answer at setup.
    """
    from optio.lanes.behavior.store_memory import InMemoryBehaviorStore

    if config.store_backend != "redis":
        return InMemoryBehaviorStore()

    from optio.lanes.behavior.store_redis import RedisBehaviorStore
    from optio.store.redis_client import RedisClient

    # `redis_url` is guaranteed non-empty: Config rejects the combination at
    # construction, so the fallback here is unreachable rather than a default.
    client = RedisClient(config.redis_url or "", timeout_ms=config.store_timeout_ms)
    client.ping()
    # No tombstone counterpart. Closing a window is not final -- a re-opened one
    # is short again, so the lane under-reports rather than misreporting, which
    # is the direction Section 6.4 requires.
    return RedisBehaviorStore(client, ttl_seconds=config.run_ttl_seconds)
