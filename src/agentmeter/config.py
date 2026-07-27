"""Configuration and feature flags.

Precedence (Section 4.3): explicit ``instrument(...)`` kwargs > ``AGENTMETER_*``
environment variables > defaults.

Flag defaults are load-bearing, not arbitrary: ``quality_lane`` is **off** by
default (ADR-003, to protect the latency budget and avoid the who-evals-the-
evaluator cost trap) and ``store_backend`` is ``memory`` by default (ADR-005, so
first value needs zero new infrastructure -- SC-1).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, cast, get_args

from agentmeter.errors import AgentMeterConfigError

StoreBackend = Literal["memory", "redis"]

_ENV_PREFIX: Final = "AGENTMETER_"

#: Orphaned-run eviction TTL in seconds (Section 7.1).
DEFAULT_RUN_TTL_SECONDS: Final = 3600.0

_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise AgentMeterConfigError(
        f"{_ENV_PREFIX}{name}={raw!r} is not a boolean; use one of {sorted(_TRUTHY | _FALSY)}"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise AgentMeterConfigError(f"{_ENV_PREFIX}{name}={raw!r} is not a number") from exc


def _env_str(name: str, default: str) -> str:
    return os.environ.get(_ENV_PREFIX + name, default)


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable runtime configuration.

    Attributes:
        cost_lane: Emit cost signals (reserve/reconcile ledger, projection).
        behavior_lane: Emit loop/repeat/retry-storm signals.
        quality_lane: Emit outcome-quality signals. **Off by default** (ADR-003);
            enabling it opts into the sampled, tiered evaluator path.
        store_backend: Where per-run state lives. ``memory`` needs no infra.
        redis_url: Connection string used when ``store_backend`` is ``redis``.
        quality_sample_rate: Fraction of runs routed to the async LLM-judge when
            the quality lane is on. Must be within ``[0.0, 1.0]``.
        run_ttl_seconds: Eviction TTL for run state orphaned by a missing run-end.
        behavior_window_size: Maximum step signatures retained per run. Bounds
            memory under long runs (Section 11).
        judge: The user's outcome evaluator, used only when ``quality_lane`` is
            on and a run is sampled. **Not settable from the environment**, and
            deliberately so: agentmeter ships no default judge and constructs no
            model client, because either would mean spending the user's money
            and using their credentials on our initiative (Section 10, ADR-003).
            Left ``None``, the quality lane runs its inline heuristic only.

    Raises:
        AgentMeterConfigError: If any value is out of range. Raised at
            construction -- that is, at setup time, never on the hot path.
    """

    cost_lane: bool = True
    behavior_lane: bool = True
    quality_lane: bool = False
    store_backend: StoreBackend = "memory"
    redis_url: str | None = None
    quality_sample_rate: float = 0.1
    run_ttl_seconds: float = DEFAULT_RUN_TTL_SECONDS
    behavior_window_size: int = 50
    # Typed as a callable rather than importing the Judge protocol: config sits
    # below lanes in the layering (Section 3.1), so it must not import one.
    judge: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        """Validate at construction so bad config fails at setup, not at runtime."""
        if not 0.0 <= self.quality_sample_rate <= 1.0:
            raise AgentMeterConfigError(
                f"quality_sample_rate must be in [0.0, 1.0], got {self.quality_sample_rate}"
            )
        if self.run_ttl_seconds <= 0:
            raise AgentMeterConfigError(
                f"run_ttl_seconds must be positive, got {self.run_ttl_seconds}"
            )
        if self.behavior_window_size <= 0:
            raise AgentMeterConfigError(
                f"behavior_window_size must be positive, got {self.behavior_window_size}"
            )
        if self.store_backend not in ("memory", "redis"):
            raise AgentMeterConfigError(
                f"store_backend must be 'memory' or 'redis', got {self.store_backend!r}"
            )
        if self.store_backend == "redis" and not self.redis_url:
            raise AgentMeterConfigError("store_backend='redis' requires redis_url")

    @classmethod
    def from_env(cls) -> Config:
        """Build a config from ``AGENTMETER_*`` environment variables.

        Returns:
            A config whose fields fall back to the documented defaults for any
            variable that is unset.

        Raises:
            AgentMeterConfigError: If a variable is present but unparseable.
        """
        backend = _env_str("STORE_BACKEND", "memory")
        if backend not in get_args(StoreBackend):
            raise AgentMeterConfigError(
                f"{_ENV_PREFIX}STORE_BACKEND must be 'memory' or 'redis', got {backend!r}"
            )
        return cls(
            cost_lane=_env_bool("COST_LANE", True),
            behavior_lane=_env_bool("BEHAVIOR_LANE", True),
            quality_lane=_env_bool("QUALITY_LANE", False),
            # cast is sound: the membership check above narrows to the Literal.
            store_backend=cast("StoreBackend", backend),
            redis_url=os.environ.get(_ENV_PREFIX + "REDIS_URL"),
            quality_sample_rate=_env_float("QUALITY_SAMPLE_RATE", 0.1),
            run_ttl_seconds=_env_float("RUN_TTL_SECONDS", DEFAULT_RUN_TTL_SECONDS),
            behavior_window_size=int(_env_float("BEHAVIOR_WINDOW_SIZE", 50)),
        )

    def merged_with(self, **overrides: object) -> Config:
        """Return a copy with explicit keyword overrides applied.

        Implements the top tier of the Section 4.3 precedence chain: values passed
        to ``instrument()`` win over environment and defaults.

        Args:
            **overrides: Field names to override. ``None`` values are ignored so
                callers can forward optional kwargs without clobbering defaults.

        Returns:
            A new validated config.

        Raises:
            AgentMeterConfigError: If a name is not a config field, or the result
                fails validation.
        """
        known = {f.name for f in self.__dataclass_fields__.values()}
        applied = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(applied) - known
        if unknown:
            raise AgentMeterConfigError(
                f"unknown config option(s): {sorted(unknown)}; valid: {sorted(known)}"
            )
        return replace(self, **applied)  # type: ignore[arg-type]  # validated above


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Per-run spend limit used by the cost lane's projection.

    agentmeter never *enforces* this limit -- exceeding it is emitted as a signal
    and the downstream policy engine decides what to do (ADR-001).

    Attributes:
        limit_usd: Ceiling used to compute ``gen_ai.run.budget_remaining``.
        max_steps: Optional step ceiling used by the worst-case projection.
    """

    limit_usd: float
    max_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate the limit at construction (setup-time failure)."""
        if self.limit_usd <= 0:
            raise AgentMeterConfigError(f"limit_usd must be positive, got {self.limit_usd}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise AgentMeterConfigError(f"max_steps must be positive, got {self.max_steps}")

    @classmethod
    def parse(cls, spec: str | float | BudgetPolicy) -> BudgetPolicy:
        """Coerce a user-friendly budget spec into a policy.

        Accepts the documented ``budget="$0.50"`` form (Section 8.1), a bare
        float, or an already-built policy.

        Args:
            spec: ``"$0.50"``, ``"0.50"``, ``0.5``, or a :class:`BudgetPolicy`.

        Returns:
            The corresponding policy.

        Raises:
            AgentMeterConfigError: If the string is not a parseable amount.
        """
        if isinstance(spec, BudgetPolicy):
            return spec
        if isinstance(spec, (int, float)):
            return cls(limit_usd=float(spec))
        cleaned = spec.strip().lstrip("$").replace(",", "")
        try:
            return cls(limit_usd=float(cleaned))
        except ValueError as exc:
            raise AgentMeterConfigError(
                f"could not parse budget {spec!r}; expected e.g. '$0.50' or 0.5"
            ) from exc


def default_config() -> Config:
    """Return the environment-resolved default configuration.

    Returns:
        A config built from ``AGENTMETER_*`` variables and documented defaults.
    """
    return Config.from_env()
