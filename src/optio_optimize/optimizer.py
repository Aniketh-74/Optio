"""The public entry point.

One class, deliberately. The pipeline, stage registry and savings ledger are all
useful to a contributor and none of them are things a user should have to
assemble to get value -- so ``Optimizer()`` with no arguments is a working,
lossless, bounded-memory optimizer, and everything else is opt-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optio_optimize.config import OptimizeConfig, config_from_mapping
from optio_optimize.pipeline import Pipeline
from optio_optimize.stages import build_stages

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider

    from optio_optimize.pipeline import AsyncProviderCall, ProviderCall
    from optio_optimize.savings import SavingsReport
    from optio_optimize.stages.base import Stage
    from optio_optimize.types import LLMRequest, LLMResponse


class Optimizer:
    """Applies the configured optimizations around a provider call.

    Attributes:
        config: The active configuration.
    """

    __slots__ = ("_pipeline", "config")

    def __init__(
        self,
        config: OptimizeConfig | None = None,
        *,
        stages: list[Stage] | None = None,
        tracer_provider: TracerProvider | None = None,
        **overrides: Any,
    ) -> None:
        """Build an optimizer.

        Args:
            config: Full configuration. Built from defaults and ``overrides``
                when omitted.
            stages: Explicit stage list, bypassing the registry. For tests and
                for users who need an order the registry does not produce;
                ordering then becomes their responsibility.
            tracer_provider: Provider ``emit_spans`` draws its tracer from
                (ADR-014). Defaults to OTel's global provider; only needed to
                target a non-global one, e.g. in tests. Has no effect unless
                ``emit_spans=True``.
            **overrides: Individual config fields, e.g. ``semantic_cache=True``.

        Raises:
            OptimizeConfigError: On invalid configuration, or if both ``config``
                and ``overrides`` are given -- silently letting one win would
                make the effective settings unpredictable.
        """
        if config is not None and overrides:
            from optio_optimize.errors import OptimizeConfigError

            raise OptimizeConfigError(
                "pass either a config object or keyword overrides, not both; "
                f"got config plus {sorted(overrides)}"
            )
        self.config = config if config is not None else config_from_mapping(overrides)
        self._pipeline = Pipeline(
            self.config,
            stages if stages is not None else build_stages(self.config),
            tracer_provider=tracer_provider,
        )

    def call(
        self,
        request: LLMRequest,
        provider: ProviderCall,
        *,
        run_id: str | None = None,
    ) -> LLMResponse:
        """Optimize ``request``, invoke ``provider`` unless served from cache.

        Args:
            request: The call to make.
            provider: Function performing the real API call.
            run_id: optio run id, so savings attribute to the same run the cost
                lane is metering.

        Returns:
            The response. Identical to what ``provider`` would have returned,
            unless a lossy stage is enabled.
        """
        return self._pipeline.execute(request, provider, run_id=run_id)

    async def acall(
        self,
        request: LLMRequest,
        provider: AsyncProviderCall,
        *,
        run_id: str | None = None,
    ) -> LLMResponse:
        """The async twin of :meth:`call`, for an async provider function.

        Every stage runs synchronously either way -- none perform I/O -- so
        this differs from :meth:`call` only in awaiting ``provider``. It
        exists because the realistic targets for adapting this package to a
        framework are themselves async: the OpenAI Agents SDK's
        ``Model.get_response`` is ``async def`` and abstract, with no
        synchronous alternative, and the same is true of most modern agent
        runtimes.

        Args:
            request: The call to make.
            provider: Async function performing the real API call.
            run_id: optio run id, so savings attribute to the same run the cost
                lane is metering.

        Returns:
            The response. Identical to what ``provider`` would have returned,
            unless a lossy stage is enabled.
        """
        return await self._pipeline.aexecute(request, provider, run_id=run_id)

    @property
    def report(self) -> SavingsReport:
        """Cumulative savings across every request this optimizer has run."""
        return self._pipeline.report

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Names of the active stages, in execution order.

        Worth checking after construction: a stage silently absent because its
        config flag defaulted off is the most common reason someone reports
        that the library "does nothing".
        """
        return tuple(s.name for s in self._pipeline.stages)
