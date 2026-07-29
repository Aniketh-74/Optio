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

    from optio_optimize.pipeline import AsyncProviderCall, Pipeline, ProviderCall
    from optio_optimize.savings import SavingsReport
    from optio_optimize.stages.base import Stage
    from optio_optimize.stages.diagnostics import PrefixFinding
    from optio_optimize.stages.semantic_cache import SimilarityFn
    from optio_optimize.stages.summarize import Summarizer
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
        summarizer: Summarizer | None = None,
        similarity_fn: SimilarityFn | None = None,
        **overrides: Any,
    ) -> None:
        """Build an optimizer.

        Args:
            config: Full configuration. Built from defaults and ``overrides``
                when omitted.
            stages: Explicit stage list, bypassing the registry. For tests and
                for users who need an order the registry does not produce;
                ordering then becomes their responsibility. Also the escape
                hatch past the ``summarizer`` requirement below, for a caller
                who wants to construct ``SummarizeHistoryStage`` themselves.
            tracer_provider: Provider ``emit_spans`` draws its tracer from
                (ADR-014). Defaults to OTel's global provider; only needed to
                target a non-global one, e.g. in tests. Has no effect unless
                ``emit_spans=True``.
            summarizer: Turns dropped-history text into a summary, for
                ``summarize_history``. This package constructs no model
                client and calls no model on its own -- same rule the core's
                quality-lane judge follows -- so there is no default.
            similarity_fn: Similarity function for ``semantic_cache``.
                Defaults to a lexical, embeddings-free metric; see
                :mod:`optio_optimize.similarity`.
            **overrides: Individual config fields, e.g. ``semantic_cache=True``.

        Raises:
            OptimizeConfigError: On invalid configuration, if both ``config``
                and ``overrides`` are given, or if ``summarize_history`` is on
                with no ``summarizer`` and no explicit ``stages`` override --
                the flag would otherwise be on and silently do nothing
                forever, indistinguishable from the library not working
                (§4.2's rule, the same one ``route_models``/``cheap_model``
                already enforces).
        """
        if config is not None and overrides:
            from optio_optimize.errors import OptimizeConfigError

            raise OptimizeConfigError(
                "pass either a config object or keyword overrides, not both; "
                f"got config plus {sorted(overrides)}"
            )
        self.config = config if config is not None else config_from_mapping(overrides)
        if self.config.summarize_history and summarizer is None and stages is None:
            from optio_optimize.errors import OptimizeConfigError

            raise OptimizeConfigError(
                "summarize_history is on but no summarizer was given; the stage "
                "would be active and always decline, which is indistinguishable "
                "from not working. Pass Optimizer(summarizer=...), or build the "
                "stage list yourself via stages=."
            )
        self._pipeline = Pipeline(
            self.config,
            stages
            if stages is not None
            else build_stages(self.config, summarizer=summarizer, similarity_fn=similarity_fn),
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
    def pipeline(self) -> Pipeline:
        """The object that runs the stages.

        Exposed for one reason: ADR-017 makes
        :class:`~optio_optimize.batch.BatchOptimizer` *own* an ``Optimizer``
        rather than reimplement it, and batch dispatch needs the two halves of
        a request separately -- run the stages now, hand back the provider's
        answer tomorrow. :meth:`~optio_optimize.pipeline.Pipeline.prepare` and
        :meth:`~optio_optimize.pipeline.Pipeline.complete` are that seam.

        Sharing this is also what makes the two surfaces share one exact cache
        and one savings report, which ADR-017 decision 3 asks for deliberately:
        a request already answered synchronously should never enter a queue,
        and an answer that arrives hours later is as cacheable as one that
        arrives immediately.
        """
        return self._pipeline

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Names of the active stages, in execution order.

        Worth checking after construction: a stage silently absent because its
        config flag defaulted off is the most common reason someone reports
        that the library "does nothing".
        """
        return tuple(s.name for s in self._pipeline.stages)

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The active stages themselves, in execution order.

        Exposed because :attr:`stage_names` is not enough to answer the one
        question the benchmark has to ask before it can claim anything: *which
        of these promise byte-identical output.* Deriving that from each
        stage's own :attr:`~optio_optimize.stages.base.Stage.fidelity` is the
        only way the answer stays right when a stage is added -- the
        hand-written list that used to encode it in the bench CLI was already
        wrong by two stages when it was replaced.
        """
        return tuple(self._pipeline.stages)

    @property
    def findings(self) -> tuple[PrefixFinding, ...]:
        """Diagnoses of why caching is not paying, if any have been reached.

        Empty until ``detect_unstable_prefix`` has seen enough requests to
        distinguish a broken prefix from a young process -- and empty forever
        if nothing is wrong, which is the ordinary case.

        Exposed as data rather than left in the log because the caller who
        needs this is often not the one reading logs: a hit rate that should be
        70% and is 0% is a production cost problem, and asserting on this tuple
        in a smoke test catches it before the invoice does.
        """
        from optio_optimize.stages.diagnostics import UnstablePrefixStage

        return tuple(
            finding
            for stage in self._pipeline.stages
            if isinstance(stage, UnstablePrefixStage)
            for finding in stage.findings
        )
