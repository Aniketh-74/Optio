"""The public entry point.

One class, deliberately. The pipeline, stage registry and savings ledger are all
useful to a contributor and none of them are things a user should have to
assemble to get value -- so ``Optimizer()`` with no arguments is a working,
lossless, bounded-memory optimizer, and everything else is opt-in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optio_optimize.cascade import CascadeRouter
from optio_optimize.config import OptimizeConfig, config_from_mapping
from optio_optimize.pipeline import Pipeline
from optio_optimize.stages import build_stages

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import TracerProvider

    from optio_optimize.cascade import CascadeStats, Verifier
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

    __slots__ = ("_cascade", "_pipeline", "config")

    def __init__(
        self,
        config: OptimizeConfig | None = None,
        *,
        stages: list[Stage] | None = None,
        tracer_provider: TracerProvider | None = None,
        summarizer: Summarizer | None = None,
        similarity_fn: SimilarityFn | None = None,
        cascade_verifier: Verifier | None = None,
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
            cascade_verifier: Acceptance check for ``cascade_routing`` -- given
                the request and the cheap model's answer, returns ``True`` to
                accept or ``False`` to escalate (ADR-023). Defaults to
                :func:`~optio_optimize.cascade.default_verifier`, which catches
                empty and truncated answers but does not judge correctness; a
                caller who needs semantic verification supplies their own.
                Ignored unless ``cascade_routing`` is on.
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
        # Cascade is not a stage (ADR-023): it wraps the provider call rather
        # than transforming the request, so it lives beside the pipeline, not in
        # it. Built only when enabled, so the common path constructs nothing.
        if self.config.cascade_routing:
            from optio_optimize.cascade import default_verifier

            self._cascade: CascadeRouter | None = CascadeRouter(
                self.config, verify=cascade_verifier or default_verifier
            )
        else:
            self._cascade = None

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
        if self._cascade is not None:
            provider = self._cascade.wrap(provider)
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
        if self._cascade is not None:
            provider = self._cascade.awrap(provider)
        return await self._pipeline.aexecute(request, provider, run_id=run_id)

    async def afan_out(
        self,
        requests: Sequence[LLMRequest],
        provider: AsyncProviderCall,
        *,
        run_id: str | None = None,
    ) -> list[LLMResponse]:
        """Optimize a concurrent fan-out and dispatch it in the cheaper order.

        N concurrent calls over a shared prompt prefix each pay to populate the
        provider's cache, because none can see another's write. Sending one first
        turns that into one write plus N-1 reads: on Anthropic the shared prefix
        goes from ``5 x 1.25`` to ``1.25 + 4 x 0.1`` for a fan-out of five, **74%
        off**, with no request altered (ADR-020). It pays on OpenAI too, whose
        automatic prefix cache still needs somebody to go first.

        **The cost is latency**, and it is the caller's to accept: one round trip
        is prepended to the batch. That is why this is a method you call rather
        than something inferred from a pattern of concurrent calls -- doubling a
        page's time to first byte to save a fraction of a cent is not a trade a
        library should make on anyone's behalf.

        Warming is skipped automatically when it cannot pay: fewer than two real
        calls, no shared prefix, or a shared prefix below
        :data:`~optio_optimize.fan_out.WARM_UP_MIN_PREFIX_TOKENS`. Below a
        provider's floor nothing is cached, so the warm-up would be pure latency.

        There is no synchronous twin, and that is not an omission. A caller
        issuing five calls in a loop **already gets this for free** -- sequential
        execution *is* warm-up ordering. The problem exists only under real
        concurrency, and serving a thread-pool caller would mean this package
        owning a thread pool, which is infrastructure ADR-016 keeps out.

        On a Claude model with ``prefix_cache`` off, Anthropic caches nothing, so
        this pays the latency and receives no discount. That is a configuration
        to fix rather than a case to detect: guessing a provider from a model
        string is the sort of proxy ``route_models`` was already caught getting
        wrong.

        Args:
            requests: The fan-out, in the caller's order.
            provider: Async function performing one real API call.
            run_id: optio run id, so savings attribute to the metered run.

        Returns:
            One response per request, in the order given. Callers correlate a
            fan-out by index.

        Raises:
            Exception: Only whatever ``provider`` raises.
        """
        from optio_optimize.fan_out import dispatch

        return await dispatch(self._pipeline, requests, provider, run_id=run_id)

    @property
    def report(self) -> SavingsReport:
        """Cumulative savings across every request this optimizer has run."""
        return self._pipeline.report

    @property
    def cascade_stats(self) -> CascadeStats | None:
        """Running cascade tally, or ``None`` when ``cascade_routing`` is off.

        Cascade savings do not fold into :attr:`report` the way a stage's do --
        a stage reports tokens avoided on one call, while cascade's economics
        are a rate across calls (a cheap answer accepted saves the gap between
        the two models; one escalated spends both). The escalation rate on this
        object is what the ADR-015 gate reads to decide whether the cascade is
        paying on a given workload.
        """
        return self._cascade.stats if self._cascade is not None else None

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
