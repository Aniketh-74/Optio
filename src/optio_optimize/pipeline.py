"""The request lifecycle.

```
request ──> [stage.before] ──> ... ──> provider call ──> [stage.after] ──> response
                  │                          ▲
                  └── short-circuit ─────────┘   (cache hit: provider never called)
```

This module owns the single most important property in the package: **a bug
here cannot break the caller's agent** (ADR-013 rule 1). The core's fail-open
guard achieves that by dropping a signal. Here it is harder, because the
pipeline sits in the request path, so "do nothing" has to mean "pass the
original request through" — and the original has to survive every stage in order
to still be available when one fails.

That is why stages return a new request rather than mutating one, and why the
pipeline keeps the last known-good request rather than trusting the running one.
A stage that raises halfway through a transformation cannot leave a partially
rewritten prompt behind, because it never had a reference to mutate.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from optio_optimize.savings import SavingsReport, StageSaving
from optio_optimize.stages.base import StageContext, StageResult
from optio_optimize.telemetry import record_span
from optio_optimize.tokens import count_request, default_counter

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider

    from optio_optimize.config import OptimizeConfig
    from optio_optimize.stages.base import Stage
    from optio_optimize.tokens import TokenCounter
    from optio_optimize.types import LLMRequest, LLMResponse

_log = logging.getLogger("optio_optimize")

#: Models warmed at construction, one per distinct tokenizer vocabulary.
#:
#: ``tiktoken`` loads each encoding's BPE table lazily and separately, so warming
#: one leaves the others cold. Real model names rather than encoding names,
#: because :class:`~optio_optimize.tokens.TokenCounter` takes models -- and
#: because naming ``cl100k_base`` here would tie this package's warm-up to
#: tiktoken's internal vocabulary names, which a different counter would not
#: share.
#:
#: ``gpt-4o`` maps to ``o200k_base`` and ``gpt-4`` to ``cl100k_base``. Anthropic
#: models resolve through the same fallback as ``gpt-4o``, so they need no third
#: entry; a counter that tokenized them separately would.
WARM_MODELS = ("gpt-4o", "gpt-4")

#: Callable that actually talks to the provider.
ProviderCall = Callable[["LLMRequest"], "LLMResponse"]

#: The async counterpart. Exists because every realistic adapter target --
#: the OpenAI Agents SDK's ``Model.get_response`` chief among them -- is
#: ``async def`` and abstract, with no synchronous alternative. Stage logic
#: itself performs no I/O, so only "call the provider" differs between the
#: two execution paths; :meth:`Pipeline._run_stages` is shared by both.
AsyncProviderCall = Callable[["LLMRequest"], "Awaitable[LLMResponse]"]


@dataclass(slots=True)
class PreparedRequest:
    """A request with every ``before`` hook applied, awaiting a provider.

    The synchronous path never has to name this: :meth:`Pipeline.execute` calls
    the provider one line after producing it. Batch dispatch does, because
    ADR-017's whole difficulty is that hours pass between the two halves --
    the provider call happens in another process, possibly on another day, and
    something has to hold the stage state in between.

    Attributes:
        request: The request as every stage that succeeded left it. This is
            what gets sent.
        ctx: The context stages ran against. Carried forward rather than
            rebuilt, because a stage's ``after`` hook must see the same
            ``scratch`` its own ``before`` populated -- the cache stages
            carry their key across the two calls this way.
        short_circuit: A response, if some stage served one without calling
            the provider. Non-``None`` means **do not send this**; a batch
            submission that queues an already-answered request pays for it
            twice and waits a day for the privilege.
        saved_input: Input tokens avoided across every stage that ran.
        saved_output: Output tokens avoided across every stage that ran.
        fired: Names of the stages that did something, in execution order.
        bypassed: Set when ``enabled`` is ``False`` and no stage ran at all.
            Distinct from "every stage declined": nothing was measured, so
            :meth:`Pipeline.complete` records nothing rather than recording
            zeroes, which would dilute the reduction ratio with requests the
            pipeline never touched.
    """

    request: LLMRequest
    ctx: StageContext
    short_circuit: LLMResponse | None = None
    saved_input: int = 0
    saved_output: int = 0
    fired: list[str] = field(default_factory=list)
    bypassed: bool = False


class Pipeline:
    """Runs stages around a provider call.

    Attributes:
        config: Active configuration.
        stages: Ordered stages. Order matters and is set by
            :func:`optio_optimize.stages.build_stages`, not by the caller.
        report: Cumulative savings across every request this pipeline has run.
    """

    __slots__ = ("_counter", "_tracer_provider", "config", "report", "stages")

    def __init__(
        self,
        config: OptimizeConfig,
        stages: list[Stage],
        *,
        tracer_provider: TracerProvider | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        """Build a pipeline.

        Args:
            config: Validated configuration.
            stages: Stages to run, in order.
            tracer_provider: Provider for `emit_spans`' tracer (ADR-014).
                Defaults to OTel's global provider; overridable for tests and
                multi-provider setups, matching ``optio.api.meter``.
            counter: Token counter to use. Defaults to the best available one.
                Injectable so the warm-up below can be tested against a counter
                with a *known* startup cost rather than against whatever
                ``tiktoken`` happens to do on the machine running the suite.
        """
        self.config = config
        self.stages = [s for s in stages if s.name not in config.disabled_stages]
        self._counter = counter if counter is not None else default_counter()
        self._tracer_provider = tracer_provider
        self.report = SavingsReport(exact=self._counter.is_exact)
        self._warm_counter()

        lossy = [s.name for s in self.stages if s.lossy]
        if lossy:
            # Running in a mode where output may differ from what the model
            # would have produced is a state worth stating once, loudly, rather
            # than leaving someone to discover it from a confusing answer.
            _log.warning(
                "optio_optimize: output-altering stages active (%s); "
                "responses may differ from an unoptimized call",
                ", ".join(sorted(lossy)),
            )

    def execute(
        self,
        request: LLMRequest,
        call: ProviderCall,
        *,
        run_id: str | None = None,
    ) -> LLMResponse:
        """Optimize a request, call the provider unless short-circuited, return.

        Args:
            request: The caller's request.
            call: Function that performs the real provider call.
            run_id: optio run this belongs to, for attributing savings.

        Returns:
            The provider's response, or one produced by a short-circuiting
            stage. Either way, exactly what the caller would have received had
            no stage failed.

        Raises:
            Exception: Only whatever ``call`` raises. Stage failures never
                propagate -- that is the guarantee this method exists to make.
        """
        prepared = self.prepare(request, run_id=run_id)
        if prepared.bypassed:
            return call(request)
        if prepared.short_circuit is not None:
            return self.complete(prepared, prepared.short_circuit, run_id=run_id)

        response = call(prepared.request)
        return self.complete(prepared, response, run_id=run_id)

    async def aexecute(
        self,
        request: LLMRequest,
        call: AsyncProviderCall,
        *,
        run_id: str | None = None,
    ) -> LLMResponse:
        """The async twin of :meth:`execute`, for an async provider call.

        Stage logic performs no I/O, so nothing about running stages differs
        between the two paths -- :meth:`_run_stages` is shared verbatim. Only
        the line that calls the provider needs an ``await``, which is the
        entire reason this method exists rather than asking every async
        caller (the OpenAI Agents SDK's ``Model.get_response`` chief among
        them, ``async def`` and abstract with no synchronous alternative) to
        bridge to :meth:`execute` themselves.

        Args:
            request: The caller's request.
            call: Async function that performs the real provider call.
            run_id: optio run this belongs to, for attributing savings.

        Returns:
            Same contract as :meth:`execute`.

        Raises:
            Exception: Only whatever ``call`` raises.
        """
        prepared = self.prepare(request, run_id=run_id)
        if prepared.bypassed:
            return await call(request)
        if prepared.short_circuit is not None:
            return self.complete(prepared, prepared.short_circuit, run_id=run_id)

        response = await call(prepared.request)
        return self.complete(prepared, response, run_id=run_id)

    def prepare(self, request: LLMRequest, *, run_id: str | None = None) -> PreparedRequest:
        """Run every ``before`` hook, without calling a provider.

        The first half of :meth:`execute`, separated so that ADR-017's batch
        surface can run the identical stages and then stop, holding the result
        until a provider answers hours later. Splitting rather than duplicating
        is the point: a second implementation of "run the stages" would be a
        second place for a stage to be skipped, and the divergence would show
        up as batch and synchronous calls being optimized differently for
        reasons nobody could see.

        Args:
            request: The caller's request.
            run_id: optio run this belongs to, for attributing savings.

        Returns:
            The prepared request. Check
            :attr:`~PreparedRequest.short_circuit` before sending anything: a
            cache hit means the answer is already in hand.
        """
        if not self.config.enabled:
            return PreparedRequest(
                request=request,
                ctx=StageContext(config=self.config, counter=self._counter, run_id=run_id),
                bypassed=True,
            )
        return self._run_stages(request, run_id)

    def complete(
        self,
        prepared: PreparedRequest,
        response: LLMResponse,
        *,
        run_id: str | None = None,
    ) -> LLMResponse:
        """Run every ``after`` hook and record what the request cost.

        The second half of :meth:`execute`. Must be called exactly once per
        :meth:`prepare`, or the read-through caches never get their write half
        and the savings report loses the request entirely.

        Args:
            prepared: What :meth:`prepare` returned.
            response: What the provider returned. Ignored when ``prepared``
                carries a short-circuit -- that response is authoritative, and
                accepting an argument that overrides it would let a caller
                report a cache hit's savings against a call they made anyway.
            run_id: Forwarded to the span emitter.

        Returns:
            The response to hand back to the caller.
        """
        if prepared.bypassed:
            return response
        if prepared.short_circuit is not None:
            return self._finish_and_emit(
                prepared, prepared.short_circuit, run_id, short_circuited=True
            )
        self._run_after(prepared.request, response, prepared.ctx)
        return self._finish_and_emit(prepared, response, run_id, short_circuited=False)

    def _warm_counter(self) -> None:
        """Pay the token counter's one-time startup here, not on request one.

        ``tiktoken`` loads its BPE vocabulary lazily, on first use: 395 ms
        against a 100 ms ``latency_budget_ms``. Whichever stage counted first
        paid it out of the deadline below, and everything after that stage was
        skipped -- five of nine on a default config, including ``prefix_cache``,
        the largest lossless saving here. The second request through the same
        process ran all nine, which is why no benchmark ever caught it (ADR-038).

        The cost is not avoided, only moved to where it is attributable: a
        one-time cost of *having* an optimizer rather than of using one.

        Failures are swallowed. This changes when a cost is paid and must never
        become a new way to fail (ADR-013 rule 1).

        **One model name is not enough.** The first version passed ``""``, which
        `encoding_for_model` rejects, so the fallback warmed `o200k_base` alone
        and every `cl100k_base` model -- `gpt-4`, `gpt-3.5-turbo` -- still paid
        the full vocabulary load out of a 100 ms deadline on request one. The
        bug ADR-038 closed, surviving for the families it did not name. Each
        distinct encoding is a separate lazy load, so each needs a real model
        name that maps to it.
        """
        for model in WARM_MODELS:
            try:
                self._counter.count_text("warm", model)
            except Exception as exc:  # noqa: BLE001 - never a new failure mode
                # Type only, never the message (§10). Continues rather than
                # returns: one unavailable vocabulary must not leave the others
                # cold, which would reintroduce the bug for a different family.
                _log.debug(
                    "optio_optimize: counter warm-up failed for %s (%s)", model, type(exc).__name__
                )

    def _run_stages(self, request: LLMRequest, run_id: str | None) -> PreparedRequest:
        """Run every stage's ``before`` hook, stopping at a short-circuit or the deadline.

        Args:
            request: The caller's request.
            run_id: Passed through to the shared :class:`StageContext`.

        Returns:
            What the run produced, up to (and including, if one fired) a
            short-circuit.
        """
        ctx = StageContext(config=self.config, counter=self._counter, run_id=run_id)
        run = PreparedRequest(request=request, ctx=ctx)
        deadline = time.perf_counter() + self.config.latency_budget_ms / 1000.0

        # `run.request` advances only on a stage that succeeds outright. A
        # stage that raises leaves it untouched, so the next stage receives
        # the last good request rather than whatever a half-finished
        # transform produced.
        for stage in self.stages:
            if time.perf_counter() >= deadline:
                # Out of budget. Everything remaining is skipped rather than
                # allowed to blow the caller's latency expectations; the
                # optimizations already applied still stand.
                _log.debug("optio_optimize: latency budget spent, skipping remaining stages")
                break

            result, elapsed_ms = self._run_before(stage, run.request, ctx)
            if result is None:
                continue

            self._account(stage.name, result, elapsed_ms)
            run.saved_input += result.saved_input_tokens
            run.saved_output += result.saved_output_tokens
            run.request = result.request
            if result.note:
                # A non-empty note is how a stage says "I did something" --
                # the same signal PrefixCacheStage's zero-token-saving marker
                # relies on to show up in reports at all.
                run.fired.append(stage.name)
            if result.short_circuited:
                run.short_circuit = result.response
                break

        return run

    def _finish_and_emit(
        self,
        run: PreparedRequest,
        response: LLMResponse,
        run_id: str | None,
        *,
        short_circuited: bool,
    ) -> LLMResponse:
        """Record savings and optionally emit a span, shared by both call paths.

        Args:
            run: What :meth:`_run_stages` produced.
            response: The response to report against -- the short-circuit, or
                what the provider returned.
            run_id: Forwarded to :func:`~optio_optimize.telemetry.record_span`.
            short_circuited: Whether the provider was ever called.

        Returns:
            ``response``, unchanged -- this method exists for its side
            effects, and returns its argument so call sites stay one-liners.
        """
        totals = (run.saved_input, run.saved_output)
        self._finish(totals, run.request, response, short_circuited=short_circuited)
        if self.config.emit_spans:
            record_span(
                run.request,
                response,
                stages=run.fired,
                saved_input_tokens=run.saved_input,
                saved_output_tokens=run.saved_output,
                short_circuited=short_circuited,
                run_id=run_id,
                tracer_provider=self._tracer_provider,
            )
        return response

    def _run_before(
        self,
        stage: Stage,
        request: LLMRequest,
        ctx: StageContext,
    ) -> tuple[StageResult | None, float]:
        """Run one stage's ``before``, absorbing any failure.

        Returns:
            ``(result, elapsed_ms)``, with ``result`` ``None`` when the stage
            failed and must be skipped.
        """
        started = time.perf_counter()
        try:
            result = stage.before(request, ctx)
        except Exception as exc:  # noqa: BLE001 - ADR-013 rule 1: never propagate
            # Type only, never the message: an exception payload can contain
            # prompt content, and §10's rule outlives the package boundary.
            _log.warning(
                "optio_optimize: stage %r failed (%s); skipping it for this request",
                stage.name,
                type(exc).__name__,
            )
            return None, (time.perf_counter() - started) * 1000.0
        return result, (time.perf_counter() - started) * 1000.0

    def _run_after(
        self,
        request: LLMRequest,
        response: LLMResponse,
        ctx: StageContext,
    ) -> None:
        """Run every stage's ``after`` hook, absorbing failures.

        Reverse order, so a stage that wrapped the request unwraps around the
        response. Failures here matter less -- the caller already has their
        answer -- but a raising cache write must not turn a successful call
        into an exception.
        """
        for stage in reversed(self.stages):
            try:
                stage.after(request, response, ctx)
            except Exception as exc:  # noqa: BLE001 - ADR-013 rule 1
                _log.warning(
                    "optio_optimize: stage %r after-hook failed (%s)",
                    stage.name,
                    type(exc).__name__,
                )

    def _account(self, name: str, result: StageResult, elapsed_ms: float) -> None:
        """Fold a stage's reported savings into the running report."""
        if result.saved_input_tokens or result.saved_output_tokens or elapsed_ms:
            self.report.record(
                StageSaving(
                    stage=name,
                    input_tokens=result.saved_input_tokens,
                    output_tokens=result.saved_output_tokens,
                    note=result.note,
                    elapsed_ms=elapsed_ms,
                )
            )

    def _finish(
        self,
        saved_before: tuple[int, int],
        sent: LLMRequest,
        response: LLMResponse,
        *,
        short_circuited: bool,
    ) -> None:
        """Record the totals for one completed request.

        **Baseline is derived, never measured independently.** The obvious
        implementation -- count the incoming request with our tokenizer, compare
        against what the provider billed -- mixes two measurement scales, and
        the difference between an estimator and a provider's real count then
        shows up as savings. The first smoke test of this pipeline produced a
        baseline *below* actual and a negative dollar figure that way.

        So: ``baseline = actual + saved``. Actual is what was billed, saved is
        what the stages report avoiding, and their sum is what would have been
        billed without them. It cannot go negative, savings cannot exceed
        baseline, and both sides come from the same source per request.
        """
        saved_input, saved_output = saved_before
        self.report.requests += 1

        if short_circuited:
            # Nothing was sent, nothing generated, nothing billed. The entire
            # cost of this request was avoided, and the stage that served it
            # already reported how much that was.
            self.report.short_circuits += 1
            self.report.baseline_input_tokens += saved_input
            self.report.baseline_output_tokens += saved_output
            return

        actual_input = response.input_tokens or count_request(sent, self._counter)
        self.report.actual_input_tokens += actual_input
        self.report.actual_output_tokens += response.output_tokens
        self.report.baseline_input_tokens += actual_input + saved_input
        self.report.baseline_output_tokens += response.output_tokens + saved_output
        self.report.provider_cached_tokens += response.cached_input_tokens
        self.report.provider_written_tokens += response.cache_write_tokens
        self.report.provider_written_1h_tokens += response.cache_write_1h_tokens
