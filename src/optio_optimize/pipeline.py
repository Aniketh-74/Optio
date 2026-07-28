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
from collections.abc import Callable
from typing import TYPE_CHECKING

from optio_optimize.savings import SavingsReport, StageSaving
from optio_optimize.stages.base import StageContext, StageResult
from optio_optimize.telemetry import record_span
from optio_optimize.tokens import count_request, default_counter

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider

    from optio_optimize.config import OptimizeConfig
    from optio_optimize.stages.base import Stage
    from optio_optimize.types import LLMRequest, LLMResponse

_log = logging.getLogger("optio_optimize")

#: Callable that actually talks to the provider.
ProviderCall = Callable[["LLMRequest"], "LLMResponse"]


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
    ) -> None:
        """Build a pipeline.

        Args:
            config: Validated configuration.
            stages: Stages to run, in order.
            tracer_provider: Provider for `emit_spans`' tracer (ADR-014).
                Defaults to OTel's global provider; overridable for tests and
                multi-provider setups, matching ``optio.api.meter``.
        """
        self.config = config
        self.stages = [s for s in stages if s.name not in config.disabled_stages]
        self._counter = default_counter()
        self._tracer_provider = tracer_provider
        self.report = SavingsReport(exact=self._counter.is_exact)

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
        if not self.config.enabled:
            return call(request)

        ctx = StageContext(config=self.config, counter=self._counter, run_id=run_id)

        # `current` advances only on a stage that succeeds outright. A stage
        # that raises leaves it untouched, so the next stage receives the last
        # good request rather than whatever a half-finished transform produced.
        current = request
        deadline = time.perf_counter() + self.config.latency_budget_ms / 1000.0
        short_circuit: LLMResponse | None = None
        saved_input = 0
        saved_output = 0
        fired: list[str] = []

        for stage in self.stages:
            if time.perf_counter() >= deadline:
                # Out of budget. Everything remaining is skipped rather than
                # allowed to blow the caller's latency expectations; the
                # optimizations already applied still stand.
                _log.debug("optio_optimize: latency budget spent, skipping remaining stages")
                break

            result, elapsed_ms = self._run_before(stage, current, ctx)
            if result is None:
                continue

            self._account(stage.name, result, elapsed_ms)
            saved_input += result.saved_input_tokens
            saved_output += result.saved_output_tokens
            current = result.request
            if result.note:
                # A non-empty note is how a stage says "I did something" --
                # the same signal PrefixCacheStage's zero-token-saving marker
                # relies on to show up in reports at all.
                fired.append(stage.name)
            if result.short_circuited:
                short_circuit = result.response
                break

        totals = (saved_input, saved_output)
        if short_circuit is not None:
            self._finish(totals, current, short_circuit, short_circuited=True)
            if self.config.emit_spans:
                record_span(
                    current,
                    short_circuit,
                    stages=fired,
                    saved_input_tokens=saved_input,
                    saved_output_tokens=saved_output,
                    short_circuited=True,
                    run_id=run_id,
                    tracer_provider=self._tracer_provider,
                )
            return short_circuit

        response = call(current)
        self._run_after(current, response, ctx)
        self._finish(totals, current, response, short_circuited=False)
        if self.config.emit_spans:
            record_span(
                current,
                response,
                stages=fired,
                saved_input_tokens=saved_input,
                saved_output_tokens=saved_output,
                short_circuited=False,
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
