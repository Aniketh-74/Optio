"""The A/B runner.

Runs a workload twice — once with the optimizer disabled, once enabled — against
the same provider, and reports the difference. Three details decide whether the
comparison means anything:

**The control arm is this library with ``enabled=False``**, not a bare provider
call. Both arms therefore pay the same instrumentation overhead, so the measured
difference is the optimizations rather than the scaffolding around them.

**The baseline runs first, and its responses are kept.** Quality comparison
needs a reference. Running optimized-first and comparing backwards would let a
warm provider-side cache flatter the second arm.

**Memory is measured with ``tracemalloc``, not RSS.** RSS is dominated by
allocator behaviour and GC timing; it would make the number flap between runs
and tell you nothing about what the cache actually holds. The core's soak tests
settled this question already (§11) and the reasoning transfers.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import TYPE_CHECKING

from optio_optimize.bench.metrics import ABResult, ArmResult, QualityResult
from optio_optimize.config import OptimizeConfig
from optio_optimize.optimizer import Optimizer

if TYPE_CHECKING:
    from optio_optimize.bench.providers import BenchProvider
    from optio_optimize.bench.workloads import Workload
    from optio_optimize.stages.semantic_cache import SimilarityFn
    from optio_optimize.stages.summarize import Summarizer
    from optio_optimize.types import LLMRequest, LLMResponse

#: Judges two responses equivalent despite differing text.
Judge = Callable[[str, str], bool]


def run_arm(
    name: str,
    requests: list[LLMRequest],
    provider: BenchProvider,
    config: OptimizeConfig,
    *,
    summarizer: Summarizer | None = None,
    similarity_fn: SimilarityFn | None = None,
) -> tuple[ArmResult, Optimizer]:
    """Run one side of the comparison.

    Args:
        name: Arm label.
        requests: The workload.
        provider: Where calls go.
        config: Optimizer settings; ``enabled=False`` makes this the control.
        summarizer: Passed through to :class:`~optio_optimize.optimizer.Optimizer`,
            for isolating ``summarize_history``. ``None`` unless that stage is
            under test -- see ``bench/__main__.py``'s ``--stage`` handling.
        similarity_fn: Passed through for ``semantic_cache``. ``None`` uses the
            stage's own lexical default.

    Returns:
        The measurements, and the optimizer used, so its per-stage report and
        cache counters can be read afterwards.
    """
    optimizer = Optimizer(config, summarizer=summarizer, similarity_fn=similarity_fn)
    provider.reset()
    result = ArmResult(name=name, live=provider.is_live)

    provider_seconds = 0.0
    calls = 0

    def timed(request: LLMRequest) -> LLMResponse:
        nonlocal provider_seconds, calls
        started = time.perf_counter()
        response = provider(request)
        provider_seconds += time.perf_counter() - started
        calls += 1
        return response

    tracemalloc.start()
    baseline_snapshot = tracemalloc.get_traced_memory()[0]
    wall_started = time.perf_counter()

    for request in requests:
        try:
            response = optimizer.call(request, timed)
        except Exception as exc:  # noqa: BLE001 - a failed call is a datum, not a crash
            result.errors.append(f"{type(exc).__name__}: {exc}")
            continue
        result.responses.append(response.content)
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        result.cached_input_tokens += response.cached_input_tokens
        result.cache_write_tokens += response.cache_write_tokens

    result.wall_seconds = time.perf_counter() - wall_started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    result.peak_memory_bytes = max(0, peak - baseline_snapshot)
    result.provider_seconds = provider_seconds
    result.provider_calls = calls
    result.requests = len(requests)
    return result, optimizer


def compare(
    workload: Workload,
    provider: BenchProvider,
    config: OptimizeConfig | None = None,
    *,
    judge: Judge | None = None,
    model: str = "gpt-4o",
    summarizer: Summarizer | None = None,
    similarity_fn: SimilarityFn | None = None,
) -> ABResult:
    """Run a workload both ways and measure the difference.

    Args:
        workload: What to run.
        provider: Where calls go. A live provider makes latency, output-length
            and quality figures meaningful; a simulator leaves them ``None``.
        config: Optimized-arm settings. Defaults apply when omitted.
        judge: Decides whether two differing responses are equivalent. Needed
            only when lossy stages are on -- without one, any difference counts
            as divergence, which is the conservative reading.
        model: Model name for pricing.
        summarizer: Passed through for isolating ``summarize_history`` (ADR-015).
            Also passed to the baseline arm's ``Optimizer`` construction --
            required there too, since ``Optimizer.__init__`` validates
            ``summarize_history`` regardless of ``enabled`` -- but the
            baseline's ``enabled=False`` means ``Pipeline.execute`` returns
            before any stage runs (``pipeline.py``), so the summarizer is
            built and never called on that arm.
        similarity_fn: Passed through for isolating ``semantic_cache`` (ADR-015).
            Same reasoning as ``summarizer`` above.

    Returns:
        The full comparison.
    """
    requests = workload.requests()
    active = config if config is not None else OptimizeConfig()

    # Baseline first: it produces the reference responses, and running it
    # second would let any provider-side warming favour it unfairly.
    from dataclasses import replace

    baseline, _ = run_arm(
        "baseline",
        requests,
        provider,
        replace(active, enabled=False),
        summarizer=summarizer,
        similarity_fn=similarity_fn,
    )
    optimized, optimizer = run_arm(
        "optimized", requests, provider, active, summarizer=summarizer, similarity_fn=similarity_fn
    )

    quality = _compare_quality(
        baseline.responses,
        optimized.responses,
        judge=judge,
        deterministic=all(r.is_deterministic for r in requests),
    )

    stage_savings = {name: saving.total_tokens for name, saving in optimizer.report.stages.items()}
    lookups, hits = _cache_counters(optimizer)

    return ABResult(
        workload=workload.name,
        baseline=baseline,
        optimized=optimized,
        quality=quality,
        model=model,
        stage_savings=stage_savings,
        cache_lookups=lookups,
        cache_hits=hits,
        models_latency=provider.models_latency,
    )


def _compare_quality(
    reference: list[str],
    candidate: list[str],
    *,
    judge: Judge | None,
    deterministic: bool,
) -> QualityResult:
    """Compare two arms' outputs."""
    result = QualityResult(deterministic_baseline=deterministic)
    for ref, cand in zip(reference, candidate, strict=False):
        result.compared += 1
        if ref == cand:
            result.identical += 1
        elif judge is not None and judge(ref, cand):
            result.equivalent += 1
        else:
            result.divergent += 1
            result.divergent_pairs.append((ref, cand))
    return result


def _cache_counters(optimizer: Optimizer) -> tuple[int, int]:
    """Read lookup and hit counts from any cache stage present.

    Reaches into the stage list because the counters belong to the backend, not
    to the savings report -- a hit rate over *lookups* needs to know about the
    requests a cache correctly declined to consider, which the report cannot
    see.
    """
    for stage in optimizer._pipeline.stages:
        backend = getattr(stage, "backend", None)
        if backend is None:
            continue
        hits = getattr(backend, "_hits", None)
        misses = getattr(backend, "_misses", None)
        if isinstance(hits, int) and isinstance(misses, int):
            return hits + misses, hits
    return 0, 0


def format_result(result: ABResult) -> list[str]:
    """Render one comparison as report lines."""
    lines = [
        f"### {result.workload}",
        f"    provider: {'LIVE' if result.live else 'simulated'}   model: {result.model}",
        "",
        f"    requests            {result.baseline.requests}",
        f"    provider calls      {result.baseline.provider_calls} -> "
        f"{result.optimized.provider_calls}",
        f"    input tokens        {result.baseline.input_tokens:,} -> "
        f"{result.optimized.input_tokens:,}   {_pct(result.input_token_reduction)}",
        f"    output tokens       {result.baseline.output_tokens:,} -> "
        f"{result.optimized.output_tokens:,}   {_pct(result.output_token_reduction)}",
        f"    total tokens        {result.baseline.total_tokens:,} -> "
        f"{result.optimized.total_tokens:,}   {_pct(result.total_token_reduction)}",
    ]

    base_cost = result.cost_usd(result.baseline)
    opt_cost = result.cost_usd(result.optimized)
    if base_cost is not None and opt_cost is not None:
        lines.append(
            f"    cost                ${base_cost:.5f} -> ${opt_cost:.5f}   "
            f"{_pct(result.cost_reduction)}"
        )

    lines.extend(
        [
            # The provider's own prefix cache, which is a different mechanism
            # from the line below it and was invisible until 2026-07-31.
            # `prefix_cache` deliberately claims no saving of its own (ADR-020's
            # rule, since the effect is measured rather than estimated), so it
            # reported 0 tokens on all twelve workloads while the number that
            # would show whether it did anything was recorded and never printed.
            # A lever this package calls its largest lossless win has to be
            # legible in its own benchmark.
            f"    provider cache      reads {result.optimized.cached_input_tokens:,}"
            f"   writes {result.optimized.cache_write_tokens:,}"
            f"   (baseline reads {result.baseline.cached_input_tokens:,})",
            f"    cache hit rate      {_pct(result.true_cache_hit_rate, signed=False)}"
            f"  ({result.cache_hits}/{result.cache_lookups} lookups)",
            f"    added overhead      {_ms(result.added_overhead_ms_per_request)} / request",
            f"    throughput          {_pct(result.throughput_change, signed=True)}",
            f"    latency (end-end)   {_pct(result.latency_change, signed=True)}",
            f"    memory overhead     {result.memory_overhead_bytes / 1024:.1f} KiB",
        ]
    )

    quality = result.quality
    if not quality.is_interpretable:
        why = "sampled baseline" if not quality.deterministic_baseline else "nothing compared"
        lines.append(f"    output quality      not interpretable ({why})")
    else:
        lines.append(
            f"    output quality      {_pct(quality.identical_rate, signed=False)} identical"
            + (f", {quality.equivalent} judged equivalent" if quality.equivalent else "")
            + (f", {quality.divergent} DIVERGENT" if quality.divergent else "")
        )

    if result.stage_savings:
        lines.append("    by stage:")
        for name, tokens in sorted(result.stage_savings.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {name:<22} {tokens:>9,} tokens")
    if result.optimized.errors:
        lines.append(f"    ERRORS: {len(result.optimized.errors)}")
        for error in result.optimized.errors[:3]:
            lines.append(f"      {error}")
    return lines


def _pct(value: float | None, *, signed: bool = False) -> str:
    """Render a fraction as a percentage, or say why it is absent."""
    if value is None:
        return "n/a"
    if signed:
        return f"{value:+.1%}"
    return f"{value:.1%}"


def _ms(value: float | None) -> str:
    """Render milliseconds."""
    return "n/a" if value is None else f"{value:+.3f} ms"
