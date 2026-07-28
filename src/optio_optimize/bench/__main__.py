"""``python -m optio_optimize.bench`` — run the A/B suite.

Simulated by default. Live runs are opt-in, capped, and print what they intend
to spend before spending it.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from optio_optimize.bench.adversarial import format_audit_report, run_semantic_cache_audit
from optio_optimize.bench.harness import compare, format_result
from optio_optimize.bench.providers import (
    DEFAULT_SPEND_CAP_USD,
    SimulatedProvider,
    SpendGuard,
    available_live_provider,
)
from optio_optimize.bench.workloads import WORKLOADS
from optio_optimize.config import CHEAP_COUNTERPART, DEFAULT_SEMANTIC_THRESHOLD, OptimizeConfig
from optio_optimize.types import LLMRequest, Message

if TYPE_CHECKING:
    from optio_optimize.bench.providers import BenchProvider
    from optio_optimize.stages.summarize import Summarizer

#: The four ADR-013-lossy stages this CLI can isolate one at a time. Order
#: matches the module docstring's own ordering rules, not alphabetical.
ALTERED_STAGES = ("route_models", "compress_prompt", "semantic_cache", "summarize_history")


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark suite.

    Returns:
        Process exit code. Non-zero when a workload diverged in output while
        only lossless stages were enabled -- that is a correctness failure, not
        a benchmark result, and it should break a build.
    """
    parser = argparse.ArgumentParser(
        prog="python -m optio_optimize.bench",
        description="A/B benchmark: optimizer on vs off, same workload, same provider.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="call a real API (needs ANTHROPIC_API_KEY or OPENAI_API_KEY). Costs money.",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=DEFAULT_SPEND_CAP_USD,
        help=f"spend cap in USD for live runs (default {DEFAULT_SPEND_CAP_USD:.2f})",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(WORKLOADS),
        help="run only this workload; repeatable. Default: all.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=ALTERED_STAGES,
        dest="stages",
        help=(
            "enable this ALTERED-tier (lossy) stage; repeatable. Each run isolates "
            "exactly the named stage(s) -- ADR-015 requires isolated evidence, since "
            "the old --aggressive flag bundled semantic_cache and compress_prompt "
            "together and produced a confounded result neither stage's evidence could "
            "actually be attributed to. Pass it more than once to test a deliberate "
            "combination; the default (omitted) is none of the four."
        ),
    )
    parser.add_argument(
        "--cheap-model",
        default=None,
        help=(
            "model for --stage route_models to downgrade to. Defaults to "
            "config.CHEAP_COUNTERPART[provider.model] if the provider's model has a "
            "known cheap counterpart; otherwise this must be set explicitly."
        ),
    )
    parser.add_argument(
        "--provider-latency",
        type=float,
        default=0.0,
        help=(
            "milliseconds to delay each simulated call. Needed for throughput and "
            "latency figures: against an instant provider those measure only our "
            "own overhead. A small hosted model is typically 300-800 ms."
        ),
    )
    parser.add_argument(
        "--strict-fidelity",
        action="store_true",
        help=(
            "run only stages that guarantee byte-identical output, and fail on any "
            "divergence. This is the CI configuration: it proves the identical-output "
            "stages really are identical."
        ),
    )
    parser.add_argument(
        "--semantic-cache-audit",
        action="store_true",
        help=(
            "run the ADR-015 adversarial audit instead of the workload suite: eight "
            "near-duplicate-but-different-answer prompt pairs measuring semantic_cache's "
            "live false-positive rate directly, plus eight same-answer controls "
            "measuring what a safe threshold costs in legitimate reuse. Ignores "
            "--workload/--stage."
        ),
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_THRESHOLD,
        help=(
            f"similarity a --semantic-cache-audit hit requires (default "
            f"{DEFAULT_SEMANTIC_THRESHOLD}, the shipped one). ADR-015 wants the default "
            "measured plus nearby values, so the margin the default actually buys is a "
            "number rather than an assumption. OptimizeConfig rejects anything below 0.9."
        ),
    )
    args = parser.parse_args(argv)

    guard = SpendGuard(args.cap) if args.live else None
    if args.live:
        provider = available_live_provider(guard)
        if provider is None:
            print(
                "No live provider available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "and install the matching SDK, or drop --live to use the simulator.",
                file=sys.stderr,
            )
            return 2
        print(f"LIVE run against {provider.label}, spend cap ${args.cap:.2f}")
        if not args.semantic_cache_audit:
            print("Both arms call the real API, so this bills roughly twice one pass.\n")
    else:
        provider = SimulatedProvider(latency_ms=args.provider_latency)
        print("Simulated run: token, cost, cache, memory and overhead figures are real.")
        if args.provider_latency > 0:
            print(f"Provider delay modelled at {args.provider_latency:.0f} ms per call.")
            print("Output-length and quality figures still need --live.\n")
        else:
            print("Output-length, latency, throughput and quality figures need --live")
            print("or --provider-latency.\n")

    if args.semantic_cache_audit:
        if not args.live:
            print(
                "note: --semantic-cache-audit without --live uses SimulatedProvider, which "
                "returns synthetic content -- it exercises the similarity/hit-rate plumbing "
                "but cannot demonstrate an actual wrong-answer collision the way a live run "
                "can. Treat this as a smoke test, not ADR-015 evidence.\n"
            )
        report = run_semantic_cache_audit(provider, threshold=args.semantic_threshold)
        print("\n".join(format_audit_report(report)))
        if guard is not None:
            print(f"spent ${guard.spent_usd:.4f} across {guard.calls} live calls")
        return 0

    selected = [WORKLOADS[n] for n in (args.workload or sorted(WORKLOADS))]

    stages = set(args.stages or ())
    if args.strict_fidelity and stages:
        print(
            "--strict-fidelity holds every stage to identical output; the ALTERED "
            f"stage(s) {', '.join(sorted(stages))} cannot honor that by definition.",
            file=sys.stderr,
        )
        return 2

    summarizer = None
    if "summarize_history" in stages:
        summarizer = _build_live_summarizer(provider, args.cheap_model)
        if not args.live:
            print(
                "note: --stage summarize_history without --live calls the simulator, "
                "which returns a fixed 'simulated-answer-<hash>' string, not a real "
                "summary. This checks the stage's plumbing, not ADR-015 evidence.\n"
            )

    cheap_model = args.cheap_model
    if "route_models" in stages:
        cheap_model = _resolve_cheap_model(args.cheap_model, provider.model)
        if cheap_model is None:
            print(
                f"--stage route_models needs a cheaper model to route to, and "
                f"{provider.model!r} has no entry in CHEAP_COUNTERPART. Pass "
                f"--cheap-model explicitly.",
                file=sys.stderr,
            )
            return 2
        print(
            f"note: route_models will retarget routable requests from "
            f"{provider.model} to {cheap_model}. This harness prices the whole "
            "optimized arm at one flat rate (ABResult.model), so the cost/token "
            "numbers below do not reflect the cheaper model actually being called "
            "-- see ADR-015's route_models section for why this stage's real "
            "evidence comes from a direct per-model capability comparison instead, "
            "not this A/B harness.\n"
        )

    config = OptimizeConfig(
        semantic_cache="semantic_cache" in stages,
        compress_prompt="compress_prompt" in stages,
        summarize_history="summarize_history" in stages,
        route_models="route_models" in stages,
        cheap_model=cheap_model,
        # Under --strict-fidelity, drop the reshaping stages so the run is a
        # clean test of the identical-output promise. trim_history, dedup and
        # pruning are SHAPED too (they change what the prompt says, not just
        # its price) and default on, so they need the same treatment or this
        # flag stops proving what it claims to.
        structured_output=not args.strict_fidelity,
        adaptive_max_tokens=not args.strict_fidelity,
        trim_history=not args.strict_fidelity,
        deduplicate=not args.strict_fidelity,
        prune_retrieval=not args.strict_fidelity,
    )

    # Only stages promising byte-identical output can be held to it. Running
    # with `structured_output` on and demanding identical responses would fail
    # every JSON workload for doing exactly what it was asked to do -- which is
    # what the first version of this check did.
    from optio_optimize.optimizer import Optimizer
    from optio_optimize.stages.base import Fidelity

    active = Optimizer(config, summarizer=summarizer)
    shaping = [s.name for s in active._pipeline.stages if s.fidelity is not Fidelity.IDENTICAL]
    strict = not shaping
    if shaping:
        print(f"note: {', '.join(shaping)} may reshape replies; divergence is expected.")
        print("      Run with --strict-fidelity to hold every stage to identical output.\n")

    divergences = 0
    for workload in selected:
        print(f"# {workload.description}")
        print(f"# expected: {workload.expectation}\n")
        # Priced against what the provider actually served, never against a
        # hardcoded name -- see BenchProvider.model.
        result = compare(workload, provider, config, model=provider.model, summarizer=summarizer)
        print("\n".join(format_result(result)))
        print()
        if strict and result.quality.divergent:
            divergences += result.quality.divergent

    if guard is not None:
        print(f"spent ${guard.spent_usd:.4f} across {guard.calls} live calls")

    if divergences:
        print(
            f"\nFAIL: {divergences} response(s) diverged while every active stage "
            "promised byte-identical output.",
            file=sys.stderr,
        )
        return 1
    return 0


def _resolve_cheap_model(explicit: str | None, provider_model: str) -> str | None:
    """Pick a ``cheap_model`` for ``route_models``, or say there is none.

    Args:
        explicit: ``--cheap-model``, if the caller set it.
        provider_model: The model actually being benchmarked.

    Returns:
        ``explicit`` if given; otherwise ``CHEAP_COUNTERPART[provider_model]``,
        or ``None`` if that model has no known cheaper counterpart -- e.g.
        ``"gpt-4o-mini"``, which is already the cheap end of its own family.
    """
    return explicit if explicit is not None else CHEAP_COUNTERPART.get(provider_model)


def _build_live_summarizer(provider: BenchProvider, cheap_model: str | None) -> Summarizer:
    """A real summarizer for isolating ``summarize_history`` (ADR-015).

    Calls the same provider the benchmark is already using, at a small,
    separate summarization request -- a real model call, not a stand-in, so
    the numbers this produces measure the stage rather than a stub. Uses
    ``cheap_model`` when given, since a summarizer's own cost should not
    swamp the tokens it is trying to save; falls back to the provider's own
    model otherwise.

    Args:
        provider: The active ``BenchProvider`` (live or simulated).
        cheap_model: Model to summarize with, if the caller set one.

    Returns:
        A ``Summarizer`` callable.
    """
    model = cheap_model or provider.model

    def summarize(text: str) -> str:
        request = LLMRequest(
            model=model,
            messages=(
                Message(
                    role="system",
                    content=(
                        "Summarize the following conversation history in 2-3 sentences. "
                        "Preserve every specific fact, number, name, and decision -- omit "
                        "only conversational filler."
                    ),
                ),
                Message(role="user", content=text),
            ),
            temperature=0.0,
        )
        return provider(request).content

    return summarize


if __name__ == "__main__":
    raise SystemExit(main())
