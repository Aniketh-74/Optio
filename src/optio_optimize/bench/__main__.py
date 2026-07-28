"""``python -m optio_optimize.bench`` — run the A/B suite.

Simulated by default. Live runs are opt-in, capped, and print what they intend
to spend before spending it.
"""

from __future__ import annotations

import argparse
import sys

from optio_optimize.bench.harness import compare, format_result
from optio_optimize.bench.providers import (
    DEFAULT_SPEND_CAP_USD,
    SimulatedProvider,
    SpendGuard,
    available_live_provider,
)
from optio_optimize.bench.workloads import WORKLOADS
from optio_optimize.config import OptimizeConfig


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
        "--aggressive",
        action="store_true",
        help="enable output-altering stages (semantic cache, compression, routing)",
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
    args = parser.parse_args(argv)

    selected = [WORKLOADS[n] for n in (args.workload or sorted(WORKLOADS))]

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

    config = OptimizeConfig(
        semantic_cache=args.aggressive,
        compress_prompt=args.aggressive,
        summarize_history=args.aggressive,
        # Under --strict-fidelity, drop the reshaping stages so the run is a
        # clean test of the identical-output promise.
        structured_output=not args.strict_fidelity,
        adaptive_max_tokens=not args.strict_fidelity,
    )

    # Only stages promising byte-identical output can be held to it. Running
    # with `structured_output` on and demanding identical responses would fail
    # every JSON workload for doing exactly what it was asked to do -- which is
    # what the first version of this check did.
    from optio_optimize.optimizer import Optimizer
    from optio_optimize.stages.base import Fidelity

    active = Optimizer(config)
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
        result = compare(workload, provider, config, model=provider.model)
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


if __name__ == "__main__":
    raise SystemExit(main())
