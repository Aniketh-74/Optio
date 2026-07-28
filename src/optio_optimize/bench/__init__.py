"""A/B benchmarking for the optimization pipeline.

Ships inside the package rather than beside it, so a user can measure this
library on *their* traffic instead of trusting numbers measured on ours. The
question "does this actually save me money" has a different answer per workload,
and a benchmark that only its author can run does not answer it for anyone else.

Run the bundled workloads::

    python -m optio_optimize.bench                  # simulated, free, instant
    python -m optio_optimize.bench --live           # real API, needs a key
    python -m optio_optimize.bench --live --cap 5   # raise the $1 spend cap

Or measure your own::

    from optio_optimize.bench import compare, Workload

    mine = Workload(
        name="my_agent",
        description="one hour of production traffic",
        build=lambda: load_my_captured_requests(),
        expectation="unknown -- that is why we are measuring",
    )
    result = compare(mine, my_provider)
"""

from __future__ import annotations

from optio_optimize.bench.harness import compare, format_result, run_arm
from optio_optimize.bench.metrics import ABResult, ArmResult, QualityResult
from optio_optimize.bench.providers import (
    AnthropicProvider,
    OpenAIProvider,
    SimulatedProvider,
    SpendGuard,
    available_live_provider,
)
from optio_optimize.bench.workloads import WORKLOADS, Workload

__all__ = [
    "WORKLOADS",
    "ABResult",
    "AnthropicProvider",
    "ArmResult",
    "OpenAIProvider",
    "QualityResult",
    "SimulatedProvider",
    "SpendGuard",
    "Workload",
    "available_live_provider",
    "compare",
    "format_result",
    "run_arm",
]
