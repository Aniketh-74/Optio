"""Does ``trim_history`` make the model more verbose? Isolate and find out.

The first full live Anthropic benchmark showed a pattern across four workloads
that only an end-to-end run with output measurement could surface:

===================== ================== ================== ==========
workload              trim_history saved output tokens      cost
===================== ================== ================== ==========
multi_turn_chat       1,188              378 -> 1,096       **-11.0%**
multi_turn_chat_long  35,673             997 -> 5,693       +10.9%
timestamped_agent     1,152              307 -> 649         **-2.8%**
rag_queries           0 (did not fire)   359 -> 303         +5.2%
===================== ================== ================== ==========

Wherever the stage fired, output roughly tripled; where it did not, output was
normal. The hypothesis is that dropping old turns removes the model's own short
replies from context, and with them the pattern it was matching -- so it reverts
to its default verbosity. Output bills at **5x input** on Haiku, so a 5.4% input
saving becomes an 11% cost *increase*.

Correlation across four workloads is suggestive and this project has been wrong
before on exactly that kind of evidence, so this isolates the stage: three arms,
one workload, same wall-clock window, only ``trim_history`` differing between
arms B and C.

* **A** -- optimizer off. The true baseline.
* **B** -- default config. What a caller gets today.
* **C** -- default config minus ``trim_history``. If output falls back toward
  A's, the stage is the cause; if it stays high, something else is.

Usage::

    python scripts/measure_trim_history_output.py

Spends real money -- roughly $0.07 on claude-haiku-4-5.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def _load_key() -> None:
    """Read the key in-process only; never onto a command line."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {"ANTHROPIC_API_KEY", "ANTHROPIC_KEY"}:
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("\"'")
            return


_load_key()

from optio_optimize.bench.harness import run_arm  # noqa: E402
from optio_optimize.bench.providers import AnthropicProvider, SpendGuard  # noqa: E402
from optio_optimize.bench.workloads import WORKLOADS  # noqa: E402
from optio_optimize.config import PRICING, OptimizeConfig  # noqa: E402
from optio_optimize.savings import _cost  # noqa: E402

WORKLOAD = "multi_turn_chat"
MODEL = AnthropicProvider.DEFAULT_MODEL

ARMS: list[tuple[str, OptimizeConfig]] = [
    ("A off (baseline)", OptimizeConfig(enabled=False)),
    ("B default", OptimizeConfig()),
    ("C no trim_history", OptimizeConfig(disabled_stages=frozenset({"trim_history"}))),
]


def main() -> None:
    """Run the three arms and print input, output and cost for each."""
    guard = SpendGuard(cap_usd=0.50)
    provider = AnthropicProvider(guard=guard)
    pricing = PRICING[MODEL]
    workload = WORKLOADS[WORKLOAD]

    print(f"{WORKLOAD} on {MODEL}, three arms\n")
    print(f"{'arm':<20}{'input':>9}{'output':>9}{'cost':>11}{'vs A':>9}")
    print("-" * 58)

    baseline_cost = None
    for label, config in ARMS:
        result, _ = run_arm(label, workload.build(), provider, config)
        cost = _cost(pricing, result.input_tokens, result.output_tokens, result.cached_input_tokens)
        if baseline_cost is None:
            baseline_cost = cost
        delta = (baseline_cost - cost) / baseline_cost if baseline_cost else 0.0
        print(
            f"{label:<20}{result.input_tokens:>9,}{result.output_tokens:>9,}"
            f"${cost:>10.5f}{delta:>+9.1%}"
        )

    print(f"\nspent ${guard.spent_usd:.4f}")
    print(
        "\nIf C's output is near A's while B's is far above it, trim_history is "
        "the cause. If B and C are both high, it is not -- and the four-workload "
        "correlation was a coincidence."
    )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    main()
