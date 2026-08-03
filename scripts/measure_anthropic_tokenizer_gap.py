"""How wrong is tiktoken about Anthropic prose (ADR-048)?

Every token count this package makes on an Anthropic model currently goes
through ``tiktoken``, whose ``encoding_for_model`` does not know Anthropic and
falls back to ``o200k_base``. That fallback is reasonable — it is much closer
than refusing — but it is OpenAI's tokenizer applied to another vendor, and this
project has already measured that the two are not interchangeable: ADR-036 found
Anthropic billing **1.29x** the raw-JSON count for tool schemas against OpenAI's
**0.65**. Opposite directions.

That measurement covered tool schemas. Nobody has measured **prose**, which is
what the other eight stages count.

This asks. ``messages.count_tokens`` is exact and **bills nothing**, so the only
cost of running it is an API key and a few seconds — no credit is consumed. Run
it, paste the ratio into a calibration table with today's date, and every
Anthropic figure this package reports stops being an OpenAI estimate.

    python scripts/measure_anthropic_tokenizer_gap.py

Prints a table and a suggested constant. It does not write one: a calibration
constant with no date beside it is the shape of a number nobody can re-check
(ADR-039).
"""

from __future__ import annotations

import sys

from optio_optimize.adapters.anthropic_tokens import AnthropicCounter
from optio_optimize.tokens import TiktokenCounter

#: Text shapes the stages actually count, rather than one lorem ipsum. Prose,
#: chat turns, JSON tool results and code fragment differently -- ``tokens.py``
#: already distinguishes prose from dense text for exactly this reason -- so a
#: single ratio measured on one of them would be calibrated for one stage and
#: wrong for the rest.
SAMPLES: dict[str, str] = {
    "prose": (
        "The claims adjuster reviews each submission against the policy schedule, "
        "confirms the deductible, and records the outcome with a short rationale. "
        "Where the schedule is silent the adjuster escalates rather than guessing. "
    )
    * 12,
    "chat turn": (
        "Can you check whether the reconciliation job finished, and if it did, "
        "tell me how many rows it rejected? I need the number before the standup. "
    )
    * 12,
    "json tool result": (
        '{"status":"ok","rows":1284,"rejected":17,"elapsed_ms":4213,'
        '"warnings":["duplicate key on row 88","null tenant on row 412"]}'
    )
    * 12,
    "code": (
        "def reconcile(rows: list[Row], *, strict: bool = False) -> Report:\n"
        "    rejected = [r for r in rows if not r.is_valid()]\n"
        "    if strict and rejected:\n"
        "        raise ReconcileError(len(rejected))\n"
        "    return Report(total=len(rows), rejected=len(rejected))\n"
    )
    * 12,
}

MODEL = "claude-haiku-4-5"


def main() -> int:
    """Compare tiktoken's estimate against Anthropic's own count."""
    try:
        exact = AnthropicCounter(default_model=MODEL)
    except ImportError:
        print("the `anthropic` package is not installed: pip install anthropic", file=sys.stderr)
        return 2

    estimate = TiktokenCounter()

    print(f"model: {MODEL}   (count_tokens is exact and bills nothing)\n")
    print(f"{'sample':<18} {'tiktoken':>10} {'anthropic':>10} {'real/est':>10}")
    print("-" * 52)

    ratios: list[float] = []
    for name, text in SAMPLES.items():
        try:
            real = exact.count_text(text, MODEL)
        except Exception as exc:  # noqa: BLE001 - a script; the message is the point
            print(f"\ncount_tokens failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("check ANTHROPIC_API_KEY. the endpoint is free but still needs a valid key.")
            return 1
        est = estimate.count_text(text, MODEL)
        ratio = real / est if est else float("nan")
        ratios.append(ratio)
        print(f"{name:<18} {est:>10,} {real:>10,} {ratio:>10.3f}")

    spread = max(ratios) - min(ratios)
    mean = sum(ratios) / len(ratios)
    print("-" * 52)
    print(f"{'mean':<18} {'':>10} {'':>10} {mean:>10.3f}")
    print(f"{'spread':<18} {'':>10} {'':>10} {spread:>10.3f}")

    print()
    if spread > 0.10:
        # One constant cannot serve four shapes that disagree this much, and
        # averaging them would be precise about the wrong thing.
        print(
            f"The ratio varies by {spread:.3f} across text shapes, which is too wide for a\n"
            "single constant. Calibrate per shape, the way TOOL_SCHEMA_CALIBRATION_BY_MODEL\n"
            "is keyed per vendor, rather than averaging these into one number."
        )
    else:
        print(
            f"Consistent across shapes (spread {spread:.3f}). A single constant of "
            f"{mean:.3f}\nwould correct tiktoken for Anthropic prose. Record it with today's "
            "date and\nthis script's name beside it, the way ADR-036's table does."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
