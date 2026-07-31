"""Does a *streamed* call reach the provider's cache? ADR-019's gate.

The unit tests prove a ``cache_control`` breakpoint appears in the request body a
mocked transport receives. That is necessary and it is not the claim. The claim
is that Anthropic *acts* on it, and only Anthropic can answer that -- the same
distinction that turned a simulated 36.3% saving into a measured -1.8% once, and
that left one earlier version of the non-streaming script measuring zero cache
reads in both arms because the prompt sat below this model's real floor.

So this sends two streamed calls sharing a long system prompt and reads
``cache_read_input_tokens`` off the second. Non-zero means the breakpoint this
package placed on a streaming request was honoured. Zero means it was not, and
the diagnostics distinguish the two ways that happens: no writes at all means the
breakpoint never landed, writes without reads means the prefix is changing
between calls.

It also checks the half that is specific to streaming: the numbers have to arrive
through the event accumulator rather than off a complete ``Message``, so
``report.provider_cached_tokens`` is the proof that
:class:`~optio_optimize.adapters.anthropic_streaming.StreamProxy` threaded the
usage through. A cache read the provider granted and this package failed to
notice is invisible in every report it prints.

``exact_cache`` is off: it would serve the second call locally and the provider
would never see it, which measures this package's own cache rather than the
provider's.

Usage::

    python scripts/measure_streaming_prefix_cache.py

Spends real money -- roughly $0.02.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

MODEL = "claude-haiku-4-5"

#: Above this model's real minimum cacheable prefix, which is **4,096 tokens** --
#: not the 1,024 in ``MIN_PREFIX_TOKENS``, and not the 2,048 that is Haiku 3.5's.
#: Below the floor Anthropic silently ignores the breakpoint, and the result reads
#: as "streaming caching does not work" when the truth is "it was never given a
#: chance". That failure has already been shipped once in this repo.
#:
#: **And then it shipped again, here.** The comment above said "above 4,096" and
#: the prompt measured 3,614, so every run of this script reported
#: ``reads 0 writes 0`` -- ADR-019's gate never once passed, because
#: ``prefix_cache`` correctly declined to mark a prefix below the floor and there
#: was never a breakpoint to measure. An intent in a comment is not a
#: measurement; :func:`_require_eligible_prefix` now checks it before spending.
SYSTEM_PROMPT = "You are a meticulous claims adjuster. Follow these rules exactly. " + (
    "Consider precedent, documentation, and the policy schedule before answering. " * 450
)

TURNS = [
    "Is water damage from a burst pipe covered?",
    "What about the resulting mould?",
]


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

from anthropic import Anthropic  # noqa: E402

from optio_optimize import Optimizer  # noqa: E402
from optio_optimize.adapters.anthropic import wrap_anthropic_client  # noqa: E402


def _require_eligible_prefix() -> int:
    """Refuse to spend money on a prompt that cannot answer the question.

    ``reads 0 / writes 0`` has two causes and this script exists to tell them
    apart: the marker failed to reach the wire (a bug in this package), or the
    prefix was below the model's floor so ``prefix_cache`` correctly declined
    and there was never a breakpoint at all (a bad fixture). Only the first is a
    finding, and a run that cannot distinguish them costs money to learn
    nothing.

    Checked here rather than asserted in a comment, because the comment above
    said "above 4,096" while the prompt measured 3,614 and every run of this
    script silently measured nothing.

    Returns:
        The stable prefix size in tokens.

    Raises:
        SystemExit: If the prefix is below this model's floor.
    """
    from optio_optimize.stages.caching import min_prefix_tokens_for
    from optio_optimize.tokens import default_counter

    floor = min_prefix_tokens_for(MODEL)
    tokens = default_counter().count_text(SYSTEM_PROMPT, MODEL)
    if tokens < floor:
        raise SystemExit(
            f"refusing to run: the system prompt is ~{tokens:,} tokens, below {MODEL}'s "
            f"{floor:,}-token cacheable minimum. prefix_cache would correctly decline to "
            f"mark it, so the run would report 'reads 0 writes 0' and prove nothing. "
            f"Lengthen SYSTEM_PROMPT."
        )
    return tokens


def run() -> tuple[list[tuple[int, int, int]], int, int]:
    """Hold a two-turn streamed conversation.

    Returns:
        ``(per_call_usage, report_cached, report_written)`` where each usage entry
        is ``(input_tokens, cache_reads, cache_writes)`` as the *provider*
        reported it in that call's ``message_start``, and the two report figures
        are what this package noticed through the stream.
    """
    client = Anthropic()
    optimizer = Optimizer(
        prefix_cache=True,
        exact_cache=False,  # or the second call never leaves the process
        trim_history=False,
        cap_tool_results=False,
        minify_tools=False,
        structured_output=False,
        adaptive_max_tokens=False,
        deduplicate=False,
        prune_retrieval=False,
        detect_unstable_prefix=False,
    )
    wrap_anthropic_client(client, optimizer=optimizer)

    usage: list[tuple[int, int, int]] = []
    for turn in TURNS:
        with client.messages.create(
            model=MODEL,
            max_tokens=120,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": turn}],
            stream=True,
        ) as stream:
            for event in stream:
                if event.type != "message_start":
                    continue
                reported = event.message.usage
                reads = int(getattr(reported, "cache_read_input_tokens", 0) or 0)
                writes = int(getattr(reported, "cache_creation_input_tokens", 0) or 0)
                usage.append((int(reported.input_tokens) + reads + writes, reads, writes))

    report = optimizer.report
    return usage, report.provider_cached_tokens, report.provider_written_tokens


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    prefix_tokens = _require_eligible_prefix()
    print(f"stable prefix: ~{prefix_tokens:,} tokens ({MODEL}'s floor cleared)")

    per_call, report_cached, report_written = run()

    print(f"\n{'call':<6} {'input':>9} {'cache reads':>12} {'cache writes':>13}")
    print("-" * 44)
    for index, (total, reads, writes) in enumerate(per_call, start=1):
        print(f"{index:<6} {total:>9,} {reads:>12,} {writes:>13,}")

    reads_total = sum(entry[1] for entry in per_call)
    writes_total = sum(entry[2] for entry in per_call)
    if reads_total:
        print(f"\nthe breakpoint reached the provider on a streamed call: {reads_total:,} read")
    elif writes_total:
        print(
            f"\n{writes_total:,} written and nothing read: the breakpoint landed but the "
            f"prefix changed between calls, so nothing could hit it"
        )
    else:
        print(
            "\nnothing written and nothing read: the breakpoint never landed. Either "
            "the marker did not reach the wire, or the prefix is below this model's "
            "4,096-token floor and Anthropic ignored it"
        )

    print(
        f"\nwhat this package noticed through the stream: {report_cached:,} cached, "
        f"{report_written:,} written"
    )
    if report_cached != reads_total or report_written != writes_total:
        print(
            "  -> MISMATCH against the provider's own numbers above. The accumulator "
            "is dropping usage, so every saving derived from a streamed call is "
            "wrong in the report even though the provider granted the discount."
        )
