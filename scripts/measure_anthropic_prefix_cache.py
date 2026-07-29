"""Is PrefixCacheStage worth what its docstring claims? Measure it.

``caching.py`` calls the prefix marker the difference between a ~90% input
discount and none, worth "roughly 30% of total spend on a long conversation".
That number has never been measured. It is also the stage this package leans on
hardest, and the same class of claim that was already wrong once: modelling only
explicit-style caching credited this library with a 36.3% saving on
``multi_turn_chat`` that OpenAI grants unconditionally, and the live run measured
**-1.8%**.

ADR-015 rule 2: isolated, one stage at a time. Everything is held constant
except ``prefix_cache``, and the disabled arm runs first so any residual
server-side cache favours the baseline rather than the result this library
wants.

Usage::

    python scripts/measure_anthropic_prefix_cache.py

Spends real money -- roughly $0.05.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

MODEL = "claude-haiku-4-5"

#: Anthropic ignores a ``cache_control`` breakpoint below a per-model floor, and
#: **that floor is 4,096 tokens on this model**. The published minimums span a
#: factor of eight -- 512 on Opus 5, 1,024 on Sonnet 5 and Opus 4.8, 2,048 on
#: Opus 4.7 and Haiku 3.5, 4,096 on Haiku 4.5 and Opus 4.6/4.5 -- so
#: ``MIN_PREFIX_TOKENS = 1024`` is not a floor for anything in particular.
#:
#: The first version of this script used a 1,449-token prompt: above that
#: constant, below this model's real floor, and it measured zero cache reads in
#: both arms. That is a broken measurement, not a negative result, and it is the
#: exact failure this length exists to prevent -- below the floor the provider
#: silently ignores the breakpoint, and a reader sees "the stage does nothing"
#: where the truth is "the stage was never given a chance". This was also
#: recorded as a 2,048 floor for one commit, which is Haiku *3.5*'s.
SYSTEM_PROMPT = "You are a meticulous claims adjuster. Follow these rules exactly. " + (
    "Consider precedent, documentation, and the policy schedule before answering. " * 300
)

TURNS = [
    "Is water damage from a burst pipe covered?",
    "What about the resulting mould?",
    "Does the deductible apply once or twice?",
    "How long does the claimant have to file?",
    "What documentation is required?",
    "Summarize your answers in three lines.",
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
from anthropic.types import MessageParam, TextBlock  # noqa: E402

from optio_optimize import Optimizer  # noqa: E402
from optio_optimize.adapters.anthropic import wrap_anthropic_client  # noqa: E402
from optio_optimize.config import PRICING  # noqa: E402
from optio_optimize.savings import _cost  # noqa: E402


def run(*, prefix_cache: bool) -> tuple[int, int, int, int, float]:
    """Hold one conversation.

    Returns:
        ``(input_tokens, cached_tokens, written_tokens, output_tokens, usd)``.
        Input includes both cache reads and cache writes, matching what this
        package means by the field everywhere else.
    """
    client = Anthropic()
    optimizer = Optimizer(
        prefix_cache=prefix_cache,
        exact_cache=False,
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

    # The SDK's own param type rather than ``list[dict[str, str]]``: the latter
    # is not assignable to ``Iterable[MessageParam]``, since a plain ``str``
    # value does not narrow to the role literal the TypedDict requires.
    history: list[MessageParam] = []
    totals = [0, 0, 0]
    created = 0
    for turn in TURNS:
        history.append({"role": "user", "content": turn})
        reply = client.messages.create(
            model=MODEL, max_tokens=300, system=SYSTEM_PROMPT, messages=history
        )
        # isinstance rather than a `.type == "text"` string check: only the
        # former lets mypy narrow the block union so `.text` is known to exist.
        # bench/providers.py made the same correction for the same reason.
        text = "".join(b.text for b in reply.content if isinstance(b, TextBlock))
        history.append({"role": "assistant", "content": text})
        cached = getattr(reply.usage, "cache_read_input_tokens", 0) or 0
        written = getattr(reply.usage, "cache_creation_input_tokens", 0) or 0
        created += written
        totals[0] += reply.usage.input_tokens + cached + written
        totals[1] += cached
        totals[2] += reply.usage.output_tokens

    # Writes are reported separately so a zero-read result can be diagnosed
    # rather than believed: reads of 0 with writes of 0 means the breakpoint was
    # ignored (below the model's floor), while reads of 0 with writes above it
    # means the prefix is changing between calls. Those are different bugs and
    # the totals alone cannot tell them apart.
    print(f"  [{'on ' if prefix_cache else 'off'}] cache writes: {created:,} tokens")
    # Writes are passed to _cost, not folded into base input. They bill at 1.25x
    # here, and the first version of this script omitted the argument -- pricing
    # them at 1.00x and reporting 53.7% where the answer is 50.1%.
    usd = _cost(PRICING[MODEL], totals[0], totals[2], totals[1], created)
    return totals[0], totals[1], created, totals[2], usd


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    off = run(prefix_cache=False)
    on = run(prefix_cache=True)

    print(f"\n{'arm':<10} {'input':>9} {'cached':>9} {'written':>9} {'output':>8} {'cost':>10}")
    print("-" * 60)
    for label, (tokens_in, cached, written, tokens_out, usd) in (("off", off), ("on", on)):
        print(
            f"{label:<10} {tokens_in:>9,} {cached:>9,} {written:>9,} {tokens_out:>8,} ${usd:>9.5f}"
        )
    if off[4]:
        print(f"\ncost reduction from prefix_cache: {(off[4] - on[4]) / off[4]:.1%}")
    print(f"cache-served input: off {off[1]:,}/{off[0]:,}, on {on[1]:,}/{on[0]:,}")
    print(
        "\nThe token counts are the measurement. The dollar figures are derived "
        f"from PRICING[{MODEL!r}] and are only as current as that table."
    )
