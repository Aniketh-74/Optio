"""What does a provider actually do when a request will not fit its window?

``OptimizeConfig.context_limit`` is accepted, validated, and read by nothing.
Before giving it a consumer, this establishes what the failure it would prevent
actually *is* -- because the two candidate behaviours want opposite designs:

* If the provider **truncates**, the caller gets a short answer and a
  ``finish_reason``, and the right response is a diagnostic saying so.
* If the provider **rejects**, the caller gets an exception, no answer, and
  fail-open re-sends the same doomed request at full price. That is worth
  preventing, not merely reporting.

It also asks the provider for the numbers rather than copying them off a
documentation page. An error that states its own limit is better evidence than
a table someone transcribed: it is current by construction, and it is the
number the API will actually enforce.

**What it found, on 2026-08-01 against ``claude-haiku-4-5`` (ADR-037):**

* ``max_tokens`` above the model's **output cap** is a hard 400, and the error
  names the cap: ``max_tokens: 1000000 > 64000``. This limit is real, enforced,
  and separate from the context window.
* A prompt longer than the **context window** is a hard 400, and the error
  names the window: ``prompt is too long: 217570 tokens > 200000 maximum``.
* ``prompt + max_tokens`` over the window is **accepted**. 158,965 + 21,000 =
  179,965 against a 200,000 window went through and generated normally. The
  provider does not add the two and reject the sum.

That third result is why this script was worth running before writing code: the
guard it was going to inform would have prevented a failure that does not
happen.

**Cost.** Two of the three probes are rejected before generation and bill
nothing. The middle one is *accepted*, so it bills its 158,965 input tokens --
about $0.16 on Haiku, plus whatever it generates. Say so plainly rather than
claiming the script is free, since being accepted is the whole finding.

Run: ``python scripts/measure_window_pressure.py``.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_key() -> None:
    """Read the key in-process only; never onto a command line."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
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

from anthropic import Anthropic, APIStatusError  # noqa: E402

MODEL = "claude-haiku-4-5"

#: A paragraph of ordinary prose, repeated to reach a target size. Content is
#: irrelevant -- only its token count matters -- but real words tokenize at a
#: realistic ratio, which a run of "aaaa" would not.
_PARAGRAPH = (
    "The billing system reconciles usage against the ledger at the close of "
    "each period, and any discrepancy is escalated to the operations team for "
    "manual review before the invoice is issued to the customer. "
)


def _count(client: Anthropic, prompt: str) -> int:
    """Exact prompt tokens, free."""
    return int(
        client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        ).input_tokens
    )


def _probe(client: Anthropic, label: str, prompt: str, max_tokens: int) -> None:
    """Send one request expected to fail, and print what the provider said."""
    print(f"\n--- {label} ---")
    try:
        exact = _count(client, prompt)
        print(f"    prompt tokens (exact, free): {exact:,}")
        print(f"    max_tokens requested       : {max_tokens:,}")
        print(f"    prompt + max_tokens        : {exact + max_tokens:,}")
    except APIStatusError as exc:
        print(f"    count_tokens rejected: {exc.status_code} {exc.message}")

    try:
        client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIStatusError as exc:
        print(f"    REJECTED  status={exc.status_code}")
        print(f"    message : {exc.message}")
    except ValueError as exc:
        print(f"    REFUSED CLIENT-SIDE by the SDK: {exc}")
    else:
        print("    ACCEPTED -- the provider did not reject this. It truncates instead.")


def _repeat_to(client: Anthropic, target: int) -> str:
    """A prompt of roughly ``target`` tokens, calibrated against the provider."""
    per = _count(client, _PARAGRAPH) / 1.0
    return _PARAGRAPH * max(1, int(target / per))


_OUTPUT_CAP = re.compile(r"maximum allowed number of output tokens", re.I)
_WINDOW = re.compile(r"prompt is too long:\s*\d+\s*tokens\s*>\s*(\d+)\s*maximum", re.I)
_CAP_VALUE = re.compile(r"max_tokens:\s*\d+\s*>\s*(\d+)")


def _limits_for(client: Anthropic, model: str, long_prompt: str) -> tuple[int | None, int | None]:
    """Ask the provider for one model's window and output cap. Costs nothing.

    Both probes are *designed* to be rejected, and a rejected request is never
    billed, so the whole table below is free. The numbers come out of the error
    messages, which makes them the provider's own arithmetic rather than a
    transcription -- the same reasoning ADR-036 used for ``count_tokens``.
    """
    window: int | None = None
    cap: int | None = None

    try:
        with client.messages.stream(
            model=model, max_tokens=10_000_000, messages=[{"role": "user", "content": "hi"}]
        ) as stream:
            stream.get_final_message()
    except APIStatusError as exc:
        found = _CAP_VALUE.search(exc.message)
        if found and _OUTPUT_CAP.search(exc.message):
            cap = int(found.group(1))

    if cap is None:
        # Without the cap there is no way to make the next probe unbillable,
        # and a probe that might bill is not one this script may send.
        return None, None

    try:
        # `max_tokens` deliberately **one above the model's own output cap**, so
        # this request is invalid however long the prompt turns out to be.
        # Validation precedes generation, so it cannot bill either way.
        #
        # The first version of this function sent `max_tokens=16` and assumed a
        # long-enough prompt guaranteed rejection. It does not: a model whose
        # window exceeds the probe simply *accepts* it and bills the lot. Seven
        # models did, for about $7.60, and the table recorded their windows as
        # "-" -- so the run paid full price for the one outcome it could not
        # read. An over-cap `max_tokens` removes the assumption entirely.
        client.messages.create(
            model=model, max_tokens=cap + 1, messages=[{"role": "user", "content": long_prompt}]
        )
    except APIStatusError as exc:
        found = _WINDOW.search(exc.message)
        if found:
            window = int(found.group(1))
        # No match means the output-cap validator answered first, which tells
        # us only that the prompt fit. Reported as unknown, never as a guess.

    return window, cap


def _table(client: Anthropic) -> None:
    """Print every priced Anthropic model's two limits, as the provider states them."""
    from optio_optimize.config import PRICING

    # Comfortably past any published window, so the rejection always fires.
    long_prompt = _PARAGRAPH * 8000
    print(f"\n{'model':<24} {'context window':>15} {'max output':>12}")
    print("-" * 53)
    for model in PRICING:
        if not model.startswith("claude"):
            continue
        window, cap = _limits_for(client, model, long_prompt)
        print(f"{model:<24} {window or '-':>15} {cap or '-':>12}")
    print("\nEvery request above was rejected before generation. Spend: nothing.")


def main() -> None:
    """Run the probes and print what each one established."""
    client = Anthropic()

    if "--table" in sys.argv:
        _table(client)
        return

    # The SDK refuses any non-streaming call whose `max_tokens` implies more
    # than ten minutes of generation: `3600 * max_tokens / 128_000 > 600`, so
    # 21,333 is the ceiling. Worth recording -- it means a guard that *lowers*
    # `max_tokens` can never itself trip this, only a caller raising one can.
    ceiling = 21_000

    # 1. The output cap, which is a different limit from the context window and
    #    must not be confused with it. Streaming, because the SDK's own guard
    #    above would otherwise refuse before the provider sees it.
    print("\n--- max_tokens far above any output cap (streamed) ---")
    try:
        with client.messages.stream(
            model=MODEL, max_tokens=1_000_000, messages=[{"role": "user", "content": "Say OK."}]
        ) as stream:
            stream.get_final_message()
    except APIStatusError as exc:
        print(f"    REJECTED  status={exc.status_code}")
        print(f"    message : {exc.message}")

    # 2. The case the guard would prevent: a prompt that fits on its own, plus
    #    a ceiling that pushes the total over. This is the shape a long agent
    #    conversation reaches naturally.
    _probe(
        client,
        "prompt fits, prompt + max_tokens does not",
        _repeat_to(client, 190_000),
        ceiling,
    )

    # 3. A prompt that does not fit on its own. Nothing can rescue this one;
    #    the question is only whether it is reported clearly.
    _probe(client, "prompt alone exceeds the window", _repeat_to(client, 260_000), 1_024)

    print(
        "\nSpend: the two rejected probes bill nothing. The middle one is accepted "
        "and bills its ~159k input tokens (~$0.16 on Haiku) -- which is the finding."
    )


if __name__ == "__main__":
    main()
