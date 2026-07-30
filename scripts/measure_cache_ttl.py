"""Does the one-hour TTL pay for itself? ADR-021's gate, and it has to wait.

A 5-minute cache entry costs 1.25x base input to write, a one-hour entry 2.0x,
and a read costs 0.1x from either. The case for the hour is an agent step slower
than five minutes: the entry expires between steps, so every step pays a fresh
write on a prefix it just wrote.

**Measuring that means actually waiting out the window.** A shorter run would
measure the case where five minutes is already sufficient, which is the case the
lever is not for. So this sleeps ~5.5 minutes between rounds, and the whole run
takes about 17 minutes for eight calls.

**Both arms run inside the same wall-clock window**, interleaved round by round
with their own nonces, so they share separate cache entries but identical timing.
Two sequential arms would have needed twice the wall time and would have let
time-of-day differences in the provider's behaviour land on only one of them.

What the arms are:

* ``5m`` -- ``cache_ttl_selection`` off. Every round writes a fresh 5-minute
  entry, because the previous one expired during the sleep. This is what the
  package does today.
* ``1h`` -- the flag on. Round 1 writes a 5-minute entry like the other arm,
  because nothing has been observed yet; round 2 sees the same prefix after a gap
  past the window, records the expiry, and writes a one-hour entry; rounds 3 and 4
  read it at 0.1x.

That first-round cost is real and deliberate. Asking for an hour on a prefix
seen once would be a 60% premium on a write that may never be read, and this is
the only lever in the package that can raise a bill.

Usage::

    python scripts/measure_cache_ttl.py

Spends real money -- roughly $0.06 -- and takes about 17 minutes.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

MODEL = "claude-haiku-4-5"

#: Rounds per arm. Four shows the compounding: the one-hour arm pays its
#: observation cost once and reads twice, while the five-minute arm writes four
#: times.
ROUNDS = 4

#: Gap between rounds. Must exceed the 300-second window by enough that clock
#: skew and request latency cannot leave the entry alive -- a gap that only just
#: clears it would make a negative result unfalsifiable.
GAP_SECONDS = 330.0

_PREFIX_BODY = "Consider precedent, documentation, and the policy schedule before answering. " * 300


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
from optio_optimize.config import PRICING, OptimizeConfig  # noqa: E402
from optio_optimize.savings import _cost  # noqa: E402


def _config(*, ttl_selection: bool) -> OptimizeConfig:
    """One flag differs; everything else is held off in both arms."""
    return OptimizeConfig(
        cache_ttl_selection=ttl_selection,
        prefix_cache=True,  # Anthropic caches nothing without the breakpoint
        exact_cache=False,  # a local hit would remove the call from the wire
        trim_history=False,
        cap_tool_results=False,
        minify_tools=False,
        structured_output=False,
        adaptive_max_tokens=False,
        deduplicate=False,
        prune_retrieval=False,
        detect_unstable_prefix=False,
    )


class Arm:
    """One arm: its own client, optimizer, nonce and running totals."""

    def __init__(self, label: str, *, ttl_selection: bool) -> None:
        self.label = label
        self.nonce = uuid.uuid4().hex[:12]
        self.rounds: list[tuple[int, int, int, int]] = []
        client = Anthropic()
        wrap_anthropic_client(client, optimizer=Optimizer(_config(ttl_selection=ttl_selection)))
        self._client = client

    @property
    def system(self) -> str:
        """The shared prefix, unique to this arm so its entry is its own."""
        return f"Session {self.nonce}. You are a meticulous claims adjuster. {_PREFIX_BODY}"

    def call(self, question: str) -> None:
        """Make one call and record what the provider billed."""
        reply = self._client.messages.create(
            model=MODEL,
            max_tokens=40,
            system=self.system,
            messages=[{"role": "user", "content": question}],
        )
        usage = reply.usage
        reads = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        writes = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        creation = getattr(usage, "cache_creation", None)
        writes_1h = int(getattr(creation, "ephemeral_1h_input_tokens", 0) or 0)
        total = int(usage.input_tokens) + reads + writes
        self.rounds.append((total, reads, writes, writes_1h))

    def usd(self) -> float:
        """Billed cost across every round, priced by band."""
        return sum(
            _cost(PRICING[MODEL], total, 0, reads, writes, writes_1h)
            for total, reads, writes, writes_1h in self.rounds
        )


QUESTIONS = [
    "Is water damage from a burst pipe covered?",
    "Is mould following a covered leak included?",
    "Does the deductible apply once or twice per event?",
    "How long does a claimant have to file?",
]


def main() -> None:
    """Run both arms interleaved, sleeping past the window between rounds."""
    arms = [Arm("5m", ttl_selection=False), Arm("1h", ttl_selection=True)]
    minutes = (ROUNDS - 1) * GAP_SECONDS / 60.0
    print(f"{ROUNDS} rounds, {GAP_SECONDS:.0f}s apart -- about {minutes:.0f} minutes of waiting\n")

    for index in range(ROUNDS):
        question = QUESTIONS[index % len(QUESTIONS)]
        for arm in arms:
            arm.call(question)
            total, reads, writes, writes_1h = arm.rounds[-1]
            band = "1h" if writes_1h else ("5m" if writes else "-")
            print(
                f"  round {index + 1} [{arm.label:>2}] input {total:>6,} "
                f"reads {reads:>6,} writes {writes:>6,} ({band})"
            )
        if index < ROUNDS - 1:
            print(f"  ... sleeping {GAP_SECONDS:.0f}s to let the 5-minute entry expire")
            time.sleep(GAP_SECONDS)

    print(f"\n{'arm':<5} {'reads':>9} {'5m writes':>11} {'1h writes':>11} {'cost':>10}")
    print("-" * 50)
    for arm in arms:
        reads = sum(r[1] for r in arm.rounds)
        writes_1h = sum(r[3] for r in arm.rounds)
        writes_5m = sum(r[2] for r in arm.rounds) - writes_1h
        print(f"{arm.label:<5} {reads:>9,} {writes_5m:>11,} {writes_1h:>11,} ${arm.usd():>9.5f}")

    baseline, treated = arms[0].usd(), arms[1].usd()
    if baseline:
        print(f"\ncost change from cache_ttl_selection: {(baseline - treated) / baseline:+.1%}")

    treated_reads = sum(r[1] for r in arms[1].rounds)
    control_reads = sum(r[1] for r in arms[0].rounds)
    if control_reads:
        print(
            "\nthe control arm read from the cache too, so the 5-minute entry did "
            "NOT expire between rounds -- the gap is too short and this measures "
            "nothing the lever is for"
        )
    elif treated_reads:
        print(
            "\nthe ordering held: the control re-wrote its prefix every round while "
            "the treated arm wrote once at 2x and then read at 0.1x"
        )
    else:
        print(
            "\nneither arm read anything. Either the breakpoint never landed, or "
            "the prefix is below this model's 4,096-token floor -- a broken "
            "measurement rather than a negative result"
        )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    main()
