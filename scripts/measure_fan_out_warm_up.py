"""Does warm-up ordering actually turn writes into reads? ADR-020's gate.

The arithmetic is easy and it is not the claim. Five cold parallel calls over a
shared prefix *should* each pay Anthropic's 1.25x write premium, and sending one
first *should* leave four reads at 0.1x -- 74% off the shared prefix. Whether the
provider behaves that way under real concurrency is a question only the provider
answers, and this package's history on that is four for four: a modelled 36.3%
saving measured -1.8%, a 53.7% figure was 50.1%, a 2,048-token floor was 4,096,
and a reasoning stage's whole safety argument dissolved on contact.

**Three arms, unwarmed / warmed / unwarmed.** The bracketing is not caution for
its own sake -- it is the only way to separate the ordering from the clock, since
the provider's cache lives for five minutes and anything the first arm writes the
later arms could read for free.

**Each arm gets its own nonce at the very start of the system prompt.** Without
it the second arm reads what the first one wrote and measures nothing at all: the
prefix would already be warm before its first call. The nonce leads the prompt so
the arms share no cacheable sub-prefix whatsoever.

Usage::

    python scripts/measure_fan_out_warm_up.py

Spends real money -- roughly $0.10.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

MODEL = "claude-haiku-4-5"

#: Branches in the fan-out. Five is enough for the effect to clear the noise and
#: small enough that three arms cost pennies.
BRANCHES = 5

#: Above this model's real 4,096-token floor. Below it Anthropic ignores the
#: breakpoint, nothing is cached, and both arms measure zero -- a broken
#: measurement that reads exactly like a negative result.
_PREFIX_BODY = "Consider precedent, documentation, and the policy schedule before answering. " * 300


def _system_prompt(nonce: str) -> str:
    """The shared prefix, made unique to one arm.

    The nonce goes first so the arms share no cacheable sub-prefix. Placed last,
    it would leave everything above it identical and the later arms would read
    the earlier arm's cache entry rather than their own.
    """
    return f"Session {nonce}. You are a meticulous claims adjuster. {_PREFIX_BODY}"


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

from anthropic import AsyncAnthropic  # noqa: E402

from optio_optimize import LLMRequest, LLMResponse, Message, Optimizer, wire  # noqa: E402
from optio_optimize.config import PRICING, OptimizeConfig  # noqa: E402
from optio_optimize.savings import _cost  # noqa: E402


def _config() -> OptimizeConfig:
    """Only ``prefix_cache`` on: Anthropic caches nothing without the marker.

    ``exact_cache`` off, and it matters here more than usual -- the fan-out's
    branches are distinct, but a local hit would remove a call from the wire
    entirely and the arms would no longer be comparable.
    """
    return OptimizeConfig(
        prefix_cache=True,
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


def _requests(nonce: str) -> list[LLMRequest]:
    """A fan-out: one shared system prefix, one distinct question each."""
    head = Message(role="system", content=_system_prompt(nonce))
    questions = [
        "Is water damage from a burst pipe covered?",
        "Is mould following a covered leak included?",
        "Does the deductible apply once or twice per event?",
        "How long does a claimant have to file?",
        "What documentation is required for a claim?",
    ]
    return [
        LLMRequest(
            model=MODEL,
            messages=(head, Message(role="user", content=question)),
            max_tokens=80,
            temperature=0.0,
        )
        for question in questions[:BRANCHES]
    ]


async def _arm(*, warmed: bool) -> tuple[int, int, int, float]:
    """Run one fan-out.

    Args:
        warmed: Whether to send one call first so the rest read its cache write.

    Returns:
        ``(input_tokens, cache_reads, cache_writes, usd)``, summed across the
        fan-out, as the provider reported them.
    """
    client = AsyncAnthropic()
    optimizer = Optimizer(_config())
    requests = _requests(uuid.uuid4().hex[:12])

    async def provider(request: LLMRequest) -> LLMResponse:
        reply = await client.messages.create(**wire.anthropic_body(request, MODEL))
        return wire.response_from_anthropic_message(reply)

    if warmed:
        responses = await optimizer.afan_out(requests, provider)
    else:
        # The honest control: same stages, same markers, same client -- every
        # request through the ordinary path, all at once, which is what a caller
        # writes today.
        responses = list(await asyncio.gather(*(optimizer.acall(r, provider) for r in requests)))

    tokens = sum(r.input_tokens for r in responses)
    reads = sum(r.cached_input_tokens for r in responses)
    writes = sum(r.cache_write_tokens for r in responses)
    output = sum(r.output_tokens for r in responses)
    usd = _cost(PRICING[MODEL], tokens, output, reads, writes)
    await client.close()
    return tokens, reads, writes, usd


async def main() -> None:
    """Run the three arms and print what the provider actually did."""
    arms: list[tuple[str, tuple[int, int, int, float]]] = []
    for label, warmed in (("cold", False), ("warmed", True), ("cold-2", False)):
        print(f"running {label} ({'warm-up first' if warmed else 'all at once'})...")
        arms.append((label, await _arm(warmed=warmed)))

    print(f"\n{'arm':<9} {'input':>9} {'reads':>9} {'writes':>9} {'cost':>10}")
    print("-" * 50)
    for label, (tokens, reads, writes, usd) in arms:
        print(f"{label:<9} {tokens:>9,} {reads:>9,} {writes:>9,} ${usd:>9.5f}")

    cold = [usd for label, (_, _, _, usd) in arms if label.startswith("cold")]
    warmed_usd = next(usd for label, (_, _, _, usd) in arms if label == "warmed")
    baseline = sum(cold) / len(cold)
    change = (baseline - warmed_usd) / baseline
    print(f"\ncost change against the mean of both cold arms: {change:+.1%}")
    print(f"spread between the two cold arms alone: {abs(cold[0] - cold[1]) / baseline:.1%}")

    warmed_reads = next(reads for label, (_, reads, _, _) in arms if label == "warmed")
    cold_reads = [reads for label, (_, reads, _, _) in arms if label.startswith("cold")]
    if warmed_reads and not any(cold_reads):
        print(
            "\nthe ordering did what ADR-020 claims: the cold arms read nothing and "
            "the warmed arm read a prefix one of its own calls had just written"
        )
    elif any(cold_reads):
        print(
            "\nthe cold arms read from the cache too, so the provider is populating "
            "it faster than concurrent calls can miss it -- the ordering is worth "
            "less than the arithmetic suggests, and this is the number that says so"
        )
    else:
        print(
            "\nno arm read anything. The breakpoint never landed, or the prefix is "
            "below this model's 4,096-token floor -- a broken measurement, not a "
            "negative result"
        )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    asyncio.run(main())
