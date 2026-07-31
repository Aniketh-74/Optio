"""Cascade against a real model: cheap first, verify, escalate on rejection.

ADR-023's technique had never been run against a live provider. Everything below
the verifier was covered by unit tests with fake providers, which proves the
control flow and proves nothing about whether a cheap model's answers actually
survive :func:`~optio_optimize.cascade.default_verifier` on real traffic -- the
only question that decides whether the technique saves money or just adds a
round trip.

**Why this does not go through the benchmark harness.** ``AnthropicProvider``
sends ``model=self.model`` and ignores ``request.model``. Cascade works by
rewriting exactly that field, so routed through the bench both the cheap attempt
and the escalation would hit the same model and the "saving" would be fiction.
The provider closure here honours the request, which is also what a real caller's
own provider function does -- so this exercises the production path (``Optimizer``
-> ``CascadeRouter`` -> provider) rather than benchmark scaffolding.

The traffic mix is deliberate. Cascade is only interesting if *both* branches
run, so it carries requests a cheap model should handle, one that is guaranteed
to be rejected (a ``max_tokens`` low enough to truncate, which
``default_verifier`` catches through `LLMResponse.was_truncated` -- and that
case is why ADR-033 exists: it silently did NOT escalate on the first run,
because Anthropic reports `max_tokens` where the check compared `length`), and two
shapes whose outcome is genuinely unknown in advance -- a strict JSON schema and
a tool call to vet.

Run: ``python scripts/measure_cascade_live.py``
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Any, cast

ROOT = pathlib.Path("c:/Users/Hp/Desktop/Side Project")
sys.path.insert(0, str(ROOT / "src"))

for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
    if "=" in line:
        name, value = line.split("=", 1)
        if name.strip() in {"ANTHROPIC_API_KEY", "ANTHROPIC_KEY"}:
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("\"'")

from anthropic import Anthropic  # noqa: E402

from optio_optimize import wire  # noqa: E402
from optio_optimize.cascade import break_even_escalation_rate  # noqa: E402
from optio_optimize.config import OptimizeConfig  # noqa: E402
from optio_optimize.optimizer import Optimizer  # noqa: E402
from optio_optimize.types import LLMRequest, LLMResponse, Message  # noqa: E402

EXPENSIVE = "claude-sonnet-4-5"
CHEAP = "claude-haiku-4-5"

client = Anthropic()
served: list[str] = []


def provider(request: LLMRequest) -> LLMResponse:
    """Serve the model the request names -- which is the whole point here."""
    system, turns = wire.anthropic_system_and_turns(request)
    optional: dict[str, Any] = {}
    if system:
        optional["system"] = system
    tools = wire.anthropic_tools(request)
    if tools:
        optional["tools"] = tools
    reply = client.messages.create(
        model=request.model,
        max_tokens=request.max_tokens or 1024,
        messages=cast("Any", turns),
        temperature=request.temperature if request.temperature is not None else 1.0,
        **optional,
    )
    served.append(reply.model)
    return wire.response_from_anthropic_message(reply)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _request(prompt: str, **overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "model": EXPENSIVE,
        "messages": (
            Message(role="system", content="You are terse and precise."),
            Message(role="user", content=prompt),
        ),
        "temperature": 0.0,
    }
    base.update(overrides)
    return LLMRequest(**base)


#: (label, request, what this shape is here to exercise)
TRAFFIC: list[tuple[str, LLMRequest, str]] = [
    ("easy-1", _request("What is the capital of France? One word."), "cheap should pass"),
    ("easy-2", _request("What is 12 times 12? Digits only."), "cheap should pass"),
    ("easy-3", _request("Name the largest ocean. One word."), "cheap should pass"),
    ("easy-4", _request("What year did the Apollo 11 landing happen? Digits only."), "cheap"),
    (
        "truncated",
        _request("Explain the causes of the French Revolution in detail.", max_tokens=16),
        "GUARANTEED escalation: finish_reason == length",
    ),
    (
        "json-strict",
        _request(
            "Give the population and founding year of Kyoto.",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "population": {"type": "number"},
                            "founded": {"type": "number"},
                        },
                        "required": ["population", "founded"],
                    }
                },
            },
        ),
        "ADR-023 step 1: schema conformance",
    ),
    (
        "tool-call",
        _request("What is the weather in Kyoto?", tools=(_WEATHER_TOOL,)),
        "ADR-023 step 3: the proposal is vetted",
    ),
    (
        "tool-call-2",
        _request("Check the weather in Oslo in celsius.", tools=(_WEATHER_TOOL,)),
        "ADR-023 step 3",
    ),
]


def main() -> int:
    """Run the traffic mix through cascade and report whether it paid."""
    config = OptimizeConfig(
        cascade_routing=True,
        cheap_model=CHEAP,
        cascade_structured_output=True,
        cascade_tools=True,
        exact_cache=False,  # every request distinct; keep cascade the only actor
        prefix_cache=False,
    )
    optimizer = Optimizer(config)

    print(f"cascade: {EXPENSIVE}  <-  cheap {CHEAP}")
    print(f"{'request':<14} {'served':<28} {'verdict':<10} note")
    print("-" * 88)

    started = time.perf_counter()
    for label, request, why in TRAFFIC:
        before = len(served)
        try:
            response = optimizer.call(request, provider)
        except Exception as exc:  # noqa: BLE001 - a failure is a datum here
            print(f"{label:<14} {'-':<28} {'ERROR':<10} {type(exc).__name__}: {exc}")
            continue
        calls = served[before:]
        verdict = "escalated" if len(calls) > 1 else "cheap"
        print(f"{label:<14} {' -> '.join(calls):<28} {verdict:<10} {why}")
        del response

    elapsed = time.perf_counter() - started
    stats = optimizer.cascade_stats
    assert stats is not None

    print(
        f"\nattempted {stats.attempted}   accepted cheap {stats.attempted - stats.escalated}   "
        f"escalated {stats.escalated}   skipped {stats.skipped}"
    )
    rate = stats.escalation_rate
    print(f"escalation rate: {'n/a' if rate is None else f'{rate:.1%}'}")
    print(
        f"latency: cheap {stats.cheap_ms:.0f} ms   escalation {stats.escalation_ms:.0f} ms   "
        f"wall {elapsed * 1000:.0f} ms"
    )

    cost = stats.cost_summary(EXPENSIVE, CHEAP)
    if cost is None:
        print("cost: unpriced model")
        return 0
    print(
        f"\ncheap spend        ${cost.cheap_spend_usd:.6f}"
        f"\nescalation spend   ${cost.escalation_spend_usd:.6f}"
        f"\ntotal spend        ${cost.total_spend_usd:.6f}"
        f"\nall-expensive      ${cost.all_expensive_baseline_usd:.6f}"
        "  (projected for cheap-accepted)"
        f"\nnet saving         ${cost.net_saving_usd:.6f}"
        f"\nescalation waste   ${cost.escalation_waste_usd:.6f}  (rejected cheap attempts)"
    )
    if cost.all_expensive_baseline_usd > 0:
        print(f"net saving         {cost.net_saving_usd / cost.all_expensive_baseline_usd:.1%}")

    # The pair that decides whether the technique paid. `escalation_rate` counts
    # requests; the bill weights them, and the requests that fail a verifier are
    # systematically the expensive ones (ADR-034).
    weighted = cost.cost_weighted_escalation_rate
    break_even = break_even_escalation_rate(EXPENSIVE, CHEAP)
    print(
        f"\nescalation by count   {'n/a' if rate is None else f'{rate:.1%}'}"
        f"\nescalation by cost    {'n/a' if weighted is None else f'{weighted:.1%}'}"
        f"\nbreak-even            {'n/a' if break_even is None else f'{break_even:.1%}'}"
    )
    if weighted is not None and break_even is not None:
        print(f"verdict               {'PAYING' if weighted < break_even else 'COSTING MORE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
