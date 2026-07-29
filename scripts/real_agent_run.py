"""Run a real agent, with real tools, through the optimizer. Twice.

Everything else that claims to test this package against a framework stops one
step short of a model. ``tests/frameworks/`` says so in its own docstring --
"Nothing here calls a model" -- because it checks adapter *recognition*. The
adapter tests build a genuine ``AsyncOpenAI`` client and mock its HTTP
transport, so the request bodies are real and the server is not. The benchmark
harness drives hand-authored message lists straight into ``Optimizer.call``.

None of those is an agent. An agent decides which tool to call, gets a payload
whose size it did not choose, appends it to a history it then re-sends, and
repeats until it is done. That loop is where this library's stages either earn
their keep or break something, and it is the one thing nothing in this repo has
ever run.

The precedent for bothering is in the adapter itself: ``_numeric_or_none``
exists because the Agents SDK's default ``ModelSettings()`` leaves optional
fields as a sentinel rather than ``None``, which made ``AdaptiveMaxTokensStage``
decline every request. Its docstring records that no hand-built test kwargs
would have hit it. This script is the general version of that discovery.

Usage::

    python scripts/real_agent_run.py            # both arms, ~$0.01
    python scripts/real_agent_run.py --arm=off  # baseline only

**This spends money.** Two arms of a tool-calling loop on gpt-4o-mini, bounded
by a spend guard.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

#: The task. Deliberately needs several tools in sequence -- the agent cannot
#: answer it from one call, so history accumulates the way it does in production
#: rather than the way a fixture does.
TASK = (
    "Customer alice@example.com reports that the blue widget in their most "
    "recent order arrived damaged. Find that order, look up its full detail, "
    "check whether the item is back in stock, and tell me the refund amount "
    "and whether we can send a replacement. Answer in two sentences."
)

MODEL = "gpt-4o-mini"


def _load_key() -> None:
    """Read the key in-process only; never onto a command line."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {"OPENAI_API_KEY", "OPEN_AI_KEY"}:
            os.environ["OPENAI_API_KEY"] = value.strip().strip("\"'")
            return


_load_key()

from agents import (  # noqa: E402
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    Tool,
    function_tool,
)
from openai import AsyncOpenAI  # noqa: E402

from optio_optimize import Optimizer  # noqa: E402
from optio_optimize.adapters.openai_agents import wrap_openai_client  # noqa: E402

# ---------------------------------------------------------------------------
# Tools. These return JSON, because real tools return JSON -- which is the
# specific thing our synthetic workloads do not do. `cap_tool_results`
# truncates by character proportion; what that does to a JSON document is a
# question no string fixture in this repo has ever asked.
# ---------------------------------------------------------------------------

_ORDERS = [
    {
        "order_id": f"ORD-{5000 + n}",
        "customer": "alice@example.com" if n % 3 == 0 else "bob@example.com",
        "placed_at": f"2026-07-{(n % 28) + 1:02d}T10:15:00Z",
        "status": ["delivered", "shipped", "processing"][n % 3],
        "total_usd": round(19.99 + n * 3.5, 2),
        "items": [
            {
                "sku": f"WID-{100 + n}",
                "name": "Blue Widget" if n % 3 == 0 else "Red Gadget",
                "qty": 1 + (n % 3),
                "unit_usd": round(9.99 + n, 2),
            }
        ],
    }
    for n in range(12)
]


@function_tool
def search_orders(customer_email: str) -> str:
    """Find orders for a customer, most recent first."""
    hits = [o for o in _ORDERS if o["customer"] == customer_email]
    hits.sort(key=lambda o: str(o["placed_at"]), reverse=True)
    return json.dumps({"count": len(hits), "orders": hits})


@function_tool
def get_order_detail(order_id: str) -> str:
    """Full detail for one order, including the shipment event log."""
    order = next((o for o in _ORDERS if o["order_id"] == order_id), None)
    if order is None:
        return json.dumps({"error": "not_found", "order_id": order_id})
    # A realistically oversized payload: a shipment event log, the kind of
    # thing a real carrier API returns and nobody trims. This is what
    # cap_tool_results exists for, and what it has never seen for real.
    events = [
        {
            "seq": i,
            "at": f"2026-07-{(i % 28) + 1:02d}T{i % 24:02d}:00:00Z",
            "facility": f"Distribution Center {i % 7}",
            "scan_type": ["ARRIVAL", "DEPARTURE", "SORT", "LOAD", "CUSTOMS"][i % 5],
            "notes": (
                "Package processed through automated sortation. Barcode read "
                "on first attempt. No exceptions recorded at this facility."
            ),
        }
        for i in range(90)
    ]
    return json.dumps({**order, "shipment_events": events, "carrier": "Globex Freight"})


@function_tool
def check_inventory(sku: str) -> str:
    """Current stock level for a SKU."""
    return json.dumps({"sku": sku, "on_hand": 3 if sku.endswith("0") else 0, "restock_eta_days": 5})


@function_tool
def refund_policy(reason: str) -> str:
    """The refund percentage that applies to a damage reason."""
    table = {"damaged": 100, "late": 25, "unwanted": 80}
    key = next((k for k in table if k in reason.lower()), "unwanted")
    return json.dumps(
        {"reason": key, "refund_percent": table[key], "replacement_allowed": key == "damaged"}
    )


#: Typed as the SDK's own union rather than list[FunctionTool]: Agent.tools is
#: an invariant list, so the narrower element type does not fit.
TOOLS: list[Tool] = [search_orders, get_order_detail, check_inventory, refund_policy]

INSTRUCTIONS = (
    "You are a customer support agent. Use the tools to establish facts before "
    "answering; never guess an order id, a stock level or a refund percentage. "
    "When a tool result says it was truncated, say so rather than assuming you "
    "saw everything."
)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    """What one run of the agent cost and produced."""

    label: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    answer: str = ""
    tool_calls: list[str] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """What the provider billed for this arm, at the published rate."""
        from optio_optimize.config import PRICING
        from optio_optimize.savings import _cost

        return _cost(PRICING[MODEL], self.input_tokens, self.output_tokens, self.cached_tokens)


def _instrument(client: AsyncOpenAI, arm: Arm) -> AsyncOpenAI:
    """Count what the provider was really sent and really billed.

    Wrapped *outside* the optimizer, so it sees the request as it went on the
    wire. Counting inside would measure our own intent instead.
    """
    original = client.chat.completions.create

    async def counting(**kwargs: Any) -> Any:
        arm.bodies.append(kwargs)
        completion = await original(**kwargs)
        usage = getattr(completion, "usage", None)
        if usage is not None:
            arm.calls += 1
            arm.input_tokens += usage.prompt_tokens
            arm.output_tokens += usage.completion_tokens
            details = getattr(usage, "prompt_tokens_details", None)
            arm.cached_tokens += getattr(details, "cached_tokens", 0) or 0
        return completion

    client.chat.completions.create = counting  # type: ignore[method-assign]
    return client


async def run_arm(label: str, optimizer: Optimizer | None) -> Arm:
    """Run the agent once, optimized or not."""
    arm = Arm(label=label)
    client = AsyncOpenAI()
    _instrument(client, arm)
    if optimizer is not None:
        wrap_openai_client(client, optimizer)

    agent = Agent(
        name="support",
        instructions=INSTRUCTIONS,
        tools=TOOLS,
        model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
        model_settings=ModelSettings(),
    )
    result = await Runner.run(agent, TASK, max_turns=12)
    arm.answer = str(result.final_output)
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        name = getattr(raw, "name", None)
        if name:
            arm.tool_calls.append(str(name))
    return arm


def report(arms: list[Arm]) -> None:
    """Print the comparison."""
    print(f"\n{'arm':<12} {'calls':>6} {'in':>8} {'cached':>8} {'out':>7} {'cost':>10}")
    print("-" * 56)
    for arm in arms:
        print(
            f"{arm.label:<12} {arm.calls:>6} {arm.input_tokens:>8,} "
            f"{arm.cached_tokens:>8,} {arm.output_tokens:>7,} ${arm.cost_usd:>9.5f}"
        )
    if len(arms) == 2:
        base, opt = arms
        if base.cost_usd:
            delta = (base.cost_usd - opt.cost_usd) / base.cost_usd
            print(f"\ncost reduction: {delta:.1%}  (negative means the optimizer cost more)")
            # Both arms share one server-side cache we cannot reset, so this
            # number moves with run order -- a baseline that ran second and
            # inherited a warm prefix looks cheaper than it is. Printed beside
            # the totals rather than hidden, because a cache-served baseline
            # is the honest comparison and also the harder one to beat.
            print(
                f"cache-served input: baseline {base.cached_tokens:,}/{base.input_tokens:,}, "
                f"optimized {opt.cached_tokens:,}/{opt.input_tokens:,}"
            )
    for arm in arms:
        print(f"\n[{arm.label}] tools: {arm.tool_calls}")
        print(f"[{arm.label}] answer: {arm.answer.strip()[:400]}")


async def main() -> None:
    """Run the requested arms and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["off", "on", "both"], default="both")
    parser.add_argument(
        "--disable",
        default="",
        help="comma-separated stage names to switch off, for isolating a regression",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="print the message list of every provider call, as sent",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    disabled = frozenset(n for n in args.disable.split(",") if n)
    arms: list[Arm] = []
    if args.arm in {"off", "both"}:
        arms.append(await run_arm("unoptimized", None))
    optimizer: Optimizer | None = None
    if args.arm in {"on", "both"}:
        optimizer = Optimizer(disabled_stages=disabled)
        arms.append(await run_arm(f"optimized{'-' + args.disable if disabled else ''}", optimizer))

    if args.dump:
        for arm in arms:
            print(f"\n===== {arm.label}: what went on the wire =====")
            for index, body in enumerate(arm.bodies):
                print(f"  call {index + 1}: {len(body.get('messages') or [])} messages")
                for message in body.get("messages") or []:
                    content = message.get("content")
                    text = content if isinstance(content, str) else repr(content)
                    marker = "  <-- TOOL_CALLS" if message.get("tool_calls") else ""
                    print(
                        f"    {message.get('role'):<10} {len(text or ''):>6} chars  "
                        f"{(text or '')[:70]!r}{marker}"
                    )

    report(arms)
    if optimizer is not None:
        print()
        for line in optimizer.report.summary_lines(MODEL):
            print(line)
        for finding in optimizer.findings:
            print(f"finding: {finding}")


if __name__ == "__main__":
    asyncio.run(main())
