"""Does ``minify_tools`` remove tokens the provider was actually billing?

The stage strips annotation-only keys (``title``, ``$schema``, ``$id``,
``$comment``) from tool schemas. Its saving has a history of being wrong in the
flattering direction: it once claimed 3,240 tokens on ``mcp_agent`` against
1,210 the provider really stopped billing -- a 2.7x overstatement -- which is
why :data:`~optio_optimize.stages.tools.ANNOTATION_STRIP_CALIBRATION` exists.

That calibration was fitted against ``gpt-4o-mini`` by differencing whole
requests. This checks it a better way, on Anthropic, at **zero cost**:
``messages.count_tokens`` is exact and free, so the before/after difference is
the provider's own arithmetic rather than an estimate of it.

Three things are worth separating and only an exact counter can:

* whether the stripped keys were billed at all (they are -- they are JSON the
  provider tokenizes);
* how much of the raw JSON difference survives into the token difference (the
  keys are punctuation-heavy, so far less than the byte count suggests);
* whether the stage's *reported* saving matches, which is the number that ends
  up in a user's report.

Run: ``python scripts/measure_minify_tools.py``. Spends nothing.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

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

from anthropic import Anthropic  # noqa: E402

from optio_optimize import wire  # noqa: E402
from optio_optimize.config import OptimizeConfig  # noqa: E402
from optio_optimize.stages.base import StageContext  # noqa: E402
from optio_optimize.stages.tools import MinifyToolsStage, _raw_tool_tokens  # noqa: E402
from optio_optimize.tokens import default_counter  # noqa: E402
from optio_optimize.types import LLMRequest, Message  # noqa: E402

MODEL = "claude-haiku-4-5"


def _mcp_shaped_tools(count: int) -> tuple[dict[str, Any], ...]:
    """Tools in the shape an MCP bridge or OpenAPI generator emits.

    The annotation keys are the point: a hand-written schema carries almost
    none of them, so measuring only hand-written tools would report that this
    stage does nothing while real generated ones carry the waste every turn.
    """
    return tuple(
        {
            "type": "function",
            "function": {
                "name": f"tool_{n}",
                "description": f"Operation {n} exposed by the platform integration layer.",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$id": f"https://example.invalid/schemas/tool_{n}.json",
                    "title": f"Tool {n} Arguments",
                    "$comment": "Generated from the OpenAPI document; do not edit by hand.",
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "title": "Record Identifier",
                            "type": "string",
                            "description": "Identifier of the record to operate on.",
                        },
                        "fields": {
                            "title": "Field Selection",
                            "type": "array",
                            "items": {"title": "Field Name", "type": "string"},
                        },
                    },
                    "required": ["record_id"],
                },
            },
        }
        for n in range(count)
    )


def _count(client: Anthropic, request: LLMRequest) -> int:
    """Anthropic's own exact token count for this request. Free."""
    system, turns = wire.anthropic_system_and_turns(request)
    optional: dict[str, Any] = {}
    if system:
        optional["system"] = system
    tools = wire.anthropic_tools(request)
    if tools:
        optional["tools"] = tools
    return int(
        client.messages.count_tokens(model=MODEL, messages=turns, **optional).input_tokens  # type: ignore[arg-type]
    )


def main() -> int:
    """Compare claimed against measured across a range of tool counts."""
    client = Anthropic()
    stage = MinifyToolsStage()
    ctx = StageContext(config=OptimizeConfig(), counter=default_counter())

    print(
        f"{'tools':>6} {'billed before':>14} {'billed after':>13} {'real':>7} {'raw json':>9} "
        f"{'claimed':>8} {'real/raw':>9}"
    )
    print("-" * 74)

    rows: list[tuple[int, int, int]] = []
    for count in (1, 3, 5, 10, 20):
        request = LLMRequest(
            model=MODEL,
            messages=(Message(role="user", content="Use a tool if one fits."),),
            tools=_mcp_shaped_tools(count),
            temperature=0.0,
        )
        result = stage.before(request, ctx)

        before = _count(client, request)
        after = _count(client, result.request)
        real = before - after
        claimed = result.saved_input_tokens
        # The uncalibrated JSON delta the stage starts from, measured rather
        # than back-computed from the constant under test.
        raw = _raw_tool_tokens(request.tools, ctx, MODEL) - _raw_tool_tokens(
            result.request.tools, ctx, MODEL
        )
        rows.append((count, real, claimed))
        print(
            f"{count:>6} {before:>14,} {after:>13,} {real:>7,} {raw:>9,} {claimed:>8,} "
            f"{real / raw if raw else float('nan'):>9.2f}"
        )

    print()
    total_real = sum(r for _, r, _ in rows)
    total_claimed = sum(c for _, _, c in rows)
    print(f"totals: real {total_real:,}   claimed {total_claimed:,}")
    if total_real <= 0:
        print("\nthe stripped keys were not billed at all -- the stage removes nothing the")
        print("provider was charging for, and any reported saving is fiction")
        return 0
    error = (total_claimed - total_real) / total_real
    print(f"claimed is {error:+.1%} against what the provider stopped billing")
    verdict = "OVERSTATED" if error > 0.10 else "understated" if error < -0.10 else "within 10%"
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    raise SystemExit(main())
