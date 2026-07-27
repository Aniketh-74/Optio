# Pricing

How `agentmeter` turns token counts into dollars, and what to do when it cannot.

## Source

Prices come from a **static in-memory table** in
[`lanes/cost/pricing.py`](../src/agentmeter/lanes/cost/pricing.py), populated
from vendors' published list prices.

The table is deliberate, not a shortcut. Cost is computed on the hot path, so a
pricing API call would add a network round trip to every LLM step — breaking the
latency budget (SC-5) and adding a failure mode to the critical path. A stale
price produces a slightly wrong number; a hanging HTTP call produces a slow
agent. The first is recoverable, the second is the thing this library promises
never to do.

Prices are stored in the published unit — **USD per million tokens** — so a
reviewer can diff the table directly against a vendor's pricing page. Conversion
to per-token happens once, at lookup.

## Model resolution

Lookup tries, in order:

1. **Exact match**, case-insensitive and whitespace-trimmed.
2. **Longest substring match**, so `gpt-4o-2024-11-20` and `openai/gpt-4o`
   resolve to the `gpt-4o` entry.

Longest-first matters: `gpt-4o-mini` must never resolve to `gpt-4o`, which costs
roughly 17× more. The resulting number would look entirely plausible, which is
what makes it dangerous.

## Unknown models emit nothing

A model not in the table produces **no cost signal**. Not a guess, not a zero,
not the price of a similar-looking model.

This is the same rule as everywhere else in the library: a fabricated cost is
worse than a missing one, because a budget policy cannot tell an invented number
from a real one and would gate real money on it. See
[signals.md](signals.md) on absence versus zero.

An unpriced step is also visible rather than silent — it stays as an open
reservation and surfaces as a **leak** at run end, with a WARN. A run reporting
leaked steps is telling you its cost is incomplete.

## Updating the table

Routine maintenance, not an architectural change:

1. Edit `_PRICES` in `pricing.py`.
2. Bump `PRICING_TABLE_VERSION` to the current date.
3. Run the tests — `test_pricing.py` checks table invariants, including that
   output is never cheaper than input (a transposed pair is the likeliest edit
   error).

The version exists so a support conversation can establish which table priced a
given run.

## Supplying your own prices

The built-in table will not cover negotiated enterprise rates, self-hosted
models, or a vendor we have not added. Implement `PricingProvider`:

```python
from agentmeter.lanes.cost.pricing import ModelPrice


class MyPricing:
    def price_for(self, model: str) -> ModelPrice | None:
        if model.startswith("our-finetune-"):
            return ModelPrice(input_per_million=0.80, output_per_million=2.40)
        return None  # unknown -> no signal, never a guess
```

Returning `None` means "I do not know this model" and produces no cost signal,
exactly as the built-in table does for an unknown id.

## What is not priced

- **Tool-call spans with no token usage.** Not every GenAI span is a billable
  model call; these are skipped rather than treated as free.
- **Negative or non-numeric token counts.** A framework reporting these has told
  us something we do not understand, and pricing it would invent a number.
- **Cached, batch, and reasoning-token tiers.** The table carries standard input
  and output rates only. Where a vendor prices cached input lower, `agentmeter`
  currently over-reports. That bias is the safe direction — over-reporting gates
  a run early rather than letting an over-budget run through — but it is a known
  gap, not a design choice, and a tiered table is the fix.
