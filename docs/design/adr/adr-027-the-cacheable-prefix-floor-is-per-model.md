# ADR-027 — The cacheable prefix floor is per-model

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-015 (isolated evidence), ADR-016 (do not change what callers send on one measurement),
ADR-020/021 (cache economics), ADR-024 (a stage may not book what it cannot attribute)

## Context

`prefix_cache` is the largest lossless saving in this package and the only reason Anthropic caches
anything at all. `MIN_PREFIX_TOKENS = 1024` decides whether it places a breakpoint.

Anthropic's real minimum spans a factor of eight, and `docs/optimize-benchmarks.md` has recorded the
table — and this defect — since 2026-07-30:

| model | minimum cacheable prefix |
|---|---|
| Opus 5 | 512 |
| Opus 4.8, Sonnet 5, Sonnet 4.6 / 4.5 | 1,024 |
| Opus 4.7, Haiku 3.5 | 2,048 |
| **Haiku 4.5, Opus 4.6 / 4.5** | **4,096** |

One constant is therefore wrong in both directions: too high for Opus 5, where it declines a
breakpoint that would have worked, and too low for four models, where it places one the provider
silently discards **while the stage's note reports success**.

That was left open deliberately, on ADR-016's grounds that a per-model table changes what every
Anthropic caller sends and one measurement should not carry it. Two things have changed since.

### The benchmark's own default model is the worst case

`AnthropicProvider` now defaults to `claude-haiku-4-5` (ADR-025's sibling fix), whose floor is
**4,096**. Measured across the suite:

```
workload                  max prompt   vs 4096 floor
mcp_agent                     11,799   clears
multi_turn_chat_long           3,712   BELOW
rag_queries                    2,555   BELOW
tool_calling_chat              2,528   BELOW
multi_turn_chat                2,496   BELOW
timestamped_agent              2,486   BELOW
tool_loop                      2,156   BELOW
retry_storm                    2,154   BELOW
fan_out                        2,166   BELOW
rag_queries_noisy              2,506   BELOW
unique_questions                  14   BELOW
sampled_creative                  22   BELOW
```

**Eleven of twelve workloads cannot cache at all on the model the benchmark runs by default**, and the
full live run duly reported zero provider cache reads on every one of them. Nothing in the report said
why. A reader would reasonably conclude the stage does not work.

### The evidence bar is now met

ADR-016's objection was one measurement. There are now four independent live measurements behind this
table: the original prefix-cache run whose first attempt placed a marker at 1,449 tokens and got zero
reads; the correction that lengthened the prompt past 4,096 and got them; the TTL run (ADR-021); and
the fan-out run (ADR-020). The floor is not in doubt — only whether to encode it.

## Decision

### 1. The floor is looked up per model, longest-prefix match

`MIN_PREFIX_TOKENS_BY_MODEL` maps a model-name prefix to its floor, matched longest-first so
`claude-haiku-4-5-20251001` resolves through `claude-haiku-4-5`. This is the same shape `PRICING`
already needs for dated ids.

### 2. An unknown model keeps today's 1,024

Not the lowest floor, and not the highest. The lowest would place markers that four known models
discard; the highest would decline breakpoints that work on most. 1,024 is what every caller gets
today, so an unrecognized model's behaviour is unchanged by this ADR — which is the property ADR-016
actually asks for.

### 3. Below the floor the stage declines *and says why*

Today a sub-floor prompt gets a breakpoint, a savings-ledger entry and a note claiming the prefix was
marked. That is the shape ADR-024 removed elsewhere: reporting work that bought nothing. The stage now
declines and names the floor and the model, so "no cache reads" is diagnosable from the report instead
of being a mystery that reads like a broken stage.

### 4. The benchmark reports what the provider actually served from cache

`ArmResult.cached_input_tokens` has been recorded all along and never printed. `prefix_cache`
correctly claims no saving of its own — ADR-020's rule, because the effect is measured rather than
estimated — so the stage line reads `0 tokens` and the number that would show whether it worked was
invisible. A lever this package calls its largest lossless win has to be legible in its own benchmark.

> **Follow-up, 2026-07-31.** This decision was **inert inside the benchmark** for its first day.
> `Workload.requests()` called `build()` with no arguments, so every workload built its requests as
> `gpt-4o` no matter what `--model` said — `--model` reached the provider and the pricing row, never
> `LLMRequest.model`. `min_prefix_tokens_for` therefore returned the unknown-model fallback of 1,024
> on every live Anthropic run, and the very failure this ADR describes went on happening on the very
> model it describes it for: a live Haiku run placed the breakpoint and came back `reads 0 writes 0`.
> It hid because gpt-4o's fallback equals Sonnet 4.5's real floor, so the one model where the bug is
> invisible is the one on which `prefix_cache` appeared to work. Fixed by `requests(model)`.
>
> **The open item below is now closed.** With the model plumbed through, `multi_turn_chat` on
> `claude-sonnet-4-5` (floor 1,024, prompts 1,406–1,769 tokens) is the suite's first end-to-end
> demonstration of `prefix_cache`: **18,300 provider cache reads, 1,872 writes, cost $0.06548 →
> $0.01672 — 74.5% — at 91.7% identical output and +0.97 ms per request.** Input *token count* is
> unchanged at 20,610, which is the point: prefix caching changes the rate, not the volume, and a
> suite reporting only token reduction would score this at 0.0%.

## Consequences

- **Opus 5 gains caching it was being denied**, on prompts between 512 and 1,024 tokens. That is the
  one direction of this change that adds a saving rather than removing a false one.
- **Haiku 4.5 and Opus 4.6/4.5 stop placing markers that do nothing.** No saving is lost, because there
  was none — what is lost is a line in the report that claimed otherwise.
- **The benchmark's headline numbers do not move**, since the discarded markers were already buying
  nothing. What moves is the explanation: eleven workloads will now say plainly that their prompt is
  below the floor.
- **The suite cannot demonstrate `prefix_cache` on its default model**, and that is a real gap this
  ADR exposes rather than fixes. Only `mcp_agent` clears 4,096. Either the workloads grow a realistic
  system prompt and tool schema — most production agents carry several thousand tokens of both — or
  the demonstration moves to a model with a lower floor. Recorded as the next open item.
- The table is data, auditable against the vendor's page, and stale the moment Anthropic changes it.
  Same standing caveat as `PRICING`, and the same mitigation: it is one dict, not logic.

## Alternatives considered

**Keep one constant.** Rejected now that the default model makes it wrong on eleven of twelve
workloads. The original reasoning — "the cost of being wrong is a marker that does nothing, never a
wrong answer" — is true about correctness and false about honesty: a marker that does nothing is
reported as a marker that did something.

**Default unknown models to 4,096.** Rejected: it silently disables the stage for every model not in
the table, including future ones with low floors, and turns an unrecognized name into a lost saving.

**Infer the floor by probing the provider once per model.** Rejected. It costs a live call at import
or first use, the answer is not observable without a second call to see whether a read occurred, and
`TokenCounter`-style determinism arguments apply — this package does not make network calls to decide
how to build a request.
