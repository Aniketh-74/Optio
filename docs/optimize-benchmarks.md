# Measured savings

Numbers from `python -m optio_optimize.bench`. Live figures come from real
`gpt-4o-mini` calls; simulated figures are marked as such. Reproduce with:

```bash
python -m optio_optimize.bench --strict-fidelity            # free, instant
python -m optio_optimize.bench --live --cap 1.00            # real API, capped
```

## Live results (gpt-4o-mini, identical-output stages only)

| workload | input ↓ | output ↓ | cost ↓ | latency | quality |
|---|---|---|---|---|---|
| `retry_storm` | 93.3% | 93.3% | **93.7%** | **−96.7%** | 100% identical |
| `tool_loop` | 80.0% | 80.0% | **80.8%** | **−78.6%** | 100% identical |
| `fan_out` | 66.7% | 66.7% | **66.9%** | −41.0% | 100% identical |
| `multi_turn_chat` | 0.0% | 0.0% | **−7.7%** | −2.6% | 100% identical |
| `rag_queries` | 0.0% | 0.0% | **6.0%** | −1.5% | 100% identical |

Total spend to produce this table: **under $0.02**.

Every workload returned byte-identical responses. That is the whole claim of the
`IDENTICAL` fidelity level, and it is checked rather than asserted.

## The result worth reading twice

**`multi_turn_chat` came in at −7.7%: the optimizer made it slightly more
expensive.** The simulated suite had reported **+36.3%** for the same workload.

The simulator was wrong, and the way it was wrong is the most instructive thing
in this document. It modelled Anthropic-style prefix caching, where nothing is
cached unless the request carries an explicit `cache_control` breakpoint — so
our `prefix_cache` stage looked like the thing unlocking a 90% input discount.

OpenAI does not work that way. Measured directly:

```
system prompt: 1387 tokens
call 1: prompt=1401  cached=1024
call 2: prompt=1401  cached=1280
call 3: prompt=1401  cached=1280
```

**OpenAI caches any prefix over ~1024 tokens automatically, with no marker.**
The baseline arm was already getting the discount. Our stage contributed
nothing, and the ±8% spread is server-side cache variance between the two arms.

So the simulator had been crediting this library with a feature the provider
grants unconditionally. That is the single easiest way to publish a benchmark
that collapses the first time someone reproduces it, and only a live call caught
it. The simulator now models both styles, and the honest comparison is:

| workload | OpenAI-style (automatic) | Anthropic-style (explicit) |
|---|---|---|
| `multi_turn_chat` | **0.0%** | **33.5%** |
| `rag_queries` | **0.0%** | **28.7%** |
| `tool_loop` | 71.6% | 85.5% |

**`prefix_cache` is worth roughly a third of spend on Anthropic and nothing on
OpenAI.** It costs nothing to leave on, but nobody should expect a discount
they were already receiving.

## What the numbers mean per stage

`exact_cache` is where the measured savings come from. It avoids the call
entirely, so the saving is 100% of that request — input, output and latency
together. Its value is therefore entirely a question of how repetitive the
workload is, which is why `retry_storm` (93%) and `unique_questions` (0%) are
both in the suite.

`prefix_cache` avoids no tokens at all and reports zero, by design. It lowers
the *price* of tokens still sent, which shows up in the cost column and in
`cached_input_tokens`, never in the token column.

`adaptive_max_tokens` and `structured_output` shape output rather than
preserving it byte-for-byte, so they are excluded from `--strict-fidelity`
runs. Their savings are real but must be measured against a quality gate, not
against an identity check.

## What these numbers are not

**They are one machine, two providers, seven synthetic workloads.** Your traffic
decides your savings, which is why the harness ships inside the package: point
it at your own captured requests rather than trusting this table.

**Live A/B slightly favours whichever arm runs second**, because the provider's
prefix cache is server-side and cannot be reset between arms. The baseline runs
first, so the bias works against the result this library would like to report.

**Output quality is checked as identity, not as correctness.** For the
`IDENTICAL` stages that is exactly the right test. For `SHAPED` and `ALTERED`
stages it is not sufficient, and the eval harness that covers them is still to
be built — until it exists, those stages should be treated as unmeasured.

## Overhead

Our own cost per request, measured with the provider's time excluded:

| workload | before memoization | after |
|---|---|---|
| `multi_turn_chat` | 1.945 ms | **0.585 ms** |
| `rag_queries` | 1.709 ms | **0.258 ms** |
| `tool_loop` | 0.384 ms | **0.131 ms** |

The bottleneck was tokenization: tiktoken costs ~2.5 ms on a 1387-token system
prompt, and the same prompt was being re-counted on every step.
`MemoizingCounter` caches by content digest — not by text, so no prompt content
is retained — at an 86.4% hit rate.

Against a provider call of 300–2000 ms, sub-millisecond overhead is under 0.3%.
