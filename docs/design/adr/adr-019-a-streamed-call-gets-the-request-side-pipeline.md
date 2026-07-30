# ADR-019 — A streamed call gets the request-side pipeline, and a replayed cache hit

**Status:** Accepted
**Date:** 2026-07-30
**Related:** ADR-013 (the package exists, fail-open is rule 1), ADR-016 (the in-scope test),
ADR-017 (`prepare`/`complete` exist because batch needed them), ADR-004 (fail-open is absolute)

## Context

Both adapters currently say the same thing, in the same words:

> **Streaming is not optimized.** A stage pipeline built around one request producing one response
> can only buffer a token stream, which defeats the reason to ask for one. A `stream=True` call
> bypasses this wrapper entirely.

The first half of that reasoning is sound and the conclusion does not follow from it. Buffering
would indeed defeat streaming. But most of this package never touches the response at all:

- `prefix_cache` places a `cache_control` breakpoint on the outgoing request. It is described in its
  own source as the largest lossless saving available, and on Anthropic it is the *only* way the
  provider caches anything. Nothing about it involves the reply.
- `trim_history`, `minify_tools`, `deduplicate`, `prune_retrieval`, `cap_tool_results`,
  `adaptive_max_tokens`, `reasoning_budget` — all request-side.
- `exact_cache` needs the reply to *store* one, but serving a hit needs nothing at all: the answer
  is already in hand.

So a streaming caller today gets **zero** of nineteen stages, and the reason given covers only the
handful that read a response. Worse, it is invisible: the library installs, the report shows nothing
saved, and nothing explains why. For anything user-facing, streaming is the dominant production
mode. "Plug and play except in the mode you actually ship" is the failure this ADR exists to fix.

The machinery is already built. ADR-017 split `Pipeline.execute` into `prepare` and `complete`
because batch dispatch has hours between the two halves. A stream has milliseconds between them and
needs the identical seam. `Optimizer.pipeline` is already public for exactly this reason.

## Decision

**A `stream=True` call runs the request-side pipeline, and the after-hooks run when the stream
finishes.** Concretely, four rules:

### 1. Every `before` hook runs, and the transformed request is what goes on the wire

No stage is skipped for being on a streaming path, because no stage knows it is on one. This is the
whole value: the marker reaches the provider, the history is trimmed, the tools are minified.

### 2. The `after` hooks run on completion, from a proxy that records as it yields

The returned object wraps the SDK's `Stream`, forwards every event to the caller **unchanged and
immediately**, accumulates the text and usage as they go past, and calls `Pipeline.complete` once
the terminal event arrives. This is not buffering: no event is withheld, and the caller sees the
first token exactly as soon as it would have without this package. What is deferred is only the
bookkeeping — a cache write, an observation, a savings row — none of which the caller waits on.

The proxy is a proxy, not a `Stream` subclass: the SDK's constructor wants a cast type, an httpx
response and a client, and faking those to satisfy `isinstance` would be worse than not satisfying
it. Iteration, `with`, `close()` and attribute forwarding all work; `isinstance(x, Stream)` does
not. That is a real behavioural difference and it is recorded here rather than discovered.

### 3. An abandoned stream completes nothing

A caller who breaks out of the loop, or closes early, gets no after-hooks at all — no cache write,
no observation, no savings row. That is deliberate and it is the safe direction: the alternative is
`exact_cache` storing half an answer and serving it, confidently and permanently, to every later
caller who asks the same question. A missing report row costs a number; a cached truncation costs
correctness. The report undercounts abandoned streams, and that is the right trade.

### 4. A cache hit is replayed as a synthesized event sequence

`message_start`, one `content_block_start`, one `content_block_delta` carrying the whole answer,
`content_block_stop`, `message_delta`, `message_stop` — built through the SDK's own pydantic types,
so a shape this package gets wrong fails here rather than in the caller's event handler.

The reassembled message is byte-identical to what the original call returned. The *chunking* is
not: one delta where the live call sent hundreds. A caller measuring inter-token latency will see a
replayed hit arrive all at once, which is not a defect — it did arrive all at once, because nothing
was generated. `Fidelity.IDENTICAL` is a claim about the message, and the message is identical.

## Scope

**Anthropic only, sync and async.** That is where the value is concentrated: `prefix_cache` is worth
zero on OpenAI, which caches long prefixes automatically for both arms of an A/B — the fact that
corrected a simulated 36.3% saving to a measured −1.8%.

**OpenAI streaming stays unoptimized, and its docstring must keep saying so.** The technique
transfers; the event synthesis does not, because `ChatCompletionChunk` is a different shape with
different rules. Shipping it as "streaming works" while one of two adapters silently does nothing
would repeat the defect this ADR is fixing, one level up. It is a follow-on with its own tests, not
a footnote to this one.

**`client.messages.stream()` is out of scope.** It is a separate method returning a higher-level
`MessageStreamManager` with its own accumulation, text helpers and event types. Wrapping it is a
third shape, and this package wraps `messages.create` because that is the one place everything built
on the SDK converges.

## Consequences

- Fail-open is unchanged and absolute (ADR-004). Any failure in translation, in `prepare`, or in
  constructing the proxy produces the caller's original streaming call, untouched. The one ordering
  constraint is that nothing which can fail may sit between the provider call and returning
  something iterable — a retry there would bill twice.
- `complete` must be called exactly once per `prepare`, and a stream gives two ways to violate that:
  a caller who iterates to the end *and* calls `close()`, and a proxy that treats both the terminal
  event and exhaustion as completion. Guarded by a one-shot flag, and it needs a named test rather
  than a careful reading.
- The savings report gains streamed requests, so its numbers move for anyone already streaming. They
  move from "nothing was measured" to "something was", which is the point.
- `emit_spans` (ADR-014) records a streamed request when the stream finishes, so its span timing
  reflects the whole generation rather than the call that started it. That is more useful and it is
  a different meaning from the non-streaming path's; worth knowing before comparing the two.

## Alternatives considered

**Keep bypassing, and document it louder.** Rejected. The gap is not a documentation problem: the
user who enables this package and streams gets nothing, and no wording makes that acceptable when
two thirds of the stages would have worked untouched.

**Buffer the stream, run the whole pipeline, then re-emit.** Rejected, and this is the option the
original docstring was right about. It would defeat the reason to stream, add latency proportional
to the reply, and buy nothing the proxy does not already get.

**Serve cache hits only, and skip the request-side stages.** Tempting as a smaller first step, and
backwards: the request-side stages are the larger and safer half, and `prefix_cache` on Anthropic is
the single biggest lossless saving in the package.
