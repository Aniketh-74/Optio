# ADR-039 — A measurement that costs money to re-check gets checked once

**Status:** Accepted
**Date:** 2026-08-01
**Related:** ADR-015 (measure, do not assume), ADR-021 (cache-write tokens are a subset),
ADR-037 (the probe that billed $7.60 and kept nothing), the `prefix_cache` minimum-size defect

## Context

Every live figure in this repository was obtained the same way: run a script, read the terminal,
copy the number into an ADR by hand, discard the response. Forty ADRs, and **not one recorded
exchange anywhere in the tree** — no fixtures, no cassettes, no stored results. A search for them
returns nothing.

The numbers are not therefore wrong. The problem is narrower and worse: **re-checking one costs
money**. So in practice it is checked once, at the moment it is written, and trusted from then on.
Every measurement here is a snapshot nobody can afford to take twice.

That is the exact shape of the longest-lived defect in the changelog. `prefix_cache` reported a
saving for months against a system prompt of 3,614 tokens — below Anthropic's cacheable minimum for
that model — so the live API returned `reads 0 writes 0` and the stage was measuring nothing. No
cheap test could have noticed, because noticing required a paid call.

It is also what made the ADR-037 probe expensive. Seven models accepted a 217,554-token prompt that
was assumed to be rejected, billing $7.60 for generation. The responses that proved the windows
exceed 217,554 were printed and dropped; the account ran dry before the scan finished, and what
survives is a sentence in a table.

## Decision

Keep the exchange. `RecordingProvider` wraps any `BenchProvider` and appends every
(request, response) pair to a JSONL file; `ReplayProvider` reads that file and serves the same
responses with no key, no network and no spend. `--record PATH` and `--replay PATH` on the bench
CLI.

**What a replay proves, stated narrowly:** the library still builds the request the provider was
measured on. Token counts, cache reads, cache writes and costs all come back exactly as the live
provider reported them.

**What it does not prove:** that the provider still answers that way. Only a fresh call does that.
So the recording carries the UTC date it was made, `ReplayProvider.label` prints it, and staleness
is a fact a reader can see rather than one they must assume.

### The key is not `cache.request_key`

`request_key` deliberately omits `max_tokens` and ignores `cacheable` markers. Both omissions are
right there and wrong here.

- `max_tokens` truncates a reply rather than changing it, so an exact cache may share one entry
  across ceilings. A recording is evidence about *one exchange*, and a truncated reply is a
  different exchange with different billed output tokens.
- `cacheable` markers do not change the answer, which is why the cache ignores them — but they are
  the entire mechanism `prefix_cache` is paid for. **A key blind to them would replay the old cache
  numbers after the stage stopped emitting them, and report the saving intact**: the original
  defect, now with a receipt attached.

A miss is the regression signal, so the digest moves whenever the wire does. `blake2b`, not
`hash()`: string hashing is salted per process, so a `hash()`-derived digest agrees with itself all
through one session and disagrees with yesterday's recording — every exchange would miss, and the
failure would look exactly like the regression this exists to report. A subprocess test with a
different `PYTHONHASHSEED` pins it, because nothing in a single process can.

### A miss raises

A replay that answered anyway would credit this library with a saving no provider ever measured.
That is worse than having no recording: it is the fabricated-number failure the whole package is
organised against, wearing evidence as a disguise. `KeyError` names the file, its date, and which
of the two causes applies — request changed, or recording stale.

### Repeats replay in recorded order

Two identical requests are how prefix caching is observed at all: the first writes the cache, the
second reads it, and **the responses differ**. Serving either one twice erases the finding. For the
same reason `reset()` is a deliberate no-op — the harness calls it at every A/B arm boundary, but a
recording is one linear run with both arms inside it in order. Rewinding there would serve the
baseline's answers to the optimized arm, and since the baseline runs first the measured saving
would silently collapse to zero.

### Written as they happen

Appended and flushed per call, not buffered to the end. Buffering is faster and would have lost the
entire ADR-037 probe when its account ran dry mid-scan: money spent, evidence gone.

## Consequences

A recorded benchmark replays to identical numbers through the real `compare()` — verified on
`multi_turn_chat`, both arms, zero misses. That round trip is itself a test of something nothing
else covers: any hidden per-run variation in how the optimizer assembles a request (a timestamp, a
set iterated in hash order) surfaces here as a replay miss and nowhere else.

`is_live` is `False` on a replay. The tokens and cache counts are the live run's and are reported
as such; the clock is a file read, so the metrics layer must not print a wall-clock comparison. A
replayed arm is regression evidence, never fresh evidence.

The version field is checked on load and an unknown one is refused rather than guessed at.
Recordings outlive releases, and evidence read through the wrong schema is indistinguishable from
evidence that was never gathered.

**This does not make measurement free — it makes it durable.** The first call still costs what it
costs. What changes is that the second, and the thousandth, cost nothing, so a live figure can be
part of the suite rather than a claim in a document.
