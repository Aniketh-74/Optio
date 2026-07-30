# ADR-022 — An image is content, and the cache key was the urgent half

**Status:** Accepted
**Date:** 2026-07-30
**Related:** ADR-013 (fail-open; never cause a cost increase), ADR-015 (evidence bar for `ALTERED`),
ADR-016 (the in-scope test), ADR-021 (fix the accounting before the lever), §10 (never log prompt
content), §4.4 (no dependency creep)

## Context

Spec item 5 was written as two things: **counting** images, which this package does not do at all,
and **reducing** them via `detail: "low"` or pre-upload downscaling. Investigating the first turned up
a third that outranks both, so this ADR orders them by severity rather than by the spec's order.

### The urgent finding: two different images hash to the same cache key

`cache.request_key` keys `[m.role, m.content, m.name]` per message. An image block never reaches
`Message.content` — `_text_from_content` extracts text blocks and deliberately ignores the rest — so
it rides through in `extra[_RAW]`, and `UNKEYED_FIELDS` excludes `extra` as *"provider transport
details, not semantics"*.

For an image that classification is simply wrong. The image **is** the semantics.

Verified rather than reasoned about. Two requests, identical prompt text, different image payloads:

```
key A: 277f0c84dc88772471a95b2f9cbe7846
key B: 277f0c84dc88772471a95b2f9cbe7846   <- identical
```

`exact_cache` is **on by default** and caches at `temperature == 0`, which is exactly the setting
deterministic vision work uses — OCR, screenshot analysis, document extraction. So "describe this
image" over two different images returns the first image's description for the second. That is a
**wrong answer**, not a mis-measurement, and it is the same defect class as the `stop` bug already
documented in `UNKEYED_FIELDS`: a field justified in prose as keyed while absent from the payload,
except here it was justified as *un*keyed on a classification that does not hold for image blocks.

`semantic_cache` shares the flaw — its `_text` joins `m.content` — but is off by default.

### What the missing token count does and does not break

This needs stating precisely, because the obvious version of the claim is wrong and I asserted it
before checking.

`pipeline` sets `actual_input = response.input_tokens or count_request(sent, counter)` and then
`baseline = actual + saved`. On a live call the provider's own number is used, and the provider counts
images. So image tokens are present on **both** sides of `reduction_ratio`, and the headline saving
percentage is **not** inflated by this gap. The reassuring reading is the correct one there.

Three real consequences remain:

1. **Window decisions are unsafe.** `count_request` returns **8 tokens** for a 1568x1568 image request
   that really bills ~1,535. `fits_in_window` exists precisely because under-counting a limit
   decision causes a provider-side rejection the user sees as a crash, and it applies a 1.15x safety
   margin to guard a few percent of estimator error while a vision request is off by two orders of
   magnitude. `trim_history` will decline to trim a request that cannot fit.
2. **Short-circuited savings are understated.** A request served from cache never reaches a provider,
   so `count_request` supplies the number and a vision cache hit reports ~8 tokens avoided against a
   true ~1,535. This is the *safe* direction and still wrong.
3. **Nothing can reason about image cost at all** — no stage, no report, no diagnostic.

### Measured, because a modelled number here would be the fifth to fail

Anthropic's `messages.count_tokens` is a **free, exact** endpoint, so this was measured rather than
taken from the docs. Synthetic PNGs of known dimensions, `claude-haiku-4-5`, image cost isolated by
differencing against a text-only baseline:

| w x h | pixels | image tokens | `w*h/750` | ratio |
|---|---|---|---|---|
| 64 x 64 | 4,096 | 13 | 5 | 2.38 |
| 200 x 200 | 40,000 | 68 | 53 | 1.27 |
| 512 x 512 | 262,144 | 365 | 350 | 1.04 |
| 800 x 600 | 480,000 | 642 | 640 | 1.00 |
| 784 x 784 | 614,656 | 788 | 820 | 0.96 |
| 1000 x 1000 | 1,000,000 | 1,300 | 1,333 | 0.98 |
| 1200 x 958 | 1,149,600 | 1,509 | 1,533 | 0.98 |
| 1092 x 1092 | 1,192,464 | 1,525 | 1,590 | 0.96 |
| **1568 x 1568** | 2,458,624 | **1,525** | **3,278** | **0.47** |
| 1568 x 784 | 1,229,312 | 1,572 | 1,639 | 0.96 |
| 2000 x 1000 | 2,000,000 | 1,572 | 2,667 | 0.59 |
| 400 x 3000 | 1,200,000 | 452 | 1,600 | 0.28 |
| 3000 x 400 | 1,200,000 | 452 | 1,600 | 0.28 |

Three things fall out, and the published `(w*h)/750` formula alone gets two of them badly wrong:

- **It is accurate in the mid-range** — 0.96 to 1.04 across 512x512 through 1200x958 — and errs
  slightly high at the top, which is the safe direction.
- **Two caps apply, and ignoring them overstates by up to 3.6x.** A max edge of 1568 explains the thin
  images exactly (400x3000 predicts 437 after edge-scaling against 452 observed), and an
  aspect-dependent maximum area caps everything else near **1,600 tokens**. Uncapped, 1568x1568
  predicts 3,278 against a real 1,525.
- **Small images cost more than the formula says** (64x64: 13 against 5), so a floor is needed.

The residual is *quantized*, not smooth: `1092x1092 -> 1,525` and `784x1568 -> 1,572` both come out at
exactly `w*h/782`, while 800x600 and 512x512 do not, which is the signature of patch-based processing
rather than a closed formula. Chasing that with a curve fit would be false precision.

## Decision

### 1. The cache key includes an image identity digest, and this ships first and alone

Same ordering rule as ADR-021: the correctness half lands before the lever, because it is independent
of everything below and is a wrong-answer bug today. `Message` gains a normalized, hashed identity for
non-text content, which `request_key` folds in.

**A digest, never the bytes.** §10's content rule covers images as squarely as prose — an image is
more identifying than most text, cache keys reach logs and metrics, and a base64 payload in a key
would put a user's screenshot in a log line. Keying on the digest also keeps keys at tens of bytes
rather than megabytes.

**Unrecognized non-text blocks are keyed too, not skipped.** The bug being fixed came from deciding a
block was semantically irrelevant; the fix should not repeat the move for `tool_use` inputs or a
future block type nobody has seen. Anything that is not a text block contributes to the digest.

### 2. Images are counted with a clamped area estimator, and its error band is documented

`count_message` gains image tokens from the block metadata. The estimator is `(w*h)/750`, clamped by
both measured caps and floored for small images, calibrated against the table above.

**Dimensions come from a header parse, with no new dependency.** PNG, JPEG, GIF and WebP all carry
dimensions in a fixed-offset header or an early marker, which is well under a hundred lines. Pillow
would be a large wheel with native code pulled in for four integers, and §4.4 exists for exactly that.

**When dimensions are unavailable, the answer is not zero.** A URL-sourced image, an unrecognized
format, or a truncated header yields no dimensions, and `absence != zero` is this project's rule.
These get a documented constant sized from the table's midpoint, and the estimate is marked inexact so
`fits_in_window` applies its margin. Returning `0` there is the current bug; returning `None` would
make every caller handle it and most would coerce to zero.

**`count_tokens` is not called from the counter.** It is exact and free, which is tempting, but the
`TokenCounter` protocol requires deterministic and side-effect-free implementations, and a network
round trip per message would add provider latency to a function whose whole purpose is to be
microseconds. It belongs in measurement scripts, where it was used to produce the table above.

**Anthropic is calibrated; OpenAI uses its published tile formula and says so.** Following
`TOOL_SCHEMA_CALIBRATION`'s precedent, which is honest about being measured on one provider: the tile
formula is unmeasured here and is still far closer than the current zero.

### 3. Reducing image cost is a separate, `ALTERED` lever, and is deferred

`detail: "low"` (~85 tokens against thousands) and pre-upload downscaling both degrade what the model
can see. That is `ALTERED` under ADR-015, whose bar is live, isolated evidence — and the evidence
required is a **vision accuracy probe**, not a token count: a hard set where a downscaled image
sometimes yields the *wrong* answer, which is the control this project's reasoning-budget run lacked
and was criticized for. Nothing here ships that lever, and no config flag is added for it, so it
cannot be turned on by accident. Item 5's reduction half stays in the spec queue with its gate named.

## Consequences

- **`exact_cache` becomes correct on vision requests and its hit rate on them drops to near zero**,
  since two calls now collide only when the image is byte-identical. That is a loss of a saving this
  package should never have been claiming: the hits it is giving up were wrong answers.
- Interaction risk is the one the spec flagged: image blocks live in content **lists**, the code path
  whose corruption made every `tool_result` turn go out as `{"role": "user", "content": ""}`. The
  round-trip tests added with that fix are the ones that must keep passing, and the digest path must
  not write to `_RAW` at all.
- `Message` gains a second public field this week. It is a digest string rather than a widening of an
  existing field, for ADR-021's reason: changing `content`'s type would break every reader of it.
- Reported savings on multimodal workloads will get **smaller and more correct** once counting lands,
  because image tokens enter the denominator on the short-circuit path where the provider never
  supplied a number.
- The estimator will be wrong by a few percent in the mid-range and more at the extremes. That is
  worth saying plainly in its docstring next to the table, because the alternative on offer is not
  exactness — it is the zero it replaces.

## Alternatives considered

**Key on the raw image bytes.** Rejected on §10: it puts user images in log lines, and makes keys
megabytes.

**Exclude vision requests from `exact_cache` entirely.** Rejected. It fixes the wrong answers and
gives up every legitimate hit, including the genuinely repeated identical-screenshot call, for no gain
over a digest. Worth reconsidering only if the digest turns out to be unreliable across SDK shapes.

**Call `count_tokens` for exact image counts.** Rejected in the counter, kept in the scripts — see
decision 2. A future opt-in exact tier for a caller who wants a network call is not foreclosed.

**Take the `(w*h)/750` formula from the documentation and skip the probe.** Rejected, and the probe is
why this ADR is not wrong: the bare formula overstates a full-size square image by **2.15x** and a
thin one by 3.6x, and it was free to find out.
