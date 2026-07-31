# ADR-032 — `cap_tool_results` is blind to the shape Anthropic callers send

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (the optimizer's contract), ADR-016 (do not mutate what the caller handed us),
ADR-022 (non-text content is content, and `wire` is the single reading of it), ADR-024 (a stage may
not book what it cannot attribute), ADR-025 (the neutral tool shape needs an Anthropic translation)

## Context

`cap_tool_results` bounds how large a single tool result may grow, because an oversized payload is
re-billed on every later turn of the conversation. On the live Sonnet 4.5 suite it saved **7,831
tokens on `mcp_agent`** — the second-largest stage saving on that workload, behind only
`trim_history`.

`mcp_agent` builds `Message(role="tool", content=payload)` directly. A real user reaching this
package through `wrap_anthropic_client` sends the Anthropic wire shape, where a tool result is a
`tool_result` **block** inside a `role="user"` turn.

Measured on both shapes with an 8,001-token payload:

```
bench shape (role='tool')   content_tokens=8001   saved=5981   "capped oversized tool result(s)"
adapter shape               content_tokens=   0   saved=   0   ""
```

**Zero, for two independent reasons.** `before` skips any message whose `role != "tool"`, and the
adapter preserves the caller's role, so a tool result arrives as `"user"`. And even with the role
filter removed the stage would still find nothing: `_text_from_content` deliberately contributes
nothing for a non-text block, so `message.content` is `""` and there is no payload to measure or
truncate. The 8,001 tokens are in `extra[RAW_CONTENT_KEY]`, where the stage never looks.

So a stage that is one of the two largest savings on the suite's flagship agent workload is a
**complete no-op for every user of the flagship integration**. That is the same shape as the defects
this measurement loop keeps surfacing — something that measures well in the benchmark and does
nothing in the product — and it also means the suite's `mcp_agent` figure overstates what an
Anthropic adapter user actually receives.

ADR-022 already settled the principle for the sibling case. A cache key that read only
`Message.content` hashed two different images identically, and the fix was that non-text content *is*
content, read through one shared helper in `wire`. The same reasoning was never carried to the stage
that exists to bound the largest non-text payload there is.

## Decision

### 1. The stage caps tool results in both shapes

Selection stops being "role is `tool`" and becomes "this message carries tool-result payload",
satisfied by either a `role="tool"` message's text or `tool_result` blocks in the raw content. The
neutral path is unchanged; the adapter path starts working.

### 2. The wire shape is read through `wire`, never parsed in the stage

`wire` gains the helper that finds tool-result payloads and the one that rebuilds content with them
capped. ADR-022's rule verbatim: a second, subtly different reading of the same wire shape is exactly
the divergence `wire` exists to prevent. `is_text_block` and `canonical_block` already live there and
are already shared with the cache key and the adapter.

### 3. Capping rebuilds the raw content; it never mutates the caller's dict

The stage produces a new `extra[RAW_CONTENT_KEY]` holding capped blocks. Editing the caller's own
param in place would be a side effect on an object we were handed, which ADR-016 forbids
independently of whether it happens to work.

This composes with the adapter without changing it. `_param_from_message` returns the raw param
untouched when `_text_from_content(original) == message.content` — for tool-result blocks both sides
are `""` either way, so the capped raw is what goes out. The existing `_rewritten_content` path,
which can only write into a single text block, is not involved and does not need to be.

### 4. The saving is the payload actually removed, in both shapes

ADR-024's rule: the ledger records tokens the provider will not be billed for, measured against what
was really sent. The adapter path measures the block payload rather than `message.content`, which is
the number that changed.

## Consequences

- **Anthropic adapter users gain a saving they were being told they had.** The savings ledger
  reported `cap_tool_results` correctly as zero on that path, so nothing was mis-reported — but the
  benchmark's `mcp_agent` result was being read as representative of adapter traffic, and it was not.
- **The cache key moves when a payload is capped**, because `non_text_digest` hashes the raw blocks.
  That is correct: a capped result is different content and must not serve a full result's cached
  answer.
- **`docs/optimize-benchmarks.md`'s `mcp_agent` figure needs its caveat removed** once re-measured
  through the adapter, and until then the honest statement is that the workload measures the neutral
  path.
- One more piece of Anthropic wire knowledge concentrated in `wire`, which is the intended
  direction: three modules now read `tool_result` blocks and exactly one of them knows their shape.
- The gap is specific to non-text blocks. `deduplicate`, `prune_retrieval` and `summarize_history`
  read text, which the adapter does populate, so they were never affected. `cap_tool_results` is the
  only stage whose target is a non-text block type.

## Alternatives considered

**Normalize a `tool_result` block into `Message(role="tool")` in the adapter.** The tidier-looking
fix, and rejected on the round trip: one Anthropic user turn may carry several `tool_result` blocks
alongside text, so the translation would be one param to many `Message`s. `_param_from_message`
depends on a 1:1 mapping to restore the caller's param — including tool_use blocks and provider
extensions it does not model — and breaking that to serve one stage trades a contained fix for a
structural one.

**Have the stage parse `extra[RAW_CONTENT_KEY]` itself.** Rejected under decision 2. It is three
lines shorter and it is the second reading of the wire shape that ADR-022 was written about.

**Leave it, and document that `cap_tool_results` is neutral-path only.** Rejected. The stage's own
docstring states the reason it exists — an oversized payload is re-billed on every later turn — and
that argument is at its strongest precisely for MCP-connected Anthropic agents, which is the traffic
that currently gets nothing.
