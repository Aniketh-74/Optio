# ADR-025 — The neutral tool shape needs an Anthropic translation

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-013 (provider-neutral request), ADR-017 (batch shares `wire`), ADR-023 (response-side
`tool_calls` normalization), ADR-024 (the report cannot see what a stage costs)

## Context

The first full live Anthropic benchmark ran 12 workloads. Two of them failed almost completely:

```
mcp_agent          9 errors,  1 of 10 calls succeeded
tool_calling_chat 19 errors,  1 of 20 calls succeeded

400 invalid_request_error: messages: Unexpected role "tool".
Allowed roles are "user" or "assistant".
```

`wire.anthropic_system_and_turns` copies `message.role` straight onto the wire. This package's neutral
request models a tool result as `role="tool"`, so every tool-using conversation is rejected.

**The same function is what ADR-017's batch submission uses**, so batch dispatch of a tool-using
conversation would fail identically. Nothing has exercised that path either.

### Why the adapter never noticed

`adapters/anthropic.py` works fine, because a real Anthropic caller's original message params ride
through in `extra[_RAW]` and are restored verbatim. The translation is only reached when a request is
*built* neutrally — the benchmark, batch submission, and any caller who constructs an `LLMRequest`
directly. So the defect sat behind the one path nothing had run.

This is also the root of a debt recorded earlier and left unfixed: `cap_tool_results` selects
`role == "tool"` and is therefore a **complete no-op on Anthropic**, because a real Anthropic
conversation never has a message with that role — it has a `user` turn carrying a `tool_result` block.
One representation gap, two symptoms.

### What the two shapes are

The neutral shape is OpenAI-flavoured, which `openai_messages` documents:

- an assistant turn carries `extra["tool_calls"]` — a list of
  `{"id", "type": "function", "function": {"name", "arguments"}}` where `arguments` is a **JSON
  string**;
- a result is `role="tool"`, `content` the payload, `extra["tool_call_id"]` the id it answers.

Anthropic wants both inside content blocks:

- `{"role": "assistant", "content": [{"type": "tool_use", "id", "name", "input": {...}}]}` — `input`
  is a **dict**, not a string;
- `{"role": "user", "content": [{"type": "tool_result", "tool_use_id", "content"}]}`.

`openai_messages`'s docstring already records this exact class of bug being paid for once on the other
vendor: *"OpenAI rejects a `tool` message with no preceding `tool_calls`, so dropping them silently —
which the first version of the live adapter did — makes every tool-calling workload fail with a 400."*
The Anthropic side never got the same treatment.

## Decision

`anthropic_system_and_turns` translates tool roles rather than forwarding them.

### 1. Direction: OpenAI shape is the neutral one, and the wire converts

ADR-023 already settled the mirror image — Anthropic's `tool_use` blocks are normalized *into* the
OpenAI-shaped `response.extra["tool_calls"]` so one verifier reads one shape. The request side follows
the same convention in reverse. Inventing a third, truly neutral tool representation would mean
translating on both vendors instead of one, for no gain.

### 2. `arguments` is parsed to a dict; unparseable arguments do not silently vanish

Anthropic's `input` must be an object. When `arguments` is not valid JSON the tool call cannot be
faithfully represented, and the two ways to be wrong are not symmetric: dropping the block leaves an
orphaned `tool_result` and a 400 that names the wrong problem, while `{}` sends a call the model never
made. So an unparseable argument string raises, and the pipeline's per-stage fail-open turns that into
"send the request unoptimized" rather than a corrupted conversation.

### 3. Consecutive tool results merge into one user turn

Anthropic requires alternating roles, so two `tool` messages in a row — which is what parallel tool
calls produce, and real agents make them constantly — must become **one** user turn carrying two
`tool_result` blocks, not two user turns. The benchmark's own workloads happen not to contain a
parallel call, which is precisely why this is worth writing down rather than discovering later.

### 4. The `cacheable` marker survives translation

A marked tool turn keeps its `cache_control`, on the last block of the turn. `PrefixCacheStage` marks
the last message of the stable prefix and that message can be a tool result in an agent loop —
dropping the marker there would silently lose the largest lossless saving in the package on exactly
the workloads that need it most. This function has already lost a marker once, on turns generally, and
the docstring records it.

### 5. `cap_tool_results` is *not* fixed here

It selects `role == "tool"`, which is right for a neutrally-built request and wrong for an
adapter-built one, where the same content arrives as a `user` turn with a `tool_result` block. That is
a stage-selection question rather than a wire question, and it needs its own measurement — the live
Anthropic figure for that stage is currently zero on every real conversation. Recorded, scoped out,
and now unblocked, because a translation has to exist before the stage can be pointed at it.

## Consequences

- **Two workloads become measurable on Anthropic for the first time**, and they are the two the
  package should be best at: `mcp_agent` and `tool_calling_chat` are where `cap_tool_results` and
  `minify_tools` live. Whatever they say, the current answer of "9 and 19 errors" says nothing.
- **Batch dispatch of tool-using conversations stops being broken**, without a separate fix, because
  ADR-017 deliberately routed both call sites through this one function.
- A tool result's payload is a string on both sides, so nothing here changes what `cap_tool_results`
  or `deduplicate` see when they run before the wire.
- The translation is one-directional. Reading an Anthropic *response* back into neutral shape is
  already done, differently, in `wire.response_from_anthropic_message` and ADR-023's provider
  surfacing; these two must agree on the `tool_calls` shape, and a test pins the round trip.

## Alternatives considered

**Make the bench build Anthropic-shaped requests.** Rejected: it hides the defect rather than fixing
it, leaves batch submission broken, and makes the workloads non-portable between vendors — the
benchmark's whole value is running the same workload against both.

**Skip `role="tool"` messages when targeting Anthropic.** Rejected. It turns a loud 400 into a
conversation missing its tool results, where the model answers from nothing and the run still reports
a cost saving. That is strictly worse than failing.

**Introduce a genuinely vendor-neutral tool representation.** Rejected as scope. It would touch every
adapter, every stage that reads `extra["tool_calls"]`, and ADR-023's verifier, to remove an asymmetry
that costs one translation function. Worth revisiting only if a third vendor arrives.
