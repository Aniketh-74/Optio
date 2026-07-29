# Plug-and-play cost reduction: an Anthropic adapter and a safety floor

**Date:** 2026-07-29
**Status:** Approved, not yet implemented
**Related:** ADR-012 (public API), ADR-013 (the optimize package), ADR-016 (what belongs here)

## Context

The project's goal is a library that **reduces token cost a lot** and is **plug and play**.
Measured against that on 2026-07-29, after running a real agent through the optimizer for the
first time:

| claim | evidence |
|---|---|
| reduces cost a lot | 47.1% on a real four-tool agent task, correct answer, default config |
| ...but | 11.4% in the run where the provider had already cached the baseline's prompt |
| ...and | 3,972 of the 4,083 saved tokens came from one stage hitting one oversized tool payload |
| plug and play | `wrap_openai_client(client)` — genuinely one line, and it optimizes a whole agent |
| ...but | that is the **only** adapter. Anthropic, Gemini and Bedrock users have nothing to plug. |

Two problems follow, and this spec addresses both.

**The reach problem.** `PrefixCacheStage` is described in its own source as *"the single largest
lossless saving available."* It does nothing on OpenAI — OpenAI caches long prefixes automatically
for both A/B arms, which is why a simulated 36.3% saving corrected to −1.8% live. On Anthropic
nothing is cached without an explicit `cache_control` breakpoint, and our `cacheable` marker is
what puts it there. **Anthropic is therefore the only place this stage can pay at all, and we have
never shipped a way for a user to reach it outside the benchmark module.**

**The safety problem.** On the same day, the first real agent run found that `trim_history` was
dropping the user's task. In a chat the oldest message is a stale question; in an agent loop it is
the entire task, and everything after it is the agent's own tool traffic. Providers accept a
conversation with no user message, so nothing failed — the model inferred a task from the evidence
and answered a question nobody asked, at higher output cost:

```
defaults          3,757 in / 288 out / $0.00074   wrong
first turn kept   3,816 in / 131 out / $0.00065   correct
```

1,304 tests did not catch this, because every fixture used the same tidy alternating-chat shape as
the code's own mental model. That is a plug-and-play failure specifically: someone plugs the
library in, gets a cheaper call, and gets a worse answer. **Default-on is only defensible if that
cannot happen quietly.**

## Goals

1. Double the library's plug-and-play reach, from one provider ecosystem to two.
2. Make the largest lossless saving in the package reachable, and measure it.
3. Make silent quality damage from a default-on stage detectable without a human reading a dump.

## Non-goals

- **No new stages, and no new config options.** The stage count stays frozen until existing stages
  have evidence on more than one workload. Every knob is something a user must understand before
  plugging in, so knobs are a plug-and-play cost, not a feature.
- **No LLM-judged answer quality.** In one measured run the *unoptimized* arm produced a wrong
  refund figure because the model passed its own tool an unmatched string. A judge would have
  reported a failure that was not ours.
- **No scenario framework, report module, CLI, or second-SDK runner.** Considered and cut: none of
  them makes the library cheaper or easier to plug in.
- **No CI gate against live providers.** Findings graduate into ordinary pytest tests; pytest is
  already the gate. What was missing was fixtures shaped like an agent, not a second gate.

## Work items

| # | Item | Serves |
|---|---|---|
| 1 | `wrap_anthropic_client()` | reach |
| 2 | Live measurement of `prefix_cache` on Anthropic | cost |
| 3 | `optio_optimize/invariants.py`, wired into pytest and the probe | safety floor |
| 4 | Four harder scenarios in `scripts/real_agent_run.py` | safety floor |

### 1. The Anthropic adapter

New module `src/optio_optimize/adapters/anthropic.py`, mirroring `openai_agents.py` closely enough
that a reader of one can read the other.

```python
from optio_optimize.adapters.anthropic import wrap_anthropic_client

wrap_anthropic_client(client)  # client.messages.create is now optimized
```

Decisions:

- **Both sync and async clients are supported.** `Anthropic` and `AsyncAnthropic` are both in
  ordinary use, unlike OpenAI where the Agents SDK forced async. `Pipeline` already has `execute`
  and `aexecute`, so the second path costs little, and "async only" is not plug and play.
- **`stream=True` bypasses the wrapper entirely**, reaching the real client unmodified. A pipeline
  built around one request producing one response can only buffer a stream, which defeats the
  reason to ask for one. Same rule as the OpenAI adapter.
- **Request translation reuses `wire.py`.** `anthropic_system_and_turns` and `anthropic_tools`
  already exist and are tested; they were written for batch dispatch. The `cacheable` →
  `cache_control` translation lives there and is the mechanism this whole item exists to expose.
- **Response translation moves into `wire.py`.** `_response_from_anthropic_message` currently
  lives in `batch_backends.py` and the adapter needs exactly the same function. One implementation,
  two callers — the third time this pattern has appeared today, after `tools` going unsent and
  batch needing the stage runner.
- **The native response object is returned verbatim** after a real call, preserving every field
  this package does not model. A response is reconstructed only for a cache hit, where handing
  back the stored original would re-bill its usage — the defect the OpenAI adapter already
  documents.
- **Credentials are never handled.** The adapter takes a client the caller already built. The SDK
  reads its own environment variable. This library does not accept, store, log or forward a key.

Unmodeled request fields ride through in `extra`, and unmodeled message fields in `extra[_RAW]`,
restored verbatim unless a stage actually changed the text — the same mechanism the OpenAI adapter
uses to avoid dropping fields it does not understand.

**No ADR is required.** This is an extension within an established pattern: `adapters/` exists,
`wrap_openai_client` sets the shape, and both `Pipeline` execution paths already exist. Nothing
here decides anything ADR-012 or ADR-013 has not already decided.

### 2. Measuring `prefix_cache` on Anthropic

The claim under test: on a provider with explicit caching, `PrefixCacheStage` converts a full-rate
prompt prefix into a discounted one, and that discount is the largest lossless saving in the
package.

Method, following ADR-015 rule 2 (isolated, one stage at a time):

- A multi-turn conversation with a large stable system prompt, run twice through the new adapter —
  once with `prefix_cache` disabled, once enabled. Everything else held constant.
- The measurement is `cache_read_input_tokens` and total billed cost from Anthropic's own usage
  reporting, not our estimate.
- Baseline runs first, so the server-side cache bias works against the result this library wants.
- Recorded in `docs/optimize-benchmarks.md` with the date, model and method, beside the OpenAI
  correction it contrasts with.

The Anthropic key lives in `.env` under the non-standard name `ANTHROPIC_KEY` (`.env` is gitignored
at `.gitignore:12`). It is mapped to the SDK's own variable **in-process only** and never placed on
a command line, where a process listing would expose it. Spend is bounded by the existing
`SpendGuard`.

If the measurement contradicts the claim, the claim changes — the docstring is edited to match the
number, exactly as it was for the 36.3% → −1.8% OpenAI correction. The number is not the goal;
knowing it is.

### 3. `invariants.py`

A pure module: no SDK, no network, no keys. Two entry points, because the two useful kinds of rule
have different shapes.

```python
check(request)                    -> tuple[Violation, ...]   # true of any single request
check_transform(original, sent)   -> tuple[Violation, ...]   # true of any rewrite
```

The second signature is the important one. "A conversation must contain a user message" is not
universally true — a caller whose history legitimately begins with an assistant message is not
malformed, and `TrimHistoryStage` is right to leave it alone. The library is not wrong to *send*
such a request; it is wrong to *create* one. That distinction is only visible with the original in
hand.

**Absolute rules** — the provider enforces these, so violating one is a rejected request:

| rule | why it is real |
|---|---|
| every `tool` message is preceded by an assistant message carrying `tool_calls` | `TrimHistoryStage` already guards this; the guard has never met live traffic |
| every tool result matches a preceding call id | one direction only: mid-loop, a call legitimately has no result yet |
| empty content is valid only alongside `tool_calls` | real agents emit `content=""` with a call attached; empty and unattached is rejected |
| at least one non-system message | an all-system request has nothing to answer |

**Preservation rules** — nothing rejects these, which is why they went unnoticed:

| rule | corresponds to |
|---|---|
| the last user message survives | the `trim_history` defect found on 2026-07-29 |
| the system prompt survives | `PrefixCacheStage` depends on its presence every call |
| surviving messages keep their relative order | no stage claims to reorder messages; `reorder_context` moves blocks inside content |
| tools are only removed or minified, never added; a called tool is never removed | `PruneToolsStage`'s stated promise, unverified against a real loop |

Constraints:

- **A `Violation` carries `(rule, message_index, role)` and never prompt content.** Violations are
  printed and may reach terminal scrollback or CI logs. §10's rule applies here exactly as it does
  to the fail-open guard, which logs an exception's type but never its message.
- **This does not run in the request path.** It is per-message work on every call, for a guarantee
  the pipeline's fail-open provides another way, and SC-5's overhead budget is not there to be
  spent on self-checking.
- Internal under ADR-012 — not exported at the top level. Promoting it later, if users want it for
  debugging, is the deliberate act that ADR describes.

### 4. Harder scenarios

Added to the existing `scripts/real_agent_run.py`, which stays a single script:

- two tools called in one turn (parallel results)
- a tool returning an empty result
- a long loop, ~15 steps, so trimming and capping both engage repeatedly
- a request whose history legitimately opens with an assistant message

Each run calls `check_transform` on every request the library rewrites, and fails the run on a
violation rather than requiring someone to read a dump.

## Architecture

```
src/optio_optimize/
  invariants.py                 pure rules. imports: our own types only.
  wire.py                       + response translation moved in from batch_backends
  adapters/anthropic.py         wrap_anthropic_client, sync and async

scripts/
  real_agent_run.py             + scenarios, + invariant checking

tests/optimize/
  test_invariants.py            rules against fixtures, both signatures
  test_adapters_anthropic.py    mocked HTTP transport, no key, no network, no spend
```

## Testing strategy

- `invariants.py` is unit-tested directly, including a test that each preservation rule fires on a
  deliberately damaged transform. A rule that cannot fail is not a rule.
- The Anthropic adapter is tested with `httpx.MockTransport` against a genuine `Anthropic` client,
  matching how `test_adapters_openai_agents.py` works: real request bodies, no network, no key, no
  spend. Coverage includes the streaming bypass, the cache-hit reconstruction path, and
  `cacheable` becoming `cache_control` on the wire.
- The regression tests added for the `trim_history` defect stay as they are; the new preservation
  rule generalises them rather than replacing them.
- Live measurement is manual and budgeted, never part of the automated suite. Nothing in `pytest`
  may spend money.

## Error handling

- Adapter translation failures fall back to the unmodified real client, matching
  `openai_agents.py`: translation runs before any provider call, so falling back cannot double-bill.
- Stage failures inside the pipeline remain fail-open per ADR-013 rule 1. This spec changes nothing
  there.
- Invariant violations raise in the probe and fail assertions in tests. They never raise in library
  code, because the checker is never called from library code.

## Risks

- **The Anthropic prefix-cache claim may not survive measurement.** That is the point of measuring
  it. The docstring changes if the number says so.
- **A preservation rule could be wrong** — asserting an invariant the library legitimately violates
  would block correct behaviour. Mitigated by deriving each rule from an actual incident or an
  actual provider requirement, and by running the full suite plus the probe before adopting one.
- **Two adapters can drift.** Mitigated by both routing through `wire.py` rather than translating
  independently, which is the specific failure that made a whole benchmark run measure nothing.

## Success criteria

1. `wrap_anthropic_client(client)` optimizes a real Anthropic call in one line, sync or async.
2. `prefix_cache` on Anthropic has a live number in `docs/optimize-benchmarks.md`, with date,
   model and method — whatever that number turns out to be.
3. Re-introducing the `trim_history` defect fails a test without anyone reading output.
4. The full gate stays green: ruff, `mypy --strict`, import contracts, and the suite.
5. No stage was added and no config option was added.
