# ADR-016 — The in-scope test for a cost technique

**Status:** Accepted
**Date:** 2026-07-29
**Related:** ADR-013 (the package exists), ADR-015 (what evidence promotes a stage), §5

## Context

The published literature on LLM cost reduction lists on the order of forty techniques. This
package implements ten. The gap has never been written down, so every time a new technique comes
up the same argument gets re-had from scratch, and — worse — it has twice been settled for the
wrong reason.

The wrong reason is **effort**. Techniques have been waved off here as "that needs a queue", "that
needs a team to operate", "that's a deployment topology". Operational burden is a reason for a
*caller* to decline a technique. It is not a reason for a library to refuse to implement one,
because the entire point of a library is that it absorbs work the caller would otherwise do. The
Batch API is the clearest case: it is a flat ~50% discount on any request whose answer is not
needed in the next second, it was declined here as "requires the caller to restructure work as
async", and that is only true if nobody writes the aggregator. Writing the aggregator is a
library's job.

There is a second, quieter failure mode in the other direction. A technique with a published
headline number is not automatically worth having. Three of the four `ALTERED`-tier stages already
shipped came back from live measurement worse than not optimizing at all (`docs/optimize-
benchmarks.md`), and in every case the field literature's own caveats had predicted it — the
caveats simply get stripped when the number gets quoted. Adding techniques on the strength of
cited results would mean shipping unverified claims, which is the one thing this package's
benchmark discipline exists to prevent.

So the scope question is genuinely two questions: *can this package express the technique at all*,
and *can it measure whether the technique worked*.

## Decision

**A technique is in scope if and only if it satisfies all three of the following.** Effort is not
one of them, and is explicitly retired as a reason to decline.

1. **Expressible against the normalized types.** The technique is a transformation of an
   `LLMRequest` into another `LLMRequest`, an observation over one, or a change in *when and how*
   a request is dispatched. If describing it requires restructuring the caller's agent, their
   retrieval pipeline, or their serving stack, this package cannot perform it from inside a single
   call.

2. **Requires no infrastructure the caller must stand up and secure.** A sandbox for
   model-generated code, a GPU fleet, a training pipeline, or a vector database the caller must
   operate are all outside what an in-process library can assume. Infrastructure *we* own and ship
   — an in-memory cache, a batch aggregator, a token counter — is not in this category.

3. **Measurable by the bench harness.** If `optio_optimize.bench` cannot produce a number for the
   technique, this package cannot make a claim about it, and per ADR-015 an unmeasurable claim does
   not ship. A technique that passes (1) and (2) but fails (3) requires the harness be extended
   first — that is a prerequisite, not an exemption.

**A technique that fails (1) or (2) is documented as a recommendation, not built.** The library's
value there is telling a user the technique exists and roughly what it is worth to them; pretending
to implement it would be worse than silence.

### Classification

**In scope — build.** Each is a request transform, an observation, or a dispatch decision:

| Technique | Shape here | Evidence in the field |
|---|---|---|
| Tool schema minification | Rewrite `request.tools` | Anthropic: 85% token cut, accuracy 49%→74% |
| Per-request tool pruning | Rewrite `request.tools` | same; `ALTERED` (a dropped tool cannot be called) |
| Tool result capping | Rewrite `tool`-role messages | "one unbounded tool response adds 50k tokens to *every subsequent turn*" |
| Output brevity instruction | Append to system message | "routinely 30–50% of output tokens in chat products" |
| Stop sequences | New `LLMRequest.stop` field — carried, never *chosen* | 5–40% of output |
| Reasoning-budget control | New `LLMRequest.thinking_budget` field | "now the dominant output lever" on reasoning models |
| Chain of Draft | Append to system message; `ALTERED` | arXiv:2502.18600 — 7.6% of CoT tokens |
| Retrieved-chunk reordering | Reorder blocks within a message | Lost-in-the-middle; zero token saving, quality only |
| Append-then-compact | Change *when* trimming fires | "append-then-compact beats slide-every-turn in almost every case" |
| Unstable-prefix detection | Observation only, transforms nothing | "the single most common production caching bug" |
| Batch dispatch | Changes when a request is sent | ~50% on every major provider |

**Out of scope — recommend, do not build:**

| Technique | Which test it fails | Why |
|---|---|---|
| Self-hosted inference, vLLM/SGLang, KV reuse | (2) | We consume inference; we do not serve it |
| Fine-tuning / prompt distillation | (2) | A training pipeline is not a request transform |
| Code execution with MCP | (2) | Requires a sandbox the caller must operate and secure — the ~99% figure is real and the security surface is also real |
| Planner/worker decomposition, externalised agent state | (1) | We see one call. We cannot restructure the agent that made it |
| AI Gateway topology | (1) | A deployment shape. Our pipeline is the in-process equivalent, and that is the correct scope for a library |
| Metadata pre-filtering, cross-encoder reranking | (1) | Both happen *before* a request exists. `prune_retrieval` is the in-request approximation and says so |
| Learned routers (RouteLLM-style) | (1), (2) | Needs training data from the caller's own traffic. We ship the routing hook; the model is theirs |

**A field, not a stage: stop sequences.** The literature scores this "5–40% savings, no accuracy
impact *if correct*", and the conditional is doing all the work. A correct stop sequence is a fact
about the caller's output format — where their answer ends — and there is no version of that this
package can infer. Stopping at `}` breaks nested JSON; stopping at a newline breaks lists. So
`LLMRequest` gains a `stop` field that adapters carry and `request_key` includes, and no stage
sets it. A stage that guessed would silently truncate answers, which is the failure mode the
truncation notice in `cap_tool_results` exists to avoid elsewhere in this same commit.

**Already covered, no new stage needed:** exact-match caching, semantic caching, prompt caching
markers, sliding window, summarisation, deduplication, retrieval pruning, model routing, structured
outputs, output ceilings. Agent loop detection for the *identical repeat* case is `exact_cache`
serving the repeat, which is strictly better than breaking the loop.

### What this ADR does not decide

It does not set any new stage's default. Every stage added under it enters at the fidelity tier its
mechanism warrants, and any `ALTERED` one is off by default and subject to ADR-015's evidence bar
before that changes. A technique being in scope is permission to build and measure it, never a
prediction that it will earn its place — three of the four stages already measured did not.

## Alternatives

**Implement everything with a published number.** Rejected: it is how the field's benchmark
literature became unreliable in the first place, and this package's only real differentiator is
that every stage carries its own measured verdict, including the losses.

**Keep declining on effort.** Rejected as the defect that motivated this ADR. It produced a scope
boundary that tracked how hard something looked rather than whether it belonged, and it cost the
package the single largest evidenced win available to it (tool schema cost) for no principled
reason.

**Split the out-of-scope techniques into a second package.** Rejected for now. `optio` and
`optio_optimize` already divide along a real seam (signals versus transforms, ADR-013); a third
package for "things that need infrastructure" would divide along an operational one, and the pieces
that would go in it — sandboxes, training, serving — have no shared abstraction between them.

## Consequences

The package roughly doubles in stage count, and every added stage carries a live measurement
obligation under ADR-015 that is more expensive than writing the stage itself. That is the intended
ratio.

`LLMRequest` gains two fields (`stop`, `thinking_budget`) that no existing stage reads, which is a
public type change — additive, defaulted, and therefore compatible, but it does widen the surface
adapters must map.

Batch dispatch does not fit the `Stage` contract at all: a stage decides how a request is shaped,
not whether it is sent now or in an hour. It needs a surface of its own alongside `Optimizer`
rather than a config flag, and that surface is asynchronous by nature. This is the one item here
that changes the package's public shape rather than extending it, and it is sequenced last for that
reason.

The recommend-don't-build list needs somewhere to live that a user will actually read, or it is
just an ADR nobody opens. It belongs in the package's own documentation as "techniques this library
deliberately does not implement, and what to do instead" — a section that makes the library more
useful precisely by describing its own boundary.
