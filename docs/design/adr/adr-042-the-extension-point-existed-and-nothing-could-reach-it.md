# ADR-042 — The extension point existed and nothing outside could reach it

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-013 (provider-neutral by design), ADR-036 (calibration is per-vendor),
ADR-038 (the counter is warmed at construction), ADR-041 (coverage should not depend on a key)

## Context

ADR-041 gave the limit tables a way to carry a vendor this project has never billed. That fixed the
tables. It did not fix the thing underneath them.

**Every savings figure this package produces is a token count**, and every token count goes through
`TokenCounter` — a two-method Protocol, `is_exact` and `count_text(text, model)`, that anything can
implement. `Pipeline` has accepted one since ADR-038, when the tokenizer warm-up needed it.

`Optimizer` did not, and `Optimizer` is the public entry point. `wrap_anthropic_client` builds one.
`Pipeline` is reachable only through the `pipeline` property, after construction, when the counter is
already fixed. So the extension point existed, was typed, was documented — and **nothing outside this
package could reach it.** Every count, for every vendor, went through `tiktoken`.

`TiktokenCounter._encoding` resolves through `tiktoken.encoding_for_model` and falls back to
`o200k_base` on `KeyError`. Every Anthropic and Google model takes that fallback. The fallback is a
reasonable default — it is much closer than refusing — but it is OpenAI's tokenizer applied to
another vendor's text, and this project has already measured that the vendors are not
interchangeable: ADR-036 found Anthropic bills **1.29×** what the raw JSON tokenizes to for tool
schemas, against OpenAI's **0.65**. Opposite directions, not merely different magnitudes.

So the situation was: the tables could now cite a vendor they had not measured, and the counter under
them still could not be told to count like that vendor.

## Decision

`Optimizer.__init__` takes `counter`, passed straight to `Pipeline`. Four lines.

The change is small because the design was already right — this is a wiring omission, not an
architectural one, and it is worth recording precisely because a wiring omission is the kind that
survives review. Nothing failed. The Protocol was there, the parameter was there one layer down, the
types checked, and the capability was unreachable.

**Provider-reported usage still wins.** `Pipeline.complete` reads
`response.input_tokens or count_request(sent, self._counter)`, and that precedence is now asserted
rather than assumed. A counter is an estimate; the provider's number is the bill. If a supplied
counter could override it, plugging in a vendor tokenizer would make reports *less* accurate on every
provider that reports usage — the exact opposite of the reason for accepting one. A counter fills a
gap the provider left; it does not contradict the provider.

**The supplied counter is the one warmed** at construction. ADR-038 moved `tiktoken`'s 395 ms
vocabulary load out of the first request's latency budget; warming the default instead would pay that
cost for a tokenizer nobody is going to use and still starve request one for the tokenizer they are.

## Consequences

2,169 tests pass. 4 of 4 mutations caught, including "accepted then dropped on the floor" — which is
what the code did before this change, and which five tests now fail on.

This does not make any figure more accurate on its own. It makes accuracy *reachable*: a caller who
can count exactly for their vendor can now say so, and that is what a multi-vendor library owes the
vendors it has not measured itself. The obvious next implementations are Anthropic's
`messages.count_tokens`, which ADR-036 established is exact and free, and HuggingFace `tokenizers`
for open-weight models, which is exact and offline.

**The general shape worth remembering:** an extension point is not an extension point until something
outside the package can construct it. A Protocol plus a parameter on an internal class is a design
for one; the test that proves it is one has to build the object a user builds.
