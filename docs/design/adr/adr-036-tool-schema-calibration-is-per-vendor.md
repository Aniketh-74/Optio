# ADR-036 — Tool-schema calibration is per-vendor, and ours was OpenAI's

**Status:** Accepted
**Date:** 2026-07-31
**Related:** ADR-015 (isolated live evidence), ADR-024 (a stage may not book what it cannot
attribute), ADR-025 / ADR-032 / ADR-033 (one vendor's shape applied globally — the same mistake,
three times before this)

## Context

`minify_tools` strips annotation-only keys from tool schemas and reports what that saved.
`ANNOTATION_STRIP_CALIBRATION = 0.37` scales the raw JSON token difference into a claimed saving,
and it exists because the stage once claimed 3,240 tokens on `mcp_agent` against 1,210 the provider
really stopped billing — a 2.7× overstatement.

That constant was fitted against **`gpt-4o-mini`**, by differencing whole requests. Its own docstring
says so.

Anthropic's `messages.count_tokens` is exact and free, so the same question can be asked far more
precisely, and on the vendor this package integrates most deeply with. Measured across five tool
counts:

| tools | billed before | billed after | real | raw JSON | claimed | real / raw |
|---|---|---|---|---|---|---|
| 1 | 694 | 605 | 89 | 69 | 25 | **1.29** |
| 3 | 1,062 | 795 | 267 | 207 | 76 | **1.29** |
| 5 | 1,430 | 985 | 445 | 345 | 127 | **1.29** |
| 10 | 2,350 | 1,460 | 890 | 690 | 255 | **1.29** |
| 20 | 4,190 | 2,410 | 1,780 | 1,380 | 510 | **1.29** |

**1.29, to three decimal places, at every size** — and identical on `claude-haiku-4-5`,
`claude-sonnet-4-5` and `claude-opus-4-5`, so it is a property of the API's tool rendering rather
than of any one model.

Against a constant of 0.37, `minify_tools` therefore **understates its own saving by 71.4%** on
Anthropic: 993 claimed against 3,471 the provider actually stopped billing.

The two vendors do not merely differ in magnitude, they differ in **direction**. OpenAI bills *less*
than the raw JSON tokenizes to (0.37); Anthropic bills *more* (1.29), because it re-renders the
schema into its own representation. No single constant can be right for both, and the one in place
was one vendor's answer applied to every vendor.

This is the fourth instance of the same mistake — after tool roles (ADR-025), tool-result blocks
(ADR-032) and truncation vocabulary (ADR-033). Those three were logic written against one provider's
wire shape; this is a *number* fitted against one provider's tokenizer. The rule generalises:
**anything measured against one vendor is that vendor's, until measured against the next.**

Unlike the other three, this one errs in the safe direction. Understating a saving breaks no
guarantee — it costs the user nothing and this package's standing rule is never to overstate. What it
does cost is the product's own claim, on precisely the traffic it targets: tool-carrying Anthropic
agents are told they saved a third of what they saved.

## Decision

### 1. The calibration is a per-vendor lookup

`ANNOTATION_STRIP_CALIBRATION_BY_MODEL` maps a model-name prefix to its ratio, longest match first —
the same shape as `MIN_PREFIX_TOKENS_BY_MODEL` and `PRICING`. Two entries today: `claude` at 1.29,
`gpt-` at 0.37.

Keyed on the vendor family rather than on individual models, because the measurement says that is
what it is: three model generations, one number. A per-model table here would imply a precision the
evidence does not support.

### 2. An unrecognised model keeps 0.37, and that choice is deliberate

Not the mean of the two, and not the higher. 0.37 is the **lowest** ratio measured, so an unknown
vendor is under-claimed rather than over-claimed. ADR-027 made the same call about prefix floors for
the same reason: when the table cannot answer, fail toward the number that cannot flatter.

### 3. The measurement script ships

`scripts/measure_minify_tools.py` costs nothing to run — `count_tokens` is free — so the constant
that had gone three months unchecked can be re-checked on any vendor at any time. A calibration that
is expensive to verify is one nobody verifies.

## Consequences

- **`minify_tools` reports roughly 3.5× more saving on Anthropic**, and that figure is now the
  provider's own arithmetic rather than a fit borrowed from another vendor. Every `mcp_agent` and
  `large_system_agent` result moves up.
- **No cost changes.** This is a reporting fix: the same tokens were already being removed and the
  same money already saved. What changes is what the report says about it.
- The OpenAI figure is untouched, so no existing OpenAI number moves.
- One more vendor-specific constant, and one more argument for the rule in Context. The next
  provider added to this package needs its tool calibration measured, not inherited — and the script
  to do it is free.

## Alternatives considered

**Keep one constant and pick the mean.** Rejected: it would be wrong for both vendors, and wrong in
the *overstating* direction for OpenAI, which is the direction this project treats as serious.

**Stop calibrating and report the raw JSON delta.** Rejected — that is precisely the 2.7×
overstatement the constant was introduced to remove. The raw delta is a property of our
serialization, not of anyone's bill.

**Derive the ratio at runtime from `count_tokens`.** Rejected on the module's standing rule: it adds
a network round trip to the decision path. The call is free in money and not free in latency, and
this package does not make network calls to decide how to build a request.
