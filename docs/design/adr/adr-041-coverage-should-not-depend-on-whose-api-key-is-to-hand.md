# ADR-041 — Coverage should not depend on whose API key is to hand

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-015 (measure, do not assume), ADR-029 (generation boundaries), ADR-031 (a published
price change is data), ADR-037 (the two limits), ADR-039 (recordings carry their date)

## Context

`CONTEXT_WINDOW` and `MAX_OUTPUT_TOKENS` carried 15 Anthropic models and nothing else. The
docstrings explained why, and the reasoning was good: every value was read out of a provider's own
400 rather than off a documentation page, which makes it the provider's arithmetic instead of a
claim about it.

That rule produced a bad outcome. Measurement needs an API key, so **coverage became a function of
whose key the maintainer happened to hold.** For a library whose whole purpose is to work against
every vendor, that is the wrong thing for coverage to depend on — and it is not a statement about
the code, which never knew what a vendor was. `_limit_for` is a string lookup; the stages mention
vendors only in docstrings. Nothing was Anthropic-only except the evidence.

The rule was being applied at one bar where two are needed. There are two ways to know a limit:

*measured* — this project sent a request and read the answer. Costs a key and some money. Cannot go
stale silently, because the probe is re-runnable.

*published* — the vendor documents it and this project has not observed it. Costs nothing, covers
every vendor with a documentation page, and can go stale without anything here noticing. **It is not
a guess.** It is a citable claim.

The tables had a slot for the first and a slot for absence, and none for the second. So a
documented-but-unobserved figure had nowhere to live and got filed next to genuine unknowns. That
single missing slot is what made this package look Anthropic-only.

## Decision

A table entry is a `Limit`, not an `int`: the number, which of the two ways it is known, a source,
and the date the source said so. A bare `int` is no longer a valid entry, which makes the citation
structural rather than a convention someone forgets under deadline.

`context_window_for` and `max_output_tokens_for` keep returning `int | None` — every caller predates
this, and a migration that quietly altered a limit would be indistinguishable from the drift the
provenance exists to expose. `context_window_provenance` and `max_output_tokens_provenance` are new.

**Three states, not two.** `None` means this package carries no limit; `PUBLISHED` means the vendor
states one. Collapsing those is exactly the mistake being fixed. Seven Anthropic models remain
absent, because a probe established only that their window exceeds 217,554 (ADR-037), and "we have
not looked" is a different claim from "the vendor says so".

A published entry must cite a URL. A measured one names its probe. Both must be dated — a published
figure with no date cannot be told from a current one, the same reason a recording carries the date
it was made (ADR-039).

### What this bought immediately

`gpt-4o` and `gpt-4o-mini`: **128,000 context, 16,384 output**, read off OpenAI's model pages on
2026-08-02. The first non-Anthropic rows either table has ever carried, and they cost nothing.

That cap matters. 16,384 against a 128,000 window is the widest gap in either table — a cap inferred
from a window would be wrong by nearly eight times — and `AdaptiveMaxTokensStage` now has something
to clamp to on OpenAI where it previously had nothing.

### What it also turned up

Looking up Gemini's limits found **Google lists `gemini-2.0-flash` as "Shut down"**, under "Previous
models". This package prices it, and it is the only Google model priced. The row is kept with a dated
note rather than deleted: removing a price silently changes what every historical report meant, and
dropping it would also drop the only evidence a third vendor was ever priced here. It carries no
limits, because the page states a context window and no output limit.

This is the ADR-029 shape — a table naming a model the API will not serve — caught by reading the
vendor's page rather than by a 404.

## Consequences

2,157 tests pass. 11 of 11 mutations caught, and one of them found a **pre-existing** hole: removing
the generation-boundary check from the *pricing* lookup left the entire pricing suite green. The
existing cases (`claude-opus-4-9`, `claude-opus-6`) share no prefix with any priced id, so a bare
`startswith` gets them right by accident. `claude-opus-4-10` does start with `claude-opus-4-1`, and
without the four-digit discriminator a tenth release would bill at the first one's rate. The
equivalent check in `_limit_for` was pinned by four tests; the older, more load-bearing one by none.

**What this does not do** is lower the bar for savings. A limit is used to warn or to clamp, and
being wrong there costs a warning. A number that appears in a savings claim still has to be
measured — that distinction is the whole reason for keeping the two words apart rather than merging
them into "known".

The honest coverage story is now sayable: *limits measured where we have billed, published where the
vendor states them, absent where neither.* That is a multi-vendor library, and more credible than
one claiming a uniform standard of evidence it never had.
