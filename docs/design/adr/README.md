# Architecture Decision Records

Summaries live in [IMPLEMENTATION.md §15](../../../IMPLEMENTATION.md); full records live here.

**No architectural decision changes without adding or superseding an ADR** (§16 rule 2). Superseding is fine; silently diverging is not.

| ADR | Title | Status |
|---|---|---|
| [000](adr-000-signal-layer-not-runtime.md) | Product is a signal layer, not a runtime | Accepted |
| [001](adr-001-emit-signals-never-enforce.md) | Emit signals, never enforce | Accepted |
| 002 | OTel GenAI semconv as the wire format | Accepted (summary in §15) |
| 003 | Quality lane is tiered, sampled, opt-in, off by default | Accepted (summary in §15) |
| [004](adr-004-fail-open-is-absolute.md) | Fail-open is absolute | Accepted |
| 005 | Pluggable state store, in-memory default | Accepted (summary in §15) |
| 006 | Two delivery surfaces: library + standalone demo | Accepted (summary in §15) |
| 007 | Enterprise control plane designed but not scheduled | Accepted (summary in §15) |
| 008 | Apache-2.0 | Accepted (summary in §15) |

ADRs 000, 001, and 004 are expanded here first because they are the ones M0 code already depends on: they determine what the library is allowed to do, and what it must never do.

## Format

```markdown
# ADR-NNN — Title

**Status:** Proposed | Accepted | Superseded by ADR-MMM
**Date:** YYYY-MM-DD

## Context      — what forced a decision
## Decision     — what was decided
## Alternatives — what was rejected, and why
## Consequences — what this costs us, including the bad parts
```
