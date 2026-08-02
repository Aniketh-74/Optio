# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

Pre-1.0, only the latest minor line receives fixes.

## Reporting a vulnerability

**Please do not open a public issue.** Use GitHub's private reporting:
[Security → Report a vulnerability](https://github.com/Aniketh-74/Optio/security/advisories/new).

This is a solo-maintained project (R-OPS-1). Expect an acknowledgement within a week; if you
have heard nothing in two, assume it was missed and escalate by opening a public issue that
says only that a report is pending — no details.

## What optio promises, and what a vulnerability would look like here

The security surface is unusual for a library, so it is worth naming precisely. optio runs
**inside** your agent process, sees your model traffic, and emits telemetry. Three properties
matter, and a break in any of them is a vulnerability, not a bug:

**1. It never emits or logs prompt or completion content.** Signals are numeric, enum, or cost
only (§7.2). Every log line in this library records an exception *type* and never its message,
because a model client's exception routinely carries the prompt in its payload. A code path
that puts trace content into a span attribute, a log record, or a metric label is a
content-leak vulnerability — report it.

**2. It stores no credentials and constructs no clients.** Pricing needs no keys (static
table). The quality judge is a callable you supply, running on your SDK and your credentials;
optio never reads an API key from the environment, never persists one, and ships no default
judge that could start calling a paid API on your behalf. Any code that reads, forwards, or
records a credential is a vulnerability.

**3. It cannot break your agent.** Fail-open is absolute (ADR-004): every lane runs behind a
guard, and an internal failure produces a missing signal rather than an exception in your call
stack. A crafted input — a hostile span, a malformed attribute, an exploding judge — that
escapes the guard and reaches the agent is a vulnerability, because availability is the
guarantee this library trades on.

### Also in scope

- Dependency vulnerabilities reachable through optio's own code paths.
- Unbounded memory or CPU growth triggerable by agent traffic — per-run state is bounded by
  design, and a way to defeat that bound is a denial-of-service issue.
- Anything that causes a **wrong** cost or quality signal rather than an absent one. A policy
  engine acts on these numbers; a silently incorrect value is more dangerous than a missing one
  (R-TECH-1), so we treat deliberate falsification as a security matter.

### Not in scope

- Vulnerabilities in the frameworks optio adapts to, or in OpenTelemetry itself — report those
  upstream.
- Your judge callable leaking your own prompts to your own model provider. That is your data
  going where you sent it; optio neither inspects nor retains it.
- Cost signals being absent for models outside the pricing table. Documented behaviour, and the
  honest answer to an unknown price.

## Disclosure

Report privately, and we will agree a disclosure timeline together — 90 days is the default.
Fixes ship with a CHANGELOG entry and a GitHub Security Advisory. If you would like credit,
say so; if you would rather not be named, that is fine too.
