# ADR-005 — Pluggable state store, in-memory default

**Status:** Accepted — with the Redis backend **deferred**, see Addendum
**Date:** 2026-07-26 (addendum 2026-07-27)
**Related:** ADR-004, R-TECH-1, SC-1, §7.1

## Context

Cost accounting is stateful. The ledger holds open reservations between a step starting and
reconciling; the behavior window holds recent step signatures; the quality lane holds spans
until run end. That state has to live somewhere.

Two requirements pull apart. **SC-1** wants first value in under five minutes, which means no
new infrastructure — a library that requires standing up Redis before it emits a single number
will not be evaluated at all. But an agent sharded across processes has one logical run whose
steps land in different memories, and per-process totals would each be a fraction of the truth.

## Decision

**A `StateStore` ABC with an in-memory default**, so the storage decision is a configuration
change rather than a rewrite.

In-memory is the default because zero-infrastructure is what makes SC-1 achievable. The ABC
exists so the distributed case has somewhere to attach.

**Store failures are lane failures.** A backend that is slow or unreachable degrades to a
dropped signal, never a blocked agent (ADR-004). Implementations raise `StateStoreError` and
the fail-open guard absorbs it.

`incr` is on the interface and must be **atomic**: the ledger's running totals go through it,
and a lost update is a wrong cost signal — the silent-wrongness failure R-TECH-1 calls the
worst possible bug.

## Alternatives

**Require Redis from the start.** Rejected: it trades the adoption story for a capability most
single-process users never need.

**In-memory only, no abstraction.** Rejected at design time. Defining the contract before there
are two implementations is what stops the second one from being contorted to fit the first.

**Hide it entirely and auto-detect.** Rejected. Where a run's cost total lives is exactly the
kind of thing an operator must be able to state, not infer.

---

## Addendum (2026-07-27): the Redis backend is not implemented

Shipping 0.1.0 revealed that this ADR had been *half* implemented, in the most dangerous way.
`Config` accepted `store_backend="redis"`, validated it, required a `redis_url`, exposed
`OPTIO_STORE_BACKEND`, and `pyproject.toml` declared a `redis` extra that installed the driver.

Nothing on the runtime path read the setting. No `StateStore` was ever constructed — per-run
state lives in `RunContext` and each lane's own structures. A user configuring Redis for a
distributed deployment would have received no error and no Redis: every process metering into
its own memory while believing state was shared, producing cost totals quietly too low in
exactly the deployment where nobody would think to check.

**Decision: reject `store_backend="redis"` at construction** until the backend exists, and drop
the extra.

Setup-time failure is correct here and does not conflict with ADR-004: fail-open governs the
*runtime* path, where optio must never break a running agent. Configuration that cannot do what
it claims is a setup error, and §4.2 says setup fails loudly.

The ABC stays. It is still the seam the distributed path attaches to, and its docstring now
says plainly that nothing constructs it yet — a design fixture, not a switch you can throw.

**What implementing it properly would take**, recorded so the next attempt does not
under-scope: atomic `incr` across processes (Lua script or `WATCH`/`MULTI`, not read-modify-
write), TTL-based eviction matching `run_ttl_seconds`, connection failures that fail open
rather than hang, and integration tests against a real Redis in CI. That is a milestone, not a
patch.

---

## Addendum (2026-08-04): the interface is superseded, the decision is not

[ADR-050](adr-050-the-store-speaks-the-domain.md) replaces the generic `StateStore` ABC with one
Protocol per lane, and builds the Redis backend this ADR deferred.

Everything decided above stands: pluggable storage, in-memory default, atomicity as an interface
requirement, store failures as lane failures. Only the *shape* is superseded — and it is
superseded because it was never exercised. No consumer ever constructed a `StateStore`, so
`get`/`set`/`incr`/`delete` was a guess, and it turned out it could not express `reconcile`
atomically, could not read a run's reservations as a collection, and collapsed `is_finalised`
into `unknown` under TTL.

The "what implementing it properly would take" list below proved accurate and complete. The ABC,
`InMemoryStateStore`, and their tests are deleted rather than left as a fixture nothing reaches.

The costs recorded below are now settled: **multi-process runs are supported**, proved by four
spawned processes metering one run to the exact total, and `store_backend='redis'` no longer
raises.

## Consequences

**Good**

- Zero-infrastructure default; `pip install` to first signal with nothing else running.
- The distributed limitation is now *loud*. A user who needs it finds out at setup, in one line,
  rather than from a finance review months later.
- The interface is designed but uncommitted, so the eventual implementation is not shaped by a
  premature guess at Redis semantics.

**Costs, accepted deliberately**

- **Multi-process runs are unsupported in 0.1.** Cost totals are per-process. Real limitation,
  stated in the README and the runbooks rather than papered over.
- **The `StateStore` ABC is currently unreached by production code**, which is dead weight until
  the backend lands. Kept because deleting and reinventing it would be worse.
- **A config option that only ever raises looks odd.** Better odd than silently wrong.
