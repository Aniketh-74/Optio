# ADR-050 — The store speaks the domain, because the generic one could not

**Status:** Accepted
**Date:** 2026-08-04
**Supersedes:** ADR-005's `StateStore` interface (its decision stands; see the addendum there)
**Related:** ADR-004 (fail-open is absolute), ADR-010 (a closed run is final),
ADR-042 (the extension point existed and nothing could reach it),
ADR-044 (a lane must not report on a run it never saw), ADR-046 (1.0.0 conditions), R-TECH-1

## Context

Per-run state lived in each lane's own structures, process-local. An agent sharded across
processes has one logical run whose steps land in different memories, so every worker meters a
fraction and believes it has the whole. **Four workers, a `$0.50` budget, and `$2.00` gets
through** — silently, because each process's arithmetic is internally consistent. That is
R-TECH-1's worst case, and it was live.

ADR-005 anticipated this and left a `StateStore` ABC as the seam. An audit on 2026-08-04 found
the seam had **never been exercised by any consumer**: nothing constructed a `StateStore`, so
its `get`/`set`/`incr`/`delete` shape was an untested guess.

That matters more than it sounds. ADR-005's stated reason for defining the interface early was
so "the second implementation is not contorted to fit the first." With zero implementations on
the runtime path, writing Redis against it would have bent the ledger around a guess at Redis
semantics — the exact inversion of the intent. And the guess was wrong in three ways:

1. **`reconcile` could not be expressed atomically.** It must check a reservation is open,
   remove it, and fold the cost into the total *together*. A caller holding primitives cannot do
   that across processes, however careful it is.
2. **Totals are derived by reading every open reservation** — deliberately, since a separately
   accumulated total is one missed decrement from drifting. The ABC could fetch one key at a
   time and had no collection read.
3. **`is_finalised`, `knows`, and "unknown" are three distinct states.** Under TTL eviction two
   of them collapse, which is precisely the failure ADR-044 exists to prevent.

## Decision

**One Protocol per lane, describing the operations that lane actually performs.** The cost lane
gets `LedgerStore` with `reserve`/`reconcile`/`snapshot`/`close_run`/`is_finalised`/`knows`/
`evict`/`run_count` — deliberately the exact method set `CostLedger` already exposed, so the
existing tests verify the extraction rather than a rewrite of it.

Each backend keeps the exactly-once promise its own way: a lock in memory, a Lua script in
Redis. The promise is enforced where the check and the mutation happen together, which is the
only place it can be.

`CostLedger` becomes a thin facade over the Protocol. Its callers are unchanged.

### Supporting decisions

**Lua, not `WATCH`/`MULTI`.** One round trip, atomic by construction, no retry loop. Optimistic
retry degrades exactly under the contention this feature exists to serve.

**The TTL is an idle timeout, refreshed on every write.** An absolute expiry is a time bomb: a
long run loses its open reservations mid-flight and `budget_remaining` jumps back to full —
ADR-044's failure arriving on a timer rather than through a bug.

**A tombstone outlives the payload.** After `close_run`, a small marker with a longer TTL keeps
`is_finalised` true. Without it a late callback finds no state, is treated as a new run, and
starts a second total under an id already reported — resurrecting a run ADR-010 declares final.
It is the in-memory backend's `_recently_closed` window expressed as a TTL rather than a bounded
FIFO.

**An unreachable store emits nothing, never a partial number.** In memory, degrading meant one
process's data, complete for that process. On a shared store it means the *other* processes'
spend is invisible, so a computed `budget_remaining` could report a full budget for an overspent
run. That is a wrong number rather than a missing one.

This needed **no new code**: the fail-open guard (ADR-004) already absorbs any `Exception` from
a lane, returns the empty signal list, warns once per component and counts the rest, and
`StoreUnavailableError` is a `StateStoreError`. Adding a second layer inside `CostLane` would
have duplicated an existing guarantee. What was missing was proof, and that is now a test.

**Setup fails loudly; runtime fails open.** An unreachable Redis at wiring time raises, because
configuration naming a backend it cannot reach is a setup error (§4.2). A backend that stops
answering later is a runtime condition, and the lane goes quiet.

## Alternatives

**Extend the generic ABC with hash operations and transactions.** Rejected, and it is the
conventional choice. The atomicity requirement *is* domain logic; pushing it above the interface
makes it unenforceable across processes, and the interface ends up exposing Redis transactions
that the in-memory backend must then emulate. It also costs three round trips per logical
operation on the agent's critical path, with a race between the first and the last.

**Keep `StateStore` and add `LedgerStore` beside it.** Rejected. Nothing constructed it and
nothing now would; a second unreachable abstraction is what ADR-042 keeps catching. It is
deleted, along with `InMemoryStateStore` and its tests, which tested a class no production code
built.

**Ship the refactor first, Redis later.** Considered seriously and rejected by the author of the
work: a half-migration sitting on main is a seam with one implementation, which is the state
that produced this ADR in the first place.

## Consequences

2,319 tests pass. The contract suite runs 20 tests against **both** backends, which is what
makes "interchangeable" a checked claim rather than an aspiration.

**The gate runs a real Redis**, as a Linux-only job — service containers do not exist on the
macOS and Windows legs of the test matrix. `OPTIO_REQUIRE_REDIS` turns the fixtures' skip into a
failure there, because a job whose whole purpose is running these tests must not pass by not
running them.

**Multi-process runs are supported and proved**: four spawned processes metering one run produce
exactly the right total, asserted rather than asserted-about. The in-memory backend's limitation
is documented by a second test that measures it — each worker sees exactly its share — instead of
by a sentence.

**Costs, accepted.** Three Protocols instead of one, so a fourth lane adds a fourth. Redis adds
an operational dependency for anyone who opts in, and `run_count` on that backend is O(keyspace)
via `SCAN` rather than O(1) — diagnostic only, never on a request path.

**Still open.** The behaviour and quality lanes remain process-local; this ADR covers the cost
ledger, which is where the money is. Their stores are the next two plans in this milestone, and
the behaviour window is the harder of the two: it maintains incremental aggregates that give it
an O(1)-in-window-size guarantee the README publishes as measured, and a naive port would turn
that into O(window) without failing anything.
