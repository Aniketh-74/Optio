# Shared state across processes, and the interface that was never exercised

**Status:** Accepted
**Date:** 2026-08-04
**Related:** ADR-004 (fail-open is absolute), ADR-005 (pluggable state store — this supersedes its
interface), ADR-010 (a closed run is final), ADR-012 (the public API is the top level),
ADR-044 (a lane must not report on a run it never saw), ADR-046 (1.0.0 conditions), R-TECH-1

## Context

`optio` keeps per-run state in three places, all process-local:

| Lane | State | Structure |
|---|---|---|
| cost | open reservations per run | `dict[str, dict[str, float]]` in `Ledger` |
| behavior | recent step signatures | `deque(maxlen=N)` **plus** derived counters in `BehaviorWindow` |
| quality | spans awaiting judgement | `dict[str, list[ReadableSpan]]` in `QualityLane` |

An agent sharded across processes has one logical run whose steps land in different memories. Each
process meters a fraction and believes it has the whole. The visible consequence is a budget that
multiplies by worker count: **four workers, a `$0.50` budget, and `$2.00` gets through** — silently,
because every process's arithmetic is internally consistent.

That is R-TECH-1's worst case (silently wrong, not loudly broken) and it is live today.

### The seam exists on paper only

ADR-005 defined a `StateStore` ABC precisely so this had somewhere to attach, and its addendum
records that the config half shipped without the runtime half. What the addendum does *not* say —
and what an audit on 2026-08-04 found — is that **the ABC has never been exercised by even one
consumer.** Nothing constructs a `StateStore`. Its shape is an untested guess.

That matters more than it sounds. ADR-005's stated reason for defining the interface early was so
"the second implementation is not contorted to fit the first." With zero implementations on the
runtime path, writing Redis against it first would bend the ledger around a guess at Redis
semantics — the exact inversion of the intent.

And the guess is wrong in three identifiable ways. The current ABC offers `get`/`set`/`incr`/
`delete`:

1. **`reconcile` cannot be expressed atomically.** It must check a reservation exists, remove it,
   and fold the actual cost into the total — together. With primitives, that logic lives in the
   caller, and a caller cannot be atomic across processes.
2. **Totals are derived by reading every open reservation for a run** (deliberately: a separately
   accumulated total is one missed decrement from drifting). The ABC can fetch one key at a time
   and has no collection read.
3. **`is_finalised`, `knows`, and "unknown" are three distinct states** the ledger depends on.
   Under TTL eviction two of them collapse into one, which is the failure ADR-044 exists to prevent.

## Goal

Multi-process runs report one correct set of signals, for all three lanes, with the difference
between backends proved by tests rather than asserted.

## Non-goals

- **No behaviour change for single-process users.** The in-memory backend stays the default and
  must remain byte-identical in output. If a single-process user can tell this shipped, it is wrong.
- **No new required infrastructure.** SC-1's five-minute first value survives; Redis is opt-in.
- **Not a distributed lock or leader election.** Runs are keyed by `run_id`; no cross-run
  coordination is introduced.
- **No change to the emitted signal names.** This is storage, not vocabulary (R-TECH-2).

## Architecture: the store speaks the domain

### Rejected: a generic key-value store

Extending the ABC with hash operations and a transaction primitive is the conventional answer and
is wrong here. The atomicity requirement *is* domain logic — "reconcile is exactly-once" can only
be guaranteed where the check and the mutation happen together. Pushing it above the interface
makes it unenforceable across processes, and the interface ends up exposing Redis transactions that
the in-memory backend must then emulate.

It also costs round trips: read reservations, compute, write back is three network hops on the
agent's critical path, and a race between the first and last.

### Accepted: one store per lane

Each lane gets a Protocol describing the operations it actually performs. Backends keep their own
promises: plain dict operations in memory, a Lua script in Redis.

```python
class LedgerStore(Protocol):
    def reserve(self, run_id: str, step_id: str, projected: float) -> None: ...
    def reconcile(self, run_id: str, step_id: str, actual: float) -> None: ...
    def snapshot(self, run_id: str) -> LedgerSnapshot: ...
    def close_run(self, run_id: str) -> LedgerSnapshot: ...
    def is_finalised(self, run_id: str) -> bool: ...
    def knows(self, run_id: str) -> bool: ...


class BehaviorStore(Protocol):
    def record(self, run_id: str, step: StepSignature, maxlen: int) -> WindowState: ...
    def state(self, run_id: str, maxlen: int) -> WindowState: ...
    def close_run(self, run_id: str) -> None: ...


class QualityStore(Protocol):
    def buffer(self, run_id: str, step: QualityStep) -> None: ...
    def drain(self, run_id: str) -> tuple[QualityStep, ...]: ...
    def close_run(self, run_id: str) -> None: ...
```

Two new value types, both frozen dataclasses and both deliberately small:

- **`WindowState`** — the aggregates classification reads, and nothing else: `total`,
  `call_counts: Mapping[tuple[str, str], int]`, `errors`, `size`. Not the steps. This is the return
  of `record` so a step costs exactly one round trip: write and read-back together.
- **`QualityStep`** — the serializable projection of a span (see below).

`maxlen` is passed per call rather than configured into the store because one store instance serves
every run in the process, while the window bound is a `Config` value the lane owns. A store that
cached it would hold configuration it has no way to invalidate.

`LedgerStore` is deliberately the **exact signature set `Ledger` already exposes**. The refactor is
"move the body behind a backend," not "redesign the contract," so every existing ledger test keeps
passing against the in-memory backend unchanged. That is the regression net for the riskiest lane.

`Ledger` becomes a thin facade over a `LedgerStore`. Lanes take their store by constructor
injection; `registry.py` builds the configured backend once and hands it to each lane.

## The three hard parts

### 1. Atomicity — Lua, not WATCH/MULTI

Every compound operation ships as a Lua script: one round trip, atomic by construction, no retry
loop. `WATCH`/`MULTI` needs optimistic retry, which degrades exactly under the contention this
feature exists to serve.

Scripts are loaded with `SCRIPT LOAD` and invoked by SHA with a fallback to `EVAL` on `NOSCRIPT`
(a restarted or failed-over Redis drops its script cache).

### 2. The behaviour window's O(1) guarantee

`BehaviorWindow` is not a list. It is a bounded deque **plus** incrementally maintained aggregates
(`_total`, `_call_counts`, `_errors`) with explicit decrement-on-eviction. That is what makes
classification O(1) in window size — a property `README.md` publishes as measured: *37 µs at 50,
38 µs at 1000, flat in window*.

A Redis port that stored the list and recomputed aggregates on read would silently convert a
documented O(1) into O(window) **and** move up to 1000 records across the network per step.

**Decision:** the Lua script maintains the aggregates in Redis, mirroring the Python eviction
logic, and returns a compact `WindowState` (the counters, not the steps). The window contents never
cross the wire on the hot path.

**This is the highest-risk correctness surface in the milestone**, because the eviction arithmetic
now exists twice. The contract test suite is what keeps the two honest, and the property test below
is aimed squarely at it.

### 3. The quality lane holds objects that cannot cross a process

`QualityLane` retains `ReadableSpan` instances; `JudgeRunner` holds `Future` objects and a thread
pool. Neither is serializable, and futures are inherently process-local.

**Decision:** define `QualityStep` — a small frozen dataclass carrying only the fields the lane
actually reads downstream — and buffer that instead of the span. The judge's in-flight futures stay
process-local by design: judging is triggered at run end, in the process that observes run end, and
that process drains a complete buffer. No future ever needs to cross a boundary.

The projection is smaller than the risk first suggests. Reading `on_run_end` today, the retained
spans are consumed for **their count** (`JudgeRequest` is built with `step_count=len(spans)` and
`content={}`) and for the tier decision. So `QualityStep` starts as span name, start/end timestamps,
and the `gen_ai.*` attributes the tier decision reads — not the span object graph.

That claim is load-bearing, so the plan's first quality task is to *verify* it against the code
rather than trust this paragraph, and to pin the outcome with a test asserting no `ReadableSpan`
ever reaches the store. If a later feature needs more of the span, it extends `QualityStep`
deliberately instead of discovering the gap in production.

## TTL, and the states it can collapse

**TTL refreshes on every write.** An absolute expiry is a time bomb: a long run loses its state
mid-flight, open reservations vanish, and `budget_remaining` jumps back to full — ADR-044's exact
failure, arriving on a timer. Idle timeout, keyed to `run_ttl_seconds`.

**A tombstone outlives the run data.** `close_run` writes a small marker with its own longer TTL so
`is_finalised` stays true after the payload expires. Without it, a late span arriving post-expiry
creates a *fresh* run record — resurrecting a run ADR-010 declares final.

Ordering of the three states after expiry is then: payload gone, tombstone present → finalised;
both gone → unknown, and per ADR-044 the lane stays silent. Silence is the safe direction.

## Failure behaviour: absence, never a partial number

ADR-004 governs: a store that is slow or unreachable degrades to a dropped signal, never a blocked
agent. But fail-open needs a sharper rule on a *shared* store than it did in memory.

In memory, degrading meant one process's data — complete for that process. On a shared store,
degrading means **the other processes' spend is invisible**, so a computed `budget_remaining` could
report a full budget for a run that is already overspent. That is a wrong number, not a missing one.

**Rule: if the store is unreachable, the lane emits no cost signal at all.** Absence is not zero,
and it is not "whatever this process happened to see."

Supporting decisions:

- **Tight timeouts, configurable, short by default.** A hung Redis must not add latency to the
  agent. Connect and read timeouts default to 50 ms.
- **One warning, not one per step.** Silence must not be mistaken for zero spend, and a warning in
  a hot loop is one people filter out.
- **No retries on the hot path.** A retry is latency the agent pays for a signal it can live
  without.

## Configuration

`store_backend="redis"` stops raising and starts working. `redis_url` returns, and the `redis`
extra is restored to `pyproject.toml`. Two additions:

- `store_timeout_ms: int = 50`
- `run_ttl_seconds` gains a documented second role: it is the idle expiry for shared state.

Validation stays at setup (§4.2): an unreachable Redis at construction is a **setup** error, loud,
because configuration that cannot do what it claims is exactly what ADR-005's addendum refused to
ship again. Unreachability *later*, on the runtime path, is fail-open.

## What this supersedes

The `StateStore` ABC is replaced by the three domain Protocols. ADR-005's decision (pluggable
storage, in-memory default, atomic increments, store failures are lane failures) stands unchanged —
only its interface shape is superseded, and it is superseded because it was never exercised.

This needs its own ADR recording that the generic-KV shape was rejected and why, so the next person
does not reintroduce it (§16 rule 2: superseding is fine, silently diverging is not).

## Testing

**One contract suite, both backends.** A single parametrised suite runs against
`[InMemory, Redis]`. This is what makes "interchangeable" a checked claim. Every existing ledger,
behaviour and quality test continues to run against the in-memory backend unchanged.

**Real Redis in CI**, as a service container. `fakeredis` may be used for local speed, but the
gate runs against the real thing: Lua semantics are precisely where a fake diverges, and Lua is
carrying the correctness here.

**The test that proves the bug is fixed.** Spawn N real processes, each reserving and reconciling
against one shared `run_id`, then assert the final total is exactly right. Against today's code
this test demonstrates `$2.00` passing a `$0.50` budget. If the milestone ships without it, nothing
has been demonstrated.

**A property test for the behaviour aggregates.** Generate random step sequences longer than the
window; assert the Redis-maintained counters equal the Python ones step for step. This is the guard
for the duplicated eviction arithmetic, and it is the test most likely to find a real defect.

**A no-spans-in-the-store test** asserting `QualityStore` never receives a `ReadableSpan`.

**Overhead re-measured**, not assumed. The README publishes per-step figures; the in-memory path
must not regress, and the Redis path gets its own row with its own number rather than inheriting
one it did not earn.

## Risks

| Risk | Mitigation |
|---|---|
| Eviction arithmetic diverges between Python and Lua | Property test comparing counters step for step |
| The quality projection omits a field the judge needs | Derive it from `on_run_end`'s actual reads; pin with a test |
| Redis latency lands on the agent's critical path | 50 ms timeouts, no retries, overhead row measured and published |
| The refactor changes single-process behaviour | `LedgerStore` mirrors today's signatures; existing tests run unchanged as the net |
| Scope: three lanes is a large diff | Sequence cost → behaviour → quality; each is independently shippable behind the same store |

## Success criteria

1. Four processes sharing one `run_id` produce one correct total, proved by a spawning test.
2. Every existing test passes unchanged against the in-memory backend.
3. The contract suite passes identically against both backends.
4. Behaviour classification stays O(1) in window size on both backends, measured.
5. An unreachable Redis produces no cost signal and one warning — never a partial number.
6. `store_backend="redis"` is documented as working, and the README's multi-process limitation is
   deleted rather than reworded.
