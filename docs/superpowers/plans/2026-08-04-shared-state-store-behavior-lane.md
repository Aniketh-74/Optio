# Shared State Store — Behaviour Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make loop detection work across processes, so four workers sharing one `run_id` classify the run from all of its steps instead of each seeing a quarter and calling it healthy.

**Architecture:** `BehaviorWindow` moves behind a `BehaviorStore` Protocol, exactly as `CostLedger` moved behind `LedgerStore` (ADR-050). The in-memory backend is today's implementation, unchanged. The Redis backend keeps the window's incremental aggregates in a hash and reduces them **server-side in Lua**, returning five scalars rather than the counter — which is what keeps the published O(1)-in-window-size guarantee true over a network.

**Tech Stack:** Python 3.10–3.14, `redis-py` (dev extra), Lua, pytest, hypothesis, real Redis service container.

## Global Constraints

- **No behaviour change for single-process users.** The in-memory backend stays byte-identical in output; every existing behaviour-lane test passes unchanged.
- **The O(1)-in-window-size guarantee is published in `README.md` as measured** ("37 µs at 50, 38 µs at 1000"). Neither backend may make classification, or its payload, grow with `behavior_window_size`.
- **Layering (import-linter):** `optio.lanes` may import `optio.store`; never the reverse. `optio.lanes.behavior` must not import `optio.lanes.cost` or `optio.lanes.quality`.
- **Fail-open governs the runtime path** (ADR-004); setup failure is loud (§4.2).
- **An unreachable store emits no behaviour signal** rather than a partial verdict — a run classified from one worker's steps is a wrong verdict, not a missing one.
- Gate every commit: `ruff check .` && `ruff format --check .` && `mypy` && `lint-imports` && `pytest -q`.
- **Never log or store prompt content** (§10). A `StepSignature` carries a call identity, never arguments.
- Redis timeouts stay at `Config.store_timeout_ms` (50 ms default). No retries on the hot path.

---

## The finding that shapes this plan

`classify` **never iterates the step signatures**. Reading `detectors.py:109-153`, it uses exactly:

| what it reads | why |
|---|---|
| `len(window)` | `size`, and the `MIN_STEPS_FOR_VERDICT` gate |
| `window.error_count` | retry-storm rate |
| `max(call_counts.values())` | `repeat_count` |
| `len(call_counts)` | `distinct_calls` |
| `sum(call_counts.most_common(2))` | `cycle_share` — `LOOP_MAX_DISTINCT` is **2** |

So the individual signatures never need to cross the wire, and neither does the counter. A naive port would return the whole `Counter` — up to `behavior_window_size` entries (1000 at the documented ceiling) **per step** — turning a published O(1) into O(window) in bytes while every test still passed.

The Redis backend therefore reduces in Lua and returns five scalars. `K` is passed in by the lane rather than hardcoded in the store, so the detector keeps its own thresholds.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/optio/lanes/behavior/store.py` | **Create.** `BehaviorStore` Protocol and the `WindowState` value type. |
| `src/optio/lanes/behavior/store_memory.py` | **Create.** `InMemoryBehaviorStore` wrapping today's `BehaviorWindow` per run. |
| `src/optio/lanes/behavior/store_redis.py` | **Create.** `RedisBehaviorStore` — the Lua that maintains aggregates and reduces them. |
| `src/optio/lanes/behavior/window.py` | **Modify.** Gains `state(k)` returning a `WindowState`; internals untouched. |
| `src/optio/lanes/behavior/detectors.py` | **Modify.** `classify` takes a `WindowState` instead of a `BehaviorWindow`. |
| `src/optio/lanes/behavior/lane.py` | **Modify.** Holds a store instead of a `dict[str, BehaviorWindow]`. |
| `src/optio/lanes/registry.py` | **Modify.** Build and inject the configured behaviour backend. |
| `tests/unit/test_behavior_store_contract.py` | **Create.** One suite, both backends. |
| `tests/property/test_behavior_aggregates_agree.py` | **Create.** Hypothesis: Redis counters equal Python's, step for step. |
| `tests/integration/test_multiprocess_behavior.py` | **Create.** Four processes, one run, one verdict. |
| `tests/bench/test_overhead.py` | **Modify.** Add the flat-in-window assertion for the Redis backend. |

---

## Task 1: `WindowState`, and `classify` reads it

**Files:**
- Create: `src/optio/lanes/behavior/store.py`
- Modify: `src/optio/lanes/behavior/window.py`, `src/optio/lanes/behavior/detectors.py`
- Test: `tests/unit/test_detectors.py` (existing — must pass unchanged where it can)

**Interfaces:**
- Produces: `WindowState(size, errors, distinct_calls, top_counts)`; `BehaviorWindow.state(k) -> WindowState`; `classify(state: WindowState) -> Verdict`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_window_state.py`:

```python
"""``classify`` reads five numbers, and that is what crosses a process boundary.

Pinned as a test because the tempting Redis port returns the whole counter --
up to ``behavior_window_size`` entries per step, 1000 at the documented
ceiling. That turns the O(1)-in-window-size guarantee the README publishes as
measured into O(window) in bytes, and every existing test still passes.
"""

from __future__ import annotations

from optio.lanes.behavior.detectors import LOOP_MAX_DISTINCT, classify
from optio.lanes.behavior.store import WindowState
from optio.lanes.behavior.window import BehaviorWindow, StepSignature


def _sig(call: str, errored: bool = False) -> StepSignature:
    return StepSignature(call=("tool", call), errored=errored)


class TestTheStateCarriesOnlyWhatClassifyUses:
    def test_state_summarises_a_window(self) -> None:
        window = BehaviorWindow(maxlen=50)
        for _ in range(3):
            window.add(_sig("read"))
        window.add(_sig("write", errored=True))

        state = window.state(LOOP_MAX_DISTINCT)

        assert state.size == 4
        assert state.errors == 1
        assert state.distinct_calls == 2
        assert state.top_counts == (3, 1)

    def test_top_counts_is_capped_at_k(self) -> None:
        """The payload must not grow with the window. Ten distinct calls still
        yield two counts, because that is all `cycle_share` sums."""
        window = BehaviorWindow(maxlen=50)
        for n in range(10):
            window.add(_sig(f"call{n}"))

        state = window.state(LOOP_MAX_DISTINCT)

        assert len(state.top_counts) == LOOP_MAX_DISTINCT
        assert state.distinct_calls == 10

    def test_an_empty_window_has_no_top_counts(self) -> None:
        state = BehaviorWindow(maxlen=50).state(LOOP_MAX_DISTINCT)

        assert state.size == 0
        assert state.top_counts == ()


class TestClassifyReadsTheState:
    def test_a_two_call_cycle_is_looping(self) -> None:
        """The textbook stuck agent: read, think, read, think. Neither call
        holds a majority alone, which is why dominance is measured over the
        top two rather than the single most frequent."""
        state = WindowState(size=10, errors=0, distinct_calls=2, top_counts=(5, 5))

        assert classify(state).state == "looping"

    def test_varied_work_is_healthy(self) -> None:
        state = WindowState(size=10, errors=0, distinct_calls=8, top_counts=(2, 2))

        assert classify(state).state == "healthy"

    def test_errors_dominating_is_a_retry_storm(self) -> None:
        state = WindowState(size=10, errors=6, distinct_calls=3, top_counts=(4, 3))

        assert classify(state).state == "retry_storm"

    def test_too_few_steps_is_never_a_pathology(self) -> None:
        state = WindowState(size=3, errors=3, distinct_calls=1, top_counts=(3,))

        assert classify(state).state == "healthy"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/unit/test_window_state.py -v`
Expected: FAIL — `optio.lanes.behavior.store` does not exist.

- [ ] **Step 3: Add `WindowState` and the Protocol**

Create `src/optio/lanes/behavior/store.py`:

```python
"""What a behaviour backend must do, and the summary it returns.

``WindowState`` is deliberately five scalars. ``classify`` reads nothing else
-- not the step signatures, not the counter itself -- so nothing else has any
business crossing a process boundary. Returning the counter instead would put
up to ``behavior_window_size`` entries on the wire per step and quietly convert
a published O(1) guarantee into O(window).

``top_counts`` is capped by the caller's ``k`` rather than by a constant here,
so ``LOOP_MAX_DISTINCT`` stays in the detector that owns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WindowState:
    """Everything ``classify`` needs about one run's recent steps.

    Attributes:
        size: Signatures currently retained -- the window, not the run.
        errors: How many of those ended in error.
        distinct_calls: Distinct call identities retained.
        top_counts: The ``k`` largest per-call counts, descending. Shorter than
            ``k`` when fewer distinct calls exist; empty for an empty window.
    """

    size: int
    errors: int
    distinct_calls: int
    top_counts: tuple[int, ...]


class BehaviorStore(Protocol):
    """Per-run step windows, for many concurrent runs."""

    def record(
        self, run_id: str, call: tuple[str, str], errored: bool, maxlen: int, k: int
    ) -> WindowState:
        """Add one step and return the resulting summary.

        One call rather than write-then-read: a step costs one round trip, and
        the summary is read on every step anyway.

        Args:
            run_id: The run's identifier.
            call: The step's call identity, never its arguments (§10).
            errored: Whether the step failed.
            maxlen: Window bound, from ``Config.behavior_window_size``.
            k: How many top counts to return (``LOOP_MAX_DISTINCT``).

        Returns:
            The window's state after adding the step.
        """
        ...

    def close_run(self, run_id: str) -> None:
        """Release a run's window. Idempotent."""
        ...

    def run_count(self) -> int:
        """How many runs currently hold a window, for leak detection."""
        ...
```

- [ ] **Step 4: Give `BehaviorWindow` a `state(k)`**

Add to `src/optio/lanes/behavior/window.py`:

```python
    def state(self, k: int) -> WindowState:
        """Summarise this window for :func:`~optio.lanes.behavior.detectors.classify`.

        Reads the aggregates ``add`` already maintains, so this stays O(k log n)
        at worst rather than O(window) -- ``Counter.most_common(k)`` uses a heap
        when ``k`` is given.

        Args:
            k: How many top counts to include.

        Returns:
            The state.
        """
        return WindowState(
            size=len(self._steps),
            errors=self._errors,
            distinct_calls=len(self._call_counts),
            top_counts=tuple(count for _, count in self._call_counts.most_common(k)),
        )
```

- [ ] **Step 5: Make `classify` take a `WindowState`**

In `detectors.py`, change the signature to `def classify(state: WindowState) -> Verdict:` and replace the three reads at the top:

```python
    size = state.size
    repeat_count = state.top_counts[0] if state.top_counts else 0
```

then `errors = state.errors`, `distinct_calls = state.distinct_calls`, and

```python
    cycle_share = sum(state.top_counts) / size
```

Nothing else in the function changes — the thresholds and their comments stay exactly as they are.

- [ ] **Step 6: Update the one call site**

In `lane.py`, `classify(window)` becomes `classify(window.state(LOOP_MAX_DISTINCT))`.

- [ ] **Step 7: Run the tests**

Run: `pytest tests/unit/test_window_state.py tests/unit/test_detectors.py tests/unit/test_behavior_lane.py -v`
Expected: PASS. Existing detector tests that construct a `BehaviorWindow` and call `classify(window)` need their call updated to `classify(window.state(LOOP_MAX_DISTINCT))` — that is a call-shape change, not a behaviour change, and no assertion may move.

- [ ] **Step 8: Gate and commit**

```bash
ruff check . && ruff format --check . && mypy && lint-imports && pytest -q
git add src/optio/lanes/behavior tests/unit/test_window_state.py tests/unit/test_detectors.py
git commit -m "classify reads five numbers, so five numbers are what travel"
```

---

## Task 2: The in-memory backend

**Files:**
- Create: `src/optio/lanes/behavior/store_memory.py`
- Create: `tests/unit/test_behavior_store_contract.py`

**Interfaces:**
- Consumes: `BehaviorStore`, `WindowState`.
- Produces: `InMemoryBehaviorStore()`.

- [ ] **Step 1: Write the contract suite**

Create `tests/unit/test_behavior_store_contract.py`, parametrised over `["memory"]` for now (Redis joins in Task 3), asserting:

```python
class TestRecordingBuildsAWindow:
    def test_the_first_step_yields_a_size_of_one(self, store) -> None:
        state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.size == 1
        assert state.distinct_calls == 1
        assert state.top_counts == (1,)

    def test_repeats_accumulate(self, store) -> None:
        for _ in range(4):
            state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.size == 4
        assert state.top_counts[0] == 4

    def test_errors_are_counted(self, store) -> None:
        store.record("run", ("tool", "read"), True, maxlen=50, k=2)
        state = store.record("run", ("tool", "read"), False, maxlen=50, k=2)

        assert state.errors == 1

    def test_runs_do_not_bleed(self, store) -> None:
        store.record("a", ("tool", "read"), False, maxlen=50, k=2)
        state = store.record("b", ("tool", "write"), False, maxlen=50, k=2)

        assert state.size == 1


class TestTheWindowIsBounded:
    def test_size_never_exceeds_maxlen(self, store) -> None:
        for n in range(20):
            state = store.record("run", ("tool", f"c{n}"), False, maxlen=5, k=2)

        assert state.size == 5

    def test_evicted_calls_leave_the_counts(self, store) -> None:
        """The subtle half: a count that hits zero must drop its key, because
        `distinct_calls` is the number of keys and a zeroed one inflates it
        forever."""
        for _ in range(5):
            store.record("run", ("tool", "old"), False, maxlen=5, k=2)
        for _ in range(5):
            state = store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert state.distinct_calls == 1
        assert state.top_counts == (5,)

    def test_evicted_errors_leave_the_tally(self, store) -> None:
        for _ in range(5):
            store.record("run", ("tool", "old"), True, maxlen=5, k=2)
        for _ in range(5):
            state = store.record("run", ("tool", "new"), False, maxlen=5, k=2)

        assert state.errors == 0


class TestLifecycle:
    def test_closing_releases_the_window(self, store) -> None:
        store.record("run", ("tool", "read"), False, maxlen=50, k=2)
        store.close_run("run")

        assert store.run_count() == 0

    def test_closing_an_unknown_run_is_not_an_error(self, store) -> None:
        store.close_run("never")

        assert store.run_count() == 0
```

- [ ] **Step 2: Run and watch it fail**

Run: `pytest tests/unit/test_behavior_store_contract.py -v`
Expected: FAIL — `store_memory` does not exist.

- [ ] **Step 3: Implement**

Create `src/optio/lanes/behavior/store_memory.py` holding `dict[str, BehaviorWindow]` behind a lock, delegating to `BehaviorWindow.add` and `BehaviorWindow.state`. Move the lane's existing `self._windows` handling here **verbatim** — same lock, same `setdefault`, same eviction on close.

- [ ] **Step 4: Run, gate, commit**

Run: `pytest -q && ruff check . && ruff format --check . && mypy && lint-imports`
Expected: clean; every existing behaviour-lane test passes unchanged.

```bash
git commit -m "The behaviour window grows a seam, and keeps its counters"
```

---

## Task 3: The Redis backend

**Files:**
- Create: `src/optio/lanes/behavior/store_redis.py`
- Modify: `tests/unit/test_behavior_store_contract.py` (add `"redis"`)
- Test: `tests/integration/test_redis_behavior.py`

**Keys per run:**

| key | type | holds |
|---|---|---|
| `optio:b:{run}:steps` | list | recent signatures, `LTRIM`-bounded |
| `optio:b:{run}:counts` | hash | call identity → count in window |
| `optio:b:{run}:meta` | hash | `errors` |

- [ ] **Step 1: Write the Lua**

One script does append, evict, and reduce:

```lua
-- KEYS: steps, counts, meta   ARGV: call, errored, maxlen, k, ttl_ms
local call, errored = ARGV[1], ARGV[2] == '1'
local maxlen, k, ttl = tonumber(ARGV[3]), tonumber(ARGV[4]), ARGV[5]

redis.call('RPUSH', KEYS[1], call .. '\0' .. ARGV[2])
redis.call('HINCRBY', KEYS[2], call, 1)
if errored then redis.call('HINCRBY', KEYS[3], 'errors', 1) end

-- Evict from the head while over the bound, decrementing what leaves. A count
-- reaching zero has its field deleted: `distinct_calls` is the number of
-- fields, and a zeroed field would inflate it for the rest of the run.
while redis.call('LLEN', KEYS[1]) > maxlen do
  local gone = redis.call('LPOP', KEYS[1])
  local sep = string.find(gone, '\0')
  local gone_call = string.sub(gone, 1, sep - 1)
  if redis.call('HINCRBY', KEYS[2], gone_call, -1) <= 0 then
    redis.call('HDEL', KEYS[2], gone_call)
  end
  if string.sub(gone, sep + 1) == '1' then
    redis.call('HINCRBY', KEYS[3], 'errors', -1)
  end
end

-- Reduce here rather than shipping the counter: the payload must not grow
-- with maxlen (the README publishes classification as flat in window size).
local counts = redis.call('HVALS', KEYS[2])
table.sort(counts, function(a, b) return tonumber(a) > tonumber(b) end)
local top = {}
for i = 1, math.min(k, #counts) do top[i] = counts[i] end

redis.call('PEXPIRE', KEYS[1], ttl)
redis.call('PEXPIRE', KEYS[2], ttl)
redis.call('PEXPIRE', KEYS[3], ttl)

return {
  tostring(redis.call('LLEN', KEYS[1])),
  redis.call('HGET', KEYS[3], 'errors') or '0',
  tostring(#counts),
  table.concat(top, ',')
}
```

- [ ] **Step 2: Write the failing integration test**

Create `tests/integration/test_redis_behavior.py`, reusing `connect_or_skip` from `tests/integration/test_redis_ledger.py`, asserting: eviction decrements counts; a count reaching zero drops its field; the TTL refreshes on each `record`; and the payload is `k` counts regardless of `maxlen`:

```python
def test_the_payload_does_not_grow_with_the_window(store) -> None:
    """The guarantee a naive port breaks silently."""
    for n in range(200):
        state = store.record("run", ("tool", f"c{n}"), False, maxlen=1000, k=2)

    assert len(state.top_counts) <= 2
    assert state.distinct_calls == 200
```

- [ ] **Step 3: Implement, then add `"redis"` to the contract fixture**

Run: `pytest tests/unit/test_behavior_store_contract.py -v`
Expected: every test passes against **both** backends.

- [ ] **Step 4: Gate and commit**

```bash
git commit -m "The window's counters move to Lua, and the payload stays flat"
```

---

## Task 4: The property test that keeps the two honest

**Files:**
- Create: `tests/property/test_behavior_aggregates_agree.py`

The eviction arithmetic now exists twice — once in Python, once in Lua. This is the highest-risk surface in the milestone, and the guard for it.

- [ ] **Step 1: Write it**

```python
"""The eviction arithmetic exists twice; this is what keeps it identical.

`BehaviorWindow.add` decrements a call's count on eviction and deletes the key
at zero, and the Lua script does the same thing in another language. A
divergence would not raise -- it would produce a slightly different verdict on
one backend, which is the silent-wrongness class this project treats as its
worst failure.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.property, pytest.mark.redis]

_CALLS = st.sampled_from(["read", "write", "search", "plan"])
_STEPS = st.lists(st.tuples(_CALLS, st.booleans()), min_size=1, max_size=80)


@given(steps=_STEPS, maxlen=st.integers(min_value=2, max_value=12))
@settings(max_examples=40, deadline=None)
def test_both_backends_agree_step_for_step(redis_store, steps, maxlen) -> None:
    from optio.lanes.behavior.store_memory import InMemoryBehaviorStore

    memory = InMemoryBehaviorStore()
    redis_store.close_run("prop")

    for call, errored in steps:
        want = memory.record("prop", ("tool", call), errored, maxlen=maxlen, k=2)
        got = redis_store.record("prop", ("tool", call), errored, maxlen=maxlen, k=2)

        assert got == want, f"backends diverged after {call!r}/{errored}"
```

- [ ] **Step 2: Run it**

Run: `OPTIO_TEST_REDIS_URL=... pytest tests/property/test_behavior_aggregates_agree.py -v`
Expected: PASS.

- [ ] **Step 3: Mutation-test the guard**

Change the Lua's `<= 0` to `< 0` so a zeroed field survives, and confirm this test fails. Revert.

- [ ] **Step 4: Commit**

```bash
git commit -m "Two implementations of one eviction rule, held to each other"
```

---

## Task 5: Wire the lane and the registry

**Files:**
- Modify: `src/optio/lanes/behavior/lane.py`, `src/optio/lanes/registry.py`

- [ ] **Step 1** `BehaviorLane.__init__` takes `store: BehaviorStore | None = None`, defaulting to `InMemoryBehaviorStore()`; `self._windows` and its lock are deleted, since the store owns both.
- [ ] **Step 2** `registry._behavior_store(config)` mirrors `_ledger(config)`: in-memory unless `store_backend == "redis"`, in which case it builds a `RedisClient` (pinging at setup) and a `RedisBehaviorStore`.
- [ ] **Step 3** Add an integration test asserting `enabled_lanes(Config(store_backend="redis", ...))` yields a `BehaviorLane` whose store is the Redis one — the object-graph assertion, because a config that parses proved nothing last time.
- [ ] **Step 4** Gate and commit.

---

## Task 6: Four processes, one verdict

**Files:**
- Create: `tests/integration/test_multiprocess_behavior.py`

- [ ] **Step 1: Write it**

Four spawned workers each add the same two-call cycle against one `run_id`. With shared state the run classifies as `looping`; with per-process windows each worker sees too few steps to pass `MIN_STEPS_FOR_VERDICT` and reports `healthy`. Pass the URL as an **argument**, not via the environment — `tests/conftest.py` strips `OPTIO_*` and the child re-imports after that.

- [ ] **Step 2: Add the in-memory counterpart** asserting four separate windows each stay below the verdict threshold, so the limitation is measured rather than described.

- [ ] **Step 3: Gate and commit.**

---

## Task 7: Prove the guarantee, then write it down

**Files:**
- Modify: `tests/bench/test_overhead.py`, `README.md`, `CHANGELOG.md`
- Modify: `docs/design/adr/adr-050-the-store-speaks-the-domain.md` (addendum)

- [ ] **Step 1: Extend the overhead bench** to record `record()` cost at `maxlen` 50 and 1000 on both backends, asserting the ratio stays flat — the same shape as the existing window-size row, which is a *structural* guarantee and must remain one.
- [ ] **Step 2: Update the README's overhead table** with the Redis row, measured rather than inherited.
- [ ] **Step 3: Add an ADR-050 addendum** recording that the behaviour lane landed, that `classify` needed only five scalars, and that returning the counter would have broken a published guarantee without failing a test.
- [ ] **Step 4: CHANGELOG**, naming the multi-process verdict bug in the same shape as the budget one.
- [ ] **Step 5: Full gate, commit, push.**

---

## Self-Review

**Spec coverage.** The spec's `BehaviorStore` section maps to Tasks 1–3; its "highest-risk correctness surface" (duplicated eviction arithmetic) to Task 4; its O(1) requirement to Tasks 1, 3 and 7. **Deferred to the quality-lane plan:** `QualityStep`, the no-spans-in-the-store test, and the final overhead re-measurement across all three lanes.

**Placeholder scan.** No TBD/TODO. Tasks 5–7 give steps rather than full listings because each is a mechanical mirror of a cost-lane change already merged and reviewable in `git log` — `_ledger(config)` for Task 5, `test_multiprocess_budget.py` for Task 6.

**Type consistency.** `WindowState(size, errors, distinct_calls, top_counts)` is constructed identically in Tasks 1, 2 and 3 and consumed by `classify` in Task 1. `record(run_id, call, errored, maxlen, k)` has the same signature in the Protocol, both backends, and every test. `close_run`/`run_count` match the lifecycle names the ledger already uses.

**Known risk carried deliberately.** Task 1 changes `classify`'s signature, so existing detector tests change shape at their call site. No assertion may move; if one needs to, the reduction to five scalars lost information and the design is wrong.
