# Shared State Store — Cost Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the cost ledger work across processes, so four workers sharing one `run_id` produce one correct total instead of each admitting the full budget.

**Architecture:** `CostLedger` becomes a thin facade over a `LedgerStore` Protocol. The current implementation moves behind an in-memory backend unchanged (so every existing test is the regression net), and a Redis backend implements the same Protocol with Lua scripts carrying the atomicity. A domain-free Redis client lives in `optio.store`; the ledger backends live in `optio.lanes.cost`, because `lanes` may import `store` and never the reverse.

**Tech Stack:** Python 3.10–3.14, `redis-py` (optional extra), Lua (server-side scripts), pytest, hypothesis, `fakeredis` for unit speed plus a real Redis service container for the gate.

## Global Constraints

- **No behaviour change for single-process users.** The in-memory backend must stay byte-identical in output. Every existing test passes unchanged.
- **Layering (import-linter):** `optio.lanes` may import `optio.store`; `optio.store` must never import `optio.lanes`. Lanes are mutually independent — `optio.lanes.cost` must not import `optio.lanes.behavior` or `optio.lanes.quality`.
- **`optio` must never import `optio_optimize`** (ADR-013).
- **Fail-open governs the runtime path** (ADR-004): an unreachable store drops the signal, never blocks the agent. **Setup** failure is loud (§4.2).
- **When the store is unreachable, emit no cost signal at all** — never a partial number computed from one process's view.
- Gate for every commit: `ruff check .` && `ruff format --check .` && `mypy` && `lint-imports` && `pytest -q`.
- Docstrings: Google style, Args/Returns/Raises. Comments explain *why*, never *what*.
- **Never log or store prompt content** (§10).
- Redis connect and read timeouts default to **50 ms**. No retries on the hot path.
- Python floor is **3.10** — no `datetime.UTC`, no `match` on 3.9-incompatible syntax, no PEP 604 in runtime `isinstance`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/optio/store/redis_client.py` | **Create.** Domain-free Redis access: connection with timeouts, script load/EVALSHA with NOSCRIPT fallback, `StoreUnavailable`. Knows nothing about ledgers. |
| `src/optio/lanes/cost/ledger_store.py` | **Create.** The `LedgerStore` Protocol — exactly today's `CostLedger` method set. |
| `src/optio/lanes/cost/ledger_memory.py` | **Create.** `InMemoryLedgerStore` — today's `CostLedger` body, moved verbatim. |
| `src/optio/lanes/cost/ledger_redis.py` | **Create.** `RedisLedgerStore` — the Lua scripts and their argument marshalling. |
| `src/optio/lanes/cost/ledger.py` | **Modify.** `CostLedger` becomes a facade delegating to a `LedgerStore`. |
| `src/optio/lanes/cost/lane.py` | **Modify.** Accept a store; emit nothing when it is unavailable. |
| `src/optio/lanes/registry.py` | **Modify.** Build the configured backend and inject it. |
| `src/optio/config.py` | **Modify.** Stop rejecting `redis`; add `store_timeout_ms`; fix the stale `Agent-Meter` URL. |
| `src/optio/store/base.py` | **Modify.** Retire the superseded `StateStore` ABC. |
| `tests/unit/test_ledger_store_contract.py` | **Create.** One suite, parametrised over backends. |
| `tests/unit/test_redis_client.py` | **Create.** Timeouts, NOSCRIPT reload, unavailability. |
| `tests/integration/test_redis_ledger.py` | **Create.** Against a real Redis: TTL refresh, tombstone. |
| `tests/integration/test_multiprocess_budget.py` | **Create.** The proof: N processes, one run, one correct total. |
| `tests/unit/test_readme_renders_on_pypi.py` | **Modify.** Extend the repo-link guard to `src/`. |

---

## Task 1: The repo-link guard misses `src/`

**Files:**
- Modify: `tests/unit/test_readme_renders_on_pypi.py`
- Modify: `src/optio/config.py` (the stale URL)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing later tasks depend on. Done first because Task 7 edits the same error message and would otherwise carry the stale link forward.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_readme_renders_on_pypi.py`:

```python
def test_no_source_file_points_at_the_old_repository_name() -> None:
    """The rename to Optio missed `src/`, and only five root files were checked.

    `config.py`'s `store_backend='redis'` error tells a user to track the work
    at `github.com/Aniketh-74/Agent-Meter/issues` — a name this project no
    longer owns. GitHub 301s a rename, so it works until someone claims the old
    name, at which point an error message in a shipped library points users at a
    stranger's issue tracker.
    """
    offenders: dict[str, str] = {}
    for path in (REPO / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for owner_repo in re.findall(r"github\.com/([\w.-]+/[\w.-]+)", text):
            if owner_repo.removesuffix(".git") != "Aniketh-74/Optio":
                offenders[str(path.relative_to(REPO))] = owner_repo

    assert not offenders, f"source files naming another repository: {offenders}"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/unit/test_readme_renders_on_pypi.py::test_no_source_file_points_at_the_old_repository_name -v`
Expected: FAIL, naming `src/optio/config.py` and `Aniketh-74/Agent-Meter`.

- [ ] **Step 3: Fix the URL**

In `src/optio/config.py`, in the `store_backend == "redis"` error message, replace:

```python
                "https://github.com/Aniketh-74/Agent-Meter/issues"
```

with:

```python
                "https://github.com/Aniketh-74/Optio/issues"
```

- [ ] **Step 4: Run the test and the suite**

Run: `pytest tests/unit/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_readme_renders_on_pypi.py src/optio/config.py
git commit -m "The rename reached five root files and stopped at src/"
```

---

## Task 2: A domain-free Redis client

**Files:**
- Create: `src/optio/store/redis_client.py`
- Test: `tests/unit/test_redis_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class StoreUnavailable(StateStoreError)` — raised when Redis cannot answer.
  - `class RedisClient` with:
    - `__init__(self, url: str, *, timeout_ms: int = 50) -> None`
    - `register_script(self, name: str, source: str) -> None`
    - `run_script(self, name: str, keys: list[str], args: list[str]) -> Any`
    - `ping(self) -> None` — raises `StoreUnavailable`; used at setup.
    - `close(self) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_redis_client.py`:

```python
"""Redis access with the two properties the ledger depends on.

The client is deliberately domain-free: it knows about connections, scripts and
timeouts, and nothing about reservations. That is what lets it sit in
`optio.store`, which the import-linter forbids from importing any lane.
"""

from __future__ import annotations

import pytest

from optio.errors import StateStoreError
from optio.store.redis_client import RedisClient, StoreUnavailable


class _FakeRedis:
    """Records calls and can be told to fail in specific ways."""

    def __init__(self) -> None:
        self.loaded: dict[str, str] = {}
        self.evalsha_calls: list[tuple[str, list[str], list[str]]] = []
        self.raise_noscript_once = False
        self.unreachable = False

    def script_load(self, source: str) -> str:
        sha = f"sha-{len(self.loaded)}"
        self.loaded[sha] = source
        return sha

    def evalsha(self, sha: str, numkeys: int, *rest: str) -> str:
        if self.unreachable:
            raise ConnectionError("no route to host")
        keys = list(rest[:numkeys])
        args = list(rest[numkeys:])
        self.evalsha_calls.append((sha, keys, args))
        if self.raise_noscript_once:
            self.raise_noscript_once = False
            raise RuntimeError("NOSCRIPT No matching script")
        return "ok"

    def ping(self) -> bool:
        if self.unreachable:
            raise ConnectionError("no route to host")
        return True

    def close(self) -> None:
        return None


def _client(fake: _FakeRedis) -> RedisClient:
    client = RedisClient.__new__(RedisClient)
    client._redis = fake  # type: ignore[attr-defined]
    client._shas = {}  # type: ignore[attr-defined]
    client._sources = {}  # type: ignore[attr-defined]
    return client


class TestScriptCaching:
    def test_a_script_is_loaded_once_and_reused(self) -> None:
        fake = _FakeRedis()
        client = _client(fake)
        client.register_script("bump", "return 1")

        client.run_script("bump", ["k"], ["1"])
        client.run_script("bump", ["k"], ["2"])

        assert len(fake.loaded) == 1, "the script was re-loaded instead of cached"
        assert len(fake.evalsha_calls) == 2

    def test_a_flushed_script_cache_is_recovered_not_fatal(self) -> None:
        """Redis drops its script cache on restart or failover.

        Treating NOSCRIPT as an error would turn an ordinary Redis restart into
        a run with no cost signals, so the client reloads and retries once.
        """
        fake = _FakeRedis()
        client = _client(fake)
        client.register_script("bump", "return 1")
        fake.raise_noscript_once = True

        result = client.run_script("bump", ["k"], ["1"])

        assert result == "ok"
        assert len(fake.loaded) == 2, "the script was not reloaded after NOSCRIPT"


class TestUnavailability:
    def test_an_unreachable_redis_raises_store_unavailable(self) -> None:
        """One exception type, so callers do not have to know redis-py's tree."""
        fake = _FakeRedis()
        fake.unreachable = True
        client = _client(fake)
        client.register_script("bump", "return 1")

        with pytest.raises(StoreUnavailable):
            client.run_script("bump", ["k"], ["1"])

    def test_store_unavailable_is_a_state_store_error(self) -> None:
        """The fail-open guard already absorbs StateStoreError; this must not
        slip past it as a new unrelated type."""
        assert issubclass(StoreUnavailable, StateStoreError)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/unit/test_redis_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'optio.store.redis_client'`.

- [ ] **Step 3: Implement the client**

Create `src/optio/store/redis_client.py`:

```python
"""Redis access, with no idea what it is storing.

Lives in ``optio.store`` because the import-linter's layering forbids this
package from importing any lane -- so it must stay domain-free. The ledger's
Lua lives with the ledger; only the mechanics of loading and invoking it live
here.

**Timeouts are the point.** A hung Redis must never add latency to an agent,
so connect and read timeouts are short by default and there are no retries on
the hot path: a retry is latency paid for a signal the caller can live without
(ADR-004).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from optio.errors import StateStoreError

if TYPE_CHECKING:
    pass

#: Connect and read timeout. Short because this sits on the agent's critical
#: path and a dropped signal is cheaper than a stalled step.
DEFAULT_TIMEOUT_MS: Final = 50


class StoreUnavailable(StateStoreError):
    """The store could not answer.

    A subclass of :class:`~optio.errors.StateStoreError` so the existing
    fail-open guard absorbs it without learning a new type.
    """


class RedisClient:
    """A Redis connection plus a cache of server-side scripts.

    Args:
        url: Redis connection string.
        timeout_ms: Connect and read timeout in milliseconds.

    Raises:
        ImportError: If ``redis`` is not installed. Loud, at construction:
            configuration that cannot do what it claims is a setup error.
    """

    def __init__(self, url: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ImportError(
                "store_backend='redis' needs the redis driver: pip install 'optio[redis]'"
            ) from exc

        seconds = timeout_ms / 1000.0
        self._redis = redis.Redis.from_url(
            url,
            socket_connect_timeout=seconds,
            socket_timeout=seconds,
            decode_responses=True,
        )
        self._shas: dict[str, str] = {}
        self._sources: dict[str, str] = {}

    def register_script(self, name: str, source: str) -> None:
        """Load a Lua script and remember its SHA under ``name``."""
        self._sources[name] = source
        self._shas[name] = self._redis.script_load(source)

    def run_script(self, name: str, keys: list[str], args: list[str]) -> Any:
        """Invoke a registered script by SHA.

        Reloads and retries **once** on ``NOSCRIPT``: Redis drops its script
        cache on restart and failover, and treating that as fatal would turn an
        ordinary restart into a run with no cost signals.

        Raises:
            StoreUnavailable: If Redis cannot be reached or errors.
        """
        try:
            return self._redis.evalsha(self._shas[name], len(keys), *keys, *args)
        except Exception as exc:
            if "NOSCRIPT" in str(exc):
                self._shas[name] = self._redis.script_load(self._sources[name])
                try:
                    return self._redis.evalsha(
                        self._shas[name], len(keys), *keys, *args
                    )
                except Exception as retry_exc:  # noqa: BLE001 - normalised below
                    raise StoreUnavailable(f"redis script failed: {name}") from retry_exc
            raise StoreUnavailable(f"redis unavailable running {name}") from exc

    def ping(self) -> None:
        """Verify the server answers.

        Called at setup so an unreachable Redis fails loudly there (§4.2)
        rather than silently on the runtime path.

        Raises:
            StoreUnavailable: If the server does not answer.
        """
        try:
            self._redis.ping()
        except Exception as exc:  # noqa: BLE001 - normalised to one type
            raise StoreUnavailable("redis did not answer ping") from exc

    def close(self) -> None:
        """Release the connection. Idempotent."""
        self._redis.close()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_redis_client.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the gate**

Run: `ruff check . && ruff format --check . && mypy && lint-imports`
Expected: all clean. `lint-imports` must still report `Contracts: 4 kept, 0 broken` — `optio.store` importing a lane would break the layering contract here.

- [ ] **Step 6: Commit**

```bash
git add src/optio/store/redis_client.py tests/unit/test_redis_client.py
git commit -m "A Redis client that knows about timeouts and nothing about ledgers"
```

---

## Task 3: Extract `LedgerStore` and the in-memory backend

**Files:**
- Create: `src/optio/lanes/cost/ledger_store.py`
- Create: `src/optio/lanes/cost/ledger_memory.py`
- Modify: `src/optio/lanes/cost/ledger.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LedgerStore` Protocol with `reserve(run_id, step_id, projected) -> None`, `reconcile(run_id, step_id, actual) -> None`, `snapshot(run_id) -> LedgerSnapshot`, `close_run(run_id) -> LedgerSnapshot`, `is_finalised(run_id) -> bool`, `knows(run_id) -> bool`, `evict(run_id) -> None`, `run_count() -> int`.
  - `InMemoryLedgerStore` implementing it.
  - `CostLedger(store: LedgerStore | None = None)` — a facade; the default is `InMemoryLedgerStore()`.

**This is the riskiest task in the plan.** The regression net is that **every existing cost-lane test must pass unchanged**. Do not edit any existing test in this task. If one fails, the extraction is wrong.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ledger_store_contract.py`:

```python
"""One suite, every backend. This is what makes "interchangeable" a checked claim.

Parametrised over the backends so a behaviour that holds in memory and not in
Redis fails here rather than in production. Redis joins the parameter list in a
later task; the shape is fixed now so adding it is one line.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from optio.errors import LedgerInvariantError
from optio.lanes.cost.ledger_memory import InMemoryLedgerStore
from optio.lanes.cost.ledger_store import LedgerStore


@pytest.fixture(params=["memory"])
def store(request: pytest.FixtureRequest) -> Iterator[LedgerStore]:
    if request.param == "memory":
        yield InMemoryLedgerStore()
        return
    raise AssertionError(f"unknown backend {request.param!r}")


class TestReserveAndReconcile:
    def test_a_reconciled_step_reports_its_actual_cost(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.reconcile("run", "step", 0.25)

        snap = store.snapshot("run")

        assert snap.actual == pytest.approx(0.25)
        assert snap.reserved == pytest.approx(0.0)
        assert snap.reconciled_steps == 1

    def test_re_reserving_replaces_rather_than_stacks(self, store: LedgerStore) -> None:
        """Frameworks retry steps and reuse ids; stacking would inflate reserved."""
        store.reserve("run", "step", 1.0)
        store.reserve("run", "step", 2.0)

        assert store.snapshot("run").reserved == pytest.approx(2.0)

    def test_a_double_reconcile_raises(self, store: LedgerStore) -> None:
        """Exactly-once, or the total double-counts money a policy gates on."""
        store.reserve("run", "step", 1.0)
        store.reconcile("run", "step", 0.25)

        with pytest.raises(LedgerInvariantError):
            store.reconcile("run", "step", 0.25)

    def test_reconciling_without_reserving_raises(self, store: LedgerStore) -> None:
        with pytest.raises(LedgerInvariantError):
            store.reconcile("run", "never-reserved", 0.25)


class TestTheThreeStates:
    def test_an_unseen_run_is_not_known(self, store: LedgerStore) -> None:
        """ADR-044: an all-zero snapshot for a run nobody metered is a lie."""
        assert store.knows("nobody") is False

    def test_a_metered_run_is_known(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)

        assert store.knows("run") is True

    def test_a_closed_run_is_finalised(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.close_run("run")

        assert store.is_finalised("run") is True

    def test_closing_is_idempotent(self, store: LedgerStore) -> None:
        """Run end can fire more than once (M1-2)."""
        store.reserve("run", "step", 1.0)
        first = store.close_run("run")
        second = store.close_run("run")

        assert first.actual == second.actual
        assert first.leaked_steps == second.leaked_steps

    def test_a_leak_is_kept_as_the_reserved_worst_case(self, store: LedgerStore) -> None:
        """Dropping it would make the run look cheaper than the evidence supports."""
        store.reserve("run", "step", 1.5)

        snap = store.close_run("run")

        assert snap.leaked_steps == 1
        assert snap.reserved == pytest.approx(1.5)

    def test_reserving_on_a_closed_run_raises(self, store: LedgerStore) -> None:
        store.reserve("run", "step", 1.0)
        store.close_run("run")

        with pytest.raises(LedgerInvariantError):
            store.reserve("run", "another", 1.0)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/unit/test_ledger_store_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'optio.lanes.cost.ledger_memory'`.

- [ ] **Step 3: Create the Protocol**

Create `src/optio/lanes/cost/ledger_store.py`:

```python
"""What a ledger backend must do (supersedes ADR-005's generic ABC).

The generic ``StateStore`` offered ``get``/``set``/``incr``/``delete`` and could
not express this: ``reconcile`` has to check a reservation exists, remove it,
and fold the cost into the total **together**, and a caller holding primitives
cannot be atomic across processes.

So the store speaks the domain. Each backend keeps the exactly-once promise its
own way -- a lock in memory, a Lua script in Redis -- and the promise is
enforced where the check and the mutation happen together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optio.lanes.cost.ledger import LedgerSnapshot


@runtime_checkable
class LedgerStore(Protocol):
    """Reserve/reconcile accounting for many concurrent runs."""

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record a step's worst-case cost before it runs.

        Re-reserving the same ``step_id`` replaces the previous reservation.

        Raises:
            LedgerInvariantError: If ``projected`` is negative or the run is
                closed.
        """
        ...

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost.

        Raises:
            LedgerInvariantError: On a double reconcile, a reconcile with no
                matching reservation, a negative cost, or a closed run.
        """
        ...

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        """Return a consistent view of a run's cost state."""
        ...

    def close_run(self, run_id: str) -> LedgerSnapshot:
        """Finalise a run, recording any leaked reservations. Idempotent."""
        ...

    def is_finalised(self, run_id: str) -> bool:
        """Whether this run has been closed, whether or not its state survives."""
        ...

    def knows(self, run_id: str) -> bool:
        """Whether this store has ever recorded anything for a run."""
        ...

    def evict(self, run_id: str) -> None:
        """Release a run's state. Finality outlives eviction."""
        ...

    def run_count(self) -> int:
        """How many runs currently hold state, for leak detection."""
        ...
```

- [ ] **Step 4: Move the implementation**

Create `src/optio/lanes/cost/ledger_memory.py` containing the **current body of `CostLedger`** from `src/optio/lanes/cost/ledger.py`, renamed to `InMemoryLedgerStore`. Move `_RunLedger`, `_CLOSED_MEMORY`, `_remember_closed`, `_is_closed`, `reserve`, `reconcile`, `snapshot`, `close_run`, `is_finalised`, `knows`, `evict`, `run_count` **verbatim** — same locking, same messages, same comments. Keep `LedgerSnapshot` and the exceptions where they are, in `ledger.py`.

The module docstring:

```python
"""The in-process ledger backend -- the default (ADR-005).

This is the implementation that shipped as ``CostLedger`` through 0.3.0, moved
behind :class:`~optio.lanes.cost.ledger_store.LedgerStore` unchanged. Moving it
verbatim is deliberate: the existing cost-lane tests then act as the regression
net for the extraction, and any behavioural difference is a bug in the move
rather than a design choice nobody wrote down.

Single-process by construction. Runs sharded across processes need the Redis
backend; each process here holds its own dictionary and would meter a fraction
of the truth while believing it had the whole.
"""
```

- [ ] **Step 5: Make `CostLedger` a facade**

In `src/optio/lanes/cost/ledger.py`, replace the class body with delegation, keeping `LedgerSnapshot` and the exception types in place:

```python
class CostLedger:
    """Per-run reserve/reconcile accounting, delegated to a backend.

    Kept as the lane's entry point so the ledger's callers are unaffected by
    where state actually lives. The backend decides that: in-process by default,
    Redis when a run spans processes.

    Args:
        store: Backend to delegate to. An in-memory store when omitted, which
            preserves the pre-0.4 constructor exactly.
        closed_memory: How many recently-closed run ids to remember. Applies to
            the default in-memory backend only; ignored when ``store`` is given,
            because a supplied backend has already chosen its own retention.
    """

    def __init__(
        self,
        store: LedgerStore | None = None,
        closed_memory: int = _CLOSED_MEMORY,
    ) -> None:
        self._store: Final[LedgerStore] = (
            store if store is not None else InMemoryLedgerStore(closed_memory=closed_memory)
        )

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record the worst-case cost of a step before it runs."""
        self._store.reserve(run_id, step_id, projected)

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost."""
        self._store.reconcile(run_id, step_id, actual)

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        """Return a consistent view of a run's cost state."""
        return self._store.snapshot(run_id)

    def close_run(self, run_id: str) -> LedgerSnapshot:
        """Finalise a run, recording any leaked reservations."""
        return self._store.close_run(run_id)

    def is_finalised(self, run_id: str) -> bool:
        """Whether this run has already been closed."""
        return self._store.is_finalised(run_id)

    def knows(self, run_id: str) -> bool:
        """Whether this ledger has ever recorded anything for a run."""
        return self._store.knows(run_id)

    def evict(self, run_id: str) -> None:
        """Release a run's state."""
        self._store.evict(run_id)

    def run_count(self) -> int:
        """How many runs currently hold state."""
        return self._store.run_count()
```

- [ ] **Step 6: Run the contract suite and the whole existing suite**

Run: `pytest tests/unit/test_ledger_store_contract.py -v`
Expected: PASS (10 tests).

Run: `pytest -q`
Expected: **2,253 passed** plus the new tests, 6 skipped. **No existing test may be edited to make this pass.** A failure here means the extraction changed behaviour.

- [ ] **Step 7: Run the gate**

Run: `ruff check . && ruff format --check . && mypy && lint-imports`
Expected: clean, `Contracts: 4 kept, 0 broken`.

- [ ] **Step 8: Commit**

```bash
git add src/optio/lanes/cost/ tests/unit/test_ledger_store_contract.py
git commit -m "The ledger grows a seam, and its own tests are the net"
```

---

## Task 4: The Redis backend

**Files:**
- Create: `src/optio/lanes/cost/ledger_redis.py`
- Modify: `tests/unit/test_ledger_store_contract.py` (add the `redis` parameter)
- Test: `tests/integration/test_redis_ledger.py`

**Interfaces:**
- Consumes: `RedisClient`, `StoreUnavailable` from Task 2; `LedgerStore` from Task 3.
- Produces: `RedisLedgerStore(client: RedisClient, *, ttl_seconds: float, tombstone_ttl_seconds: float)`.

**Key layout in Redis**, per run:

| Key | Type | Holds |
|---|---|---|
| `optio:{run}:open` | hash | `step_id -> projected` |
| `optio:{run}:totals` | hash | `actual`, `reconciled`, `leaked`, `closed` |
| `optio:{run}:done` | string | tombstone; longer TTL than the two above |

- [ ] **Step 1: Add a `redis` marker**

In `pyproject.toml`, add to `markers`:

```toml
    "redis: needs a real Redis server (the gate runs one as a service container)",
```

- [ ] **Step 2: Write the failing integration tests**

Create `tests/integration/test_redis_ledger.py`:

```python
"""The Redis backend against a real server.

`fakeredis` is fine for speed elsewhere, but not here: Lua is what carries the
atomicity, and a fake's Lua is exactly where it diverges from the real thing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from optio.lanes.cost.ledger_redis import RedisLedgerStore
from optio.store.redis_client import RedisClient

pytestmark = pytest.mark.redis

REDIS_URL = os.environ.get("OPTIO_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def store() -> Iterator[RedisLedgerStore]:
    client = RedisClient(REDIS_URL)
    try:
        client.ping()
    except Exception:  # noqa: BLE001 - the skip message is the point
        pytest.skip(f"no Redis at {REDIS_URL}")
    client._redis.flushdb()  # type: ignore[attr-defined]
    yield RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    client._redis.flushdb()  # type: ignore[attr-defined]
    client.close()


class TestTtlIsAnIdleTimeoutNotADeadline:
    def test_writing_refreshes_the_expiry(self, store: RedisLedgerStore) -> None:
        """An absolute expiry is a time bomb on a long run.

        State would vanish mid-flight, open reservations with it, and
        `budget_remaining` would jump back to full -- ADR-044's failure on a
        timer. So every write pushes the expiry out.
        """
        store.reserve("run", "a", 1.0)
        first = store.ttl_seconds_remaining("run")

        store.reserve("run", "b", 1.0)
        second = store.ttl_seconds_remaining("run")

        assert second >= first, "a second write did not refresh the TTL"


class TestTheTombstoneOutlivesThePayload:
    def test_a_closed_run_stays_finalised_after_its_payload_is_gone(
        self, store: RedisLedgerStore
    ) -> None:
        """Without this, a late span after expiry starts a *fresh* run record --
        resurrecting a run ADR-010 declares final."""
        store.reserve("run", "a", 1.0)
        store.close_run("run")
        store.drop_payload_for_test("run")

        assert store.is_finalised("run") is True

    def test_a_run_that_never_existed_is_not_finalised(self, store: RedisLedgerStore) -> None:
        assert store.is_finalised("never") is False
```

- [ ] **Step 3: Run them and watch them fail**

Run: `pytest tests/integration/test_redis_ledger.py -v -m redis`
Expected: FAIL with `ModuleNotFoundError: No module named 'optio.lanes.cost.ledger_redis'` (or SKIP if no Redis is running — start one first: `docker run -p 6379:6379 --rm redis:7`).

- [ ] **Step 4: Implement the backend**

Create `src/optio/lanes/cost/ledger_redis.py`. The two scripts that carry the correctness:

```python
#: Reserve. Rejects a closed run, then sets the field -- replacing any previous
#: value for this step, because frameworks retry steps and reuse their ids.
#: One script rather than EXISTS-then-HSET: between two commands another worker
#: can close the run, and the reservation would land on a finalised total.
_RESERVE = """
if redis.call('EXISTS', KEYS[3]) == 1 then return 'CLOSED' end
if redis.call('HGET', KEYS[2], 'closed') == '1' then return 'CLOSED' end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HSETNX', KEYS[2], 'actual', '0')
redis.call('PEXPIRE', KEYS[1], ARGV[3])
redis.call('PEXPIRE', KEYS[2], ARGV[3])
return 'OK'
"""

#: Reconcile. The whole reason this file exists: check the reservation is open,
#: remove it, and fold the cost into the total -- atomically. Split across
#: round trips, two workers interleave and the total is wrong rather than
#: missing, which is the failure R-TECH-1 calls the worst kind.
_RECONCILE = """
if redis.call('EXISTS', KEYS[3]) == 1 then return 'CLOSED' end
if redis.call('HGET', KEYS[2], 'closed') == '1' then return 'CLOSED' end
if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 0 then return 'NOTOPEN' end
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('HINCRBYFLOAT', KEYS[2], 'actual', ARGV[2])
redis.call('HINCRBY', KEYS[2], 'reconciled', 1)
redis.call('PEXPIRE', KEYS[1], ARGV[3])
redis.call('PEXPIRE', KEYS[2], ARGV[3])
return 'OK'
"""

#: Close. Counts what was still open as leaked and writes the tombstone, whose
#: TTL is longer than the payload's so `is_finalised` survives expiry.
_CLOSE = """
if redis.call('HGET', KEYS[2], 'closed') ~= '1' then
  local leaked = redis.call('HLEN', KEYS[1])
  redis.call('HSET', KEYS[2], 'leaked', leaked, 'closed', '1')
end
redis.call('SET', KEYS[3], '1', 'PX', ARGV[1])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('PEXPIRE', KEYS[2], ARGV[2])
return 'OK'
"""
```

The Python wrapper translates the sentinel returns into the same exceptions the in-memory backend raises, so the contract suite cannot tell the backends apart:

```python
    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        """Record a step's worst-case cost before it runs."""
        if projected < 0:
            raise LedgerInvariantError(
                f"cannot reserve a negative cost ({projected}) for {run_id}/{step_id}"
            )
        result = self._client.run_script(
            "reserve",
            self._keys(run_id),
            [step_id, repr(float(projected)), str(self._ttl_ms)],
        )
        if result == "CLOSED":
            raise LedgerInvariantError(
                f"cannot reserve on closed run {run_id!r}; the run's cost "
                f"has already been reported"
            )

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        """Replace a step's reservation with its actual cost."""
        if actual < 0:
            raise LedgerInvariantError(
                f"cannot reconcile a negative cost ({actual}) for {run_id}/{step_id}"
            )
        result = self._client.run_script(
            "reconcile",
            self._keys(run_id),
            [step_id, repr(float(actual)), str(self._ttl_ms)],
        )
        if result == "CLOSED":
            raise LedgerInvariantError(
                f"cannot reconcile {run_id}/{step_id} on a closed run; "
                f"the run's cost has already been reported"
            )
        if result == "NOTOPEN":
            raise LedgerInvariantError(
                f"no open reservation for {run_id}/{step_id}; "
                f"either reconciled twice or never reserved"
            )
```

Also implement `snapshot`, `close_run`, `is_finalised`, `knows`, `evict`, `run_count`, plus two test-only helpers used above: `ttl_seconds_remaining(run_id)` and `drop_payload_for_test(run_id)`.

- [ ] **Step 5: Add Redis to the contract suite**

In `tests/unit/test_ledger_store_contract.py`, change the fixture:

```python
@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> Iterator[LedgerStore]:
    if request.param == "memory":
        yield InMemoryLedgerStore()
        return

    redis_url = os.environ.get("OPTIO_TEST_REDIS_URL", "redis://localhost:6379/15")
    client = RedisClient(redis_url)
    try:
        client.ping()
    except StoreUnavailable:
        pytest.skip(f"no Redis at {redis_url}")
    client._redis.flushdb()  # type: ignore[attr-defined]
    yield RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    client._redis.flushdb()  # type: ignore[attr-defined]
    client.close()
```

- [ ] **Step 6: Run the whole contract suite against both backends**

Run: `pytest tests/unit/test_ledger_store_contract.py -v`
Expected: 20 passed (10 tests × 2 backends). Every difference between the backends shows up here.

- [ ] **Step 7: Run the gate**

Run: `ruff check . && ruff format --check . && mypy && lint-imports && pytest -q`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/optio/lanes/cost/ledger_redis.py tests/ pyproject.toml
git commit -m "Reconcile is one round trip or it is a race"
```

---

## Task 5: Configuration stops lying

**Files:**
- Modify: `src/optio/config.py`
- Modify: `src/optio/lanes/registry.py`
- Modify: `pyproject.toml` (restore the `redis` extra)
- Test: `tests/unit/test_config.py` (existing file — add cases)

**Interfaces:**
- Consumes: `RedisClient`, `RedisLedgerStore`, `InMemoryLedgerStore`.
- Produces: `Config.store_timeout_ms: int = 50`; `store_backend="redis"` builds a real backend.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config.py`:

```python
class TestRedisIsNoLongerRejected:
    def test_redis_backend_is_accepted(self) -> None:
        """It raised through 0.3.0 because nothing implemented it (ADR-005's
        addendum). Now something does."""
        config = Config(store_backend="redis", redis_url="redis://localhost:6379/15")

        assert config.store_backend == "redis"

    def test_redis_without_a_url_is_a_setup_error(self) -> None:
        """Configuration that cannot do what it claims fails loudly at setup."""
        with pytest.raises(OptioConfigError, match="redis_url"):
            Config(store_backend="redis")

    def test_the_timeout_must_be_positive(self) -> None:
        with pytest.raises(OptioConfigError, match="store_timeout_ms"):
            Config(store_timeout_ms=0)

    def test_the_timeout_defaults_to_fifty_milliseconds(self) -> None:
        """Short on purpose: a hung Redis must not add latency to an agent."""
        assert Config().store_timeout_ms == 50
```

- [ ] **Step 2: Run and watch them fail**

Run: `pytest tests/unit/test_config.py -k Redis -v`
Expected: FAIL — the first raises `OptioConfigError`, the rest `TypeError: unexpected keyword argument 'store_timeout_ms'`.

- [ ] **Step 3: Update `Config`**

In `src/optio/config.py`: add the field, delete the rejection block, require a URL.

```python
    store_timeout_ms: int = 50
```

Replace the `if self.store_backend == "redis":` rejection with:

```python
        if self.store_backend == "redis" and not self.redis_url:
            # Setup-time failure, per §4.2: fail-open governs the runtime path,
            # not configuration that names a backend it cannot reach.
            raise OptioConfigError(
                "store_backend='redis' needs redis_url. Set it, or use the "
                "default store_backend='memory'."
            )
        if self.store_timeout_ms <= 0:
            raise OptioConfigError(
                f"store_timeout_ms must be positive, got {self.store_timeout_ms}"
            )
```

Add `OPTIO_STORE_TIMEOUT_MS` to `from_env`, and update the `store_backend`/`redis_url` docstring lines to describe a backend that works.

- [ ] **Step 4: Build the backend in the registry**

In `src/optio/lanes/registry.py`, replace `lanes.append(CostLane(config))` with a store built from config:

```python
    if config.cost_lane:
        from optio.lanes.cost.lane import CostLane

        lanes.append(CostLane(config, store=_ledger_store(config)))
```

and add:

```python
def _ledger_store(config: Config) -> LedgerStore:
    """Build the configured ledger backend.

    An unreachable Redis raises here, at setup, rather than on the agent's
    path: a backend that cannot answer is a configuration error (§4.2), while
    a backend that stops answering later is a runtime one and fails open.
    """
    if config.store_backend != "redis":
        return InMemoryLedgerStore()

    from optio.lanes.cost.ledger_redis import RedisLedgerStore
    from optio.store.redis_client import RedisClient

    # `redis_url` is guaranteed non-empty: Config rejects the combination.
    client = RedisClient(config.redis_url or "", timeout_ms=config.store_timeout_ms)
    client.ping()
    return RedisLedgerStore(
        client,
        ttl_seconds=config.run_ttl_seconds,
        tombstone_ttl_seconds=config.run_ttl_seconds * 10,
    )
```

- [ ] **Step 5: Restore the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
redis = ["redis>=5.0"]
```

- [ ] **Step 6: Run the tests and the gate**

Run: `pytest -q && ruff check . && ruff format --check . && mypy && lint-imports`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/optio/config.py src/optio/lanes/registry.py pyproject.toml tests/unit/test_config.py
git commit -m "store_backend='redis' stops raising and starts working"
```

---

## Task 6: Unreachable means no signal, not a partial one

**Files:**
- Modify: `src/optio/lanes/cost/lane.py`
- Test: `tests/failinject/test_store_unavailable.py`

**Interfaces:**
- Consumes: `StoreUnavailable`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/failinject/test_store_unavailable.py`:

```python
"""A shared store that stops answering must not produce a number.

In memory, degrading meant one process's data -- complete for that process. On
a shared store it means the *other* processes' spend is invisible, so a
computed `budget_remaining` could report a full budget for a run that is
already overspent. That is a wrong number rather than a missing one, and this
project treats those differently.
"""

from __future__ import annotations

import logging

import pytest

from optio.config import Config
from optio.lanes.cost.lane import CostLane
from optio.store.redis_client import StoreUnavailable

pytestmark = pytest.mark.failinject


class _DeadStore:
    """Every operation fails the way an unreachable Redis does."""

    def reserve(self, run_id: str, step_id: str, projected: float) -> None:
        raise StoreUnavailable("redis unavailable")

    def reconcile(self, run_id: str, step_id: str, actual: float) -> None:
        raise StoreUnavailable("redis unavailable")

    def snapshot(self, run_id: str) -> object:
        raise StoreUnavailable("redis unavailable")

    def close_run(self, run_id: str) -> object:
        raise StoreUnavailable("redis unavailable")

    def is_finalised(self, run_id: str) -> bool:
        raise StoreUnavailable("redis unavailable")

    def knows(self, run_id: str) -> bool:
        raise StoreUnavailable("redis unavailable")

    def evict(self, run_id: str) -> None:
        raise StoreUnavailable("redis unavailable")

    def run_count(self) -> int:
        raise StoreUnavailable("redis unavailable")


def test_an_unreachable_store_emits_no_cost_signal(run_fixture) -> None:
    """Not a zero, not a full budget -- nothing."""
    lane = CostLane(Config(), store=_DeadStore())

    signals = lane.on_run_end(run_fixture)

    assert signals == [], f"emitted {[s.name for s in signals]} from a store it could not read"


def test_it_warns_once_rather_than_every_step(
    run_fixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Silence must not be mistaken for zero spend -- but a warning in a hot
    loop is one people filter out."""
    lane = CostLane(Config(), store=_DeadStore())

    with caplog.at_level(logging.WARNING, logger="optio"):
        for _ in range(5):
            lane.on_run_end(run_fixture)

    assert len(caplog.records) == 1
```

*(`run_fixture` is the existing run-double fixture used by the other cost-lane tests; import it from the same conftest those use.)*

- [ ] **Step 2: Run and watch it fail**

Run: `pytest tests/failinject/test_store_unavailable.py -v`
Expected: FAIL — `CostLane` does not accept `store` yet, and does not catch `StoreUnavailable`.

- [ ] **Step 3: Implement**

In `src/optio/lanes/cost/lane.py`: accept `store: LedgerStore | None = None` and pass it to `CostLedger`. Wrap the signal-producing path so `StoreUnavailable` yields `[]` and logs once, guarded by an instance flag so the warning does not repeat.

- [ ] **Step 4: Run the tests and the gate**

Run: `pytest -q && ruff check . && ruff format --check . && mypy && lint-imports`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/optio/lanes/cost/lane.py tests/failinject/test_store_unavailable.py
git commit -m "A store that cannot answer produces silence, not a full budget"
```

---

## Task 7: The test that proves the bug is fixed

**Files:**
- Create: `tests/integration/test_multiprocess_budget.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the milestone's success criterion.

- [ ] **Step 1: Write the test**

```python
"""Four processes, one run, one correct total.

This is the test the milestone exists for. Run against the in-memory backend it
demonstrates the bug: each process meters into its own dictionary, so a $0.50
budget admits $2.00 while every process's arithmetic is internally consistent.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from optio.lanes.cost.ledger_redis import RedisLedgerStore
from optio.store.redis_client import RedisClient

pytestmark = pytest.mark.redis

REDIS_URL = os.environ.get("OPTIO_TEST_REDIS_URL", "redis://localhost:6379/15")
WORKERS = 4
STEPS_PER_WORKER = 25
COST_PER_STEP = 0.01


def _meter(worker: int) -> None:
    """Reserve and reconcile a fixed number of steps against the shared run."""
    client = RedisClient(REDIS_URL)
    store = RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    for step in range(STEPS_PER_WORKER):
        step_id = f"w{worker}-s{step}"
        store.reserve("shared-run", step_id, COST_PER_STEP)
        store.reconcile("shared-run", step_id, COST_PER_STEP)
    client.close()


def test_four_processes_produce_one_correct_total() -> None:
    client = RedisClient(REDIS_URL)
    try:
        client.ping()
    except Exception:  # noqa: BLE001 - the skip message is the point
        pytest.skip(f"no Redis at {REDIS_URL}")
    client._redis.flushdb()  # type: ignore[attr-defined]

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_meter, args=(w,)) for w in range(WORKERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker failed with exit code {p.exitcode}"

    store = RedisLedgerStore(client, ttl_seconds=60.0, tombstone_ttl_seconds=300.0)
    snap = store.snapshot("shared-run")

    expected = WORKERS * STEPS_PER_WORKER * COST_PER_STEP
    assert snap.actual == pytest.approx(expected), (
        f"four processes metered {snap.actual} against an expected {expected}; "
        "the shared ledger lost or duplicated updates"
    )
    assert snap.reconciled_steps == WORKERS * STEPS_PER_WORKER
    assert snap.reserved == pytest.approx(0.0)

    client._redis.flushdb()  # type: ignore[attr-defined]
    client.close()
```

- [ ] **Step 2: Run it**

Run: `pytest tests/integration/test_multiprocess_budget.py -v -m redis`
Expected: PASS — `snap.actual == 1.0` exactly.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_multiprocess_budget.py
git commit -m "Four processes, one run, one total"
```

---

## Task 8: A real Redis in CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add the service and the marked run**

Add to the main test job:

```yaml
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 5s
          --health-timeout 3s
          --health-retries 10
```

and a step after the existing test run:

```yaml
      - name: Redis-backed store
        # Marked tests are excluded from the default run because they need a
        # server. Skipping silently when Redis is absent is right locally and
        # wrong here, so the gate runs them explicitly and they must not skip.
        env:
          OPTIO_TEST_REDIS_URL: redis://localhost:6379/15
        run: pytest -m redis -q --no-header -p no:randomly
```

- [ ] **Step 2: Verify locally with a container**

```bash
docker run -d -p 6379:6379 --name optio-redis redis:7-alpine
OPTIO_TEST_REDIS_URL=redis://localhost:6379/15 pytest -m redis -q
docker rm -f optio-redis
```

Expected: all redis-marked tests pass, none skip.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "The gate runs a real Redis, because Lua is where a fake diverges"
```

---

## Task 9: Write it down

**Files:**
- Create: `docs/design/adr/adr-050-the-store-speaks-the-domain.md`
- Modify: `docs/design/adr/README.md`, `docs/design/adr/adr-005-pluggable-state-store-in-memory-default.md`
- Modify: `src/optio/store/base.py` (retire the ABC)
- Modify: `README.md`, `CHANGELOG.md`, `docs/runbooks.md`

- [ ] **Step 1: Write ADR-050**

Record: the generic KV shape was rejected because `reconcile` cannot be atomic above the interface; the ABC had never been exercised by any consumer, so its shape was an untested guess; the store now speaks the domain, one Protocol per lane; TTL is an idle timeout and the tombstone outlives the payload; unreachable means no signal rather than a partial one. Mark ADR-005's **interface** superseded, its decision intact.

- [ ] **Step 2: Add the addendum to ADR-005**

```markdown
## Addendum (2026-08-04): the interface is superseded, the decision is not

ADR-050 replaces the generic `StateStore` ABC with one Protocol per lane. The
decision recorded above — pluggable storage, in-memory default, atomic
increments, store failures are lane failures — stands unchanged. Only the shape
is superseded, and it is superseded because it was never exercised: no consumer
ever constructed a `StateStore`, so `get`/`set`/`incr`/`delete` was a guess, and
it could not express `reconcile` atomically.
```

- [ ] **Step 3: Retire the ABC**

Delete `StateStore` from `src/optio/store/base.py`, leaving the module docstring pointing at ADR-050. Keep `InMemoryStateStore` only if something still imports it; if nothing does, delete `src/optio/store/memory.py` and `tests/unit/test_memory_store.py` with it — a fixture with no consumer is the thing ADR-042 keeps catching.

- [ ] **Step 4: Update the README**

Delete the multi-process limitation from the Status block rather than rewording it:

```markdown
> - **State is in-process by default.** Set `store_backend="redis"` for runs
>   sharded across processes ([ADR-050](...)); the in-memory default needs no
>   infrastructure and meters one process.
```

Update the test count in the Development block to the new number.

- [ ] **Step 5: Add the CHANGELOG entry**

Under `## [Unreleased]`, an `### Added` entry naming the measured bug (four workers, `$0.50` budget, `$2.00` through) and the proof test.

- [ ] **Step 6: Run the full gate**

Run: `ruff check . && ruff format --check . && mypy && lint-imports && pytest -q`
Expected: clean, `Contracts: 4 kept, 0 broken`.

- [ ] **Step 7: Commit**

```bash
git add docs/ src/optio/store/ README.md CHANGELOG.md
git commit -m "ADR-050: the store speaks the domain, and ADR-005's shape retires"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: domain Protocol → 3; Lua atomicity → 4; TTL refresh + tombstone → 4; fail-open-to-absence → 6; config → 5; contract suite both backends → 3 and 4; real Redis in CI → 8; multi-process proof → 7; supersession ADR → 9. **Deliberately deferred to the behaviour and quality plans:** the `WindowState`/`QualityStep` types, the hypothesis property test for the window aggregates, the no-spans-in-the-store test, and the overhead re-measurement (it needs all three lanes on the store before the published per-step figure can be re-derived honestly).

**Placeholder scan.** No TBD/TODO. Task 4 Step 4 and Task 6 Step 3 describe the remaining methods rather than printing every line — the signatures they must satisfy are fixed by the Protocol in Task 3 and the exception messages are given verbatim where behaviour depends on them.

**Type consistency.** `LedgerStore` names in Task 3 match their uses in 4, 5 and 6. `RedisClient.run_script(name, keys, args)` in Task 2 matches the calls in Task 4. `StoreUnavailable` subclasses `StateStoreError` in Task 2 and is caught as such in Task 6. `RedisLedgerStore(client, *, ttl_seconds, tombstone_ttl_seconds)` is constructed identically in Tasks 4, 5 and 7.

**Known risk carried deliberately.** Task 3 moves a locking implementation verbatim; the regression net is that no existing test may be edited. If the extraction tempts anyone to change a test, that is the signal to stop and re-read the diff.
