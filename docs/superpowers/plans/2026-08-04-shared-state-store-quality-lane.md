# Shared State Store — Quality Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make outcome scoring work across processes, so a run sharded over four workers is scored from the step that actually finished last and reports the number of steps it actually took.

**Architecture:** `QualityLane`'s `dict[str, list[ReadableSpan]]` moves behind a `QualityStore` Protocol, the third and last of ADR-050's per-lane Protocols. The in-memory backend is today's implementation. The Redis backend keeps a counter and one projected step per run, because that is provably all the run-end consumers read.

**Tech Stack:** Python 3.10–3.14, `redis-py` (dev extra), Lua, pytest, hypothesis, real Redis service container.

## Global Constraints

- **No behaviour change for single-process users** *except the `step_count` fix in Task 1*, which is a correction and is called out in the CHANGELOG. Every other existing quality-lane test passes unchanged.
- **No `ReadableSpan` may reach a store.** Spans are not serializable and the spec makes this a named test.
- **Layering (import-linter):** `optio.lanes` may import `optio.store`; never the reverse. `optio.lanes.quality` must not import `optio.lanes.cost` or `optio.lanes.behavior`.
- **Fail-open governs the runtime path** (ADR-004); setup failure is loud (§4.2).
- **The quality lane is off by default** (ADR-003). Nothing here may change that.
- **Never log or store prompt content** (§10). The projection carries counts, booleans and enum strings — never output text.
- Redis timeouts stay at `Config.store_timeout_ms` (50 ms default). No retries on the hot path.
- Gate every commit: `ruff check .` && `ruff format --check .` && `mypy` && `lint-imports` && `pytest -q`.

---

## The findings that shape this plan

The spec required its own projection claim to be **verified against the code rather than trusted**. It was, and it did not survive.

> *"the retained spans are consumed for their count … and for the tier decision. So `QualityStep` starts as span name, start/end timestamps, and the `gen_ai.*` attributes the tier decision reads"*

**1. The tier decision reads no spans at all.** `sampling.decide(run, config)` takes the run and the config. Span name and timestamps are read by nobody.

**2. The real consumer is `heuristic.score(spans)`, and it reads only `spans[-1]`.** Three fields off one span:

| what it reads | where |
|---|---|
| `span.status.status_code is StatusCode.ERROR` | `heuristic._errored` |
| `gen_ai.response.finish_reasons` | `heuristic._incomplete_finish_reason` |
| `gen_ai.usage.output_tokens` | `heuristic.score` |

So the projection is *narrower* than the spec guessed, along a different axis. A store holding 64 spans per run holds 63 that nothing will ever read.

**3. A live bug, found by asking what `len(spans)` means.** `on_run_end` builds `JudgeRequest(step_count=len(spans))`, and `spans` is capped at `MAX_RETAINED_SPANS = 64`. But `judge.py:73` documents `step_count` as *"How many steps the run took"*, and `docs/quality.md:70` shows a user passing it straight to their evaluator as `steps=request.step_count`.

**A 500-step run tells the user's judge it took 64 steps.** Wrong rather than missing, plausible enough that nobody would query it, and no test pins it — `tests/unit/test_quality_judge.py` builds a `JudgeRequest` directly. Fixed in Task 1, ahead of the extraction, so the store is built against the corrected contract.

**4. `buffer`/`drain` becomes `record`/`close_run`, for the third time.** The spec's Protocol drains and then closes. That is two operations with a gap another worker's step can land in, so the summary would describe a run that had already moved on. `LedgerStore.close_run` returns its snapshot and `BehaviorStore.close_run` returns its state for exactly this reason. Three lanes, one shape — worth recording in the ADR rather than re-deriving a fourth time.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/optio/lanes/quality/store.py` | **Create.** `QualityStore` Protocol, `QualityStep`, `QualitySummary`. |
| `src/optio/lanes/quality/store_memory.py` | **Create.** `InMemoryQualityStore`. |
| `src/optio/lanes/quality/store_redis.py` | **Create.** `RedisQualityStore` — counter plus one projected step. |
| `src/optio/lanes/quality/heuristic.py` | **Modify.** `score` takes a `QualityStep | None`; the span-reading moves to a `project` function. |
| `src/optio/lanes/quality/lane.py` | **Modify.** Holds a store; `step_count` becomes the true count. |
| `src/optio/lanes/registry.py` | **Modify.** Build and inject the configured quality backend. |
| `tests/unit/test_quality_store_contract.py` | **Create.** One suite, both backends. |
| `tests/integration/test_redis_quality.py` | **Create.** What only a real server shows. |
| `tests/integration/test_multiprocess_quality.py` | **Create.** Four processes, one score. |
| `tests/unit/test_no_spans_in_stores.py` | **Create.** The spec's named test, across all three lanes. |

---

## Task 1: `step_count` means what it says

Ahead of the extraction, so the store is built against a corrected contract rather than inheriting a wrong one.

**Files:**
- Modify: `src/optio/lanes/quality/lane.py`
- Test: `tests/unit/test_quality_lane.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_judge_is_told_the_real_step_count() -> None:
    """`step_count` is documented as "how many steps the run took" and
    docs/quality.md shows a user passing it to their evaluator as `steps=`.

    It was `len(spans)`, and spans are capped at MAX_RETAINED_SPANS -- so any
    run longer than 64 steps reported 64. Wrong rather than missing, and
    plausible enough that a user would build on it.
    """
    seen: list[int] = []

    def judge(request: JudgeRequest) -> JudgeScores:
        seen.append(request.step_count)
        return JudgeScores(task_success=1.0)

    lane = QualityLane(Config(quality_lane=True, quality_sample_rate=1.0), judge=judge)
    run = Run("long")
    for _ in range(MAX_RETAINED_SPANS + 40):
        lane.process_span(span(), run)
    lane.on_run_end(run)

    assert seen == [MAX_RETAINED_SPANS + 40]
```

- [ ] **Step 2: Run it and watch it fail**

Expected: FAIL, `seen == [64]`.

- [ ] **Step 3: Count steps rather than measuring the buffer**

`process_span` increments a per-run counter alongside the buffer; `on_run_end` reads the counter. The buffer stays capped — it is a bound on retention, and the count is not a measurement of it.

- [ ] **Step 4: Run the whole quality suite**

Expected: PASS, and every other quality test unchanged.

- [ ] **Step 5: Mutation**

Put `len(spans)` back; confirm the new test fails and nothing else does — which is what makes it worth having.

- [ ] **Step 6: Gate and commit**

---

## Task 2: `QualityStep`, and the heuristic reads it

**Files:**
- Create: `src/optio/lanes/quality/store.py`
- Modify: `src/optio/lanes/quality/heuristic.py`, `src/optio/lanes/quality/lane.py`
- Test: `tests/unit/test_quality_step.py`

**Interfaces:**
- Produces: `QualityStep(errored, finish_reasons, output_tokens)`; `QualitySummary(step_count, last)`; `project(span) -> QualityStep`; `heuristic.score(step: QualityStep | None) -> HeuristicResult`.

- [ ] **Step 1: Write the failing test**

Assert `project` captures exactly the three fields the heuristic reads, that `finish_reasons` normalises both the array-valued and the flattened-string form the current code accepts, and — the one that matters — that **no output text is carried**:

```python
def test_the_projection_carries_no_content() -> None:
    """Section 10. The projection is counts, booleans and enum strings.

    Asserted structurally rather than by listing fields, so a field added
    later has to justify itself here."""
    step = project(span_with(output="the model's actual answer"))

    for value in astuple(step):
        assert not isinstance(value, str), "a bare string field can carry content"
```

- [ ] **Step 2: Run it and watch it fail** — `optio.lanes.quality.store` does not exist.

- [ ] **Step 3: Add the value types and the Protocol**

`QualityStep` and `QualitySummary` are frozen slotted dataclasses. `QualityStore` declares `record(run_id, step) -> None`, `close_run(run_id) -> QualitySummary | None`, `run_count() -> int`. `close_run` returns the summary for the reason recorded in finding 4; `None` means *no run*, never *a run with no steps*.

- [ ] **Step 4: Split `project` out of `heuristic`**

`project(span) -> QualityStep` holds today's attribute reads verbatim. `score` takes `QualityStep | None` and keeps every threshold and comment exactly as they are — this is a change of input type, not of logic, and no assertion in `tests/unit/test_quality_heuristic.py` may move. Where a test builds spans and calls `score(spans)`, it becomes `score(project(span))`.

- [ ] **Step 5: Run the heuristic suite**

Expected: PASS with assertions untouched. If one has to change, the projection lost information and the design is wrong.

- [ ] **Step 6: Gate and commit**

---

## Task 3: The in-memory backend

**Files:**
- Create: `src/optio/lanes/quality/store_memory.py`, `tests/unit/test_quality_store_contract.py`

- [ ] **Step 1: Write the contract suite**, parametrised over `["memory"]` for now (Redis joins in Task 4):

```python
class TestRecording:
    def test_the_summary_counts_every_step(self, store) -> None:
        for _ in range(100):
            store.record("run", _step())

        summary = store.close_run("run")

        assert summary.step_count == 100

    def test_the_summary_keeps_the_last_step(self, store) -> None:
        """The heuristic judges the final answer, so "last" is the whole
        contract -- keeping the first would score the run's opening move."""
        store.record("run", _step(output_tokens=10))
        store.record("run", _step(output_tokens=99))

        assert store.close_run("run").last.output_tokens == 99

    def test_runs_do_not_bleed(self, store) -> None:
        store.record("a", _step(output_tokens=1))
        store.record("b", _step(output_tokens=2))

        assert store.close_run("a").last.output_tokens == 1


class TestLifecycle:
    def test_closing_releases_the_run(self, store) -> None:
        store.record("run", _step())
        store.close_run("run")

        assert store.run_count() == 0

    def test_closing_twice_reports_nothing_the_second_time(self, store) -> None:
        """Run end fires more than once (M1-2). A second close reporting an
        empty summary would let the lane re-score from no evidence and emit a
        weaker verdict over the first -- the failure the behavior lane hit."""
        store.record("run", _step())
        store.close_run("run")

        assert store.close_run("run") is None

    def test_closing_an_unknown_run_is_not_an_error(self, store) -> None:
        assert store.close_run("never") is None
```

- [ ] **Step 2: Run and watch it fail.**
- [ ] **Step 3: Implement** — a lock and `dict[str, QualitySummary]`, replacing the last step and incrementing the count on each `record`.
- [ ] **Step 4: Gate and commit.**

---

## Task 4: The Redis backend

**Files:**
- Create: `src/optio/lanes/quality/store_redis.py`, `tests/integration/test_redis_quality.py`
- Modify: `tests/unit/test_quality_store_contract.py` (add `"redis"`)

**Keys per run:** one hash, `optio:q:{run}`, holding `steps`, `errored`, `finish`, `tokens`.

- [ ] **Step 1: Write the Lua**

`record` is `HINCRBY steps 1`, three `HSET`s for the last step, and one `PEXPIRE` — the payload is fixed size regardless of run length, exactly as the behaviour lane's is.

`close` reads the hash, deletes it, and returns `false` when it was absent. One script, so the read and the release are one event.

- [ ] **Step 2: Write the integration test** — every key carries a TTL, the TTL refreshes on each step, `close` removes the hash, and `run_count` scans only `optio:q:*` so the ledger's and the behaviour lane's keys are not counted as quality runs.

- [ ] **Step 3: Add `"redis"` to the contract fixture.**

Expected: every contract test passes against **both** backends.

- [ ] **Step 4: Mutation-test the reduction** — make `record` overwrite the counter rather than increment it, and confirm `test_the_summary_counts_every_step` fails on the Redis leg.

- [ ] **Step 5: Gate and commit.**

---

## Task 5: Wire the lane and the registry

**Files:**
- Modify: `src/optio/lanes/quality/lane.py`, `src/optio/lanes/registry.py`

- [ ] **Step 1** `QualityLane.__init__` takes `store: QualityStore | None = None`, defaulting to `InMemoryQualityStore()`; `self._spans`, its lock and the step counter are deleted, since the store owns all three. `MAX_RETAINED_SPANS` goes with them — there is no buffer left to bound, which is the clearest evidence the projection was right.
- [ ] **Step 2** `registry._quality_store(config)` mirrors `_behavior_store(config)`, pinging at setup, with its own client.
- [ ] **Step 3** Add the object-graph test: `enabled_lanes(Config(quality_lane=True, store_backend="redis", ...))` yields a `QualityLane` whose store is the Redis one, and the default config's is not. The assertion that a setting is *read* — ADR-005's addendum exists because one was not.
- [ ] **Step 4** Gate and commit.

---

## Task 6: No spans in any store

**Files:**
- Create: `tests/unit/test_no_spans_in_stores.py`

The spec names this test. It is written once for all three lanes, because the property is about the boundary rather than about quality.

- [ ] **Step 1: Write it** — drive each lane through `process_span` and `on_run_end` with a recording store that asserts every argument it receives is JSON-round-trippable, and that no argument is or contains a `ReadableSpan`.

- [ ] **Step 2: Mutation** — pass the span itself into the quality store; confirm it fails.

- [ ] **Step 3: Gate and commit.**

---

## Task 7: Four processes, one score

**Files:**
- Create: `tests/integration/test_multiprocess_quality.py`

- [ ] **Step 1: Write it** — four spawned workers each record steps against one run; the run is then scored from a step count that is the sum, not one worker's share. Pass the URL as an **argument**, not via the environment: `tests/conftest.py` strips `OPTIO_*` and the child re-imports after that.

- [ ] **Step 2: Add the in-memory counterpart** asserting each worker sees only its own share, so the limitation is measured rather than described.

- [ ] **Step 3: Gate and commit.**

---

## Task 8: Measure it, then write it down

**Files:**
- Modify: `tests/bench/test_overhead.py`, `README.md`, `CHANGELOG.md`, `docs/quality.md`
- Modify: `docs/design/adr/adr-050-the-store-speaks-the-domain.md` (addendum)

- [ ] **Step 1: Benchmark the quality lane's hot path on both backends.** This lane emits nothing per step, so a per-step round trip buys nothing until run end — the honest question is what it costs, and the answer goes in the table rather than in a reassurance.
- [ ] **Step 2: Update the README's overhead table** with the measured row.
- [ ] **Step 3: `docs/quality.md`** — correct what `step_count` means, and document `store_backend="redis"` for sharded runs.
- [ ] **Step 4: ADR-050 addendum** recording that all three lanes have landed, that `close_run` returning its summary was re-derived three times, and that the per-step-write design was chosen over accumulate-and-merge with the measured reason.
- [ ] **Step 5: CHANGELOG** — the `step_count` fix as a **Fixed** entry in its own right, separately from the shared-store **Added** entry.
- [ ] **Step 6: Full gate, commit, push.**

---

## Self-Review

**Spec coverage.** The spec's `QualityStore` section maps to Tasks 2–5; its "verify the projection against the code" requirement to the findings section and Task 2; its named no-spans test to Task 6; its overhead re-measurement to Task 8. Nothing in the spec's quality scope is unassigned.

**Where this plan deliberately departs from the spec, and why.** Three places, each recorded above: the projection's contents (finding 2), `buffer`/`drain` → `record`/`close_run` (finding 4), and the store holding a summary rather than a list (finding 2 again — a 64-item buffer of which one item is read). The spec asked for the first to be verified rather than trusted; the second and third follow from it.

**Placeholder scan.** No TBD/TODO. Tasks 4–8 give steps rather than full listings because each is a mechanical mirror of a behaviour-lane change already merged and reviewable in `git log` — `store_redis.py` for Task 4, `_behavior_store(config)` for Task 5, `test_multiprocess_behavior.py` for Task 7.

**Type consistency.** `QualityStep(errored, finish_reasons, output_tokens)` and `QualitySummary(step_count, last)` are constructed identically in Tasks 2, 3 and 4 and consumed by `heuristic.score` in Task 2. `record(run_id, step)` / `close_run(run_id)` / `run_count()` match across the Protocol, both backends and every test, and the lifecycle names match the two lanes already merged.

**Known risk carried deliberately.** Task 1 changes a number a user's judge already receives. It is a correction, not a refactor, and it ships as a `Fixed` entry — but anyone who calibrated against the capped value sees a different input. That is the right trade: the documented meaning was always the true count, and continuing to send 64 would keep a wrong number in a user's evaluation loop.
