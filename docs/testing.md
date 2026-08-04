# Testing

Because the implementer is an AI coding agent and the reviewer is one human, tests are the
primary defence against confident-but-wrong code (§9) rather than an afterthought. This
document records what is tested, what each layer is *for*, and — the part usually left out —
how we know the tests would actually catch a bug.

## The layers

| Layer | Scope | Gate |
|---|---|---|
| Unit | every module, pure logic | ≥ 85% core, **100%** on `ledger.py` and `failopen.py` |
| Property (`hypothesis`) | ledger reserve/reconcile invariant, window bounds | blocking |
| Fault injection | every internal exception at every lane boundary | blocking (SC-4) |
| Contract | emitted names/types vs pinned semconv | blocking (SC-2) |
| Integration | real span flow, end to end | per adapter |
| Benchmark | per-step overhead | blocking, < 5 ms p99 (SC-5) |
| Policy pack | Rego/Cedar/AGT against the real engines | blocking (SC-3) |
| Demo smoke | `docker compose up` produces the signals | blocking (ADR-006) |
| **Real frameworks** | adapters against actual LangGraph/CrewAI/OpenAI/Claude | blocking (R-TECH-3) |
| **Portability** | 5 Pythons × 3 OSes, dependency floor, built wheel | blocking |
| **Soak** | memory over 50k steps and 40k runs | blocking (§11) |
| **Mutation** | do the tests detect an injected bug? | periodic, see below |

## Soak testing

§11 budgets memory per run and names a soak test as its verification. It guards the failure no
other suite can see: nothing breaks, it just grows. A few hundred bytes leaked per run is
invisible in a unit test and fatal in an agent process that stays up for a week.

Measured with `tracemalloc` rather than RSS — RSS is dominated by allocator behaviour and GC
timing and would make the test flap; tracemalloc counts what Python actually holds.

| Scenario | Result |
|---|---|
| One run, 50,000 steps | **+0.4 KiB** total |
| Quality lane on, 20,000 steps | **+0.0 KiB** |
| 40,000 short runs, steady state | **+0.0 bytes/run** |

The first row is the design claim made good: cost is O(window), not O(steps), so a long-lived
agent does not die overnight.

The third needed care to interpret. A naive measurement reports ~74 bytes per run, which looks
like a slow leak. Measuring the *rate* in successive blocks shows what it really is: ~114 B/run
in the first block, ~34 in the second, then **0.0 for the following 30,000 runs**. That is the
ledger's closed-run FIFO filling to its 4,096-id cap (ADR-010) and then holding — bounded state,
not a leak.

A ceiling alone would not have caught the difference, since a genuine leak that happens to be
small passes any per-run ceiling. `test_growth_stops_rather_than_merely_being_slow` asserts the
rate falls to zero, which is the property that actually matters.

## Mutation testing

100% line coverage means every line *ran*. It does not mean any behaviour was *asserted* — a
test can execute a line and check nothing about it. Mutation testing closes that gap by
changing the source (flip `>` to `>=`, `True` to `False`, delete a condition) and asking
whether any test notices.

This is not theoretical here. Three real defects this project shipped or nearly shipped were
of exactly that shape: a Cedar guard test that kept passing with the guard deleted, a
`budget_remaining` that reported a full budget for a run it could not price, and — found by
this exercise — a leak warning nothing asserted.

### Running it

```bash
python scripts/mutate.py --list      # available targets
python scripts/mutate.py ledger      # run one
```

That is the only supported way to do it, and the reason is worth understanding.

`cosmic-ray` (chosen over `mutmut`, which has no Windows support) mutates source files **in
place** and restores them with a context manager. That works when a run finishes or raises, and
does nothing at all when the process is killed — a Ctrl-C at the wrong moment, a terminated
tool call, a closed lid. The mutated file simply stays on disk.

This is not hypothetical. This project's first mutation run was interrupted and left:

```python
return snapshot.committed - remaining * estimate  # should be +
```

Every worst-case cost projection with its sign flipped, sitting in the working tree, one
`git commit -a` away from being real. It was caught by reading `git status` — luck, not process.

`scripts/mutate.py` removes the failure mode rather than warning about it. It exports `HEAD`
with `git archive` into a temp directory, builds an isolated venv there, and runs the mutation
against that copy. **The working tree is never opened for writing**, so no kill signal, crash,
or full disk can corrupt it. Verified by killing a run mid-execution and confirming the source
hash was unchanged.

Using `git archive HEAD` also means mutations run against committed code, never against
half-finished edits — a mutation report about code that exists nowhere else is worse than none.

Two smaller things it handles, both learned the hard way: the generated config uses forward
slashes and TOML literal strings, because a Windows path in a basic string is an invalid escape
sequence that cosmic-ray reports as a *failed baseline* — sending you off to debug a test suite
that is fine. And it separates annotation noise from real survivors in its output, so the
signal is not buried.

### The backstop

`tests/contract/test_source_is_not_mutated.py` asserts the specific arithmetic and boolean
expressions that mutation operators target — the ones where a flip yields a plausible wrong
number rather than an error. It runs in the blocking `contract` gate, so a leftover mutation
cannot reach `main` even if someone bypasses the script and runs `cosmic-ray` directly.

Verified by re-injecting both real corruptions this project experienced (the projection sign
flip and an inverted leak warning); the guard fails on each.

### Results (2026-07-27)

| Module | Mutants | Survived | Real gaps |
|---|---|---|---|
| `failopen.py` | 54 | 13 | **2**, both equivalent |
| `ledger.py` | 80 | 15 | **1**, now fixed |
| `project.py` | 224 | 81 | **0** |

Raw survival rates look alarming and mostly are not. Two categories of survivor cannot be
killed, and reading them as failures leads to writing worthless tests:

**Annotation noise.** `float | None` in a signature contains a `|`, which cosmic-ray dutifully
mutates to `+`, `-`, `*`, and so on. With `from __future__ import annotations` these are
strings that never evaluate. This accounts for 72 of `project.py`'s 81 survivors and 11 of
`failopen.py`'s 13.

**Genuine equivalent mutants.** `if TYPE_CHECKING:` inverted still executes nothing at runtime.
`name == component` versus `name is component` behave identically for interned strings.
`reconciled_steps > 0` versus `!= 0` differ only for negative counts, which the ledger rejects
at the boundary — verified, not assumed.

**The one real gap** was in `ledger.py`: inverting `if leaked:` suppressed the warning that a
run ended with unreconciled reservations, and no test noticed. The count was thoroughly
property-tested; the *warning* was not. It matters because that warning is the only thing
telling an operator the reported cost is a reserved worst case rather than measured spend —
silently presenting an estimate as a measurement is the R-TECH-1 failure mode. Now covered by
three tests, verified to fail against the mutation.

### Why it is periodic rather than per-PR

A full run takes minutes per module and the signal is dominated by equivalents that need human
judgement to dismiss. Wiring it as a blocking gate would train contributors to add
`# pragma: no mutate` until it went quiet. It belongs in the maintainer's periodic review of
the two modules where silent wrongness is worst — the ledger and the guard — and after any
change to cost arithmetic.

## Profiling

Section 16 rule 10 forbids optimizing without a benchmark to justify it, so the hot path was
profiled before anything was changed. Two findings were worth acting on; recording them here
because the *shape* of both is easy to miss.

**Classification was O(window) per step.** `classify` rebuilt a `Counter` over the whole window
on every step — 1,018,775 generator calls for 20,000 steps. Every existing test passed: the
cost is bounded by config, so it never becomes O(run), and
`test_classification_is_flat_in_run_length` checks exactly that. What no test checked was the
cost against *window size*:

| window | before | after |
|---|---|---|
| 50 (default) | 53 µs | **37 µs** |
| 200 | 105 µs | **41 µs** |
| 1000 | 370 µs | **38 µs** |

The window is now counted incrementally on append. Widening it to catch longer cycles is a
memory decision rather than a latency one.

**What the row above does *not* say, found while making it a test (ADR-050).** Per-step cost is
flat in window *size*, and that is the claim — but the driver it is flat against is the number of
**distinct calls** the window holds, not the window bound. Finding the top `k` counts means
scanning the live ones on both backends: `Counter.most_common(k)` over the keys in memory, `HVALS`
plus a bounded insertion in Lua. The two coincide only when a workload's tool diversity is bounded,
which the benchmark's was — 64 tool names.

A first version of the test let every step name a new tool, making `distinct == window`, and
"found" a 4.7× regression that was really the workload. Measured with diversity held at 64 and the
window varied, and then the other way around:

| axis | 50 | 200 | 1000 |
|---|---|---|---|
| window size (64 distinct calls) | 12.6 µs | 14.7 µs | 13.2 µs |
| distinct calls (window 1000) | — | — | 8.3 µs at 8 → **35.1 µs at 1000** |

The growth is real and points the safe way: it peaks when every step is different, which is the
case with no loop to detect, and it is cheapest on the tight cycles the lane exists to catch. Even
the worst case is two orders of magnitude inside SC-5. Both axes are now asserted separately in
`tests/bench/test_overhead.py` so the distinction cannot silently collapse again.

**Over Redis**, the same step costs ~620 µs against a ~500 µs bare `PING` on this machine (Windows,
Docker Desktop NAT — a Linux loopback is several times faster). The benchmark asserts the *ratio*,
under three round trips, rather than the absolute: a step is one Lua script by construction, so a
regression there means the script started doing real work, and the ratio means the same thing on
any hardware.

The risk that swap introduces is drift — stored counts diverging from the deque — and drift is
invisible to every other test in the suite, because the deque itself stays correct. Three
property tests in `TestIncrementalCountsMatchARecount` compare against a full recount on *every*
add, not just at the end, since a decrement error that a later eviction cancels out would pass a
final-state check having produced wrong verdicts throughout. Verified by removing the eviction
decrement: all three fail.

**The cost lane took two ledger snapshots per priced step.** Halving that is the smaller half of
the fix. The larger half is consistency: taken separately, a concurrent step can land between
them, so `actual_cost` and `budget_remaining` describe different ledger states — two numbers
that were never simultaneously true. A policy checking `remaining == limit - actual` would see a
contradiction, which is worse than either number being merely stale.

**Import time: 211 ms → 158 ms.** `importlib.metadata` cost 88 ms of it — 42% — pulling in
`zipfile`, `email.message`, `pathlib` and `inspect` to read a version string most programs never
touch. It resolves lazily now (PEP 562). The pre-existing 500 ms budget test is far too loose to
catch that returning, so `test_version_lookup_stays_off_the_import_path` names the specific
modules instead; re-adding the eager import fails it while the budget test stays green.

Nothing else in the profile justifies changing. It is flat, the leader does inherent work
(sorting and hashing span attributes), and per-step overhead sits at ~67 µs against a 5 ms
budget — 75× headroom. Further optimization would be the premature kind rule 10 names.

## Concurrency

Agent frameworks are overwhelmingly concurrent — LangGraph fans out to parallel branches,
CrewAI runs agents on a pool — so single-threaded correctness says little about how the library
behaves where it actually runs.

Building these tests produced a result worth recording, because it contradicts the obvious
approach. **Removing the ledger's lock entirely and hammering it with 64 threads at a 100 ns
switch interval produces no lost update at all.** CPython's GIL makes the individual dict writes
atomic, so the natural "many threads, check the total" test passes whether the lock exists or
not. It looks like a concurrency test and proves nothing.

What the lock actually guards is the **composite** read-modify-write in `close_run`: it reads
`len(ledger.open)`, then writes `leaked_steps` and `closed` from that read. A `reserve` landing
in between yields a leak count that disagrees with the ledger's own state — and the leak count
is what tells an operator whether a reported cost is measured spend or a reserved worst case.

Measured against that specific interleaving: **75 inconsistent closes in 400 rounds with the
lock removed, 0 with it.** That is the test worth having, and every concurrency test here was
validated the same way — by deleting the synchronisation and confirming the suite goes red.
A concurrency test never checked against a broken lock is decoration.

The suite also covers 16 simultaneous runs asserting no cross-run cost contamination (two
plausible numbers is worse than one obviously wrong one), all three lanes writing at once,
concurrent `install_tap` calls resolving to a single tap, and every test bounded by a timeout so
a deadlock fails CI rather than hanging it.

### The behavior lane's lock became load-bearing too

Making classification O(1) introduced a second read-modify-write: `add` decrements the evicted
call's count and increments the new one. Applying the same method to it — remove the lock, run 16
threads at a 100 ns switch interval — produced two failures, the second of them unanticipated:

* **The counts drift from the deque.** Silent, exactly like the ledger's: the deque stays correct,
  so nothing else in the suite looks wrong while every subsequent verdict is computed from a
  window that never existed.
* **`classify` raises `RuntimeError: dictionary changed size during iteration`**, from
  `Counter.most_common` iterating the counter while another thread mutates it. The fail-open
  guard absorbs it, so the agent survives — but the behavior signal disappears for those steps
  and only `optio.internal.lane_errors` says why.

Both are contained because `add` and `classify` both run inside the lane's existing lock.
`TestTheBehaviorLaneLockIsLoadBearing` exists so that *narrowing* that lock — a reasonable-looking
change now that classify is O(1) — fails here rather than in someone's agent. Verified the usual
way: it goes red with the lock removed.

This is the argument for re-running the concurrency method after an optimization rather than
assuming the existing tests still cover it. The optimization was correct; it just moved where the
lock had to be.

## What is deliberately not tested

**Detector accuracy against real agent traffic.** The false-positive rate (0/1200) is measured
against *synthetic* healthy workloads — polling, paging, bounded retries, fan-out. Real traffic
will differ, and the number is published as a regression gate rather than a claim about your
agent.

**The judge's judgement.** We validate that scores are numeric and in range and drop what is
not. Whether the user's judge is any good is the "who evaluates the evaluator" problem
(R-TECH-5), and no test here can speak to it.

**Multi-process runs.** Not supported in 0.1 (ADR-005), so there is nothing to test yet.
