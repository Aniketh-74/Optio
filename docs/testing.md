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
| **Mutation** | do the tests detect an injected bug? | periodic, see below |

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

`cosmic-ray` rather than `mutmut`, because mutmut does not support Windows.

```bash
pip install cosmic-ray
cosmic-ray init cr-ledger.toml crl.sqlite
cosmic-ray exec cr-ledger.toml crl.sqlite
cr-report crl.sqlite
```

**Run it against a throwaway clone, never your working tree.** cosmic-ray mutates files in
place and restores them on exit; if it is interrupted it leaves corrupted source behind. That
happened during this project's first run and left `committed - remaining * estimate` in
`project.py` — the projection with its sign flipped.

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

## What is deliberately not tested

**Detector accuracy against real agent traffic.** The false-positive rate (0/1200) is measured
against *synthetic* healthy workloads — polling, paging, bounded retries, fan-out. Real traffic
will differ, and the number is published as a regression gate rather than a claim about your
agent.

**The judge's judgement.** We validate that scores are numeric and in range and drop what is
not. Whether the user's judge is any good is the "who evaluates the evaluator" problem
(R-TECH-5), and no test here can speak to it.

**Multi-process runs.** Not supported in 0.1 (ADR-005), so there is nothing to test yet.
