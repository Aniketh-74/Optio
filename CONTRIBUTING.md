# Contributing

[IMPLEMENTATION.md](IMPLEMENTATION.md) is the source of truth. Read the relevant section and ADRs before writing code. If a decision you need is missing, **stop and propose an ADR** — do not decide architecture inline (§16 rule 1).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

## The gates

Every one of these runs in CI and blocks merge. Run them locally first:

```bash
ruff check . && ruff format --check .   # lint + format
mypy                                    # --strict, configured in pyproject
pytest                                  # unit + contract
lint-imports                            # §3.1 layer boundaries
coverage report --fail-under=85         # after pytest --cov
```

## Rules that are not negotiable

These come from §16. They exist because the failure they prevent is *silent* — it passes review and shows up as a wrong number in production.

1. **Fail-open is inviolable.** Nothing in `runtime/` or `lanes/` may raise past `failopen.py`. A monitoring layer that can break the agent is worse than no monitoring layer (ADR-004). Any PR touching these paths must pass the fault-injection suite.
2. **Signal names come from `semconv.py` only** — never string literals. Adding or renaming a signal requires `docs/signals.md` updated, a contract test, and (if breaking) an ADR plus a version bump. Downstream OPA/Cedar/AGT policies are written against those exact strings.
3. **The ledger invariant is sacred.** Reserve precedes the step; reconcile replaces the reservation exactly once. A leak or a double-count is invisible without the property tests — keep them green.
4. **Lanes stay independent.** No lane imports another (`lint-imports` enforces it). This is what lets cost, behavior, and quality ship and fail separately.
5. **No architecture change without an ADR.** Superseding one is fine; silently diverging is not.
6. **Setup fails loudly, runtime fails open.** Config errors raise at `instrument()` time. Nothing raises on the hot path.
7. **No new dependency without justification** in the PR. Hot-path deps additionally need a benchmark within the 5 ms p99 budget. Framework deps go behind extras and are never imported at core import.

## Definition of done

A task is done when acceptance criteria are met, **tests are green, docs are updated, and it's within the performance budget** — not when the code works (§16 rule 7).

Per §9, a lane or adapter is not complete until its unit + contract + (for stateful components) property + fault-injection + benchmark tests pass. No signal counts as "emitted" until a contract test asserts its name and type against `docs/signals.md`.

## Commits

Small, incremental, one logical change each. The human reviewer's capacity is the bottleneck — optimize for reviewability, not for a tidy final diff (§16 rule 9).

## Tests

| Location | Purpose |
|---|---|
| `tests/unit/` | Pure logic, every module. |
| `tests/property/` | Hypothesis suites — ledger interleavings, window bounds. |
| `tests/contract/` | Signal names and types match the pinned semconv (blocking every PR). |
| `tests/integration/` | Real span flow, per adapter. |
| `tests/failinject/` | Proof the agent survives any internal failure. |
| `tests/bench/` | Per-step overhead against the SC-5 budget. |

Mark tests with the matching marker (`contract`, `property`, `failinject`, `bench`, `integration`) so CI can select them.
