# ADR-012 — The public API is the top-level package only

**Status:** Accepted
**Date:** 2026-07-28
**Related:** ADR-002, ADR-008, §8.1, §16 rule 12, CHANGELOG

## Context

The CHANGELOG commits this project to Semantic Versioning and names "the public API (§8.1) and
the emitted signal names (§7.2)" as the compatibility surface. §8.1 lists five names:
`instrument`, `meter`, `RunContext`, `Config`, and (with `BudgetPolicy` and
`GENAI_SEMCONV_VERSION` alongside them) that is what `optio/__init__.py` exports.

What no document states is what happens to everything *else*. The package ships 31 submodules,
all importable, several with obviously useful contents — `CostLedger`, `BehaviorWindow`,
`cost_of`, `classify`. Nothing marks them private: no leading underscore on the module names,
full docstrings, complete type annotations, and `py.typed` telling a type checker they are
supported.

A user who writes `from optio.lanes.cost.pricing import cost_of` is behaving reasonably. They
will discover it in the source, it will work, it will type-check, and their editor will
autocomplete it. When a minor release moves it, SemVer says their build should not have broken —
and by any reading available to them at the time, they were right.

This is the ordinary way a library loses the ability to change its own internals: not by a
decision to make them public, but by never saying they were not.

The obvious mechanism is `__all__` on every module. It is the wrong one here. `__all__` is a
statement about a module's *exports*, and adding it to `optio.lanes.cost.ledger` reads as
"these are the supported names in this module" — which advertises the module as an integration
point rather than denying it. Renaming every internal module to `_lanes`, `_runtime`, `_store`
would be unambiguous, but it makes tracebacks, profiles, and the import-linter contracts that
enforce §3.1 harder to read, for a signal a sentence can carry.

## Decision

**Only names exported from the top-level `optio` package are public. Everything under
`optio.*` submodules is internal and may change in any release, including a patch.**

Concretely:

* The public API is exactly `optio.__all__`. It is enforced by
  `tests/unit/test_public_api.py`, which already fails when a name appears in the package
  namespace without being declared — that test is what caught `version` and
  `PackageNotFoundError` leaking out of an `importlib.metadata` import during the rename, and
  `TYPE_CHECKING` leaking during the lazy-version change.
* Submodules keep their unprefixed names and their docstrings. They are documented because
  the people reading them are contributors and reviewers, and because §9 makes tests the
  primary defence — undocumented internals are harder to review, and review is the control.
* The boundary is stated where a user will actually meet it: README, CHANGELOG, and the
  module docstring of `optio/__init__.py`.
* A test asserts the promise is honoured in the direction that matters — that the documented
  names really are importable and really are what `__all__` says — so the guarantee cannot
  quietly become false.

The signal names (§7.2) remain the *other* half of the compatibility surface and are
unaffected: they are a stronger promise than the Python API, since a downstream Rego or Cedar
policy matching on `gen_ai.run.projected_cost` breaks silently rather than loudly (ADR-002).

## Consequences

**We can refactor the internals.** The two optimizations that prompted writing this ADR both
changed internal signatures — `BehaviorWindow` gained `call_counts` and `error_count`,
`CostLane._forward_signals` gained a parameter. Under an unstated boundary, each is arguably a
breaking change to somebody's import. Under this one, neither is.

**Users who need an internal have a documented path.** Opening an issue is the intended
response, not a monkeypatch. If a use case is real, the name can be promoted to the top level
deliberately — which is a decision with an ADR behind it rather than an accident of what
happened to be importable.

**A dependency on an internal will still happen.** Someone will import `CostLedger` regardless.
The difference is that this ADR makes that their risk rather than our constraint, and makes the
answer to "you broke my build" a citation rather than an argument.

**`py.typed` still covers everything**, which is mildly in tension with the boundary: a type
checker will happily resolve `optio.lanes.cost.ledger.CostLedger`. Splitting type coverage to
match the API boundary is possible but would mean shipping stubs that disagree with the source,
and losing type checking on the internals for contributors. Not worth it — the marker means
"this package ships types", not "every typed name is stable".

## Alternatives considered

**`__all__` on every module.** Rejected above: it advertises submodules as export surfaces,
which is the opposite of the intent.

**Underscore-prefix every internal package** (`optio._lanes`, `optio._runtime`). Unambiguous
and self-enforcing, and genuinely tempting. Rejected because it degrades every traceback,
profile, and import-linter contract in the project for a signal that prose plus a test carries
adequately — and because the rename would touch every file for zero behavioural change, which
§16 rule 10's spirit argues against.

**A `optio.internal` namespace** holding everything non-public. Same enforcement benefit as
underscores with the same churn, plus it flattens the §3.1 lane structure that the import
boundaries depend on.
