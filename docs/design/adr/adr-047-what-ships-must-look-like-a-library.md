# ADR-047 — What ships must look like a library to the tools that consume it

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-012 (the public API is the top-level package only),
ADR-045 (the README is a distribution artifact)

## Context

`src/optio/py.typed` existed. `src/optio_optimize/py.typed` did not.

Both packages are declared in `[tool.hatch.build.targets.wheel]`, the whole tree is `mypy --strict`
clean, and `pyproject` advertises `Typing :: Typed`. So under PEP 561 a consumer running a type
checker against `optio_optimize` had **every annotation in 47 modules silently ignored**, with
nothing anywhere to explain why.

Nothing in this repository could have noticed. Our own `mypy` run reads the *source tree*, where the
annotations are plainly present. The marker only matters once the package is installed from a wheel
— the one place we never looked.

That is the general shape: the wheel is the only artifact most users ever see, and several of its
properties are invisible from inside the repo. A file present in `src/` is not evidence that it ships.

## Decision

Test the built wheel, not the source tree:

- **Every package `pyproject` ships must carry `py.typed`.** Without it, annotations are invisible
  and no error says so.
- **Every package it declares must actually be in the artifact** — the packaging mistake an editable
  install cannot show you, because an editable install resolves from `src/` regardless.
- **The wheel carries nothing but the library.** The sdist ships tests and docs deliberately, so a
  downstream packager can verify what they are shipping; `site-packages` is not the place for either.

The package list is parsed out of `pyproject` rather than hardcoded, so a third package added later
is covered automatically rather than quietly exempt. A separate test asserts that list is non-empty,
for the reason ADR-045 records: an empty `parametrize` is a skip, and a skip is not a pass.

## Consequences

The wheel-building fixture skips rather than fails where `build` is unavailable — the sdist
verification job runs the suite from an extracted tarball in a bare venv, and a test that cannot
build a wheel there has nothing to say rather than something to fail about.

**The general rule, stated once:** anything listed in `pyproject` is shipped, and shipped things are
read by people who will never see this repository. `py.typed`, the README, the license, the project
URLs and the classifiers are all part of the product. Until 0.2.0 none of them had a test.
