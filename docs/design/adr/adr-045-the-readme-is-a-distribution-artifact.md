# ADR-045 — The README is a distribution artifact, not a repo file

**Status:** Accepted
**Date:** 2026-08-02
**Related:** ADR-039 (evidence carries its date), ADR-040 (replace a lucky case with the property)

## Context

`pyproject.toml` sets `readme = "README.md"`, so that file is not documentation *about* the
distribution — it **is** part of it, rendered as the project description at
`pypi.org/project/optio`. Preparing 0.2.0, three faults were found in it, none caught by any
existing check:

**Relative links do not resolve on PyPI.** It renders the markdown against its own origin, so
`](docs/testing.md)` arrives as `pypi.org/project/optio/docs/testing.md` and 404s. Fifteen links
were relative and none were absolute. All fifteen targets exist on disk and work perfectly on
GitHub, which is exactly why nobody noticed.

**The version banner had gone a release behind, for the second time.** It read
`Status: alpha (0.1.0)` while the package said `0.2.0` — a wrong version on the first line of the
first page anyone sees. `61c31c2` had already fixed this once.

**A repository rename left 27 references behind.** `Agent-Meter` → `Optio` touched five files.
GitHub 301s renamed repositories, so nothing broke and nothing would have — until someone claims
the vacated name, at which point the landing page, the PyPI sidebar links from `pyproject`, and the
address in `SECURITY.md` all point at a stranger's repository.

`twine check --strict` passes throughout. It validates that the description *parses*; it has no
opinion on whether anything in it points anywhere.

## Decision

Treat the README as shipped output and test it like code:

- **No relative links.** They break specifically on the page users land on.
- **Every `blob/main` or `tree/main` link decomposes back to a path that must exist here.**
  Converting a link from relative to absolute moves the failure from "obviously broken" to "broken
  only when someone clicks it", and something has to hold the target.
- **Every GitHub URL across the five user-facing files names one repository.** Those five —
  README, `pyproject`, CHANGELOG, SECURITY, RELEASING — are the ones whose URLs reach people who
  cannot see this repo.
- **The banner version is checked against `pyproject`** rather than kept in step by hand.

## The guard failed the way it was written to catch

The link check hardcoded `Aniketh-74/Agent-Meter` in its `parametrize` set. The rename emptied that
set — and **pytest reports an empty parametrize as a skip, not a failure**. The suite stayed green
with the check completely inert, one commit after it was written.

So the pattern no longer names a repository, and a separate test asserts the parameter set is
non-empty. That second test is the general lesson: *"there was nothing to check"* and *"everything
checked out"* are indistinguishable in a green run, and only one of them is good news. Any
data-driven test needs something asserting its data exists.

## Consequences

2,204 tests pass. Each of the five checks was verified by reintroducing its fault and watching the
suite fail — including the empty-parameter case, which is how the inert guard was found at all.

**The general rule:** anything listed in `pyproject` is shipped. The README, the license, the
project URLs and the classifiers are all read by people who will never see this repository, and
they are the only part of the project most users ever look at. They deserve the same tests as the
code, and until 0.2.0 they had none.
