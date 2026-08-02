# Releasing

Publishing is the one irreversible step in this project: a version number on PyPI can be
yanked but never reused. The workflow is built so every check happens *before* anything
leaves the machine, and the upload itself waits for a human.

## One-time setup

You only do this once, and only you can do it — it needs your PyPI account.

### 1. Reserve the name and configure trusted publishing

1. Create a PyPI account and enable 2FA.
2. Go to **[PyPI → Publishing](https://pypi.org/manage/account/publishing/)** and add a
   *pending* trusted publisher:

   | Field | Value |
   |---|---|
   | PyPI project name | `optio` |
   | Owner | `Aniketh-74` |
   | Repository name | `Optio` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   "Pending" means the project does not exist yet; the first successful publish creates it.

**Trusted publishing rather than an API token**, deliberately: there is no long-lived secret
stored in this repository, so there is nothing to leak, rotate, or accidentally print in a log.
The workflow mints a short-lived OIDC token scoped to this repo and workflow.

### 2. Create the `pypi` environment

In **Settings → Environments → New environment**, name it `pypi`, and add yourself as a
**required reviewer**.

This is what makes publishing a deliberate act. A tag runs the full gate, builds the artifacts,
generates the SBOM, and verifies the wheel installs — then stops and waits for you to click
approve. Without the environment the upload would happen automatically on any tag, including
one pushed by mistake.

## Cutting a release

1. **Update the version** in `pyproject.toml`. `__version__` reads from installed metadata, so
   there is only one place to change.

2. **Update `CHANGELOG.md`**: move `[Unreleased]` entries under the new version with today's
   date, and add the comparison links at the bottom.

3. **Commit and push to `main`.** Wait for CI to go green — the release workflow re-runs the
   gate, but finding a problem before tagging is cheaper than after.

4. **Tag and push:**

   ```bash
   git tag -a v0.1.0 -m "optio 0.1.0"
   git push origin v0.1.0
   ```

5. **Watch the Actions tab.** The workflow will:
   - run the full §17 gate (lint, types, imports, tests, coverage, the two 100% modules)
   - assert the tag, `pyproject.toml`, and `CHANGELOG.md` all agree
   - build the sdist and wheel, and `twine check --strict` them
   - generate a CycloneDX SBOM and verify it describes optio rather than the toolchain
   - install the built wheel into a clean venv and confirm it imports and works
   - **pause**, waiting for your approval on the `pypi` environment

6. **Approve.** The upload runs and prints the artifact hashes.

7. **Verify from the outside**, on a machine that has never seen the source:

   ```bash
   pip install optio
   python -c "import optio; print(optio.__version__)"
   ```

## Dry runs

Use **Actions → Release → Run workflow** with `dry_run` left checked. Everything runs except
the publish, which is the way to exercise a change to the release pipeline without spending a
version number.

## If something goes wrong

**Before approval:** nothing has been published. Delete the tag, fix, and start again.

```bash
git tag -d v0.1.0 && git push origin :refs/tags/v0.1.0
```

**After approval:** the version is gone forever. Yank it on PyPI (which hides it from resolvers
without breaking anyone who already pinned it) and release a patch. Do not try to reuse the
number — PyPI will refuse, and it should.

## Versioning

SemVer, with one project-specific rule: **the emitted signal names are part of the public API.**
Renaming or removing a `gen_ai.run.*` attribute is a breaking change even though no Python
signature moved, because downstream OPA, Cedar, and AGT policies are written against those
exact strings. A policy that silently stops matching is worse than one that fails to load, so
that change needs a major bump and an ADR (R-TECH-2).
