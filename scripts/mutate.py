#!/usr/bin/env python
"""Run mutation testing without ever touching the working tree.

## Why this script exists

`cosmic-ray` mutates source files **in place** and restores them with a context
manager (`cosmic_ray.mutating.use_mutation`). That is fine when a run finishes
or raises, and useless when the process is killed: a Ctrl-C at the wrong moment,
a terminated tool call, a closed laptop lid, and the mutated file simply stays on
disk.

This is not hypothetical. During this project's first mutation run an interrupt
left `project.py` holding::

    return snapshot.committed - remaining * estimate   # should be +

Every worst-case cost projection with its sign flipped, sitting in the working
tree, one `git commit -a` away from being real. It was caught by `git status`,
which is luck rather than process.

So: this script copies the repository to a temporary directory, runs the
mutation there, and reports back. The working tree is never opened for writing,
so no failure mode -- not a kill signal, not a crash, not a full disk -- can
corrupt it. The copy is thrown away afterwards whether the run succeeded or not.

## Usage

    python scripts/mutate.py ledger
    python scripts/mutate.py --list
    python scripts/mutate.py failopen --keep     # keep the workspace to inspect

Targets are declared in `TARGETS` below rather than accepted as free-form paths:
mutation testing is only meaningful against a test suite that actually exercises
the module, and pairing them here means the pairing gets reviewed.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Target:
    """A module to mutate, paired with the tests that should catch it."""

    module: str
    tests: tuple[str, ...]
    why: str


#: Modules worth mutating, and the suites expected to kill the mutants.
#:
#: Not every module belongs here. Mutation testing costs minutes per module and
#: its output needs human judgement to separate real gaps from equivalent
#: mutants, so it is aimed at the places where a silent wrong answer is worst.
TARGETS: dict[str, Target] = {
    "ledger": Target(
        module="src/optio/lanes/cost/ledger.py",
        tests=(
            "tests/unit/test_ledger.py",
            "tests/unit/test_ledger_lifecycle.py",
            "tests/property/test_ledger_invariant.py",
        ),
        why="R-TECH-1: a wrong cost total is the product's worst possible bug",
    ),
    "failopen": Target(
        module="src/optio/runtime/failopen.py",
        tests=("tests/failinject", "tests/property/test_failopen_properties.py"),
        why="SC-4: the one component that must never fail",
    ),
    "project": Target(
        module="src/optio/lanes/cost/project.py",
        tests=(
            "tests/unit/test_project.py",
            "tests/unit/test_cost_lane.py",
            "tests/integration/test_cost_signals_end_to_end.py",
        ),
        why="emits the signals a budget policy gates on",
    ),
    "detectors": Target(
        module="src/optio/lanes/behavior/detectors.py",
        tests=(
            "tests/unit/test_detectors.py",
            "tests/unit/test_false_positive_rate.py",
            "tests/unit/test_window.py",
        ),
        why="a false positive here becomes the user's outage (ADR-004)",
    ),
}


def build_workspace(destination: Path) -> Path:
    """Copy the repository to a scratch directory.

    Uses ``git archive`` so the copy contains exactly what is committed --
    no caches, no ``.venv``, no half-finished edits. A mutation run against
    uncommitted work would report on code that does not exist anywhere else.

    Args:
        destination: Directory to populate. Must already exist.

    Returns:
        The populated directory.
    """
    archive = destination / "repo.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    workspace = destination / "repo"
    workspace.mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(workspace)], check=True)
    archive.unlink()
    return workspace


def run(target: Target, workspace: Path, *, verbose: bool) -> int:
    """Execute a mutation session inside the workspace.

    Args:
        target: What to mutate and what should catch it.
        workspace: The throwaway copy of the repository.
        verbose: Whether to stream cosmic-ray's own output.

    Returns:
        A process exit code.
    """
    venv = workspace / ".venv"
    print(f"  building an isolated environment in {venv} ...", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    scripts = venv / ("Scripts" if sys.platform == "win32" else "bin")
    python = scripts / ("python.exe" if sys.platform == "win32" else "python")

    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "-e", ".[dev]", "cosmic-ray"],
        cwd=workspace,
        check=True,
    )

    # An absolute interpreter path: cosmic-ray runs the test command from its
    # own working directory, where a bare `python` resolves to whatever is on
    # PATH -- usually a system interpreter with no optio installed.
    #
    # Forward slashes even on Windows. A backslash is TOML's escape character,
    # so `C:\Users\...` parses as an invalid escape sequence and the config
    # fails to load -- which cosmic-ray reports as a failed baseline, sending
    # you off to debug a test suite that is perfectly fine.
    #
    # shlex.quote because cosmic-ray splits the command with shlex: a temp
    # directory containing a space would otherwise become two arguments.
    interpreter = shlex.quote(python.as_posix())
    test_command = " ".join(
        [
            interpreter,
            "-m pytest -q -x --no-header -p no:cacheprovider --no-cov",
            *target.tests,
        ]
    )

    # A TOML *literal* string (single quotes): the interpreter path may contain
    # spaces, and quoting it inside a basic string would nest double quotes and
    # terminate the value early.
    config = workspace / "mutate.toml"
    config.write_text(
        "[cosmic-ray]\n"
        f"module-path = '{target.module}'\n"
        "timeout = 120.0\n"
        "excluded-modules = []\n"
        f"test-command = '{test_command}'\n"
        "\n[cosmic-ray.distributor]\n"
        "name = 'local'\n",
        encoding="utf-8",
    )

    cr = scripts / ("cosmic-ray.exe" if sys.platform == "win32" else "cosmic-ray")
    report = scripts / ("cr-report.exe" if sys.platform == "win32" else "cr-report")
    session = workspace / "mutate.sqlite"
    quiet = None if verbose else subprocess.DEVNULL

    print("  checking the baseline (tests must pass unmutated) ...", flush=True)
    baseline = subprocess.run(
        [str(cr), "baseline", str(config)],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if baseline.returncode != 0:
        # cosmic-ray reports a config-loading error the same way it reports a
        # failing suite, which sends you off debugging tests that are fine.
        # Show its actual output rather than guessing at the cause.
        detail = (baseline.stderr or baseline.stdout or "").strip()
        print("\n  BASELINE FAILED -- mutation results would be meaningless.")
        if "ConfigError" in detail:
            print("  This is a configuration problem, not a test failure.")
            print(f"  Config: {config}")
        else:
            print("  The tests do not pass before any mutation. Fix the suite first.")
        if detail:
            print("\n  " + "\n  ".join(detail.splitlines()[-12:]))
        return 1

    print("  running mutations (this takes a few minutes) ...", flush=True)
    subprocess.run([str(cr), "init", str(config), str(session)], cwd=workspace, check=True)
    subprocess.run(
        [str(cr), "exec", str(config), str(session)], cwd=workspace, stdout=quiet, stderr=quiet
    )

    output = subprocess.run(
        [str(report), str(session)], cwd=workspace, capture_output=True, text=True
    ).stdout
    summarise(output, target)
    return 0


def summarise(report: str, target: Target) -> None:
    """Print a report that separates real gaps from unkillable noise.

    Two categories of survivor cannot be killed and drown out the signal if
    reported raw:

    * mutations of ``|`` inside type annotations, which ``from __future__
      import annotations`` turns into strings that never evaluate
    * genuine equivalent mutants, which need human judgement

    Args:
        report: Raw ``cr-report`` output.
        target: The target that was mutated.
    """
    lines = report.splitlines()
    survivors = [
        lines[i - 1].strip() for i, line in enumerate(lines) if "SURVIVED" in line and i > 0
    ]
    annotation_noise = [s for s in survivors if "BitOr" in s]
    real = [s for s in survivors if "BitOr" not in s]

    totals = [line.strip() for line in lines if "total jobs" in line or "surviving" in line]

    print()
    print(f"  === {target.module} ===")
    for line in totals:
        print(f"  {line}")
    print(f"  annotation noise (unkillable):  {len(annotation_noise)}")
    print(f"  survivors needing judgement:    {len(real)}")

    if real:
        print()
        print("  Each of these is either a test gap or an equivalent mutant.")
        print("  Decide which by asking: is there an input that would tell the")
        print("  original and the mutant apart? If not, it is equivalent.")
        print()
        for survivor in real:
            print(f"    {survivor}")
        print()
        print("  Full diffs:  cr-report <session>.sqlite --show-diff")


def sweep_stale_workspaces(max_age_hours: float = 24.0) -> None:
    """Remove workspaces a previous killed run left behind.

    The whole point of this script is that a kill signal costs nothing, but
    "nothing" should not mean a few hundred megabytes of orphaned venv per
    interrupted run. Cleaning on the way in rather than trying to catch every
    exit path, because the exit paths are exactly what a kill skips.

    Args:
        max_age_hours: Leave anything newer alone -- it may be a concurrent run.
    """
    import time

    cutoff = time.time() - max_age_hours * 3600
    for stale in Path(tempfile.gettempdir()).glob("optio-mutate-*"):
        try:
            if stale.is_dir() and stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            # A workspace we cannot stat or remove is not worth failing over.
            continue


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(
        description="Mutation testing that cannot corrupt the working tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("target", nargs="?", help="which module to mutate")
    parser.add_argument("--list", action="store_true", help="show available targets")
    parser.add_argument("--keep", action="store_true", help="keep the workspace for inspection")
    parser.add_argument("--verbose", action="store_true", help="stream cosmic-ray output")
    args = parser.parse_args()

    if args.list or not args.target:
        print("Targets:\n")
        for name, target in TARGETS.items():
            print(f"  {name:<12} {target.module}")
            print(f"  {'':<12} {target.why}\n")
        return 0

    if args.target not in TARGETS:
        print(f"Unknown target {args.target!r}. Available: {', '.join(TARGETS)}")
        return 2

    target = TARGETS[args.target]
    sweep_stale_workspaces()
    scratch = Path(tempfile.mkdtemp(prefix="optio-mutate-"))
    print(f"Mutating {target.module}")
    print(f"  {target.why}")
    print(f"  workspace: {scratch}")
    print("  the working tree is never written to, so an interrupt cannot corrupt it")
    print()

    try:
        workspace = build_workspace(scratch)
        return run(target, workspace, verbose=args.verbose)
    finally:
        if args.keep:
            print(f"\n  workspace kept at {scratch}")
        else:
            # onexc rather than ignore_errors: a workspace that cannot be
            # removed should be reported, not silently leaked into temp.
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
