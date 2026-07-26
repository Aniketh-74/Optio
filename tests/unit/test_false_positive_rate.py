"""Measured false-positive rate for the behavior detectors (Section 6.4).

Section 6.4 requires the FP rate to be a *published* metric. A number written
into the docs by hand would be worthless -- it would drift the moment a
threshold changed, and nobody would notice. So it is measured here, printed, and
gated: `docs/behavior.md` quotes what this suite produces.

The corpus is synthetic, and the docs say so. It is built from the healthy
patterns most likely to be misread as pathology -- polling, paged retrieval,
bounded retries, fan-out, and mixed work -- rather than from production traces
we do not have. That makes this a measurement of *known confusable shapes*, not
a population estimate. It is still the useful number: those shapes are where a
false positive would actually come from.

A false positive here is a healthy run classified as anything other than
`healthy`. Under Section 6.4 that is the error that matters, because a
downstream policy may kill the run -- turning our detection error into the
user's outage.
"""

from __future__ import annotations

import random
from typing import Final

import pytest

from agentmeter import semconv
from agentmeter.lanes.behavior.detectors import classify
from agentmeter.lanes.behavior.window import BehaviorWindow, StepSignature

#: Deterministic corpus: the published number must be reproducible.
SEED: Final = 20260727

#: Healthy runs generated per pattern.
RUNS_PER_PATTERN: Final = 200

#: The gate. Section 6.4 sets no numeric target, so this is our own commitment:
#: healthy runs of known-confusable shape are flagged at most this often.
MAX_FALSE_POSITIVE_RATE: Final = 0.01

TOOLS: Final = ["search", "read_file", "write_file", "summarise", "fetch", "plan", "verify"]


def _step(rng: random.Random, tool: str, args: str, *, errored: bool = False) -> StepSignature:
    return StepSignature(tool=tool, args_digest=args, errored=errored)


def _varied_work(rng: random.Random) -> list[StepSignature]:
    """An agent doing genuinely different things."""
    length = rng.randint(5, 60)
    return [_step(rng, rng.choice(TOOLS), f"d{n}") for n in range(length)]


def _paged_retrieval(rng: random.Random) -> list[StepSignature]:
    """One tool, many pages. Repetitive and entirely correct."""
    pages = rng.randint(5, 40)
    return [_step(rng, "fetch", f"page={n}") for n in range(pages)]


def _polling(rng: random.Random) -> list[StepSignature]:
    """Poll a job to completion, interleaved with real work."""
    steps: list[StepSignature] = []
    for n in range(rng.randint(5, 30)):
        steps.append(_step(rng, "check_status", f"job=1&t={n}"))
        if n % 3 == 0:
            steps.append(_step(rng, rng.choice(TOOLS), f"d{n}"))
    return steps


def _bounded_retries(rng: random.Random) -> list[StepSignature]:
    """Varied work where a minority of calls fail and are retried."""
    steps: list[StepSignature] = []
    for n in range(rng.randint(6, 40)):
        tool = rng.choice(TOOLS)
        if rng.random() < 0.15:
            steps.append(_step(rng, tool, f"d{n}", errored=True))
        steps.append(_step(rng, tool, f"d{n}"))
    return steps


def _fan_out(rng: random.Random) -> list[StepSignature]:
    """Same tool across many distinct inputs -- map over a list."""
    return [_step(rng, "summarise", f"doc={n}") for n in range(rng.randint(5, 50))]


def _mixed(rng: random.Random) -> list[StepSignature]:
    """A long run combining every healthy pattern above."""
    steps: list[StepSignature] = []
    for builder in (_varied_work, _paged_retrieval, _polling, _bounded_retries):
        steps.extend(builder(rng))
    rng.shuffle(steps)
    return steps


HEALTHY_PATTERNS: Final = {
    "varied work": _varied_work,
    "paged retrieval": _paged_retrieval,
    "polling": _polling,
    "bounded retries": _bounded_retries,
    "fan-out": _fan_out,
    "mixed": _mixed,
}


def _classify_run(steps: list[StepSignature], window_size: int = 50) -> str:
    """Run steps through a window the way the lane does, returning the verdict."""
    window = BehaviorWindow(window_size)
    for step in steps:
        window.add(step)
    return classify(window).state


def _false_positives(name: str, rng: random.Random) -> tuple[int, int]:
    """Return (flagged, total) for one healthy pattern."""
    builder = HEALTHY_PATTERNS[name]
    flagged = 0
    for _ in range(RUNS_PER_PATTERN):
        if _classify_run(builder(rng)) != semconv.LOOP_STATE_HEALTHY:
            flagged += 1
    return flagged, RUNS_PER_PATTERN


@pytest.mark.parametrize("pattern", sorted(HEALTHY_PATTERNS))
def test_each_healthy_pattern_stays_within_the_fp_budget(pattern: str) -> None:
    """No single healthy pattern may be systematically misread.

    Parametrised rather than aggregated so a pattern that is always flagged
    cannot hide behind five that never are.
    """
    flagged, total = _false_positives(pattern, random.Random(SEED))
    rate = flagged / total

    print(f"\n  {pattern:<18} FP {flagged}/{total} = {rate:.3%}")
    assert rate <= MAX_FALSE_POSITIVE_RATE, (
        f"healthy pattern {pattern!r} flagged {rate:.2%} of the time"
    )


def test_overall_false_positive_rate_is_published() -> None:
    """Measure and print the aggregate rate quoted in docs/behavior.md."""
    rng = random.Random(SEED)
    flagged = total = 0
    for pattern in sorted(HEALTHY_PATTERNS):
        pattern_flagged, pattern_total = _false_positives(pattern, rng)
        flagged += pattern_flagged
        total += pattern_total

    rate = flagged / total
    print(f"\nfalse-positive rate: {flagged}/{total} = {rate:.3%} healthy runs flagged")

    assert rate <= MAX_FALSE_POSITIVE_RATE


def test_the_detector_still_catches_real_pathologies() -> None:
    """The FP gate must not be satisfiable by never detecting anything.

    Without this, `return HEALTHY` would score a perfect zero.
    """
    rng = random.Random(SEED)

    stuck = [_step(rng, "search", "same") for _ in range(20)]
    cycle = [_step(rng, "read" if n % 2 else "think", "x") for n in range(20)]
    storm = [_step(rng, "call_api", "x", errored=True) for _ in range(20)]

    assert _classify_run(stuck) == semconv.LOOP_STATE_LOOPING
    assert _classify_run(cycle) == semconv.LOOP_STATE_LOOPING
    assert _classify_run(storm) == semconv.LOOP_STATE_RETRY_STORM


def test_detection_rate_on_pathological_runs_is_published() -> None:
    """The other half of the picture: how often real pathologies are caught."""
    rng = random.Random(SEED)
    caught = total = 0

    def stuck() -> list[StepSignature]:
        return [_step(rng, "stuck", "same") for _ in range(rng.randint(6, 40))]

    def cycle() -> list[StepSignature]:
        return [_step(rng, "a" if n % 2 else "b", "x") for n in range(rng.randint(6, 40))]

    def storm() -> list[StepSignature]:
        return [_step(rng, "api", "x", errored=True) for _ in range(rng.randint(6, 40))]

    for _ in range(RUNS_PER_PATTERN):
        for builder in (stuck, cycle, storm):
            total += 1
            if _classify_run(builder()) != semconv.LOOP_STATE_HEALTHY:
                caught += 1

    rate = caught / total
    print(f"\ndetection rate: {caught}/{total} = {rate:.1%} of pathological runs flagged")
    assert rate > 0.95
