"""Detecting the caching bug that leaves no trace.

Provider prefix caching is worth 90% off the tokens it covers and requires only
that the head of the prompt be **byte-identical** between calls. That is a
fragile requirement to meet by accident and an easy one to break silently:
inject a timestamp into the system prompt, serialize a tool list from a set, or
splice a user's name in above the instructions, and the hit rate goes to zero
while every test still passes and every response still looks right. The bill
goes up by roughly an order of magnitude on the affected tokens and nothing
anywhere reports an error.

This module is the only thing in the package that **changes nothing**. It reads
each request, compares it against what it has seen before, and says so when the
comparison shows a prefix being broken. That posture is deliberate: the fix is
always in the caller's prompt-assembly code, not in the request, and a stage
that "helpfully" reordered a tool list or stripped a timestamp would be
rewriting application logic it does not understand.

**No prompt content is retained, ever** (§10). Findings are computed from
16-byte digests, and the warnings name the *shape* of the problem -- "your
system prompt differs on nearly every call" -- never the text that differs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from optio_optimize.stages.base import Fidelity, Stage, StageResult

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest

_log = logging.getLogger("optio_optimize")

#: Requests to observe before any finding is reported. Below this the ratios
#: are noise: three requests with three distinct system prompts is equally
#: consistent with a bug and with an agent that has only just started.
MIN_OBSERVATIONS = 10

#: Distinct-prefix ratio above which the system prompt is judged unstable.
#: Deliberately high. A service handling several features or tenants through
#: one optimizer legitimately sees a handful of distinct system prompts, and
#: that alternation is fine -- each one caches separately. The failure this
#: catches is different in kind: a prompt that is *never* the same twice,
#: which is what a timestamp or a request id at the top produces.
UNSTABLE_PREFIX_RATIO = 0.8

#: Observations retained. Bounded because this runs in a long-lived agent
#: process; the ratio is computed over a window rather than over all time, so a
#: prompt fixed mid-run stops being reported rather than being tarred forever.
WINDOW = 128


def _digest(text: str) -> str:
    """A 16-byte digest. Never reversible to content, which is the point."""
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


@dataclass(slots=True)
class PrefixFinding:
    """One diagnosis about why prefix caching is not paying.

    Attributes:
        kind: Machine-readable identifier, stable across releases.
        detail: One line a human can act on. Contains no prompt content.
        observations: Requests the finding is based on.
    """

    kind: str
    detail: str
    observations: int


@dataclass(slots=True)
class _Window:
    """Bounded history of digests for one aspect of the prefix."""

    seen: list[str] = field(default_factory=list)

    def record(self, digest: str) -> None:
        """Add an observation, discarding the oldest beyond :data:`WINDOW`."""
        self.seen.append(digest)
        if len(self.seen) > WINDOW:
            del self.seen[0]

    @property
    def distinct_ratio(self) -> float:
        """Distinct digests over total observations, ``0.0`` when empty."""
        return len(set(self.seen)) / len(self.seen) if self.seen else 0.0


class UnstablePrefixStage(Stage):
    """Report prompts whose cacheable head changes when it should not.

    Two findings, each with a specific and different fix:

    ``unstable_system_prompt``
        The first message differs on nearly every request. Almost always an
        interpolated timestamp, request id, or per-user detail sitting above
        the instructions. The fix is to move the varying part *below* the
        stable block, not to remove it.

    ``unstable_tool_order``
        The same tools arrive in a different order between calls -- the
        signature of a schema list built from a ``set`` or a ``dict`` whose
        iteration order is not pinned. Detected by comparing each request's
        tool list against its own sorted form: when the sorted digest never
        varies and the unsorted one does, the tools are identical and only
        their order moved. That comparison is what makes this finding safe to
        state plainly, since it distinguishes "your ordering is unstable" from
        "you are genuinely sending different tools" -- and the second is a
        legitimate design this stage must not scold anyone for.

    Reports zero savings and never modifies a request, so it is
    ``Fidelity.IDENTICAL`` trivially: it is not an optimization, it is the
    thing that tells you why your optimizations are not working.

    Each finding is logged once per stage instance. A warning that repeats on
    every request through a hot loop is one people filter out, which would
    defeat the purpose.
    """

    fidelity = Fidelity.IDENTICAL

    def __init__(self) -> None:
        """Build the stage with empty observation windows."""
        self._system = _Window()
        self._tools_ordered = _Window()
        self._tools_sorted = _Window()
        self._reported: set[str] = set()
        self.findings: list[PrefixFinding] = []

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "unstable_prefix"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Observe the request's prefix. Always returns it unchanged."""
        del ctx
        self._observe(request)
        for finding in self._diagnose():
            if finding.kind in self._reported:
                continue
            self._reported.add(finding.kind)
            self.findings.append(finding)
            _log.warning("optio_optimize: %s", finding.detail)
        # Deliberately no note: a note marks a stage as having *done* something,
        # and this one never does. Its output is the finding, not a saving.
        return self.declines(request)

    def _observe(self, request: LLMRequest) -> None:
        """Record digests for this request's cacheable head."""
        if request.messages and request.messages[0].role == "system":
            self._system.record(_digest(request.messages[0].content))
        if request.tools:
            rendered = [json.dumps(t, sort_keys=True, separators=(",", ":")) for t in request.tools]
            self._tools_ordered.record(_digest("".join(rendered)))
            self._tools_sorted.record(_digest("".join(sorted(rendered))))

    def _diagnose(self) -> list[PrefixFinding]:
        """Findings supported by the observations so far."""
        found: list[PrefixFinding] = []

        count = len(self._system.seen)
        if count >= MIN_OBSERVATIONS and self._system.distinct_ratio > UNSTABLE_PREFIX_RATIO:
            found.append(
                PrefixFinding(
                    kind="unstable_system_prompt",
                    detail=(
                        f"the system prompt differed on {self._system.distinct_ratio:.0%} of the "
                        f"last {count} requests, so no provider prefix cache can hit. Something "
                        "varying -- a timestamp, request id, or per-user detail -- is above the "
                        "stable instructions. Moving it below them restores the discount."
                    ),
                    observations=count,
                )
            )

        # Tool ordering is *not* judged by ratio, unlike the system prompt.
        # There are only n! orderings of n tools, so with a small tool set a
        # genuinely random ordering bug repeats itself constantly: five tools
        # permuted at random across twenty requests can easily show a distinct
        # ratio of 0.25, far under any sensible threshold. The first version of
        # this check used the ratio and could not detect the very bug it was
        # written for -- a test caught it.
        #
        # The correct test is exact rather than statistical, because unstable
        # ordering is never something a caller wants: if the tool *set* never
        # changed and the order did, that is the bug, at any frequency.
        orderings = len(set(self._tools_ordered.seen))
        tool_count = len(self._tools_ordered.seen)
        if (
            tool_count >= MIN_OBSERVATIONS
            and len(set(self._tools_sorted.seen)) == 1
            and orderings > 1
        ):
            found.append(
                PrefixFinding(
                    kind="unstable_tool_order",
                    detail=(
                        f"one unchanging set of tool schemas arrived in {orderings} different "
                        f"orders across {tool_count} requests. The tools themselves never "
                        "changed, so this is an unordered container -- a set, or a dict whose "
                        "iteration order is not pinned -- being serialized. Sorting the list "
                        "once makes the whole schema block cacheable."
                    ),
                    observations=tool_count,
                )
            )

        return found


__all__ = [
    "MIN_OBSERVATIONS",
    "UNSTABLE_PREFIX_RATIO",
    "WINDOW",
    "PrefixFinding",
    "UnstablePrefixStage",
]
