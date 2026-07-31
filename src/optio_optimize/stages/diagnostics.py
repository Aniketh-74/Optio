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

from optio_optimize.stages.base import PREFIX_IS_UNSTABLE, Fidelity, Stage, StageResult

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
        self._observe(request)
        # Published for PrefixCacheStage, which otherwise pays a cache-write
        # premium on a prefix this stage has already established can never be
        # read back. Set only once the window supports the claim -- absence
        # means "not established", never "stable" (ADR-030).
        if self._system_is_unstable():
            ctx.scratch[PREFIX_IS_UNSTABLE] = True
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

    def _system_is_unstable(self) -> bool:
        """Whether the system prompt has been seen changing on nearly every call.

        The same test the ``unstable_system_prompt`` finding uses, named
        separately because it now has a second consumer that acts on it rather
        than reporting it.
        """
        return (
            len(self._system.seen) >= MIN_OBSERVATIONS
            and self._system.distinct_ratio > UNSTABLE_PREFIX_RATIO
        )

    def _diagnose(self) -> list[PrefixFinding]:
        """Findings supported by the observations so far."""
        found: list[PrefixFinding] = []

        count = len(self._system.seen)
        if self._system_is_unstable():
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


#: Fraction of the context window at which a prompt is reported as under
#: pressure. High on purpose: a warning that fires on ordinary traffic is one
#: people filter out, and the failure it predicts is still several turns away
#: at 90%.
PRESSURE_RATIO = 0.9


#: Characters attributed to one image when deciding whether a real count is
#: worth doing. Images cost roughly 1,600 tokens at Anthropic's cap (ADR-022)
#: and carry almost no characters, so counting them as characters would let an
#: image-heavy request slip past the guard. One token is at least one
#: character, so charging an image *more* characters than it can cost tokens
#: keeps the value an upper bound -- the direction that cannot hide a finding.
IMAGE_CHARS = 2_000


def _cheap_upper_bound(request: LLMRequest) -> int:
    """An upper bound on the request's tokens, computed without a tokenizer.

    Every token consumes at least one character, so a character count bounds a
    token count from above -- but only if it covers everything the real counter
    covers. This covers message text, tool schemas and images; ``count_request``
    counts all three.
    """
    from optio_optimize.wire import RAW_CONTENT_KEY, is_text_block

    total = sum(len(m.content) for m in request.messages)
    total += sum(len(json.dumps(tool, separators=(",", ":"))) for tool in request.tools)
    for message in request.messages:
        # Read through the same key and the same text/non-text test that
        # `message_image_tokens` uses. A second, private notion of "an image
        # block" here would drift from the real counter's, which is the defect
        # class this bound exists to avoid rather than to repeat.
        raw = message.extra.get(RAW_CONTENT_KEY)
        if not isinstance(raw, dict):
            continue
        blocks = raw.get("content")
        if isinstance(blocks, list):
            total += IMAGE_CHARS * sum(1 for block in blocks if not is_text_block(block))
    return total


class WindowPressureStage(Stage):
    """Report prompts approaching the limit that will reject them.

    A model's context window binds the **prompt**, and exceeding it is a hard
    400 before any generation::

        prompt is too long: 217570 tokens > 200000 maximum

    Two findings:

    ``prompt_exceeds_context_window``
        The prompt is already over. This request will be rejected and nothing in
        this package can rescue it -- trimming enough history to fit would be an
        ``ALTERED`` change to the caller's conversation that ADR-015 has no
        evidence for. Reporting it is the honest action.

    ``prompt_near_context_window``
        Within :data:`PRESSURE_RATIO` of the limit. Predicts the failure above
        while the caller can still act on it.

    **Says nothing about ``max_tokens``, and that is a measured decision.** The
    obvious design guards ``prompt + max_tokens`` against the window; the
    provider accepts that request. 158,965 prompt tokens plus a 21,000 ceiling
    against a 200,000 window generated normally, so a guard there would trade a
    real truncation for an error that does not occur (ADR-037).

    The limit comes from ``config.context_limit`` when the caller set it, then
    from :data:`~optio_optimize.config.CONTEXT_WINDOW`. When neither answers the
    stage is silent -- seven Anthropic models sit in exactly that position, with
    a window known only to exceed 217,554, and a guessed window would make this
    stage wrong on the largest models in the table.

    **Runs last**, unlike :class:`UnstablePrefixStage`, because it reports on
    the request that is actually about to be sent rather than on the one the
    caller assembled. Placed first it also cost 152 ms on a 2.7 MB conversation
    and starved the trim that would have fixed that request -- see
    :func:`~optio_optimize.stages.build_stages` and ADR-037.

    Like :class:`UnstablePrefixStage`, it changes nothing and claims nothing.
    **No prompt content is retained or reported** (§10): the findings carry
    token counts and a limit, never text.
    """

    fidelity = Fidelity.IDENTICAL

    def __init__(self) -> None:
        """Build the stage with nothing reported yet."""
        self._reported: set[str] = set()
        self.findings: list[PrefixFinding] = []

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "window_pressure"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Measure the prompt against the window. Always returns it unchanged."""
        for finding in self._diagnose(request, ctx):
            if finding.kind in self._reported:
                continue
            self._reported.add(finding.kind)
            self.findings.append(finding)
            _log.warning("optio_optimize: %s", finding.detail)
        # No note, for the reason UnstablePrefixStage gives: a note marks a
        # stage as having done something, and this one never does.
        return self.declines(request)

    def _diagnose(self, request: LLMRequest, ctx: StageContext) -> list[PrefixFinding]:
        """Findings this request supports, which is at most one."""
        from optio_optimize.config import context_window_for
        from optio_optimize.tokens import count_request, fits_in_window

        limit = ctx.config.context_limit or context_window_for(request.model)
        if limit is None:
            return []

        # A character count is a strict upper bound on a token count -- every
        # token consumes at least one character -- so a prompt with fewer
        # characters than the threshold cannot possibly reach it, and no
        # tokenizer needs to run. Exact, not heuristic: this skips work, it
        # never skips a finding.
        #
        # Worth the lines because the alternative is not cheap. Counting an
        # 81-turn, 2.7 MB conversation costs 152 ms, and this stage saves
        # nothing -- so on ordinary traffic it must cost nothing too. The
        # character sum is 0.01 ms on that same request.
        #
        # The bound has to cover everything `count_request` counts, and the
        # first version covered only messages. Tool schemas live in
        # `request.tools` and contribute nothing to any message's content, and
        # an image contributes ~1,600 tokens while `Message.content` holds only
        # extracted text (ADR-022) -- so both could carry a request over the
        # window while the guard, reading messages alone, waved it through.
        # That is the same flattering assumption that made the ADR-037 probe
        # bill $7.60, and it is a false negative in the one direction a
        # diagnostic must not have.
        if _cheap_upper_bound(request) < limit * PRESSURE_RATIO:
            return []

        tokens = count_request(request, ctx.counter)
        if not fits_in_window(tokens, limit, ctx.counter):
            # `fits_in_window` inflates an inexact count rather than trusting
            # it, and this stage inherits that asymmetry rather than
            # re-deciding it: warning once unnecessarily costs a log line,
            # while missing a real overflow costs the caller a crash.
            return [
                PrefixFinding(
                    kind="prompt_exceeds_context_window",
                    detail=(
                        f"the prompt measures {tokens:,} tokens against a context window of "
                        f"{limit:,}, so the provider will reject this request outright. No "
                        "stage here can shorten it without changing what you asked -- the fix "
                        "is in how the conversation is assembled."
                    ),
                    observations=1,
                )
            ]

        # Asked as "does it fit in 90% of the window?" rather than by comparing
        # against `limit * PRESSURE_RATIO` directly, so the estimator margin is
        # applied to both thresholds by the one function that owns it.
        #
        # Comparing raw tokens here made this finding **unreachable**: the
        # margin is 1.15 and 1/1.15 = 0.87, which is below PRESSURE_RATIO, so
        # every prompt large enough to be "near" had already been inflated past
        # the hard limit and reported as exceeding. A test caught it, which is
        # the second time in this module a threshold written one way could not
        # detect the thing it was written for (see the tool-ordering comment
        # above).
        if not fits_in_window(tokens, int(limit * PRESSURE_RATIO), ctx.counter):
            return [
                PrefixFinding(
                    kind="prompt_near_context_window",
                    detail=(
                        f"the prompt measures {tokens:,} tokens against a context window of "
                        f"{limit:,} -- {tokens / limit:.0%} of it. A few more turns will be "
                        "rejected outright rather than truncated."
                    ),
                    observations=1,
                )
            ]

        return []


__all__ = [
    "MIN_OBSERVATIONS",
    "PRESSURE_RATIO",
    "UNSTABLE_PREFIX_RATIO",
    "WINDOW",
    "PrefixFinding",
    "UnstablePrefixStage",
    "WindowPressureStage",
]
