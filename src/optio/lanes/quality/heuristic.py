"""Cheap inline outcome checks (M5-2).

The free tier of the quality lane. Deterministic, no network, no model call, and
therefore safe to run on every run of an enabled lane (ADR-003).

## What this can and cannot tell you

These checks catch **obvious non-completion**: an empty answer, a truncated
generation, a run that ended in an error, a model that returned nothing but a
refusal stub. They do not judge whether an answer is *correct* -- that needs the
LLM-judge (M5-3), and even then only on sampled runs.

That limit is the whole reason the tier exists. Section 1.3 names "well-formed
but wrong" as the failure permission-based governance cannot see, and a
heuristic cannot see it either. Pretending otherwise would be worse than not
scoring: a confident `success=true` on a confidently-wrong run is exactly the
signal that would get a bad answer shipped.

So the heuristic is deliberately **one-sided**. It reports failure when it has
evidence of failure, and abstains otherwise. It never asserts success on its own
-- see :func:`score`.

## Privacy

Section 10 forbids emitting or logging prompt/completion content. This module
*reads* output to measure it, and retains nothing: it returns counts and
booleans, never substrings. The one number derived from content -- length -- is
compared against a threshold and discarded.

## The split

:func:`project` reads the span; :func:`score` reads the projection. They live
together because they have to agree on which fields matter -- the projection is
defined as *what the scorer reads*, and a field added to one without the other
is either dead weight on the wire or a check that silently stopped running.

The split is what lets a run be scored in a process that never saw its spans
(ADR-050). It also puts every guard against a malformed attribute at the
boundary where the attribute arrives, leaving :func:`score` as logic over types
that are already clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from opentelemetry.trace import StatusCode

from optio import semconv
from optio.lanes.quality.store import QualityStep

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

#: Below this many characters an answer is treated as non-substantive. Set low
#: on purpose: "yes" and "42" are legitimate complete answers to some questions,
#: so this only catches output that is empty or near-empty.
MIN_ANSWER_CHARS: Final = 2

#: Finish reasons that mean the model stopped before it was done. A truncated
#: answer is a failed task even when the text reads fine up to the cut.
_INCOMPLETE_FINISH_REASONS: Final[frozenset[str]] = frozenset(
    {"length", "max_tokens", "content_filter"}
)


@dataclass(frozen=True, slots=True)
class HeuristicResult:
    """What the inline checks concluded.

    Attributes:
        failed: ``True`` when there is positive evidence the run did not
            complete. ``False`` means "no evidence of failure", which is *not*
            the same as success -- see :attr:`conclusive`.
        conclusive: Whether the checks had enough information to judge at all.
            ``False`` when no output span was observed.
        reason: Short machine-readable cause, or ``None``. Never contains run
            content (Section 10).
    """

    failed: bool
    conclusive: bool
    reason: str | None = None

    @property
    def succeeded(self) -> bool | None:
        """Tri-state outcome: ``True``, ``False``, or ``None`` for unknown.

        The heuristic never returns ``True`` here on its own -- absence of
        evidence of failure is not evidence of success, and a run that produced
        a fluent wrong answer looks identical to one that produced a fluent
        right answer from the outside.
        """
        if not self.conclusive:
            return None
        return False if self.failed else None


#: The verdict when nothing was observed.
UNKNOWN: Final = HeuristicResult(failed=False, conclusive=False, reason=None)


def project(span: ReadableSpan) -> QualityStep:
    """Reduce a finished span to the fields :func:`score` reads.

    The boundary where an attribute of unknown shape becomes a typed value. A
    framework can put anything in an attribute, so every guard lives here and
    every unreadable value degrades to "not reported" rather than to a guess --
    absence is unknown, not zero.

    Args:
        span: The span that just finished.

    Returns:
        Its projection. Carries no run content (Section 10): a boolean, a tuple
        of provider finish-reason tokens, and a token count.
    """
    attributes = span.attributes or {}
    return QualityStep(
        errored=_errored(span),
        finish_reasons=_finish_reasons(attributes.get(semconv.GEN_AI_RESPONSE_FINISH_REASONS)),
        output_tokens=_output_tokens(attributes.get(semconv.GEN_AI_USAGE_OUTPUT_TOKENS)),
    )


def score(step: QualityStep | None) -> HeuristicResult:
    """Run the inline checks over a completed run's final step.

    The final step, because that is where a run's answer is. An error the agent
    hit mid-run and recovered from is not a failed task, so which step this is
    handed matters as much as what it checks -- the store keeps the last one.

    Args:
        step: The run's last step, or ``None`` when it had none.

    Returns:
        The verdict. ``UNKNOWN`` when the run produced nothing to judge.
    """
    if step is None:
        return UNKNOWN

    if step.errored:
        return HeuristicResult(failed=True, conclusive=True, reason="final step errored")

    truncated = _incomplete_finish_reason(step.finish_reasons)
    if truncated is not None:
        return HeuristicResult(
            failed=True, conclusive=True, reason=f"incomplete generation ({truncated})"
        )

    if step.output_tokens is not None:
        if step.output_tokens <= 0:
            return HeuristicResult(failed=True, conclusive=True, reason="no output produced")
        # Positive token count is evidence the model answered, but says nothing
        # about whether the answer was right. Conclusive that it ran; not
        # conclusive that it succeeded.
        return HeuristicResult(failed=False, conclusive=True, reason=None)

    # No usage attribute at all: the framework did not report one. Absence is
    # unknown, not zero (docs/signals.md) -- reporting failure here would flag
    # every run from an instrumentation that omits token counts.
    return UNKNOWN


def _finish_reasons(value: object) -> tuple[str, ...]:
    """Normalise the finish-reasons attribute to a tuple of strings.

    The upstream attribute is array-valued (one entry per choice), but several
    instrumentations flatten it to a bare string. Both are accepted: rejecting
    the flattened form would silently disable truncation detection for those
    users, and a missing check is invisible in a way a type error is not.

    Args:
        value: The raw attribute value, of unknown shape.

    Returns:
        The reasons, or empty when the attribute was absent or unreadable.
        Non-string entries are dropped rather than coerced -- a reason that is
        not a string is not a reason.
    """
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _output_tokens(value: object) -> int | None:
    """Read a reported output-token count, or ``None``.

    Args:
        value: The raw attribute value, of unknown shape.

    Returns:
        The count, or ``None`` when it was absent or not a genuine integer.
        ``bool`` is excluded explicitly: it is a subclass of ``int``, so ``True``
        would otherwise be read as "1 token" and turn an instrumentation quirk
        into evidence that the model answered.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _incomplete_finish_reason(reasons: tuple[str, ...]) -> str | None:
    """Return the finish reason indicating truncation, if any.

    Args:
        reasons: Normalised finish reasons.

    Returns:
        The offending reason, or ``None`` when the generation was not truncated.
    """
    for reason in reasons:
        if reason.lower() in _INCOMPLETE_FINISH_REASONS:
            return reason
    return None


def _errored(span: ReadableSpan) -> bool:
    """Whether a span carries an error status.

    Args:
        span: The span to check.

    Returns:
        ``True`` when the span's status is ERROR.
    """
    status = getattr(span, "status", None)
    code = getattr(status, "status_code", None)
    return code is StatusCode.ERROR
