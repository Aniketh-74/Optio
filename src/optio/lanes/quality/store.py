"""What a quality backend must do, and the values it carries.

Follows ADR-050: the store speaks the domain rather than offering primitives,
so each backend keeps its own promises where the check and the mutation happen
together.

**A ``ReadableSpan`` never reaches a store.** Spans are not serializable, and
the lane's buffer of them was the reason the quality lane could not be shared
across processes at all. :class:`QualityStep` is the projection that replaces
it, and it is *small* -- three fields, derived from what the run-end consumers
provably read rather than from what a span happens to offer.

That derivation is the part worth stating, because the design spec guessed it
wrong. It supposed the retained spans were read for their count and by the tier
decision, so the projection would need the span name, its timestamps and the
attributes that decision consults. In fact ``sampling.decide`` takes the run and
the config and no spans at all, and the only consumer is
:func:`~optio.lanes.quality.heuristic.score`, which reads **the last span
alone**. So a store never needs a list: a counter and one projected step answer
every question the lane asks at run end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class QualityStep:
    """The serializable projection of one finished step.

    Exactly the fields :func:`~optio.lanes.quality.heuristic.score` reads, and
    no others. Adding one is a deliberate act rather than a discovery made in
    production, which is why the projection is pinned by its own test.

    Attributes:
        errored: Whether the step's span carried an ``ERROR`` status.
        finish_reasons: Normalised ``gen_ai.response.finish_reasons``. Upstream
            types this as an array and several instrumentations flatten it to a
            bare string; both arrive here as a tuple, and an unreadable value as
            an empty one. The raw reasons are carried rather than a
            pre-computed "was truncated" flag, so which reasons count as
            truncation stays in the scorer that owns that policy.
        output_tokens: ``gen_ai.usage.output_tokens`` when it was reported as a
            genuine integer. ``None`` covers absent, non-integer, and ``bool``
            -- absence is unknown, not zero (``docs/signals.md``), and ``bool``
            is a subclass of ``int`` that would otherwise pass as "1 token".
    """

    errored: bool
    finish_reasons: tuple[str, ...]
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class QualitySummary:
    """Everything the lane needs about a finished run.

    Attributes:
        step_count: How many steps the run took. The counted total, not the
            size of any buffer -- ``docs/quality.md`` shows users passing this
            straight into their own evaluator, and it reported a retention cap
            until 0.3.1.
        last: The final step, which is the only one the heuristic scores. Never
            ``None`` for a run that recorded a step; the type admits it so a
            backend cannot be forced to invent one.
    """

    step_count: int
    last: QualityStep | None


class QualityStore(Protocol):
    """Per-run scoring state, for many concurrent runs.

    Implementations must be safe under concurrent use. In-process that means a
    lock; across processes it means the backend's own atomicity, because a lock
    held in one process says nothing to another.
    """

    def record(self, run_id: str, step: QualityStep) -> None:
        """Note that a step happened, and that it was the latest.

        Returns nothing, unlike the other lanes' ``record``: quality is a
        run-scoped property and cannot be judged from one step, so there is
        nothing to read back and no reason to pay for a reply.

        Args:
            run_id: The run's identifier.
            step: The step's projection. Never a span (Section 10, and spans do
                not serialize).
        """
        ...

    def close_run(self, run_id: str) -> QualitySummary | None:
        """Release a run's state and return what it held.

        One call rather than read-then-release: on a shared backend those are
        two round trips with a gap another worker's step can land in, so the
        summary would describe a run that had already moved on.

        Args:
            run_id: The run's identifier.

        Returns:
            The summary, or ``None`` if the run held no state. ``None`` is not
            a run with no steps: run end fires more than once (M1-2), and a
            second close reporting an empty summary would let the lane score
            from no evidence and emit a weaker verdict over the first.
        """
        ...

    def run_count(self) -> int:
        """How many runs currently hold state.

        Exposed for leak detection: unbounded growth means runs are not being
        released at run end.

        Returns:
            Number of tracked runs.
        """
        ...
