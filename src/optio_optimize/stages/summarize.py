"""Replace aged-out history with a summary instead of discarding it.

Same problem :class:`~optio_optimize.stages.history.TrimHistoryStage` solves,
with a different trade: trimming keeps nothing of what ages out, summarizing
keeps a compressed trace of it, at the cost of an actual model call and the
risk any summary carries -- it might omit or misstate something a later turn
needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from optio_optimize.stages.base import Fidelity, Stage, StageResult
from optio_optimize.tokens import count_message

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest

#: A synchronous summarizer: raw "role: content" history text in, summary
#: text out. Synchronous because every :class:`Stage` in this package is --
#: ``Optimizer.acall`` exists so the *provider* call can be async, not so
#: stage bodies can be (``pipeline.py``: "stage logic performs no I/O").
#:
#: This stage is the one exception to that "no I/O" premise in practice: a
#: real summarizer calls a model, which is exactly the I/O the premise
#: assumed away. Under :meth:`~optio_optimize.optimizer.Optimizer.call` that
#: is unremarkable -- it is one more blocking call among others. Under
#: :meth:`~optio_optimize.optimizer.Optimizer.acall`, a genuinely blocking
#: summarizer (a synchronous HTTP call, ``asyncio.run(...)`` used to bridge
#: an async one) stalls the event loop for its duration, same as calling any
#: other blocking function from async code. A caller with an async
#: summarizer needs a non-blocking bridge (a thread pool, not ``asyncio.run``)
#: to avoid that -- this package does not supply one, the same way it
#: supplies no summarizer at all.
Summarizer = Callable[[str], str]


class SummarizeHistoryStage(Stage):
    """Summarize aged-out history instead of dropping it.

    **Ships no summarizer.** Same rule optio core's quality-lane LLM-judge
    follows (0.1.0 CHANGELOG: "optio ships no default and constructs no
    model client, so enabling the lane cannot spend your money on our
    initiative") -- this stage calls whatever synchronous ``summarizer``
    the caller supplies and nothing else. With none, it always declines,
    which is why ``summarize_history=True`` alone is safe to set: the flag
    spends nothing by itself. Supply one via ``Optimizer(summarizer=...)``.

    Shares :class:`TrimHistoryStage`'s tool-call safety invariant: the cut
    point never separates an assistant's ``tool_calls`` message from its
    results, and if no safe cut exists this declines rather than force one.

    ``Fidelity.ALTERED``: the summary is generated text, not the history
    itself, and a summarizer -- however good -- can omit or misstate
    something a later turn depended on. Off by default (ADR-013).

    **What one live measurement found (2026-07-29, ADR-015).** On a
    conversation with four load-bearing facts planted before the
    ``recent_turns`` window and asked back afterwards, this stage recovered
    **4 of 4** where ``trim_history`` recovered **0 of 4**, with **zero**
    silent errors -- no fact was misstated, which is the failure that would
    matter most. The stage does what it claims.

    **It still cost more than not optimizing at all**, and the reason lives
    in this class rather than in the numbers: :meth:`before` calls the
    summarizer *unconditionally, on every request*. There is no memoization
    keyed on the dropped history, so the same aged-out turns are re-summarized
    on every turn of a conversation. Measured: the summarized prompt was 261
    tokens against the full history's 466 -- a real reduction -- but the
    summarizer call added 361 tokens nobody was spending, for 622 total. The
    prompt is bounded; the *cost* is not, because it scales with the dropped
    history just as the full prompt does, so the bounded-prompt advantage
    cannot catch up.

    A summary computed once and reused across turns would change that
    arithmetic entirely. This stage does not do that, and a caller enabling it
    should know the bill is paid per request. See
    ``docs/optimize-benchmarks.md``.
    """

    fidelity = Fidelity.ALTERED

    def __init__(self, summarizer: Summarizer | None = None) -> None:
        """Build the stage.

        Args:
            summarizer: Turns dropped-history text into a summary. ``None``
                (the default) means this stage always declines.
        """
        self._summarizer = summarizer

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "summarize_history"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Replace history older than the recent window with a summary."""
        if self._summarizer is None:
            return self.declines(request)

        messages = request.messages
        boundary = 0
        while boundary < len(messages) and messages[boundary].role == "system":
            boundary += 1

        history = messages[boundary:]
        keep = ctx.config.recent_turns
        if len(history) <= keep:
            return self.declines(request)

        cut = len(history) - keep
        while cut > 0 and history[cut].role == "tool":
            cut -= 1
        if cut == 0:
            return self.declines(request)

        dropped, kept = history[:cut], history[cut:]

        # Permitted to raise (stages/base.py): the pipeline's own guard
        # absorbs a broken summarizer and skips this stage for the request,
        # falling back to whatever the previous stage produced -- the same
        # outcome as trim_history would give, not a special case here.
        text = "\n".join(f"{m.role}: {m.content}" for m in dropped)
        summary_text = self._summarizer(text)
        if not summary_text:
            return self.declines(request)

        from optio_optimize.types import Message

        summary_message = Message(
            role="system", content=f"[Earlier conversation, summarized] {summary_text}"
        )

        saved_before = sum(count_message(m, ctx.counter, request.model) for m in dropped)
        saved_after = count_message(summary_message, ctx.counter, request.model)
        saved = max(0, saved_before - saved_after)

        new_history = (summary_message, *kept)
        return StageResult(
            request=request.with_messages(messages[:boundary] + new_history),
            saved_input_tokens=saved,
            note=f"summarized {len(dropped)} of {len(history)} turns",
        )
