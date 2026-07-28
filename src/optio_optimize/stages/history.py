"""Bounding conversation history to a recent window.

Multi-turn chat is the workload where the naive integration is worst: every
step resends the whole conversation so far, so cost grows with the square of
its length rather than staying flat. Provider-side prefix caching (see
``stages/caching.py``) discounts the *price* of that resend where the
provider supports it, but changes nothing about *how much* gets sent -- and on
OpenAI it changes nothing at all, since the discount is automatic and this
package's marker adds no benefit on top of it (``docs/optimize-benchmarks.md``).
Trimming attacks the other half of the problem: send less.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio_optimize.stages.base import Fidelity, Stage, StageResult
from optio_optimize.tokens import count_message

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest


class TrimHistoryStage(Stage):
    """Keep the system prompt and only the most recent turns.

    Everything older than :attr:`~optio_optimize.config.OptimizeConfig.
    recent_turns` is dropped from what actually gets sent -- not summarized,
    not compressed, just left out. That is a real change to what the model can
    see, which is why this is :attr:`Fidelity.SHAPED` rather than
    ``IDENTICAL``: an answer that depended on turn 2 of a 20-turn conversation
    will not be able to reference it once turn 2 has aged out of the window.

    It stops short of ``ALTERED``, the tier ADR-013 gates behind an eval
    suite: unlike summarization, this never invents text the model did not
    already produce, and a caller who needs more lookback than the default
    affords can simply raise ``recent_turns``. Dropping context is the
    accepted trade (config.py: "bounded-risk... they drop context rather than
    invent it"); inventing a replacement for it is a different stage
    (``summarize_history``) and a different default.

    **A simulated run predicted this stage would cost money on OpenAI; a live
    run against the real API says otherwise.** The simulator's automatic-cache
    model matches a growing prefix by exact string comparison, and a sliding
    window breaks that match on nearly every turn -- so the simulation showed
    fewer tokens sent (6.5%) but a *higher* bill (cost up 34.8%), on the theory
    that trimming forfeits a discount the untrimmed baseline was already
    earning. Live against ``gpt-4o-mini`` (``python -m optio_optimize.bench
    --live``, ``multi_turn_chat``), that did not happen: cost fell 8.4% and
    output tokens fell 35% along with it -- a shorter prompt produced shorter
    answers too, which the simulator cannot model because it always returns a
    fixed synthetic completion length. Total spend to measure this: under
    $0.02. See ``docs/optimize-benchmarks.md`` for the full comparison and
    the corrected numbers.

    The simulator was not lying, it was answering a narrower question than the
    one that matters: whether *this specific coarse model* of automatic
    caching predicts what the provider actually does. It does not, at least
    not closely enough to trust for a cost claim -- the same lesson
    ``prefix_cache``'s docstring already draws from the 36.3%-to-0% correction,
    now confirmed a second time. Simulated figures in this package should be
    read as a reason to run ``--live`` on your own traffic, never as the
    number to ship.

    **Never starts the kept window on an orphaned tool result.** A ``tool``
    role message is, by every major provider's protocol, always the direct
    response to a ``tool_calls`` entry on the *immediately preceding*
    assistant message. If a naive suffix cut landed between them -- entirely
    possible once a window boundary lands inside a run of parallel tool
    calls -- the trimmed request would carry a tool result with no matching
    call, which providers reject the same way they reject ``fan_out``'s
    missing ``"json"`` literal (``docs/optimize-benchmarks.md``). This stage
    walks the cut point backward past any leading run of ``tool`` messages so
    the assistant message that issued them, and every one of its results,
    survive together or not at all.
    """

    fidelity = Fidelity.SHAPED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "trim_history"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Drop history older than the configured recent-turn window."""
        messages = request.messages

        # System messages are never history to trim -- they are the one part
        # of the prompt PrefixCacheStage relies on being present every call.
        boundary = 0
        while boundary < len(messages) and messages[boundary].role == "system":
            boundary += 1

        history = messages[boundary:]
        keep = ctx.config.recent_turns
        if len(history) <= keep:
            return self.declines(request)

        cut = len(history) - keep

        # Never cut between a tool_calls assistant message and its results:
        # walk the boundary back past any leading run of "tool" messages so
        # the assistant that issued them stays paired with all of them.
        while cut > 0 and history[cut].role == "tool":
            cut -= 1
        if cut == 0:
            # Extending all the way back to the start of history means no cut
            # point is safe -- the whole thing is one tool exchange. Trimming
            # nothing is correct here, not a bug.
            return self.declines(request)

        dropped, kept = history[:cut], history[cut:]
        saved = sum(count_message(m, ctx.counter, request.model) for m in dropped)
        return StageResult(
            request=request.with_messages(messages[:boundary] + kept),
            saved_input_tokens=saved,
            note=f"dropped {len(dropped)} of {len(history)} turns",
        )
