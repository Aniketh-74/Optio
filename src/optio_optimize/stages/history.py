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
from optio_optimize.tokens import count_message, count_request
from optio_optimize.types import Message

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest


#: Stands in for the turns an anchored cut removed. Without it the model sees
#: the opening of a conversation followed by a much later exchange and no sign
#: that anything is missing, which invites it to invent the connection. One
#: line is cheap insurance against that.
_ELISION = Message(
    role="system",
    content=(
        "[earlier turns omitted to stay within budget; the opening and the recent turns are shown]"
    ),
)


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

    **Anchoring: keep the oldest turns and drop the middle.** With
    ``anchor_turns`` set, the cut takes the middle of the conversation instead
    of its front. Two independent measurements motivate it and neither is
    obvious alone: the provider's cached region is ``system + oldest turns``
    (87% of ``multi_turn_chat``'s prompt was served from cache before trimming
    touched it), and the recall audit found load-bearing facts stated in the
    *first* exchange and never repeated, of which plain trimming recovered 0
    of 4. A front cut therefore discards the cheapest and the most valuable
    context in one move.

    Measured live, ``multi_turn_chat_long`` at 50 turns, ``gpt-4o-mini``:

    ==========================  ==============  ================
    mode                        cost reduction  identical replies
    ==========================  ==============  ================
    slide (``anchor_turns=0``)  26.3%           25/50
    anchored (``anchor_turns=2``) 16.8%         **50/50**
    ==========================  ==============  ================

    So anchoring trades roughly nine points of saving for *every* reply
    matching the unoptimized baseline. It does not make trimming free; it
    converts a quality loss into a smaller, visible cost. Off by default
    (``anchor_turns=0``) because that is a change to what every existing
    caller sends, and one good measurement on one workload is not grounds for
    it -- see ADR-016.

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

        # Append-then-compact: with a threshold set, hold off entirely until
        # the prompt crosses it, then cut in one go.
        #
        # The reason is the interaction between this stage and provider prefix
        # caching, and it is not obvious. Trimming on *every* turn moves the
        # start of the message list every turn, so the prompt's head is never
        # what it was last call and the provider's cache matches nothing. Left
        # alone, a conversation only ever appends -- the head stays byte-stable
        # and the whole of it bills at the cached rate. So a smaller prompt at
        # the full rate can lose to a larger prompt at a tenth of it, and the
        # published guidance is that append-then-compact wins "in almost every
        # case" despite holding more tokens on average.
        #
        # Off by default, and the simulated numbers are *not* the reason to
        # turn it on. Simulated over 50 turns, sliding every turn costs 12.6%
        # more than never trimming at all under automatic caching (cached
        # tokens 97,280 -> 2,688) and compaction recovers nearly all of it.
        # That looks like a clean confirmation of the published guidance, and
        # it is the same artifact that has already been disproved once: this
        # class's own docstring records the simulator predicting cost *up*
        # 34.8% from trimming where the live API measured it *down* 8.4%. The
        # simulator matches a growing prefix by exact string comparison, so it
        # over-punishes any change to the head, which is precisely what this
        # comparison turns on. Reproducing a known-wrong model more carefully
        # does not make it evidence (ADR-015 rule 1). The option ships, the
        # default stays, and the question stays open until a live run settles
        # it.
        threshold = ctx.config.compact_at_tokens
        if threshold is not None and count_request(request, ctx.counter) < threshold:
            return self.declines(request)

        # System messages are never history to trim -- they are the one part
        # of the prompt PrefixCacheStage relies on being present every call.
        boundary = 0
        while boundary < len(messages) and messages[boundary].role == "system":
            boundary += 1

        history = messages[boundary:]
        keep = ctx.config.recent_turns

        # The first user turn is not history either, and for a stronger reason
        # than the system prompt above it.
        #
        # In a chat, turn 1 is an old question that has been answered and
        # superseded. In an agent loop it is *the task*, and every message
        # after it is the agent's own tool traffic. Dropping it leaves the
        # model a system prompt, a pile of tool results, and no statement of
        # what it was asked to do -- so it infers a task from the evidence and
        # answers a question nobody asked.
        #
        # Found by running a real Agents SDK agent (scripts/real_agent_run.py),
        # not by any test here: providers accept a conversation with no user
        # message at all, so this fails silently. On a four-tool support task
        # the trimmed arm dumped a markdown table of order fields instead of
        # the two-sentence answer requested, and cost *more* than trimming
        # correctly did, because an unfocused model writes longer:
        #
        #   defaults            3,757 in / 288 out / $0.00074   wrong
        #   first turn kept     3,816 in / 142 out / $0.00066   correct
        #
        # A floor rather than a default, so `anchor_turns=0` still means "no
        # anchoring beyond this". There is no workload where discarding the
        # question is the cheap option.
        task_anchor = 1 if history and history[0].role == "user" else 0
        anchor = max(ctx.config.anchor_turns, task_anchor)
        if len(history) <= keep + anchor:
            return self.declines(request)

        cut = len(history) - keep

        # Never cut between a tool_calls assistant message and its results:
        # walk the boundary back past any leading run of "tool" messages so
        # the assistant that issued them stays paired with all of them.
        while cut > 0 and history[cut].role == "tool":
            cut -= 1
        if cut <= anchor:
            # Extending all the way back to the start of history means no cut
            # point is safe -- the whole thing is one tool exchange. Trimming
            # nothing is correct here, not a bug.
            return self.declines(request)

        # The anchor keeps the *oldest* turns and drops the middle instead of
        # the front. Two measurements point the same way and neither is
        # obvious on its own:
        #
        # * The provider's cached region is `system + oldest turns` -- it
        #   matches the longest common prefix, and the oldest turns are the
        #   part that never changes. A front cut therefore discards precisely
        #   the tokens that were already billing at half rate, and shifts
        #   everything below it out of the provider's 128-token block
        #   alignment. Measured: 87% of `multi_turn_chat`'s prompt was served
        #   from cache before trimming touched it.
        # * The recall audit found load-bearing facts -- a budget, a deadline,
        #   a decision -- stated in the *first* exchange and never repeated.
        #   `trim_history` recalled 0 of 4 of them; those are exactly the turns
        #   a front cut removes first.
        #
        # So a front cut throws away the cheapest and the most valuable
        # context in the same move. Anchoring keeps both ends and takes the
        # middle, where the filler is.
        anchored = history[:anchor]
        dropped, kept = history[anchor:cut], history[cut:]

        # A cut that now starts on an orphaned tool result is the same hazard
        # the front cut already guards against, one boundary further in.
        while dropped and kept and kept[0].role == "tool":
            dropped, kept = dropped[:-1], (dropped[-1], *kept)
        if not dropped:
            return self.declines(request)

        saved = sum(count_message(m, ctx.counter, request.model) for m in dropped)
        gap = (
            (_ELISION,)
            if anchored
            # The marker is only needed when a gap is actually created. A pure
            # front cut leaves a conversation that reads as though it started
            # later, which is coherent; an anchored cut leaves one that jumps,
            # and a model told nothing was removed will try to reconcile the
            # jump instead of accepting it.
            else ()
        )
        return StageResult(
            request=request.with_messages(messages[:boundary] + anchored + gap + kept),
            saved_input_tokens=saved,
            note=f"dropped {len(dropped)} of {len(history)} turns"
            + (f", anchored on the first {len(anchored)}" if anchored else ""),
        )
