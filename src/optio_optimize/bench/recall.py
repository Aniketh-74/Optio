"""``summarize_history``'s audit: does the summary keep what mattered?

ADR-015's bar for this stage is unusually specific, because the stage cannot
be evaluated at all without a real summarizer and a workload built to probe
the one thing it claims over ``trim_history``. Both are here.

**The comparison that decides it is against trimming, not against nothing.**
``trim_history`` solves the same problem, is ``SHAPED`` rather than
``ALTERED``, and is free -- it drops aged-out turns instead of paying a model
call to compress them. So "the summary preserved the fact" is not on its own
a result: trimming might have kept it too (if it never aged out), or the
summary might have lost it just like trimming would, in which case the extra
call bought nothing. Three arms run on identical requests:

``full``
    The whole conversation, no history stage. The control -- what the right
    answer looks like when nothing has been removed. A probe the model gets
    wrong even here is measuring the model, not the stage, and is excluded.
``trim``
    ``trim_history`` only. The free alternative this stage has to beat.
``summarize``
    ``summarize_history`` only, with a real live-calling summarizer.

**The planted fact is the whole design.** A long conversation whose *first*
exchange states something load-bearing -- a budget figure, a date, a decision,
a constraint -- followed by enough filler to push it out of the
``recent_turns`` window, then a final question that can only be answered from
it. "Does the conversation still make sense" is not the question; "is the
specific fact still correct" is, and only a planted fact with a known answer
can tell you.

**Why a wrong answer here is worse than a missing one.** ``trim_history``
losing the fact is visible: the model says it does not have that context.
A summary that *misstates* it is not -- the number is present, confident, and
wrong. That asymmetry is why this stage is ``ALTERED`` and trimming is not,
so the report distinguishes "declined to answer" from "answered incorrectly"
rather than scoring both as a miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from optio_optimize.config import DEFAULT_RECENT_TURNS, OptimizeConfig
from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, Message

if TYPE_CHECKING:
    from optio_optimize.bench.providers import BenchProvider
    from optio_optimize.stages.summarize import Summarizer

_SYSTEM = (
    "You are a project assistant. Answer from the conversation so far. "
    "If the conversation does not contain the answer, say exactly: NOT IN CONTEXT."
)

#: Stated in the first exchange and never repeated. Each is a single
#: unambiguous token of information a later turn can be asked to recover.
_PLANTED = (
    "The approved project budget is $47,500.",
    "The hard deadline is March 14th.",
    "We chose PostgreSQL over MySQL for the datastore.",
    "The client requires SOC 2 Type II compliance before launch.",
)

#: Filler exchanges, deliberately on-topic but carrying no planted fact --
#: an off-topic filler would make the summarizer's job artificially easy by
#: letting it discard whole turns as irrelevant.
_FILLER = (
    "Can you outline the rollout phases?",
    "What should the staging environment look like?",
    "How should we handle database migrations?",
    "What is a reasonable code review turnaround?",
    "How do we structure the on-call rotation?",
    "What monitoring should be in place at launch?",
    "How should we document the API?",
    "What is a sensible branching strategy?",
)


@dataclass(frozen=True, slots=True)
class RecallProbe:
    """One question whose answer was planted early and then aged out.

    Attributes:
        question: The final turn, answerable only from the planted facts.
        expected: Accepted answers, matched the same way the routing audit
            matches -- see :func:`~optio_optimize.bench.routing.grade`.
        fact: The planted statement this probe is trying to recover, for the
            report.
    """

    question: str
    expected: tuple[str, ...]
    fact: str


RECALL_PROBES: tuple[RecallProbe, ...] = (
    RecallProbe("What is the approved project budget?", ("47 500", "47500"), _PLANTED[0]),
    # "march 14th" as well as "march 14": the grader's word-boundary check
    # treats the ordinal suffix as part of the token, so "March 14th" does not
    # match "march 14". Listing both spellings is what `expected` is for -- the
    # alternative would be teaching the matcher about English ordinals, which
    # is a lot of machinery to avoid typing four characters. Caught live: this
    # probe was excluded as control-arm-wrong when the control had in fact
    # answered it correctly.
    RecallProbe("What is the hard deadline?", ("march 14", "march 14th"), _PLANTED[1]),
    RecallProbe("Which datastore did we choose?", ("postgresql", "postgres"), _PLANTED[2]),
    RecallProbe(
        "What compliance certification does the client require before launch?",
        ("soc 2",),
        _PLANTED[3],
    ),
)

#: Answer text meaning "I was not given that", as opposed to an answer that is
#: simply wrong. The distinction is the point of this audit: a declined answer
#: is a visible loss, a wrong one is a silent one.
_DECLINED = re.compile(r"not in context|do not have|don't have|no information", re.I)


def build_conversation(model: str) -> tuple[Message, ...]:
    """The shared history: facts planted first, then enough filler to bury them.

    Args:
        model: Only used by the caller for the request; kept in the signature
            so conversation construction and request construction stay
            together at the call site.

    Returns:
        System prompt, the planting exchange, then the filler exchanges.
    """
    del model
    messages: list[Message] = [Message(role="system", content=_SYSTEM)]
    messages.append(
        Message(
            role="user",
            content="Here is the project brief. " + " ".join(_PLANTED),
        )
    )
    messages.append(
        Message(
            role="assistant",
            content="Understood. I have noted the brief and will refer to it.",
        )
    )
    for question in _FILLER:
        messages.append(Message(role="user", content=question))
        messages.append(
            Message(
                role="assistant",
                # A real assistant reply, not a single clause. The filler's job
                # is to bury the planted facts deeply enough that recovering
                # them is a genuine test, and since ADR-026 it has a second job:
                # trimming is gated on the value of what it removes, so filler
                # this thin left the trim arm declining to trim and the audit
                # measuring nothing at all about trimming.
                content=(
                    f"For that: proceed incrementally, document the decision, and "
                    f"review it at the next checkpoint. ({question.rstrip('?')}) "
                    f"In practice that means agreeing the acceptance criteria up "
                    f"front, keeping the change small enough to reason about, and "
                    f"writing down what you expected to happen before you look at "
                    f"what did. If the result disagrees with the expectation, the "
                    f"expectation is the thing to examine first. Record the "
                    f"outcome either way so the next person does not repeat the "
                    f"experiment, and raise it at the checkpoint if it changes "
                    f"any assumption the plan depends on."
                ),
            )
        )
    return tuple(messages)


@dataclass(slots=True)
class ArmAnswer:
    """One arm's answer to one probe.

    Attributes:
        answer: What the model said.
        correct: Whether it recovered the planted fact.
        declined: Whether it said it did not have the context, as opposed to
            answering incorrectly.
        input_tokens: Prompt tokens actually sent on the answer call.
        extra_calls: Model calls this arm made beyond the one answer -- the
            summarizer's, for the ``summarize`` arm. The cost of the stage.
        summarizer_tokens: Tokens the summarizer call itself consumed, in and
            out. Counted because ``input_tokens`` alone flatters this stage
            badly: the summarized prompt really is smaller than the full one,
            and the call that made it smaller is invisible in that column.
            Any honest comparison against ``trim_history`` -- which is free --
            or against sending the full history has to add this back.
    """

    answer: str
    correct: bool
    declined: bool
    input_tokens: int
    extra_calls: int = 0
    summarizer_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Everything this arm spent to produce one answer."""
        return self.input_tokens + self.summarizer_tokens


@dataclass(slots=True)
class RecallResult:
    """One probe run through every arm.

    Attributes:
        probe: The question asked.
        arms: Arm name -> what that arm answered.
    """

    probe: RecallProbe
    arms: dict[str, ArmAnswer] = field(default_factory=dict)

    @property
    def is_interpretable(self) -> bool:
        """Whether the control got it right, making the other arms meaningful.

        A probe the model fails even with the whole conversation in front of
        it is measuring the model, not the history stage, and folding it into
        either arm's score would be attributing a model limitation to a
        library.
        """
        full = self.arms.get("full")
        return full is not None and full.correct


@dataclass(slots=True)
class RecallAuditReport:
    """Every probe across every arm, plus the tool-boundary check.

    Attributes:
        results: One entry per probe.
        model: Model answered against.
        recent_turns: The window both history stages were configured with.
        tool_boundary_safe: Whether the stage refused to cut a tool_calls
            message away from its results -- the invariant it shares with
            ``trim_history``, re-confirmed rather than assumed.
    """

    results: list[RecallResult] = field(default_factory=list)
    model: str = ""
    recent_turns: int = DEFAULT_RECENT_TURNS
    tool_boundary_safe: bool = False

    @property
    def interpretable(self) -> list[RecallResult]:
        """Probes the control arm answered correctly."""
        return [r for r in self.results if r.is_interpretable]

    def recall_rate(self, arm: str) -> float:
        """Fraction of interpretable probes ``arm`` answered correctly."""
        usable = self.interpretable
        if not usable:
            return 0.0
        return sum(1 for r in usable if r.arms[arm].correct) / len(usable)

    def silent_error_rate(self, arm: str) -> float:
        """Fraction of interpretable probes ``arm`` got wrong *without* saying so.

        The number that separates this stage's risk from ``trim_history``'s.
        A declined answer is a visible loss the caller can react to; a
        confidently wrong one is not.
        """
        usable = self.interpretable
        if not usable:
            return 0.0
        wrong_and_quiet = sum(
            1 for r in usable if not r.arms[arm].correct and not r.arms[arm].declined
        )
        return wrong_and_quiet / len(usable)

    def mean_input_tokens(self, arm: str) -> float:
        """Average prompt tokens ``arm`` sent on the answer call alone."""
        usable = self.interpretable
        if not usable:
            return 0.0
        return sum(r.arms[arm].input_tokens for r in usable) / len(usable)

    def mean_total_tokens(self, arm: str) -> float:
        """Average tokens ``arm`` spent in total, summarizer call included.

        The column that decides whether this stage is worth it. For ``full``
        and ``trim`` it equals :meth:`mean_input_tokens`; for ``summarize`` it
        does not, and the difference is the whole argument.
        """
        usable = self.interpretable
        if not usable:
            return 0.0
        return sum(r.arms[arm].total_tokens for r in usable) / len(usable)

    def extra_calls(self, arm: str) -> int:
        """Model calls ``arm`` made beyond answering -- the stage's own cost."""
        return sum(r.arms[arm].extra_calls for r in self.results)


def _tool_boundary_holds(summarizer: Summarizer) -> bool:
    """Confirm the stage never cuts a ``tool_calls`` message from its results.

    ``tool_calling_chat`` did this for ``trim_history`` live; the same
    invariant is asserted in ``SummarizeHistoryStage``'s docstring and is
    checked here rather than inherited. Free -- no model call is needed to
    see where the cut landed.

    Args:
        summarizer: Any summarizer; its output is irrelevant to the boundary.

    Returns:
        True when every produced request keeps each assistant ``tool_calls``
        message adjacent to the ``tool`` messages answering it.
    """
    from optio_optimize.stages.base import StageContext
    from optio_optimize.stages.summarize import SummarizeHistoryStage
    from optio_optimize.tokens import default_counter

    stage = SummarizeHistoryStage(summarizer)
    ctx = StageContext(
        config=OptimizeConfig(summarize_history=True, recent_turns=3),
        counter=default_counter(),
    )

    history: list[Message] = [Message(role="system", content=_SYSTEM)]
    for turn in range(6):
        history.append(Message(role="user", content=f"step {turn}"))
        history.append(
            Message(
                role="assistant",
                content="",
                extra={"tool_calls": [{"id": f"call_{turn}", "type": "function"}]},
            )
        )
        history.append(
            Message(role="tool", content=f"result {turn}", extra={"tool_call_id": f"call_{turn}"})
        )

    for cut_at in range(1, len(history)):
        request = LLMRequest(model="gpt-4o", messages=tuple(history[:cut_at]), temperature=0.0)
        produced = stage.before(request, ctx).request
        for index, message in enumerate(produced.messages):
            if message.role != "tool":
                continue
            previous = produced.messages[index - 1] if index else None
            if previous is None:
                return False
            if previous.role not in ("assistant", "tool"):
                return False
    return True


def run_recall_audit(
    provider: BenchProvider,
    summarizer: Summarizer,
    *,
    probes: tuple[RecallProbe, ...] = RECALL_PROBES,
    recent_turns: int = DEFAULT_RECENT_TURNS,
) -> RecallAuditReport:
    """Run every probe through the full, trimmed and summarized arms.

    Args:
        provider: Where answer calls go.
        summarizer: A real summarizer for the ``summarize`` arm. A stub would
            measure the stub -- the exact trap ADR-015 names for this stage.
        probes: Probes to run.
        recent_turns: Window for both history stages. The default is the
            shipped one, so the numbers describe what a caller actually gets.

    Returns:
        Per-arm recall, silent-error rates, token counts and the boundary check.
    """
    from optio_optimize.bench.routing import grade as _grade

    base = OptimizeConfig(
        exact_cache=False,
        prefix_cache=False,
        adaptive_max_tokens=False,
        structured_output=False,
        trim_history=False,
        deduplicate=False,
        prune_retrieval=False,
        recent_turns=recent_turns,
    )
    from dataclasses import replace as _replace

    from optio_optimize.tokens import default_counter

    counter = default_counter()
    calls: dict[str, int] = {}
    summarizer_tokens: dict[str, int] = {}

    def counting_summarizer(text: str) -> str:
        calls["n"] = calls.get("n", 0) + 1
        summary = summarizer(text)
        # Counted locally rather than read off a response: `Summarizer` is
        # `str -> str` by design, so the usage numbers of the call it makes
        # are not observable from here. An estimate from the same tokenizer
        # the rest of this package uses is the honest available answer, and
        # it is stated as an estimate wherever it is reported.
        summarizer_tokens["n"] = (
            summarizer_tokens.get("n", 0)
            + counter.count_text(text, provider.model)
            + counter.count_text(summary, provider.model)
        )
        return summary

    arms = {
        "full": (_replace(base, enabled=False), None),
        "trim": (_replace(base, trim_history=True), None),
        "summarize": (_replace(base, summarize_history=True), counting_summarizer),
    }

    report = RecallAuditReport(
        model=provider.model,
        recent_turns=recent_turns,
        tool_boundary_safe=_tool_boundary_holds(summarizer),
    )
    history = build_conversation(provider.model)

    for probe in probes:
        result = RecallResult(probe=probe)
        for arm, (config, arm_summarizer) in arms.items():
            calls["n"] = 0
            summarizer_tokens["n"] = 0
            optimizer = Optimizer(config, summarizer=arm_summarizer)
            request = LLMRequest(
                model=provider.model,
                messages=(*history, Message(role="user", content=probe.question)),
                temperature=0.0,
                max_tokens=64,
            )
            response = optimizer.call(request, provider)
            answer = response.content
            result.arms[arm] = ArmAnswer(
                answer=answer,
                correct=_grade(answer, probe),
                declined=bool(_DECLINED.search(answer)),
                input_tokens=response.input_tokens,
                extra_calls=calls["n"],
                summarizer_tokens=summarizer_tokens["n"],
            )
        report.results.append(result)
    return report


def format_recall_report(report: RecallAuditReport) -> list[str]:
    """Render a recall audit as human-readable lines."""
    usable = report.interpretable
    excluded = len(report.results) - len(usable)
    lines = [
        f"summarize_history recall audit -- model {report.model}, "
        f"recent_turns={report.recent_turns}",
        f"{len(usable)}/{len(report.results)} probes interpretable "
        f"(the control arm answered correctly)",
        "",
        f"{'arm':<12}{'recalled':>10}{'silently wrong':>16}{'prompt tok':>12}"
        f"{'+summarizer':>13}{'TOTAL tok':>11}{'extra calls':>13}",
    ]
    for arm in ("full", "trim", "summarize"):
        overhead = report.mean_total_tokens(arm) - report.mean_input_tokens(arm)
        lines.append(
            f"{arm:<12}"
            f"{report.recall_rate(arm):>9.0%} "
            f"{report.silent_error_rate(arm):>15.0%} "
            f"{report.mean_input_tokens(arm):>11.0f} "
            f"{overhead:>12.0f} "
            f"{report.mean_total_tokens(arm):>10.0f} "
            f"{report.extra_calls(arm):>12}"
        )
    lines.append(
        "  (summarizer tokens are counted with this package's own tokenizer, not read "
        "from a usage field: Summarizer is str -> str, so its call is not observable here)"
    )
    if excluded:
        lines.append(
            f"  note: {excluded} probe(s) excluded -- the model missed them even with "
            "the whole conversation, so they measure the model, not the stage."
        )
    lines.append("")

    for result in report.results:
        marker = "" if result.is_interpretable else "  [EXCLUDED: control arm wrong]"
        lines.append(f"# {result.probe.question}{marker}")
        lines.append(f"    planted:    {result.probe.fact}")
        for arm in ("full", "trim", "summarize"):
            answer = result.arms[arm]
            if answer.correct:
                verdict = "recalled"
            elif answer.declined:
                verdict = "lost (said so)"
            else:
                verdict = "SILENTLY WRONG"
            lines.append(f"    {arm:<10} [{verdict}] {answer.answer!r}")
        lines.append("")

    lines.append(
        f"tool-call boundary: {'held' if report.tool_boundary_safe else 'VIOLATED'} "
        "(no tool result was ever separated from the assistant message that called it)"
    )
    return lines


__all__ = [
    "RECALL_PROBES",
    "ArmAnswer",
    "RecallAuditReport",
    "RecallProbe",
    "RecallResult",
    "build_conversation",
    "format_recall_report",
    "run_recall_audit",
]
