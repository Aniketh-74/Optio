"""``route_models``'s audit: does the cheap model actually answer as well?

ADR-015 asks a question this package's A/B harness structurally cannot
answer. ``ABResult`` prices a whole arm at one flat rate, so a run where
*some* requests were served by a cheaper model reports a blended number that
reflects neither model -- and more importantly, a routed request is never
expected to be output-identical, so the identity check the rest of the suite
leans on does not apply here at all. The evidence this stage needs is a
direct per-model comparison: ask both models the same short question and see
who gets it right.

**Graded against ground truth, not a judge.** Every probe here has one
unambiguous correct answer, checked by string match against a terse expected
response. That is a deliberate departure from ``compress_prompt``'s
judge-based grading: a judge is itself a model, and using one to decide
whether a weaker model is good enough puts a capability question inside the
very thing being measured. Where a correct answer can just be *stated in
advance*, stating it is strictly better evidence.

**The workload has to contain hard questions, or it proves nothing.** The
stage routes on length alone (:data:`~optio_optimize.stages.routing.
MAX_ROUTABLE_TOKENS`, 500 tokens) on the theory that short means easy. A
probe set of only genuine lookups would confirm that theory by construction
and tell an operator nothing about their traffic. So half these probes are
deliberately short *and hard*: counting letters, comparing decimals,
transitive ordering, and the bat-and-ball question -- all under a dozen
words, all things a weaker model is known to trip on. If the length heuristic
is sound, the two categories should separate; if it is not, they will.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from optio_optimize.config import OptimizeConfig, pricing_for
from optio_optimize.optimizer import Optimizer
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.routing import RouteModelsStage
from optio_optimize.tokens import default_counter
from optio_optimize.types import LLMRequest, Message

if TYPE_CHECKING:
    from optio_optimize.bench.providers import BenchProvider

#: Appended to every probe so answers are terse enough to grade by match
#: rather than by judgment. Not a trick to help the cheap model: both models
#: get the identical instruction, so any accuracy gap is capability, not
#: formatting luck.
_TERSE = " Answer with as few words as possible and no explanation."


@dataclass(frozen=True, slots=True)
class RoutingProbe:
    """One short question with a known correct answer.

    Attributes:
        category: ``easy`` (a genuine lookup or single-step conversion) or
            ``hard`` (short wording, real reasoning). The split is the point:
            see the module docstring.
        question: The prompt, short enough that ``route_models`` will route it.
        expected: Accepted answers, matched case-insensitively against the
            response with punctuation stripped. More than one entry where a
            correct answer has several equally right spellings ("0.05" and
            "5 cents"), never to widen a wrong answer into a right one.
        note: Why this probe is here, for the report.
    """

    category: str
    question: str
    expected: tuple[str, ...]
    note: str = ""

    def request(self, model: str) -> LLMRequest:
        """This probe as a deterministic request against ``model``."""
        return LLMRequest(
            model=model,
            messages=(Message(role="user", content=self.question + _TERSE),),
            temperature=0.0,
            max_tokens=64,
        )


#: A period that is *not* a decimal point -- i.e. not flanked by digits on
#: both sides. Sentence-ending periods have to go before matching, and
#: decimal points have to stay, and the first version of this module did not
#: distinguish them: it kept every ``.``, so ``"Tokyo."`` normalized to
#: ``"tokyo."`` and the word-boundary check for ``"tokyo"`` failed on the
#: trailing period. That scored three correct live answers wrong, including
#: one that was reported as a ``route_models`` REGRESSION -- a plausible-
#: looking 12.5% regression rate that was entirely this bug.
_NON_DECIMAL_PERIOD = re.compile(r"(?<!\d)\.|\.(?!\d)")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation except decimal points, collapse whitespace."""
    lowered = _NON_DECIMAL_PERIOD.sub(" ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]", " ", lowered)).strip()


class Gradable(Protocol):
    """Anything carrying a set of accepted answers.

    :func:`grade` reads nothing else, so widening it to a protocol lets the
    recall audit's probes reuse this matcher instead of being cast into a
    ``RoutingProbe`` they are not. One matcher for both audits is also the
    point: a grader bug found by one is fixed for the other, and both have now
    found one.
    """

    @property
    def expected(self) -> tuple[str, ...]:
        """Accepted answers."""
        ...


def grade(response: str, probe: Gradable) -> bool:
    """Whether ``response`` answers ``probe`` correctly.

    Accepts an answer that *is* one of the expected strings, or that contains
    one as a whole token-run. Containment is needed because a model may reply
    "Tokyo." or "The ball costs $0.05" despite being asked for brevity, and
    scoring those wrong would measure instruction-following rather than
    capability -- the thing this audit is specifically not about.

    Args:
        response: What the model said.
        probe: The probe, carrying its accepted answers.

    Returns:
        True when the response matches an expected answer.
    """
    got = _normalize(response)
    for candidate in probe.expected:
        want = _normalize(candidate)
        if got == want or re.search(rf"(?<![a-z0-9.]){re.escape(want)}(?![a-z0-9.])", got):
            return True
    return False


#: Twelve probes: four genuine lookups, and eight short-but-hard split between
#: famous reasoning traps and ordinary multi-step problems. Every answer is
#: checkable without a model in the loop.
ROUTING_PROBES: tuple[RoutingProbe, ...] = (
    RoutingProbe("easy", "What is the capital of Japan?", ("tokyo",), "plain lookup"),
    RoutingProbe("easy", "What is the chemical symbol for gold?", ("au",), "plain lookup"),
    RoutingProbe("easy", "What is 15% of 200?", ("30",), "single-step arithmetic"),
    RoutingProbe("easy", "How many days are in a leap year?", ("366",), "plain lookup"),
    RoutingProbe(
        "hard",
        "How many times does the letter r appear in the word strawberry?",
        ("3", "three"),
        "character counting: short, and a known weak spot for small models",
    ),
    RoutingProbe(
        "hard",
        "Which is larger, 9.11 or 9.9? Reply with just the number.",
        ("9.9",),
        "decimal comparison: reads as version-number ordering to a weak model",
    ),
    RoutingProbe(
        "hard",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
        "the ball. How much does the ball cost?",
        ("0.05", "5 cents", "five cents"),
        "the intuitive wrong answer ($0.10) is strongly available",
    ),
    RoutingProbe(
        "hard",
        "Alice is taller than Bob. Bob is taller than Carol. Is Carol taller than Alice?",
        ("no",),
        "transitive ordering stated tersely",
    ),
    # The four above are famous traps, and a 2026-era cheap model answers all
    # of them -- they are in everyone's training data by now, so passing them
    # says nothing about a *novel* short-but-hard request. These four are
    # ordinary multi-step problems with no canonical phrasing to recall.
    RoutingProbe(
        "hard",
        "What is 17 times 24, minus 89?",
        ("319",),
        "two-step arithmetic, no memorable phrasing to recall",
    ),
    RoutingProbe(
        "hard",
        "If today is Wednesday, what day of the week is it 100 days from now?",
        ("friday",),
        "modular arithmetic plus a calendar mapping",
    ),
    RoutingProbe(
        "hard",
        "What is the 4th letter from the end of the word extraordinary?",
        ("n",),
        "indexed character access, counting backwards",
    ),
    RoutingProbe(
        "hard",
        "A car travels 60 km in 45 minutes. What is its speed in km per hour?",
        ("80",),
        "unit conversion where the intuitive answer (60) is wrong",
    ),
)


@dataclass(slots=True)
class ProbeResult:
    """One probe answered by both models.

    Attributes:
        probe: The question asked.
        routed: Whether ``RouteModelsStage`` actually retargets this request.
            A probe the stage declines is not evidence about routing.
        expensive_answer: What the requested model said.
        cheap_answer: What ``cheap_model`` said.
        expensive_correct: Whether the requested model was right.
        cheap_correct: Whether the cheap model was right.
    """

    probe: RoutingProbe
    routed: bool
    expensive_answer: str
    cheap_answer: str
    expensive_correct: bool
    cheap_correct: bool

    @property
    def regressed(self) -> bool:
        """The failure this stage risks: right before routing, wrong after."""
        return self.expensive_correct and not self.cheap_correct


@dataclass(slots=True)
class RoutingAuditReport:
    """Both models' accuracy across the probe set, split by category.

    Attributes:
        results: One entry per probe, in the order run.
        expensive_model: The model a caller asked for.
        cheap_model: The model ``route_models`` would substitute.
        declines: Decline-case name -> whether the stage correctly declined.
    """

    results: list[ProbeResult] = field(default_factory=list)
    expensive_model: str = ""
    cheap_model: str = ""
    declines: dict[str, bool] = field(default_factory=dict)

    def by_category(self) -> dict[str, list[ProbeResult]]:
        """Group results by ``easy``/``hard``."""
        grouped: dict[str, list[ProbeResult]] = {}
        for result in self.results:
            grouped.setdefault(result.probe.category, []).append(result)
        return grouped

    def accuracy(self, results: list[ProbeResult], *, cheap: bool) -> float:
        """Fraction of ``results`` the named model answered correctly."""
        if not results:
            return 0.0
        correct = sum(1 for r in results if (r.cheap_correct if cheap else r.expensive_correct))
        return correct / len(results)

    @property
    def regression_rate(self) -> float:
        """Fraction of probes the routing would have turned from right to wrong."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.regressed) / len(self.results)

    @property
    def cost_ratio(self) -> float | None:
        """How much cheaper the cheap model's input tokens are, if both priced."""
        expensive = pricing_for(self.expensive_model)
        cheap = pricing_for(self.cheap_model)
        if expensive is None or cheap is None or cheap.input_usd_per_m == 0:
            return None
        return expensive.input_usd_per_m / cheap.input_usd_per_m


def _declines_hold(cheap_model: str, expensive_model: str) -> dict[str, bool]:
    """Check every documented decline live, against real request shapes.

    ADR-015 asks for confirmation that the stage's guards "actually hold
    under real request shapes and none of them leak a case they were meant to
    protect". Free to check -- the stage makes no call to decide -- so it runs
    on every audit rather than being taken on trust from the eval gate.

    Args:
        cheap_model: The model routing would substitute.
        expensive_model: The model a caller asked for.

    Returns:
        Decline-case name -> whether the request was correctly left alone.
    """
    stage = RouteModelsStage()
    ctx = StageContext(
        config=OptimizeConfig(route_models=True, cheap_model=cheap_model),
        counter=default_counter(),
    )

    def declined(request: LLMRequest) -> bool:
        return stage.before(request, ctx).request.model == request.model

    base = ROUTING_PROBES[0].request(expensive_model)
    from dataclasses import replace

    return {
        "tools attached": declined(
            replace(base, tools=({"type": "function", "function": {"name": "search"}},))
        ),
        "response_format set": declined(replace(base, response_format={"type": "json_object"})),
        "already the cheap model": declined(replace(base, model=cheap_model)),
        "over the token ceiling": declined(
            replace(
                base,
                messages=(Message(role="user", content="word " * 2000),),
            )
        ),
        "routable request is routed": not declined(base),
    }


def run_routing_audit(
    expensive_provider: BenchProvider,
    cheap_provider: BenchProvider,
    *,
    probes: tuple[RoutingProbe, ...] = ROUTING_PROBES,
) -> RoutingAuditReport:
    """Ask both models every probe and grade them against known answers.

    Two calls per probe, one per model, through **two separate providers**.
    That is not incidental plumbing: ``OpenAIProvider.__call__`` sends
    ``self.model`` and ignores ``request.model`` entirely (deliberately -- it
    is what makes its pricing honest, see ``BenchProvider.model``). An earlier
    version of this function took one provider and varied ``request.model``,
    which would have sent both "arms" to the same model and produced a
    perfectly clean, entirely meaningless 0% regression rate.

    The models are called directly rather than through ``Optimizer`` because
    the question here is a capability comparison. Whether the *stage* would
    route a given probe is a separate question, checked against the stage and
    reported per probe as ``routed``.

    Args:
        expensive_provider: Serves the model a caller asked for.
        cheap_provider: Serves the model ``route_models`` would substitute.
        probes: Probes to run. Defaults to the full set.

    Returns:
        Per-probe grades, per-category accuracy, and the decline checks.

    Raises:
        ValueError: If both providers serve the same model, which would make
            every comparison below trivially equal.
    """
    expensive_model = expensive_provider.model
    cheap_model = cheap_provider.model
    if expensive_model == cheap_model:
        raise ValueError(
            f"both providers serve {expensive_model!r}; a routing audit comparing a "
            "model with itself measures nothing. Pass --cheap-model explicitly."
        )

    report = RoutingAuditReport(
        expensive_model=expensive_model,
        cheap_model=cheap_model,
        declines=_declines_hold(cheap_model, expensive_model),
    )
    optimizer = Optimizer(
        OptimizeConfig(
            route_models=True,
            cheap_model=cheap_model,
            exact_cache=False,
            prefix_cache=False,
            adaptive_max_tokens=False,
            structured_output=False,
            trim_history=False,
            deduplicate=False,
            prune_retrieval=False,
        )
    )
    stage = next(s for s in optimizer._pipeline.stages if isinstance(s, RouteModelsStage))
    ctx = StageContext(config=optimizer.config, counter=default_counter())

    for probe in probes:
        expensive_request = probe.request(expensive_model)
        routed = stage.before(expensive_request, ctx).request.model == cheap_model

        expensive = expensive_provider(expensive_request).content
        cheap = cheap_provider(probe.request(cheap_model)).content

        report.results.append(
            ProbeResult(
                probe=probe,
                routed=routed,
                expensive_answer=expensive,
                cheap_answer=cheap,
                expensive_correct=grade(expensive, probe),
                cheap_correct=grade(cheap, probe),
            )
        )
    return report


def format_routing_report(report: RoutingAuditReport) -> list[str]:
    """Render a routing audit as human-readable lines."""
    ratio = report.cost_ratio
    lines = [
        f"route_models audit -- {report.expensive_model} vs {report.cheap_model}"
        + (f" ({ratio:.1f}x cheaper input)" if ratio else ""),
        "",
        "accuracy on short prompts, graded against known answers:",
    ]
    grouped = report.by_category()
    for category in ("easy", "hard"):
        results = grouped.get(category, [])
        if not results:
            continue
        lines.append(
            f"  {category:<5} {report.expensive_model}: "
            f"{report.accuracy(results, cheap=False):.0%}   "
            f"{report.cheap_model}: {report.accuracy(results, cheap=True):.0%}   "
            f"({len(results)} probes)"
        )
    lines.append("")
    lines.append(
        f"regression rate: {report.regression_rate:.1%} "
        f"({sum(1 for r in report.results if r.regressed)}/{len(report.results)} probes "
        "the requested model got right and the cheap one did not)"
    )
    lines.append("")

    for category in ("easy", "hard"):
        results = grouped.get(category, [])
        if not results:
            continue
        lines.append(f"# {category}")
        for result in results:
            if result.regressed:
                marker = "REGRESSION"
            elif result.cheap_correct:
                marker = "cheap model ok"
            else:
                marker = "both wrong"
            lines.append(f"  [{marker}] {result.probe.question}")
            if not result.routed:
                lines.append("    (stage would NOT route this -- not routing evidence)")
            lines.append(f"    {report.expensive_model:<12} {result.expensive_answer!r}")
            lines.append(f"    {report.cheap_model:<12} {result.cheap_answer!r}")
            lines.append(f"    expected     {' | '.join(result.probe.expected)}")
        lines.append("")

    lines.append("decline guards, checked live against real request shapes:")
    for case, held in report.declines.items():
        lines.append(f"  [{'ok' if held else 'LEAKED'}] {case}")
    return lines


__all__ = [
    "ROUTING_PROBES",
    "ProbeResult",
    "RoutingAuditReport",
    "RoutingProbe",
    "format_routing_report",
    "grade",
    "run_routing_audit",
]
