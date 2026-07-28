"""``semantic_cache``'s adversarial workload: near-duplicate, different-answer pairs.

ADR-015 names this the highest-stakes stage in the package: a false positive
here does not shrink or reword an answer, it serves a **complete, confident,
wrong answer to a different question**, with nothing marking it as such. The
eval gate (``eval/cases.py``) checks the stage's *logic* deterministically and
for free; it cannot tell you whether real prompt shapes actually collide at
the shipped 0.97 threshold. This module is the live, direct measurement ADR-015
requires before that question gets an answer either way.

**Design.** Each pair shares a long, realistic context block (a support-policy
passage, the kind of thing a RAG pipeline retrieves) differing in exactly one
embedded fact -- a number, an entity, a negation, or a date -- the four shapes
a Jaccard-style word-overlap metric is structurally blind to, per ADR-015. The
shared context is deliberately long: a short question like "what is the
capital of France?" vs "...of Germany?" only shares 5 of 7 unique words
(Jaccard ~0.71), safely under the 0.97 default and proving nothing about the
threshold's real margin. A hundred-word shared passage with one differing
sentence pushes similarity close to 1.0, which is the actual production risk
shape -- a long retrieved context, one changed detail -- and the only shape
worth measuring against.

**Why no judge is required here.** Unlike ``compress_prompt``'s quality
question ("is the reworded answer still good"), a semantic-cache hit returns
the *seed* pair's stored text verbatim for the *near-duplicate* question.
Every pair is constructed so the two questions have different correct
answers, so a hit serves a wrong answer -- observable directly (served text
equals the seed's stored text, word for word) without a model or a human
grading anything.

**With one honest caveat, which the audit measures rather than assumes.**
"Constructed to have different answers" is a claim about the prompts, not an
observation about the model. If the model answers both halves identically --
most plausibly by declining, "the context does not specify" -- then a hit
serves an answer that happens to be right, and scoring it as a wrong answer
would inflate the failure rate. So every pair also makes a third,
cache-disabled call for the near-duplicate: when that baseline differs from
the seed's answer, the two questions provably differ live and a hit really is
wrong; when it does not, the probe is reported as **degenerate** and excluded
from the false-positive denominator instead of quietly counted either way.
That is also why the report prints a *hit rate* next to the false-positive
rate: the first characterises the threshold, the second characterises the
risk, and on a degenerate probe they are not the same number.

**And why there is a benign control set.** An adversarial workload alone can
only ever answer "how often is this wrong", which makes ``semantic_threshold
= 1.0`` look like a perfect score instead of what it is -- a stage that never
fires, whose byte-identical case ``exact_cache`` already serves losslessly and
for free. :data:`SEMANTIC_CACHE_BENIGN_PAIRS` mirrors the adversarial
questions one for one with a *reworded question over an identical context*:
the legitimate reuse this stage exists to capture. Reporting both rates at a
given threshold turns "is it safe" into the question actually worth asking --
"is there a setting where it is both safe and useful" -- and, live at
2026-07-29, the answer on this workload shape was no. See
``docs/optimize-benchmarks.md`` for the measured sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from optio_optimize.config import DEFAULT_SEMANTIC_THRESHOLD, OptimizeConfig
from optio_optimize.optimizer import Optimizer
from optio_optimize.stages.semantic_cache import SemanticCacheStage, _prompt_text
from optio_optimize.types import LLMRequest, Message

if TYPE_CHECKING:
    from optio_optimize.bench.providers import BenchProvider

#: Model the pairs are built at when the caller names none. Only matters for
#: token counting and the stage's per-model match scoping -- a live provider
#: serves ``provider.model`` regardless of what a request asks for -- but
#: ``run_semantic_cache_audit`` rebuilds the pairs at the provider's own model
#: anyway, so a report never claims a model that was not the one called.
DEFAULT_AUDIT_MODEL = "gpt-4o-mini"

#: Shared context every pair embeds, with exactly one fact differing between
#: the seed and near-duplicate half. ~100 words: long enough that a single
#: differing sentence is a small fraction of total vocabulary, which is what
#: pushes word-overlap similarity close to 1.0 -- the actual stress case.
_POLICY_CONTEXT = (
    "Per the current support agreement, response times are guaranteed within "
    "four business hours for priority tickets and within one business day "
    "for standard tickets. The agreement covers software defects, "
    "configuration assistance, and integration troubleshooting for the "
    "primary deployment. Support requests should include the account "
    "identifier, a description of the issue, and any relevant error logs "
    "so the assigned engineer can reproduce the problem without a back and "
    "forth. Escalations that miss the guaranteed window are routed "
    "automatically to a senior engineer on call. {fact} Renewal terms are "
    "reviewed annually and any changes are communicated at least sixty "
    "days in advance of the renewal date."
)


@dataclass(frozen=True, slots=True)
class NearDuplicatePair:
    """One adversarial probe: two prompts, one changed fact, different answers.

    Attributes:
        category: One of ``number``, ``entity``, ``negation``, ``time`` --
            the four ways a single word can flip the correct answer while
            leaving lexical similarity to the rest of the prompt almost
            untouched.
        seed: The request stored first, whose response becomes the cache
            entry a false positive would incorrectly reuse.
        near_duplicate: A request sharing nearly all of ``seed``'s wording
            but with a different correct answer.
        seed_fact: The embedded fact in ``seed``, for reporting.
        near_duplicate_fact: The embedded fact in ``near_duplicate``, for
            reporting -- what a false positive silently discards.
    """

    category: str
    seed: LLMRequest
    near_duplicate: LLMRequest
    seed_fact: str
    near_duplicate_fact: str

    def at_model(self, model: str) -> NearDuplicatePair:
        """Return this same probe with both requests retargeted to ``model``.

        ``SemanticCacheStage`` scopes matching per ``request.model`` and the
        token counter picks an encoding from it, so an audit run against a
        provider serving something other than :data:`DEFAULT_AUDIT_MODEL`
        must restate its requests rather than leave a stale name on them --
        otherwise the report names one model and the calls went to another.

        Only ``.model`` moves. An earlier version regenerated both prompts
        from this module's own template, which is correct for the built-in
        pairs and silently substitutes different prompts for a caller-supplied
        one -- the kind of quiet, plausible-looking substitution this whole
        module exists to detect, so it is worth naming here.

        Args:
            model: Model name to retarget both halves at.

        Returns:
            An equivalent pair whose two requests target ``model``, or
            ``self`` when it already does.
        """
        if self.seed.model == model and self.near_duplicate.model == model:
            return self
        return replace(
            self,
            seed=replace(self.seed, model=model),
            near_duplicate=replace(self.near_duplicate, model=model),
        )


def _request(question: str, fact: str, model: str) -> LLMRequest:
    context = _POLICY_CONTEXT.format(fact=fact)
    return LLMRequest(
        model=model,
        messages=(
            Message(role="system", content="Answer using only the context provided."),
            Message(role="user", content=f"Context:\n{context}\n\nQuestion: {question}"),
        ),
        temperature=0.0,
    )


def _pair(
    category: str,
    question: str,
    seed_fact: str,
    near_duplicate_fact: str,
    *,
    model: str = DEFAULT_AUDIT_MODEL,
) -> NearDuplicatePair:
    return NearDuplicatePair(
        category=category,
        seed=_request(question, seed_fact, model),
        near_duplicate=_request(question, near_duplicate_fact, model),
        seed_fact=seed_fact,
        near_duplicate_fact=near_duplicate_fact,
    )


def _benign_pair(
    question: str,
    rephrased: str,
    fact: str,
    *,
    model: str = DEFAULT_AUDIT_MODEL,
) -> NearDuplicatePair:
    """A control probe: same context, same fact, question reworded.

    The mirror image of an adversarial pair, and the half that keeps the
    audit honest. A threshold high enough to reject every adversarial pair
    also rejects these -- and these are the *legitimate* reuse
    ``semantic_cache`` exists to capture, so a hit here is the win, not the
    failure. Measuring only the adversarial set would make "threshold 1.0"
    look like a perfect answer instead of what it is: a stage that never
    fires.

    Carries ``category="benign"``, which
    :attr:`SemanticCacheAuditReport.false_positive_rate` never counts.
    """
    return NearDuplicatePair(
        category="benign",
        seed=_request(question, fact, model),
        near_duplicate=_request(rephrased, fact, model),
        seed_fact=fact,
        near_duplicate_fact=fact,
    )


#: Two pairs per category, eight total. Three live calls each (seed,
#: near-duplicate under semantic_cache, near-duplicate baseline), less one
#: per cache hit -- so a full default audit including the eight controls
#: costs 32-48 calls against a cheap model, measured at $0.0010-$0.0015.
#: Small enough to run often, real enough to mean something.
SEMANTIC_CACHE_ADVERSARIAL_PAIRS: tuple[NearDuplicatePair, ...] = (
    _pair(
        "number",
        "How many support seats does the plan include?",
        "The plan includes 50 support seats.",
        "The plan includes 500 support seats.",
    ),
    _pair(
        "number",
        "What is the guaranteed response time in hours for priority tickets?",
        "Priority tickets outside the standard window are guaranteed a response within 4 hours.",
        "Priority tickets outside the standard window are guaranteed a response within 12 hours.",
    ),
    _pair(
        "entity",
        "Which region is the primary deployment region?",
        "The primary deployment region is us-east-1.",
        "The primary deployment region is eu-west-1.",
    ),
    _pair(
        "entity",
        "Which team handles escalations that miss the guaranteed window?",
        "Escalations that miss the guaranteed window are handled by the Platform team.",
        "Escalations that miss the guaranteed window are handled by the Security team.",
    ),
    _pair(
        "negation",
        "Are weekend escalations covered under this plan?",
        "Weekend escalations are covered under this plan at no extra cost.",
        "Weekend escalations are not covered under this plan at no extra cost.",
    ),
    _pair(
        "negation",
        "Does this plan include on-call phone support?",
        "This plan includes on-call phone support for priority tickets.",
        "This plan does not include on-call phone support for priority tickets.",
    ),
    _pair(
        "time",
        "When is the next contract renewal due?",
        "The next contract renewal is due by March 1st.",
        "The next contract renewal is due by September 1st.",
    ),
    _pair(
        "time",
        "By what date must configuration changes be submitted for the next release?",
        "Configuration changes for the next release must be submitted by the 5th of the month.",
        "Configuration changes for the next release must be submitted by the 25th of the month.",
    ),
)

#: The control set: eight probes whose two halves ask the *same* question in
#: different words against an identical context, so a hit is the win rather
#: than the failure. Deliberately mirrors the adversarial questions one for
#: one, so the two rates at a given threshold are comparable rather than
#: measured on different material -- the comparison is the whole point. A
#: threshold that rejects every adversarial pair and every benign one has not
#: made the stage safe, it has switched it off, and ``exact_cache`` already
#: handles the byte-identical case losslessly and for free.
SEMANTIC_CACHE_BENIGN_PAIRS: tuple[NearDuplicatePair, ...] = (
    _benign_pair(
        "How many support seats does the plan include?",
        "What is the number of support seats included in the plan?",
        "The plan includes 50 support seats.",
    ),
    _benign_pair(
        "What is the guaranteed response time in hours for priority tickets?",
        "How many hours is the guaranteed response time for priority tickets?",
        "Priority tickets outside the standard window are guaranteed a response within 4 hours.",
    ),
    _benign_pair(
        "Which region is the primary deployment region?",
        "What region is the primary deployment in?",
        "The primary deployment region is us-east-1.",
    ),
    _benign_pair(
        "Which team handles escalations that miss the guaranteed window?",
        "What team is responsible for escalations that miss the guaranteed window?",
        "Escalations that miss the guaranteed window are handled by the Platform team.",
    ),
    _benign_pair(
        "Are weekend escalations covered under this plan?",
        "Does this plan cover weekend escalations?",
        "Weekend escalations are covered under this plan at no extra cost.",
    ),
    _benign_pair(
        "Does this plan include on-call phone support?",
        "Is on-call phone support included in this plan?",
        "This plan includes on-call phone support for priority tickets.",
    ),
    _benign_pair(
        "When is the next contract renewal due?",
        "What is the due date of the next contract renewal?",
        "The next contract renewal is due by March 1st.",
    ),
    _benign_pair(
        "By what date must configuration changes be submitted for the next release?",
        "What is the submission deadline for configuration changes in the next release?",
        "Configuration changes for the next release must be submitted by the 5th of the month.",
    ),
)

#: What ``--semantic-cache-audit`` runs by default: both halves, because
#: either one alone is misleading. The adversarial set alone makes a very
#: high threshold look perfect; the benign set alone makes a low one look
#: excellent.
SEMANTIC_CACHE_AUDIT_PAIRS: tuple[NearDuplicatePair, ...] = (
    SEMANTIC_CACHE_ADVERSARIAL_PAIRS + SEMANTIC_CACHE_BENIGN_PAIRS
)


@dataclass(slots=True)
class PairResult:
    """Outcome of running one adversarial pair through a live semantic cache.

    Attributes:
        pair: The probe that produced this result.
        threshold: ``semantic_threshold`` the stage was configured with.
        similarity: The actual lexical similarity the stage computed between
            the two prompts, regardless of whether it cleared the threshold --
            the number that determines whether this pair stress-tests the
            default at all.
        seed_response: What the model said for ``pair.seed``.
        near_duplicate_response: What was actually returned for
            ``pair.near_duplicate`` -- the seed's cached text, verbatim, if
            the stage fired.
        near_duplicate_baseline: What the model actually says for
            ``pair.near_duplicate`` when asked directly, semantic cache off.
            The answer a real (uncached) call would have produced.
        served_from_cache: Whether ``SemanticCacheStage`` served the
            near-duplicate request from the seed's cache entry.
    """

    pair: NearDuplicatePair
    threshold: float
    similarity: float
    seed_response: str
    near_duplicate_response: str
    near_duplicate_baseline: str
    served_from_cache: bool

    @property
    def is_benign(self) -> bool:
        """Whether this is a control probe rather than an adversarial one.

        A benign pair's two halves have the *same* correct answer, so a hit
        is the behaviour the stage exists for, not a failure. Benign results
        never enter the false-positive numerator or denominator; they are
        reported separately as a hit rate, which is what says whether a
        threshold safe enough to reject the adversarial set leaves the stage
        able to do anything at all.
        """
        return self.pair.category == "benign"

    @property
    def probe_is_valid(self) -> bool:
        """Whether this adversarial pair actually probed anything, live.

        The pairs are *constructed* so the two questions have different
        correct answers, but construction is a claim about the prompts, not
        an observation about the model. If the model answers both halves the
        same way -- most plausibly by declining ("the context does not
        specify") -- then a cache hit on this pair serves an answer that
        happens to be right, and counting it as a wrong answer would
        overstate the failure rate.

        ``near_duplicate_baseline`` is the call that settles it: what the
        model really says for the near-duplicate, with the cache off. When
        that differs from ``seed_response``, the two questions demonstrably
        have different answers live and a hit here really would be wrong.
        When it does not, the probe is degenerate and is reported as such
        rather than folded into the headline number. Always ``False`` for a
        benign control, which is not an adversarial probe at all.
        """
        if self.is_benign:
            return False
        return self.near_duplicate_baseline.strip() != self.seed_response.strip()

    @property
    def false_positive(self) -> bool:
        """Whether this pair demonstrates the failure mode ADR-015 names.

        True when the stage served the near-duplicate from cache *and* the
        probe is valid -- i.e. a stored answer was reused for a question the
        model provably answers differently. A hit on a degenerate probe is
        still a hit (see :attr:`served_from_cache`, which is the number that
        characterises the *threshold*) but it is not evidence of a wrong
        answer, which is the number that characterises the *risk*.
        """
        return self.served_from_cache and self.probe_is_valid


@dataclass(slots=True)
class SemanticCacheAuditReport:
    """The full run: every pair, every category, the aggregate rate.

    Attributes:
        results: One entry per pair, in the order run.
        model: Model the audit was run against, for pricing/attribution.
    """

    results: list[PairResult] = field(default_factory=list)
    model: str = ""

    @property
    def valid_probes(self) -> list[PairResult]:
        """Adversarial probes the model demonstrably answers two different ways.

        The denominator for :attr:`false_positive_rate`. A degenerate probe
        (see :attr:`PairResult.probe_is_valid`) cannot distinguish a wrong
        answer from a right one, so counting it either way would be a made-up
        number. Benign controls are never in here -- they are not adversarial
        probes and have their own rate.
        """
        return [r for r in self.results if r.probe_is_valid]

    @property
    def adversarial_results(self) -> list[PairResult]:
        """Every probe built to have two different correct answers."""
        return [r for r in self.results if not r.is_benign]

    @property
    def benign_results(self) -> list[PairResult]:
        """Every control probe: reworded question, same correct answer."""
        return [r for r in self.results if r.is_benign]

    @property
    def hit_rate(self) -> float:
        """Fraction of *adversarial* pairs where the stage fired at all.

        Characterises the *threshold*: how often a near-duplicate clears it.
        Reported next to, never instead of, :attr:`false_positive_rate` --
        they answer different questions and on a degenerate probe they differ.
        """
        adversarial = self.adversarial_results
        if not adversarial:
            return 0.0
        return sum(1 for r in adversarial if r.served_from_cache) / len(adversarial)

    @property
    def benign_hit_rate(self) -> float:
        """Fraction of control probes where the stage fired -- the *win* rate.

        The number that stops a high threshold from looking free. Raising
        ``semantic_threshold`` until the adversarial set stops colliding is
        only progress if legitimate paraphrases still hit; if this falls to
        zero at the same time, the "safe" setting is one where the stage
        does nothing, and ``exact_cache`` (lossless, on by default) already
        covers the byte-identical case for free.

        Note this counts *firing*, not answer correctness: a benign pair's
        two halves share the same context and the same embedded fact, so the
        answer being reused is the answer to the same question. That is a
        weaker claim than a judge would make and it is the only one this
        measurement supports.
        """
        benign = self.benign_results
        if not benign:
            return 0.0
        return sum(1 for r in benign if r.served_from_cache) / len(benign)

    @property
    def false_positive_rate(self) -> float:
        """Fraction of *valid* probes where the cache served a wrong answer.

        Zero when no probe was valid -- which is not a safety result and is
        why :attr:`valid_probes` is reported alongside it. A run whose probes
        all turned out degenerate measured nothing, and the report says so
        rather than printing a reassuring 0.0%.
        """
        valid = self.valid_probes
        if not valid:
            return 0.0
        return sum(1 for r in valid if r.false_positive) / len(valid)

    def by_category(self) -> dict[str, list[PairResult]]:
        """Group results by adversarial category, for a per-shape breakdown."""
        grouped: dict[str, list[PairResult]] = {}
        for result in self.results:
            grouped.setdefault(result.pair.category, []).append(result)
        return grouped


def run_semantic_cache_audit(
    provider: BenchProvider,
    *,
    pairs: tuple[NearDuplicatePair, ...] = SEMANTIC_CACHE_AUDIT_PAIRS,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> SemanticCacheAuditReport:
    """Run every adversarial pair live and measure the false-positive rate.

    Three calls per pair: the seed (stored), the near-duplicate through a
    fresh ``SemanticCacheStage`` (does it fire?), and the near-duplicate again
    through the bare provider (what the correct answer actually looks like,
    for the report -- not needed to *detect* a false positive, only to show
    what it cost).

    A fresh ``Optimizer`` per pair, not one shared across all eight: sharing
    one would let an earlier pair's seed collide with a later pair's
    near-duplicate, measuring cross-pair contamination instead of the
    within-pair failure mode this audit targets.

    Args:
        provider: Where calls go. Live for real evidence; a simulator can
            exercise the plumbing but returns synthetic content, so it cannot
            demonstrate a wrong-answer collision the way a live run can.
        pairs: Probes to run. Defaults to the full adversarial set.
        threshold: ``semantic_threshold`` to configure the stage with -- run
            once at the shipped default and again at a lower value to
            characterize margin, per ADR-015.

    Returns:
        Every pair's outcome and the aggregate false-positive rate.
    """
    config = OptimizeConfig(
        exact_cache=False,
        prefix_cache=False,
        adaptive_max_tokens=False,
        structured_output=False,
        trim_history=False,
        deduplicate=False,
        prune_retrieval=False,
        semantic_cache=True,
        semantic_threshold=threshold,
    )

    report = SemanticCacheAuditReport(model=provider.model)
    for base_pair in pairs:
        # A live provider serves its own model whatever the request names, so
        # restate the request at that model rather than let the report claim
        # a pairing that never happened.
        pair = base_pair.at_model(provider.model)
        optimizer = Optimizer(config)
        provider.reset()

        seed_response = optimizer.call(pair.seed, provider)
        near_dup_response = optimizer.call(pair.near_duplicate, provider)
        baseline_response = provider(pair.near_duplicate)

        # Reaches into the stage for its own similarity function, the same
        # way harness._cache_counters() reaches into a cache stage's backend
        # for hit/miss counts: the number belongs to the stage, not anything
        # this module tracks independently. Found by name rather than by
        # index -- config makes it the only stage today, but a similarity
        # number silently taken off some *other* stage is exactly the kind of
        # wrong-but-plausible figure this audit exists to catch.
        stage = next(s for s in optimizer._pipeline.stages if isinstance(s, SemanticCacheStage))
        # _prompt_text, not a local re-join: if the stage ever changes how it
        # renders a request for comparison, a copy here would keep reporting
        # a similarity the stage did not actually use.
        similarity = stage._similarity_fn(
            _prompt_text(pair.seed),
            _prompt_text(pair.near_duplicate),
        )

        report.results.append(
            PairResult(
                pair=pair,
                threshold=threshold,
                similarity=similarity,
                seed_response=seed_response.content,
                near_duplicate_response=near_dup_response.content,
                near_duplicate_baseline=baseline_response.content,
                served_from_cache=near_dup_response.served_from == "semantic_cache",
            )
        )
    return report


def format_audit_report(report: SemanticCacheAuditReport) -> list[str]:
    """Render an audit report as human-readable lines."""
    valid = report.valid_probes
    adversarial = report.adversarial_results
    benign = report.benign_results
    degenerate = len(adversarial) - len(valid)
    hits = sum(1 for r in adversarial if r.served_from_cache)
    benign_hits = sum(1 for r in benign if r.served_from_cache)
    threshold = report.results[0].threshold if report.results else 0.0
    lines = [
        f"semantic_cache adversarial audit -- model {report.model}, threshold {threshold:.2f}",
        f"false-positive rate: {report.false_positive_rate:.1%} "
        f"({sum(1 for r in valid if r.false_positive)}/{len(valid)} valid adversarial probes)"
        "   <- wrong answers served",
        f"benign hit rate:     {report.benign_hit_rate:.1%} "
        f"({benign_hits}/{len(benign)} control probes)"
        "   <- legitimate reuse captured",
        f"adversarial hits:    {report.hit_rate:.1%} "
        f"({hits}/{len(adversarial)} cleared the threshold)",
    ]
    if degenerate:
        lines.append(
            f"  note: {degenerate} adversarial probe(s) excluded as degenerate -- the model "
            "gave the same answer to both halves, so a hit there proves nothing either way."
        )
    if not valid and adversarial:
        lines.append(
            "  WARNING: no adversarial probe was valid. The 0.0% above measured nothing; "
            "it is not a safety result."
        )
    if benign and not benign_hits and not hits:
        lines.append(
            "  WARNING: this threshold fired on nothing at all, adversarial or benign. "
            "That is not a safe setting, it is an off one -- and exact_cache already "
            "covers the byte-identical case losslessly."
        )
    lines.append("")
    for category, results in report.by_category().items():
        lines.append(f"# {category}")
        for result in results:
            if result.is_benign:
                marker = "hit (the win)" if result.served_from_cache else "missed (no reuse)"
            elif not result.probe_is_valid:
                marker = "DEGENERATE PROBE"
            elif result.false_positive:
                marker = "FALSE POSITIVE"
            else:
                marker = "correctly missed"
            lines.append(
                f"  [{marker}] similarity={result.similarity:.4f} "
                f"(threshold={result.threshold:.2f}, "
                f"{'fired' if result.served_from_cache else 'declined'})"
            )
            if result.is_benign:
                lines.append(f"    shared fact:        {result.pair.seed_fact}")
                lines.append(f"    seed answer:        {result.seed_response!r}")
                lines.append(f"    reworded answer:    {result.near_duplicate_response!r}")
                continue
            lines.append(f"    seed fact:          {result.pair.seed_fact}")
            lines.append(f"    near-dup fact:      {result.pair.near_duplicate_fact}")
            lines.append(f"    seed answer:        {result.seed_response!r}")
            if result.false_positive:
                lines.append(f"    SERVED (wrong):     {result.near_duplicate_response!r}")
                lines.append(f"    should have said:   {result.near_duplicate_baseline!r}")
            else:
                lines.append(f"    near-dup answer:    {result.near_duplicate_response!r}")
        lines.append("")
    return lines
