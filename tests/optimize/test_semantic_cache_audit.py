"""`semantic_cache`'s adversarial audit (ADR-015) -- the plumbing, for free.

These run against `SimulatedProvider`, so they check that the audit correctly
detects a hit/miss and computes the false-positive rate from it -- not
whether a real model would actually be fooled, which needs `--live` and is
not something a deterministic test can demonstrate (see the module docstring
in `bench/adversarial.py`).
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.adversarial import (
    SEMANTIC_CACHE_ADVERSARIAL_PAIRS,
    SEMANTIC_CACHE_AUDIT_PAIRS,
    SEMANTIC_CACHE_BENIGN_PAIRS,
    NearDuplicatePair,
    format_audit_report,
    run_semantic_cache_audit,
)
from optio_optimize.bench.providers import SimulatedProvider
from optio_optimize.similarity import jaccard
from optio_optimize.stages.semantic_cache import _prompt_text
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


class TestAdversarialPairs:
    def test_eight_pairs_across_four_categories(self) -> None:
        assert len(SEMANTIC_CACHE_ADVERSARIAL_PAIRS) == 8
        categories = {p.category for p in SEMANTIC_CACHE_ADVERSARIAL_PAIRS}
        assert categories == {"number", "entity", "negation", "time"}

    def test_every_pair_has_two_of_its_category(self) -> None:
        counts: dict[str, int] = {}
        for pair in SEMANTIC_CACHE_ADVERSARIAL_PAIRS:
            counts[pair.category] = counts.get(pair.category, 0) + 1
        assert all(count == 2 for count in counts.values())

    def test_seed_and_near_duplicate_are_deterministic_requests(self) -> None:
        # The stage only ever stores/matches temperature==0 requests; a pair
        # built at temperature != 0 would silently never exercise anything.
        for pair in SEMANTIC_CACHE_ADVERSARIAL_PAIRS:
            assert pair.seed.temperature == 0.0
            assert pair.near_duplicate.temperature == 0.0

    def test_seed_and_near_duplicate_differ_in_content(self) -> None:
        for pair in SEMANTIC_CACHE_ADVERSARIAL_PAIRS:
            assert pair.seed.messages != pair.near_duplicate.messages
            assert pair.seed_fact != pair.near_duplicate_fact


class TestRunSemanticCacheAudit:
    def test_reports_one_result_per_pair(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())

        assert len(report.results) == len(SEMANTIC_CACHE_AUDIT_PAIRS)

    def test_a_hit_serves_the_seeds_exact_response_text(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())

        hits = [r for r in report.adversarial_results if r.served_from_cache]
        assert hits, "expected at least one collision at the default threshold"
        for hit in hits:
            assert hit.near_duplicate_response == hit.seed_response
            assert hit.false_positive is True

    def test_every_simulated_adversarial_probe_is_valid(self) -> None:
        # SimulatedProvider hashes the request text, so the near-duplicate's
        # uncached answer can never coincide with the seed's. Degenerate
        # probes are a live-model phenomenon (both halves declining the same
        # way); this asserts the simulator cannot manufacture one, which is
        # what makes the other tests here interpretable.
        report = run_semantic_cache_audit(SimulatedProvider())

        assert all(r.probe_is_valid for r in report.adversarial_results)
        assert len(report.valid_probes) == len(report.adversarial_results)

    def test_a_degenerate_probe_is_excluded_from_the_false_positive_rate(self) -> None:
        # A hit whose baseline answer matches the seed's proves nothing: the
        # served text was right anyway. Counted in hit_rate, excluded from
        # false_positive_rate.
        report = run_semantic_cache_audit(SimulatedProvider())
        hit = next(r for r in report.results if r.served_from_cache)
        hit.near_duplicate_baseline = hit.seed_response

        assert hit.probe_is_valid is False
        assert hit.false_positive is False
        assert hit not in report.valid_probes
        assert report.hit_rate > 0.0

    def test_an_all_degenerate_run_reports_zero_without_claiming_safety(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())
        for result in report.results:
            result.near_duplicate_baseline = result.seed_response

        assert report.valid_probes == []
        assert report.false_positive_rate == 0.0
        assert "WARNING: no adversarial probe was valid" in "\n".join(format_audit_report(report))

    def test_pairs_are_rebuilt_at_the_providers_own_model(self) -> None:
        # The stage scopes matching per request.model and the token counter
        # picks an encoding from it, so a report naming one model while the
        # requests named another would be measuring something it did not say.
        provider = SimulatedProvider(model="gpt-4o")

        report = run_semantic_cache_audit(provider)

        assert report.model == "gpt-4o"
        assert all(r.pair.seed.model == "gpt-4o" for r in report.results)
        assert all(r.pair.near_duplicate.model == "gpt-4o" for r in report.results)

    def test_reported_similarity_is_the_stages_own_metric(self) -> None:
        # Not a re-implementation of the join: the number must come from the
        # same text the stage compared, or the margin it characterises is not
        # the margin the stage actually had.
        report = run_semantic_cache_audit(SimulatedProvider())

        for result in report.results:
            expected = jaccard(
                _prompt_text(result.pair.seed),
                _prompt_text(result.pair.near_duplicate),
            )
            assert result.similarity == pytest.approx(expected)
            assert result.served_from_cache == (result.similarity >= result.threshold)

    def test_a_miss_returns_a_response_matching_the_near_duplicate_prompt(self) -> None:
        # SimulatedProvider's content is a deterministic hash of the request
        # text, so a real (uncached) call for the near-duplicate must differ
        # from the seed's -- proving a "miss" actually asked the provider
        # rather than silently reusing something.
        report = run_semantic_cache_audit(SimulatedProvider())

        misses = [r for r in report.results if not r.served_from_cache]
        for miss in misses:
            assert miss.near_duplicate_response != miss.seed_response
            assert miss.false_positive is False

    def test_the_configs_own_hard_floor_collides_on_every_pair(self) -> None:
        # OptimizeConfig rejects semantic_threshold below 0.9 outright
        # (config.py: "below 0.9 unrelated prompts match"), so 0.9 -- not an
        # arbitrarily low number -- is the lowest value this audit can even
        # construct. Every pair here was built with a shared ~100-word
        # context and one differing sentence specifically to clear even the
        # *default* 0.97, so all eight colliding at the floor is expected,
        # not a bug in the test.
        report = run_semantic_cache_audit(SimulatedProvider(), threshold=0.9)

        assert report.false_positive_rate == 1.0

    def test_a_perfect_threshold_never_collides(self) -> None:
        # 1.0 requires the two prompts to be byte-identical, which no pair
        # here is by construction.
        report = run_semantic_cache_audit(SimulatedProvider(), threshold=1.0)

        assert report.false_positive_rate == 0.0

    def test_false_positive_rate_matches_the_adversarial_hit_fraction(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())

        adversarial = report.adversarial_results
        hits = sum(1 for r in adversarial if r.served_from_cache)
        assert report.false_positive_rate == hits / len(adversarial)

    def test_by_category_groups_every_result_exactly_once(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())

        grouped = report.by_category()
        assert sum(len(v) for v in grouped.values()) == len(report.results)
        assert set(grouped) == {"number", "entity", "negation", "time", "benign"}

    def test_a_custom_pair_set_is_honoured(self) -> None:
        custom = (
            NearDuplicatePair(
                category="number",
                seed=LLMRequest(
                    model="gpt-4o-mini",
                    messages=(Message(role="user", content="The value is 1. What is the value?"),),
                    temperature=0.0,
                ),
                near_duplicate=LLMRequest(
                    model="gpt-4o-mini",
                    messages=(Message(role="user", content="The value is 2. What is the value?"),),
                    temperature=0.0,
                ),
                seed_fact="The value is 1.",
                near_duplicate_fact="The value is 2.",
            ),
        )

        report = run_semantic_cache_audit(SimulatedProvider(), pairs=custom)

        assert len(report.results) == 1
        # Retargeted to the provider's model, but otherwise the caller's own
        # pair: an earlier version of at_model() regenerated prompts from this
        # module's built-in policy template, silently replacing a custom
        # probe's text with something it never asked for.
        ran = report.results[0].pair
        assert ran.seed.messages == custom[0].seed.messages
        assert ran.near_duplicate.messages == custom[0].near_duplicate.messages
        assert ran.seed_fact == custom[0].seed_fact
        assert ran.seed.model == SimulatedProvider().model

    def test_at_model_is_identity_when_the_model_already_matches(self) -> None:
        pair = SEMANTIC_CACHE_ADVERSARIAL_PAIRS[0]

        assert pair.at_model(pair.seed.model) is pair


class TestBenignControlSet:
    """The half that stops a high threshold from looking free."""

    def test_the_control_set_mirrors_the_adversarial_one(self) -> None:
        assert len(SEMANTIC_CACHE_BENIGN_PAIRS) == len(SEMANTIC_CACHE_ADVERSARIAL_PAIRS)
        assert all(p.category == "benign" for p in SEMANTIC_CACHE_BENIGN_PAIRS)

    def test_both_halves_of_a_control_pair_carry_the_same_fact(self) -> None:
        # The point of a control: identical context, identical embedded fact,
        # only the question's wording differs. A control whose fact differed
        # would be an adversarial pair wearing the wrong label.
        for pair in SEMANTIC_CACHE_BENIGN_PAIRS:
            assert pair.seed_fact == pair.near_duplicate_fact
            assert pair.seed.messages != pair.near_duplicate.messages

    def test_the_default_audit_runs_both_halves(self) -> None:
        assert SEMANTIC_CACHE_AUDIT_PAIRS == (
            SEMANTIC_CACHE_ADVERSARIAL_PAIRS + SEMANTIC_CACHE_BENIGN_PAIRS
        )

        report = run_semantic_cache_audit(SimulatedProvider())

        assert len(report.adversarial_results) == len(SEMANTIC_CACHE_ADVERSARIAL_PAIRS)
        assert len(report.benign_results) == len(SEMANTIC_CACHE_BENIGN_PAIRS)

    def test_a_control_hit_is_never_counted_as_a_false_positive(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider(), pairs=SEMANTIC_CACHE_BENIGN_PAIRS)

        assert report.valid_probes == []
        assert report.false_positive_rate == 0.0
        assert all(not r.probe_is_valid and not r.false_positive for r in report.results)

    def test_benign_hit_rate_counts_only_control_probes(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())

        benign = report.benign_results
        expected = sum(1 for r in benign if r.served_from_cache) / len(benign)
        assert report.benign_hit_rate == expected

    def test_raising_the_threshold_never_helps_the_control_set_more_than_the_adversarial(
        self,
    ) -> None:
        # The finding this control set exists to make visible, asserted as a
        # property rather than left in a doc: on this workload shape there is
        # no threshold that rejects near-duplicates while still capturing
        # legitimate paraphrases, because the two sets score in the same
        # similarity band. If a future similarity_fn ever separated them, this
        # test failing is the signal that the recommendation should be revisited.
        for threshold in (0.90, 0.95, 0.97, 0.98, 0.99, 1.0):
            report = run_semantic_cache_audit(SimulatedProvider(), threshold=threshold)
            assert report.benign_hit_rate <= report.hit_rate, (
                f"at threshold {threshold} the control set outscored the adversarial "
                "set -- the similarity metric now separates them and ADR-015's "
                "semantic_cache resolution needs re-deriving"
            )

    def test_a_threshold_that_fires_on_nothing_is_flagged_not_congratulated(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider(), threshold=1.0)
        text = "\n".join(format_audit_report(report))

        assert report.false_positive_rate == 0.0
        assert report.benign_hit_rate == 0.0
        assert "fired on nothing at all" in text


class TestFormatAuditReport:
    def test_renders_the_false_positive_rate_and_every_category(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())

        lines = format_audit_report(report)
        text = "\n".join(lines)

        assert "false-positive rate" in text
        for category in ("number", "entity", "negation", "time"):
            assert f"# {category}" in text

    def test_marks_false_positives_distinctly_from_correct_misses(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider())
        text = "\n".join(format_audit_report(report))

        has_hit = any(r.false_positive for r in report.results)
        has_miss = any(not r.false_positive for r in report.results)
        if has_hit:
            assert "FALSE POSITIVE" in text
        if has_miss:
            assert "correctly missed" in text

    def test_an_empty_report_does_not_crash(self) -> None:
        report = run_semantic_cache_audit(SimulatedProvider(), pairs=())

        lines = format_audit_report(report)

        assert any("0/0" in line for line in lines)
