"""`route_models`'s audit (ADR-015): grading, guards, and the two-provider rule.

The accuracy numbers themselves need `--live`; what is testable for free is
that the grader says the right thing about a given answer, that the audit
refuses to compare a model with itself, and that every decline guard the
stage documents still holds.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.providers import SimulatedProvider
from optio_optimize.bench.routing import (
    ROUTING_PROBES,
    ProbeResult,
    RoutingProbe,
    format_routing_report,
    grade,
    run_routing_audit,
)
from optio_optimize.types import LLMRequest, LLMResponse

pytestmark = pytest.mark.optimize


def _probe(*expected: str) -> RoutingProbe:
    return RoutingProbe("easy", "q?", expected)


class _EchoingProvider:
    """A provider whose answer *is* its model name, so callers are traceable."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.seen: list[LLMRequest] = []

    @property
    def is_live(self) -> bool:
        return False

    @property
    def models_latency(self) -> bool:
        return False

    @property
    def label(self) -> str:
        return f"echo({self.model})"

    def reset(self) -> None:
        self.seen.clear()

    def __call__(self, request: LLMRequest) -> LLMResponse:
        self.seen.append(request)
        return LLMResponse(content=self.model, model=self.model, input_tokens=1, output_tokens=1)


class TestGrade:
    """The grader that a live run's every number depends on."""

    def test_an_exact_answer_is_correct(self) -> None:
        assert grade("Tokyo", _probe("tokyo")) is True

    def test_a_trailing_period_does_not_make_a_right_answer_wrong(self) -> None:
        # The bug this test exists for: normalization kept every '.' so it
        # could preserve decimals, which left "tokyo." failing a word-boundary
        # check for "tokyo". Three correct live answers were scored wrong and
        # one was reported as a route_models REGRESSION.
        assert grade("Tokyo.", _probe("tokyo")) is True
        assert grade("No.", _probe("no")) is True
        assert grade("366 days.", _probe("366")) is True

    def test_an_answer_embedded_in_a_sentence_is_correct(self) -> None:
        # Both models get the same "be terse" instruction, so ignoring one
        # that answers in a sentence anyway would measure instruction-
        # following rather than capability.
        assert grade("The ball costs $0.05.", _probe("0.05")) is True

    def test_decimals_still_compare_as_decimals(self) -> None:
        assert grade("9.9", _probe("9.9")) is True
        assert grade("9.11", _probe("9.9")) is False

    def test_a_longer_number_is_not_a_match(self) -> None:
        # "3" must not match inside "30", or every counting probe passes.
        assert grade("30", _probe("3")) is False
        assert grade("366", _probe("36")) is False

    def test_a_substring_of_a_word_is_not_a_match(self) -> None:
        assert grade("gold", _probe("au")) is False
        assert grade("Australia", _probe("au")) is False

    def test_any_accepted_spelling_counts(self) -> None:
        probe = _probe("0.05", "5 cents", "five cents")
        assert grade("five cents", probe) is True
        assert grade("5 cents", probe) is True
        assert grade("$0.10", probe) is False

    def test_a_wrong_answer_is_wrong(self) -> None:
        assert grade("Kyoto", _probe("tokyo")) is False
        assert grade("", _probe("tokyo")) is False


class TestProbeSet:
    def test_hard_probes_outnumber_easy_ones(self) -> None:
        # A probe set of only lookups would confirm the length heuristic by
        # construction. The hard ones are what can falsify it, and there are
        # more of them because the first four turned out not to be hard for a
        # 2026 cheap model at all -- they are famous enough to be memorized.
        categories = [p.category for p in ROUTING_PROBES]
        assert categories.count("easy") == 4
        assert categories.count("hard") == 8

    def test_every_probe_is_deterministic_and_short_enough_to_route(self) -> None:
        from optio_optimize.stages.routing import MAX_ROUTABLE_TOKENS
        from optio_optimize.tokens import count_request, default_counter

        counter = default_counter()
        for probe in ROUTING_PROBES:
            request = probe.request("gpt-4o")
            assert request.temperature == 0.0
            assert count_request(request, counter) <= MAX_ROUTABLE_TOKENS

    def test_every_probe_states_its_expected_answer(self) -> None:
        for probe in ROUTING_PROBES:
            assert probe.expected
            assert all(e.strip() for e in probe.expected)


class TestTwoProviderRule:
    def test_comparing_a_model_with_itself_is_refused(self) -> None:
        # OpenAIProvider.__call__ sends self.model and ignores request.model,
        # so a single-provider version of this audit would have sent both
        # arms to the same place and reported a clean, meaningless 0%.
        provider = SimulatedProvider(model="gpt-4o")

        with pytest.raises(ValueError, match="measures nothing"):
            run_routing_audit(provider, provider)

    def test_each_provider_answers_for_its_own_model(self) -> None:
        # SimulatedProvider hashes message content only and ignores the model,
        # so it cannot show which provider answered -- these stubs can, and
        # that is the property that matters: a single-provider audit would
        # send both arms to the same model and never say so.
        expensive = _EchoingProvider("gpt-4o")
        cheap = _EchoingProvider("gpt-4o-mini")

        report = run_routing_audit(expensive, cheap)

        assert report.expensive_model == "gpt-4o"
        assert report.cheap_model == "gpt-4o-mini"
        assert all(r.expensive_answer == "gpt-4o" for r in report.results)
        assert all(r.cheap_answer == "gpt-4o-mini" for r in report.results)
        assert len(expensive.seen) == len(cheap.seen) == len(ROUTING_PROBES)

    def test_the_cost_ratio_comes_from_the_pricing_table(self) -> None:
        report = run_routing_audit(
            SimulatedProvider(model="gpt-4o"), SimulatedProvider(model="gpt-4o-mini")
        )

        assert report.cost_ratio == pytest.approx(2.50 / 0.15)

    def test_an_unpriced_model_reports_no_ratio_rather_than_a_guess(self) -> None:
        report = run_routing_audit(
            SimulatedProvider(model="gpt-4o"), SimulatedProvider(model="some-local-model")
        )

        assert report.cost_ratio is None


class TestDeclineGuards:
    def test_every_documented_decline_holds(self) -> None:
        # ADR-015 asks for confirmation that the guards "actually hold under
        # real request shapes and none of them leak a case they were meant to
        # protect". Free to check, so it runs on every audit.
        report = run_routing_audit(
            SimulatedProvider(model="gpt-4o"), SimulatedProvider(model="gpt-4o-mini")
        )

        assert report.declines == {
            "tools attached": True,
            "response_format set": True,
            "already the cheap model": True,
            "over the token ceiling": True,
            "routable request is routed": True,
        }

    def test_a_leaked_guard_is_rendered_distinctly(self) -> None:
        report = run_routing_audit(
            SimulatedProvider(model="gpt-4o"), SimulatedProvider(model="gpt-4o-mini")
        )
        report.declines["tools attached"] = False

        text = "\n".join(format_routing_report(report))

        assert "[LEAKED] tools attached" in text


class TestRegressionAccounting:
    @pytest.mark.parametrize(
        ("expensive_correct", "cheap_correct", "regressed", "why"),
        [
            (True, False, True, "right before routing, wrong after -- the failure"),
            (False, False, False, "both wrong: routing did not cause it"),
            (False, True, False, "cheap model right where the expensive one was not"),
            (True, True, False, "both right: routing was free"),
        ],
    )
    def test_only_right_then_wrong_counts_as_a_regression(
        self, expensive_correct: bool, cheap_correct: bool, regressed: bool, why: str
    ) -> None:
        result = ProbeResult(
            probe=ROUTING_PROBES[0],
            routed=True,
            expensive_answer="a",
            cheap_answer="b",
            expensive_correct=expensive_correct,
            cheap_correct=cheap_correct,
        )

        assert result.regressed is regressed, why

    def test_the_report_renders_both_categories_and_the_rate(self) -> None:
        report = run_routing_audit(
            SimulatedProvider(model="gpt-4o"), SimulatedProvider(model="gpt-4o-mini")
        )

        text = "\n".join(format_routing_report(report))

        assert "# easy" in text
        assert "# hard" in text
        assert "regression rate:" in text
        assert "16.7x cheaper input" in text
