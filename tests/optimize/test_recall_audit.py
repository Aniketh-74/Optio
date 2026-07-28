"""`summarize_history`'s recall audit (ADR-015): the three arms and the exclusions.

Recall numbers need `--live` and a real summarizer. What is testable for free
is that the workload actually buries its planted facts, that the control arm
gates the other two, that a wrong answer is counted differently from a missing
one, and that the tool-call boundary the stage promises still holds.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.providers import SimulatedProvider
from optio_optimize.bench.recall import (
    RECALL_PROBES,
    ArmAnswer,
    RecallAuditReport,
    RecallProbe,
    RecallResult,
    build_conversation,
    format_recall_report,
    run_recall_audit,
)
from optio_optimize.config import DEFAULT_RECENT_TURNS

pytestmark = pytest.mark.optimize


def _summarizer(text: str) -> str:
    return f"summary of {len(text)} chars"


def _answer(correct: bool, declined: bool = False) -> ArmAnswer:
    return ArmAnswer(answer="x", correct=correct, declined=declined, input_tokens=10)


def _report(*rows: tuple[bool, bool, bool, bool]) -> RecallAuditReport:
    """Build a report from (full, trim, summarize, summarize_declined) tuples."""
    report = RecallAuditReport(model="gpt-4o")
    for full, trim, summarize, declined in rows:
        result = RecallResult(probe=RECALL_PROBES[0])
        result.arms = {
            "full": _answer(full),
            "trim": _answer(trim, declined=not trim),
            "summarize": _answer(summarize, declined=declined),
        }
        report.results.append(result)
    return report


class TestTheWorkloadActuallyBuriesItsFacts:
    def test_the_planted_facts_are_in_the_first_exchange(self) -> None:
        messages = build_conversation("gpt-4o")

        assert messages[0].role == "system"
        for probe in RECALL_PROBES:
            assert probe.fact in messages[1].content

    def test_no_planted_fact_is_repeated_later(self) -> None:
        # If a fact appeared again inside the recent window, trimming would
        # keep it and every arm would score the same for the wrong reason.
        messages = build_conversation("gpt-4o")

        for probe in RECALL_PROBES:
            later = [m for m in messages[2:] if probe.fact in m.content]
            assert later == []

    def test_the_facts_fall_outside_the_recent_turns_window(self) -> None:
        # The load-bearing assertion of this whole audit: if the planting
        # exchange survived trimming, the summarize arm would have nothing to
        # prove and a 100% recall score would mean nothing.
        messages = build_conversation("gpt-4o")
        history = [m for m in messages if m.role != "system"]

        window = history[-DEFAULT_RECENT_TURNS:]
        for probe in RECALL_PROBES:
            assert all(probe.fact not in m.content for m in window)

    def test_every_probe_states_an_expected_answer(self) -> None:
        for probe in RECALL_PROBES:
            assert probe.expected
            assert probe.fact


class TestControlArmGating:
    def test_a_probe_the_control_arm_missed_is_excluded(self) -> None:
        # Scoring a probe the model fails even with the whole conversation
        # in front of it would attribute a model limitation to this library.
        report = _report((False, False, False, False))

        assert report.interpretable == []
        assert report.recall_rate("summarize") == 0.0

    def test_rates_are_over_interpretable_probes_only(self) -> None:
        report = _report(
            (True, False, True, False),  # counts: summarize recalled
            (False, False, False, False),  # excluded
        )

        assert len(report.interpretable) == 1
        assert report.recall_rate("summarize") == 1.0
        assert report.recall_rate("trim") == 0.0

    def test_the_report_marks_excluded_probes(self) -> None:
        text = "\n".join(format_recall_report(_report((False, False, False, False))))

        assert "[EXCLUDED: control arm wrong]" in text
        assert "measure the model, not the stage" in text


class TestSilentErrorsAreCountedSeparately:
    def test_a_declined_answer_is_a_visible_loss_not_a_silent_one(self) -> None:
        # trim_history losing a fact is the acceptable failure: the model says
        # it does not have the context.
        report = _report((True, False, False, True))

        assert report.recall_rate("summarize") == 0.0
        assert report.silent_error_rate("summarize") == 0.0

    def test_a_confidently_wrong_answer_is_the_silent_failure(self) -> None:
        report = _report((True, False, False, False))

        assert report.recall_rate("summarize") == 0.0
        assert report.silent_error_rate("summarize") == 1.0

    def test_a_correct_answer_is_never_a_silent_error(self) -> None:
        report = _report((True, True, True, False))

        assert report.recall_rate("summarize") == 1.0
        assert report.silent_error_rate("summarize") == 0.0

    def test_the_report_names_the_three_verdicts_distinctly(self) -> None:
        text = "\n".join(
            format_recall_report(_report((True, True, True, False), (True, False, False, False)))
        )

        assert "[recalled]" in text
        assert "[lost (said so)]" in text
        assert "[SILENTLY WRONG]" in text


class TestArms:
    def test_all_three_arms_run_on_every_probe(self) -> None:
        report = run_recall_audit(SimulatedProvider(), _summarizer)

        assert len(report.results) == len(RECALL_PROBES)
        for result in report.results:
            assert set(result.arms) == {"full", "trim", "summarize"}

    def test_only_the_summarize_arm_pays_for_extra_calls(self) -> None:
        # The cost side of the comparison: trim_history is free, this is not.
        report = run_recall_audit(SimulatedProvider(), _summarizer)

        assert report.extra_calls("summarize") == len(RECALL_PROBES)
        assert report.extra_calls("trim") == 0
        assert report.extra_calls("full") == 0

    def test_the_trimmed_arm_sends_less_than_the_full_one(self) -> None:
        report = run_recall_audit(SimulatedProvider(), _summarizer)

        full = sum(r.arms["full"].input_tokens for r in report.results)
        trim = sum(r.arms["trim"].input_tokens for r in report.results)
        summarize = sum(r.arms["summarize"].input_tokens for r in report.results)
        assert trim < full
        assert summarize < full


class TestToolCallBoundary:
    def test_the_boundary_the_stage_promises_holds(self) -> None:
        # tool_calling_chat confirmed this live for trim_history;
        # SummarizeHistoryStage asserts the same invariant in its docstring,
        # so it is checked rather than inherited.
        report = run_recall_audit(SimulatedProvider(), _summarizer)

        assert report.tool_boundary_safe is True

    def test_a_violation_is_rendered_distinctly(self) -> None:
        report = _report((True, True, True, False))
        report.tool_boundary_safe = False

        assert "tool-call boundary: VIOLATED" in "\n".join(format_recall_report(report))


class TestGradingReusesTheRoutingMatcher:
    def test_an_answer_in_a_sentence_still_counts(self) -> None:
        from optio_optimize.bench.routing import RoutingProbe, grade

        probe = RecallProbe("q?", ("47 500", "47500"), "f")
        as_routing = RoutingProbe("easy", probe.question, probe.expected)

        assert grade("The approved budget is $47,500.", as_routing) is True
        assert grade("The approved budget is $45,000.", as_routing) is False

    def test_an_ordinal_suffix_does_not_break_a_date_match(self) -> None:
        # The grader's word-boundary check counts "th" as part of the token,
        # so "March 14th" does not match "march 14". Caught live: the deadline
        # probe was excluded as control-arm-wrong when the control had
        # answered it correctly. Both spellings are listed rather than
        # teaching the matcher about English ordinals.
        from optio_optimize.bench.routing import RoutingProbe, grade

        deadline = next(p for p in RECALL_PROBES if "deadline" in p.question)
        as_routing = RoutingProbe("easy", deadline.question, deadline.expected)

        assert grade("The hard deadline is March 14th.", as_routing) is True
        assert grade("The hard deadline is March 14.", as_routing) is True
        assert grade("The hard deadline is March 20th.", as_routing) is False
