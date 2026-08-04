"""Inline outcome heuristics (M5-2).

The load-bearing property is the one that looks like a gap: **the heuristic never
reports success.** It reports failure on evidence and abstains otherwise.

That asymmetry is the point. Section 1.3 names "well-formed but wrong" as the
failure permission-based governance cannot see, and a heuristic cannot see it
either -- a fluent wrong answer is indistinguishable from a fluent right one
without reading the content. A heuristic that emitted `success=true` because
nothing looked broken would manufacture exactly the false assurance the quality
lane exists to replace, and would do it on every run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from opentelemetry.trace import StatusCode

from optio import semconv
from optio.lanes.quality.heuristic import UNKNOWN, HeuristicResult, project, score

if TYPE_CHECKING:
    from collections.abc import Mapping


def span(
    attributes: Mapping[str, object] | None = None,
    *,
    errored: bool = False,
) -> Mock:
    """Build a stand-in for a finished ReadableSpan."""
    mock = Mock()
    mock.name = "step"
    mock.attributes = attributes or {}
    mock.status = Mock(status_code=StatusCode.ERROR if errored else StatusCode.OK)
    return mock


def answered(tokens: int = 50) -> Mock:
    """A span that looks like a normal completed generation."""
    return span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: tokens})


class TestFailureIsDetected:
    def test_an_errored_step_is_a_failure(self) -> None:
        result = score(project(span({}, errored=True)))
        assert result.failed
        assert result.succeeded is False

    def test_no_output_tokens_is_a_failure(self) -> None:
        result = score(project(answered(tokens=0)))
        assert result.failed
        assert result.succeeded is False

    def test_a_truncated_generation_is_a_failure(self) -> None:
        # Reads as complete text up to the cut, but the task did not finish.
        result = score(project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: ["length"]})))
        assert result.failed
        assert "incomplete" in (result.reason or "")

    def test_a_flattened_finish_reason_is_still_read(self) -> None:
        # Upstream types this as an array, but several instrumentations flatten
        # it. Rejecting the string form would silently disable this check.
        result = score(project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: "max_tokens"})))
        assert result.failed

    def test_a_content_filter_stop_is_a_failure(self) -> None:
        result = score(project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: ["content_filter"]})))
        assert result.failed


class TestSuccessIsNeverClaimed:
    """The asymmetry, tested directly."""

    def test_a_normal_answer_does_not_report_success(self) -> None:
        result = score(project(answered()))
        assert result.succeeded is None, "heuristic must never assert success"

    def test_a_long_answer_does_not_report_success(self) -> None:
        # More output is not more correct. A confidently wrong essay scores the
        # same as a confidently right one here, because they are identical from
        # the outside.
        assert score(project(answered(tokens=100_000))).succeeded is None

    def test_a_completed_run_is_conclusive_but_not_successful(self) -> None:
        # "It ran and produced output" is a real conclusion; "it did the right
        # thing" is not one this tier can reach.
        result = score(project(answered()))
        assert result.conclusive
        assert not result.failed
        assert result.succeeded is None


class TestAbsenceIsUnknown:
    def test_no_spans_is_unknown(self) -> None:
        assert score(None) == UNKNOWN
        assert score(None).succeeded is None

    def test_a_span_with_no_usage_attribute_is_unknown(self) -> None:
        # Some instrumentations omit token counts. Reporting failure here would
        # flag every run from those users.
        result = score(project(span({semconv.GEN_AI_REQUEST_MODEL: "gpt-4o"})))
        assert not result.conclusive
        assert result.succeeded is None

    def test_a_non_integer_token_count_is_unknown(self) -> None:
        result = score(project(span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: "fifty"})))
        assert not result.conclusive

    def test_a_boolean_token_count_is_not_read_as_a_number(self) -> None:
        # bool is a subclass of int; True would otherwise pass as "1 token".
        result = score(project(span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: True})))
        assert not result.conclusive

    def test_an_unreadable_finish_reason_does_not_crash(self) -> None:
        for value in (42, {"a": 1}, None, [None, 7]):
            result = score(project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: value})))
            assert not result.failed


class TestNoContentIsRetained:
    """Section 10: the heuristic reads output but keeps none of it."""

    def test_the_result_holds_no_run_content(self) -> None:
        secret = "patient SSN 123-45-6789"
        result = score(
            project(
                span(
                    {
                        semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10,
                        "gen_ai.completion": secret,
                        "gen_ai.prompt": secret,
                    }
                )
            )
        )

        assert secret not in repr(result)
        assert secret not in str(result.reason)

    def test_the_reason_is_a_fixed_vocabulary(self) -> None:
        # Reasons reach logs. They must describe a category, never quote the run.
        results = [
            score(project(span({}, errored=True))),
            score(project(answered(tokens=0))),
            score(project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: ["length"]}))),
        ]
        for result in results:
            assert result.reason is not None
            assert len(result.reason) < 60


class TestResultIsInert:
    def test_the_result_is_frozen(self) -> None:
        result = HeuristicResult(failed=False, conclusive=True)
        try:
            result.failed = True  # type: ignore[misc]
        except AttributeError:
            return
        raise AssertionError("HeuristicResult should be immutable")
