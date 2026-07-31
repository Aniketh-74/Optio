"""A cost delta is only a measurement when the arms differ (ADR-028).

The live Anthropic run of 2026-07-31 reported a cost percentage for all twelve
workloads. Five of them send **byte-identical requests in both arms and make the
same number of provider calls** -- verified offline by capturing every request
each arm hands the provider:

===================== ==================== =========================
workload              reported             optimizer actually did
===================== ==================== =========================
``timestamped_agent`` **-1.6%**            nothing
``sampled_creative``  **-4.7%**            nothing
``unique_questions``  **+2.8%**            nothing
``multi_turn_chat``   0.0%                 nothing
``tool_calling_chat`` 0.0%                 nothing
===================== ==================== =========================

Those percentages are the provider's own output nondeterminism, and they fail in
both directions at once. The two negatives read as ADR-013 rule 1 violations --
a cost increase caused by the optimizer, the one outcome this package treats as
unacceptable -- and the previous measurement iteration opened by planning a live
isolation run to find the stage responsible. No stage was responsible. The
positive landed on ``unique_questions``, whose stated purpose is "Included so the
suite reports its own limits", and had it claiming a saving where the library did
nothing at all.

``--control`` exists to measure exactly this nondeterminism, and
``QualityResult.is_interpretable`` already applies the lesson to the quality
line. Nothing carried it across to cost.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.metrics import ABResult, ArmResult, QualityResult, sent_fingerprint
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _request(**overrides: object) -> LLMRequest:
    base: dict[str, object] = {
        "model": "claude-haiku-4-5",
        "messages": (
            Message(role="system", content="You are terse."),
            Message(role="user", content="Question 0."),
        ),
        "temperature": 0.0,
    }
    base.update(overrides)
    return LLMRequest(**base)  # type: ignore[arg-type]


def _result(*, base_digest: str, opt_digest: str, base_calls: int, opt_calls: int) -> ABResult:
    return ABResult(
        workload="w",
        model="claude-haiku-4-5",
        baseline=ArmResult(
            name="baseline",
            requests=12,
            provider_calls=base_calls,
            input_tokens=19_200,
            output_tokens=600,
            sent_digest=base_digest,
        ),
        optimized=ArmResult(
            name="optimized",
            requests=12,
            provider_calls=opt_calls,
            input_tokens=19_200,
            output_tokens=610,
            sent_digest=opt_digest,
        ),
        quality=QualityResult(),
    )


class TestTheFingerprintSeesWhatReachesTheWire:
    def test_the_same_request_fingerprints_the_same(self) -> None:
        assert sent_fingerprint(_request()) == sent_fingerprint(_request())

    def test_changed_content_changes_the_fingerprint(self) -> None:
        other = _request(
            messages=(
                Message(role="system", content="You are terse."),
                Message(role="user", content="Question 1."),
            )
        )

        assert sent_fingerprint(_request()) != sent_fingerprint(other)

    def test_a_cache_marker_changes_the_fingerprint(self) -> None:
        """``prefix_cache`` changes nothing but ``cacheable``, and it changes spend.

        A breakpoint moves prompt tokens onto the write and read rates. A
        fingerprint blind to it would call the run a no-op and throw away the
        largest lossless saving this package has.
        """
        marked = _request(
            messages=(
                Message(role="system", content="You are terse.", cacheable=True),
                Message(role="user", content="Question 0."),
            )
        )

        assert sent_fingerprint(_request()) != sent_fingerprint(marked)

    def test_max_tokens_is_included_though_request_key_omits_it(self) -> None:
        """ADR-028 decision 2, and the one case a borrowed definition breaks.

        ``cache.request_key`` leaves ``max_tokens`` out on purpose -- a cached
        completion can be reused across differing limits by checking
        ``finish_reason``. That reasoning is about reuse, not change detection.
        ``adaptive_max_tokens`` is on by default and changes that field and no
        other, so a fingerprint borrowed from ``request_key`` would file a run
        where only that stage fired as a no-op and discard a real saving.
        """
        capped = _request(max_tokens=256)

        assert sent_fingerprint(_request()) != sent_fingerprint(capped)

    def test_the_fingerprint_is_a_fixed_width_digest(self) -> None:
        """Section 10 binds the benchmark too: a digest, never the bytes.

        Asserting "the prompt is not in the output" would pass no matter what
        this function does, because the final ``blake2b`` masks anything the
        payload contains -- the identical mis-aimed test that let a raw base64
        image payload sit inside a cache key until mutation testing caught it.
        The falsifiable property is the one §10 actually relies on: whatever
        goes in, a fixed-width hex digest comes out.
        """
        long_prompt = "clause " * 5_000
        fingerprint = sent_fingerprint(
            _request(messages=(Message(role="user", content=long_prompt),))
        )

        assert len(fingerprint) == 32
        assert set(fingerprint) <= set("0123456789abcdef")


class TestANoOpRunIsNotAMeasurement:
    def test_identical_sends_and_call_counts_are_not_attributable(self) -> None:
        result = _result(base_digest="abc", opt_digest="abc", base_calls=12, opt_calls=12)

        assert result.cost_is_attributable is False

    def test_a_changed_request_is_attributable(self) -> None:
        result = _result(base_digest="abc", opt_digest="def", base_calls=12, opt_calls=12)

        assert result.cost_is_attributable is True

    def test_a_saved_provider_call_is_attributable(self) -> None:
        """A cache hit changes the call count without changing what was sent."""
        result = _result(base_digest="abc", opt_digest="abc", base_calls=15, opt_calls=1)

        assert result.cost_is_attributable is True

    def test_the_delta_itself_is_still_reported(self) -> None:
        """The dollars were really spent; both figures are true observations.

        What is false is reading the ratio as a saving, and a large delta on a
        no-op run is itself a signal -- it means the provider is noisier than
        assumed.
        """
        result = _result(base_digest="abc", opt_digest="abc", base_calls=12, opt_calls=12)

        assert result.cost_reduction is not None


class TestTheReportNamesNoiseRatherThanPrintingAPercentage:
    def test_an_unattributable_cost_line_says_so(self) -> None:
        from optio_optimize.bench.harness import format_result

        lines = format_result(
            _result(base_digest="abc", opt_digest="abc", base_calls=12, opt_calls=12)
        )

        cost_line = next(line for line in lines if "cost" in line)
        assert "NOT ATTRIBUTABLE" in cost_line

    def test_an_unattributable_cost_line_omits_the_percentage(self) -> None:
        """-1.6% is what a reader mistakes for an ADR-013 violation."""
        from optio_optimize.bench.harness import format_result

        lines = format_result(
            _result(base_digest="abc", opt_digest="abc", base_calls=12, opt_calls=12)
        )

        cost_line = next(line for line in lines if "cost" in line)
        assert "%" not in cost_line

    def test_an_unattributable_cost_line_keeps_both_dollar_figures(self) -> None:
        from optio_optimize.bench.harness import format_result

        lines = format_result(
            _result(base_digest="abc", opt_digest="abc", base_calls=12, opt_calls=12)
        )

        cost_line = next(line for line in lines if "cost" in line)
        assert cost_line.count("$") == 2

    def test_an_attributable_cost_line_still_reports_its_percentage(self) -> None:
        from optio_optimize.bench.harness import format_result

        lines = format_result(
            _result(base_digest="abc", opt_digest="def", base_calls=12, opt_calls=12)
        )

        cost_line = next(line for line in lines if "cost" in line)
        assert "%" in cost_line
        assert "NOT ATTRIBUTABLE" not in cost_line
