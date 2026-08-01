"""A caller can supply their vendor's tokenizer (ADR-042).

Every savings figure this package produces is a token count, and every token
count comes from :class:`~optio_optimize.tokens.TokenCounter` -- a two-method
Protocol that anything can implement. ``Pipeline`` has accepted one since
ADR-038.

:class:`~optio_optimize.optimizer.Optimizer` did not, and it is the public entry
point. So the extension point existed and **nothing outside this package could
reach it**: every count, on every vendor, went through ``tiktoken``.

That is not a small approximation. ADR-036 measured Anthropic billing **1.29x**
what the raw JSON tokenizes to for tool schemas, against OpenAI's 0.65 -- the
two vendors differ in direction, not just magnitude. Applying one tokenizer to
both is the same error, one layer down, and it is the layer every reported
figure rests on.

Nothing here claims a default counter is wrong. It claims a caller who knows
better must be able to say so, which is what a multi-vendor library owes the
vendors it does not measure.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


class _CountingCounter:
    """A stand-in for a vendor tokenizer, which records what it was asked."""

    is_exact = True

    def __init__(self, chars_per_token: int = 4) -> None:
        self.calls: list[tuple[str, str]] = []
        self._ratio = chars_per_token

    def count_text(self, text: str, model: str = "") -> int:
        self.calls.append((text, model))
        return len(text) // self._ratio


def _answer(request: LLMRequest) -> LLMResponse:
    """A provider that costs nothing and reports plausible usage."""
    return LLMResponse(content="ok", input_tokens=1_000, output_tokens=5)


def _request(words: int = 4_000, model: str = "claude-haiku-4-5") -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(
            Message(role="system", content="Follow the schedule exactly. " * 200),
            Message(role="user", content=" ".join(["reconcile"] * words)),
            Message(role="assistant", content="noted"),
            Message(role="user", content="continue"),
        ),
        temperature=0.0,
    )


class TestTheOptimizerAcceptsACounter:
    def test_a_supplied_counter_is_the_one_used(self) -> None:
        counter = _CountingCounter()

        Optimizer(counter=counter).call(_request(), _answer)

        assert counter.calls, "the supplied counter was never consulted"

    def test_it_is_the_one_the_stages_see(self) -> None:
        """Reaching the pipeline is not enough; the stages do the counting.

        ``trim_history`` and ``prefix_cache`` both size the prompt, and a
        counter that arrived at the pipeline but not at ``StageContext`` would
        satisfy a shallower test while changing nothing about the numbers.
        """
        counter = _CountingCounter()
        optimizer = Optimizer(counter=counter)

        optimizer.call(_request(), _answer)

        # The system prompt is what `prefix_cache` sizes against its floor, and
        # the history is what `trim_history` measures. Both must have gone
        # through the caller's counter, not a private one.
        counted = "".join(text for text, _ in counter.calls)
        assert "Follow the schedule exactly." in counted

    def test_omitting_it_still_works(self) -> None:
        """The default is unchanged for every existing caller."""
        assert Optimizer().call(_request(), _answer).content == "ok"

    def test_a_counter_changes_the_reported_numbers(self) -> None:
        """The point. Two tokenizers disagreeing must produce two answers.

        If a supplied counter could be accepted and then ignored, this test is
        the only thing that notices -- and "accepted then ignored" is exactly
        the shape of an extension point nobody wired up.

        Uses a provider that reports **no** usage, because one that reports
        usage should override any estimate and does: ``Pipeline.complete`` reads
        ``response.input_tokens or count_request(...)``. That precedence is
        correct and worth stating -- a counter only ever fills a gap the
        provider left, so this is the case where its choice is visible.
        """

        def _answer_without_usage(request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="ok")

        generous = Optimizer(counter=_CountingCounter(chars_per_token=2))
        frugal = Optimizer(counter=_CountingCounter(chars_per_token=8))
        request = _request()

        generous.call(request, _answer_without_usage)
        frugal.call(request, _answer_without_usage)

        assert generous.report.baseline_input_tokens != frugal.report.baseline_input_tokens

    def test_a_provider_that_reports_usage_still_wins(self) -> None:
        """The precedence the test above relies on, asserted rather than assumed.

        A counter is an estimate; the provider's own number is the bill. If a
        supplied counter could override it, plugging in a vendor tokenizer would
        make reports *less* accurate on every provider that reports usage --
        the opposite of the reason for accepting one.
        """
        optimizer = Optimizer(counter=_CountingCounter(chars_per_token=2))

        optimizer.call(_request(), _answer)

        assert optimizer.report.actual_input_tokens == 1_000

    def test_the_warm_up_uses_it_too(self) -> None:
        """ADR-038 warms the counter at construction. A supplied counter is the
        one that needs warming; warming the default instead pays the cost twice
        and still starves request one."""
        counter = _CountingCounter()

        Optimizer(counter=counter)

        assert counter.calls, "construction did not warm the supplied counter"


class TestItComposesWithTheRestOfTheConstructor:
    def test_it_works_alongside_config_overrides(self) -> None:
        counter = _CountingCounter()

        optimizer = Optimizer(counter=counter, trim_history=False)

        assert optimizer.config.trim_history is False

    def test_it_works_alongside_an_explicit_stage_list(self) -> None:
        """``stages=`` bypasses the registry; the counter must still arrive."""
        from optio_optimize.stages import build_stages

        counter = _CountingCounter()
        config = OptimizeConfig()

        Optimizer(config, stages=build_stages(config), counter=counter).call(_request(), _answer)

        assert counter.calls
