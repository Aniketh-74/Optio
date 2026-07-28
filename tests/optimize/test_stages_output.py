"""AdaptiveMaxTokensStage and StructuredOutputStage: the output-side stages.

Both had only indirect coverage through the pipeline/benchmark suites before
this file -- output.py sat at 54% coverage, the lowest of any stage module,
despite billing at 3-5x the input rate per its own module docstring.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.output import (
    FLOOR_TOKENS,
    MIN_OBSERVATIONS,
    AdaptiveMaxTokensStage,
    StructuredOutputStage,
    _percentile,
)
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(**overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": "gpt-4o",
        "messages": (Message(role="user", content="hi"),),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


def _response(output_tokens: int, **overrides: object) -> LLMResponse:
    defaults: dict[str, object] = {
        "content": "x",
        "input_tokens": 10,
        "output_tokens": output_tokens,
        "model": "gpt-4o",
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)  # type: ignore[arg-type]


class TestAdaptiveMaxTokensStage:
    def test_declines_when_the_caller_already_set_a_ceiling(self) -> None:
        """Overriding an explicit instruction, in either direction,
        substitutes our guess for what the caller actually asked for."""
        stage = AdaptiveMaxTokensStage()

        result = stage.before(_request(max_tokens=500), _ctx())

        assert not result.short_circuited
        assert result.request.max_tokens == 500

    def test_declines_before_enough_observations_exist(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        for _ in range(MIN_OBSERVATIONS - 1):
            stage.after(_request(), _response(300), ctx)

        result = stage.before(_request(), ctx)

        assert result.request.max_tokens is None
        assert result.note == ""

    def test_imposes_a_ceiling_once_enough_observations_exist(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        for _ in range(MIN_OBSERVATIONS):
            stage.after(_request(), _response(300), ctx)

        result = stage.before(_request(), ctx)

        assert result.request.max_tokens is not None
        assert result.request.max_tokens >= FLOOR_TOKENS
        assert "ceiling" in result.note
        assert result.saved_output_tokens >= 0

    def test_the_ceiling_never_goes_below_the_floor(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        for _ in range(MIN_OBSERVATIONS):
            stage.after(_request(), _response(10), ctx)  # tiny observed lengths

        result = stage.before(_request(), ctx)

        assert result.request.max_tokens == FLOOR_TOKENS

    def test_a_cached_response_is_not_recorded_as_an_observation(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        cached = _response(9999, served_from="exact_cache")
        for _ in range(MIN_OBSERVATIONS):
            stage.after(_request(), cached, ctx)

        # None of those counted, so the stage still has no real sample.
        result = stage.before(_request(), ctx)
        assert result.request.max_tokens is None

    def test_a_zero_length_response_is_not_recorded(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        for _ in range(MIN_OBSERVATIONS):
            stage.after(_request(), _response(0), ctx)

        result = stage.before(_request(), ctx)
        assert result.request.max_tokens is None

    def test_the_observation_history_stays_bounded(self) -> None:
        stage = AdaptiveMaxTokensStage()
        ctx = _ctx()
        for _ in range(1500):
            stage.after(_request(), _response(300), ctx)

        assert len(stage._lengths) <= 1000


class TestStructuredOutputStage:
    def test_declines_when_no_response_format_or_tools(self) -> None:
        stage = StructuredOutputStage()

        result = stage.before(_request(), _ctx())

        assert not result.short_circuited
        assert result.note == ""

    def test_fires_when_response_format_is_set(self) -> None:
        stage = StructuredOutputStage()
        request = _request(response_format={"type": "json_object"})

        result = stage.before(request, _ctx())

        assert result.note == "preamble suppressed"
        assert stage.INSTRUCTION in result.request.messages[0].content

    def test_fires_when_tools_are_set_even_without_a_response_format(self) -> None:
        stage = StructuredOutputStage()
        request = _request(tools=({"name": "search"},))

        result = stage.before(request, _ctx())

        assert result.note == "preamble suppressed"

    def test_appends_to_an_existing_system_message_rather_than_inserting_a_new_one(self) -> None:
        stage = StructuredOutputStage()
        request = LLMRequest(
            model="gpt-4o",
            messages=(
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ),
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = stage.before(request, _ctx())

        assert len(result.request.messages) == 2
        assert result.request.messages[0].content == f"be terse\n{stage.INSTRUCTION}"
        assert result.request.messages[0].role == "system"

    def test_inserts_a_new_system_message_when_none_exists(self) -> None:
        stage = StructuredOutputStage()
        request = _request(response_format={"type": "json_object"})  # only a user message

        result = stage.before(request, _ctx())

        assert len(result.request.messages) == 2
        assert result.request.messages[0].role == "system"
        assert result.request.messages[0].content == stage.INSTRUCTION

    def test_does_not_apply_twice(self) -> None:
        stage = StructuredOutputStage()
        request = _request(response_format={"type": "json_object"})

        first = stage.before(request, _ctx())
        second = stage.before(first.request, _ctx())

        assert second.note == ""
        assert second.request.messages == first.request.messages

    def test_saved_output_tokens_never_goes_negative(self) -> None:
        """If the instruction itself costs more than the typical preamble it
        suppresses, the net saving floors at zero rather than going negative.
        """

        class HugeCounter(HeuristicCounter):
            def count_text(self, text: str, model: str = "") -> int:
                return 1000

        stage = StructuredOutputStage()
        request = _request(response_format={"type": "json_object"})
        ctx = StageContext(config=OptimizeConfig(), counter=HugeCounter())

        result = stage.before(request, ctx)

        assert result.saved_output_tokens == 0


class TestPercentile:
    def test_empty_values_return_zero(self) -> None:
        assert _percentile([], 0.5) == 0.0

    def test_the_median_of_a_list(self) -> None:
        assert _percentile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_a_high_percentile_never_goes_out_of_bounds(self) -> None:
        assert _percentile([10, 20], 0.99) == 20.0
