"""SemanticCacheStage: serving near-matching deterministic requests."""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.semantic_cache import SemanticCacheStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx(*, threshold: float = 0.9) -> StageContext:
    return StageContext(
        config=OptimizeConfig(semantic_threshold=threshold), counter=HeuristicCounter()
    )


def _request(content: str, *, model: str = "gpt-4o", temperature: float | None = 0.0) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(Message(role="user", content=content),),
        temperature=temperature,
    )


def _response(content: str = "answer", **overrides: object) -> LLMResponse:
    defaults: dict[str, object] = {
        "content": content,
        "input_tokens": 50,
        "output_tokens": 10,
        "model": "gpt-4o",
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)  # type: ignore[arg-type]


class TestSemanticCacheStage:
    def test_a_miss_declines(self) -> None:
        stage = SemanticCacheStage()

        result = stage.before(_request("what is the capital of france"), _ctx())

        assert not result.short_circuited

    def test_sampled_requests_are_never_cached(self) -> None:
        stage = SemanticCacheStage()
        request = _request("hello", temperature=0.9)
        ctx = _ctx()
        stage.after(request, _response(), ctx)

        result = stage.before(request, ctx)

        assert not result.short_circuited

    def test_a_none_temperature_is_not_treated_as_deterministic(self) -> None:
        stage = SemanticCacheStage()
        request = _request("hello", temperature=None)
        ctx = _ctx()
        stage.after(request, _response(), ctx)

        assert not stage.before(request, ctx).short_circuited

    def test_a_byte_identical_request_hits(self) -> None:
        stage = SemanticCacheStage()
        text = "what is the capital of france"
        ctx = _ctx(threshold=0.97)
        stage.before(_request(text), ctx)
        stage.after(_request(text), _response("Paris"), ctx)

        result = stage.before(_request(text), ctx)

        assert result.short_circuited
        assert result.response is not None
        assert result.response.content == "Paris"
        assert result.response.served_from == "semantic_cache"
        assert "semantic hit" in result.note

    def test_an_unrelated_request_never_hits(self) -> None:
        stage = SemanticCacheStage()
        ctx = _ctx(threshold=0.5)  # generous threshold; still must not match
        stage.before(_request("what is the capital of france"), ctx)
        stage.after(_request("what is the capital of france"), _response("Paris"), ctx)

        result = stage.before(_request("explain quantum entanglement"), ctx)

        assert not result.short_circuited

    def test_a_stored_response_is_zeroed_on_the_hit(self) -> None:
        stage = SemanticCacheStage()
        text = "what is the capital of france"
        ctx = _ctx(threshold=0.97)
        stage.before(_request(text), ctx)
        stage.after(_request(text), _response(input_tokens=100, output_tokens=20), ctx)

        result = stage.before(_request(text), ctx)

        assert result.response is not None
        assert result.response.input_tokens == 0
        assert result.response.output_tokens == 0
        assert result.saved_input_tokens == 100
        assert result.saved_output_tokens == 20

    def test_entries_are_scoped_per_model(self) -> None:
        """A response from one model must not answer for another -- different
        capability and style, not just a cached fact."""
        stage = SemanticCacheStage()
        text = "what is the capital of france"
        ctx = _ctx(threshold=0.97)
        stage.before(_request(text, model="gpt-4o"), ctx)
        stage.after(_request(text, model="gpt-4o"), _response("Paris"), ctx)

        result = stage.before(_request(text, model="claude-sonnet-4"), ctx)

        assert not result.short_circuited

    def test_after_does_not_store_a_response_already_served_from_a_cache(self) -> None:
        stage = SemanticCacheStage()
        text = "what is the capital of france"
        ctx = _ctx()
        already_cached = _response(served_from="exact_cache")

        stage.after(_request(text), already_cached, ctx)

        assert not stage.before(_request(text), ctx).short_circuited

    def test_a_custom_similarity_function_is_used_instead_of_the_default(self) -> None:
        always_match = SemanticCacheStage(similarity_fn=lambda a, b: 1.0)
        ctx = _ctx(threshold=0.97)
        always_match.before(_request("apples"), ctx)
        always_match.after(_request("apples"), _response("fruit"), ctx)

        result = always_match.before(_request("completely different text"), ctx)

        assert result.short_circuited
        assert result.response is not None
        assert result.response.content == "fruit"

    def test_the_store_stays_bounded(self) -> None:
        stage = SemanticCacheStage(max_entries=10)
        ctx = _ctx()
        for i in range(50):
            request = _request(f"unique question number {i}")
            stage.after(request, _response(f"answer {i}"), ctx)

        assert len(stage._entries) <= 10
