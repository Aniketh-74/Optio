"""ExactCacheStage and PrefixCacheStage: the two lossless cache mechanisms.

Both stages already have indirect coverage through test_pipeline.py and the
benchmark suite, but neither had a dedicated unit test until now -- unusual
given caching.py is this package's own "moat" analog. These exercise each
stage directly, including the paths only reachable through specific
before()/after() sequencing (a truncated stored reply, a response that
already came from a cache reaching after() a second time).
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.caching import ExactCacheStage, PrefixCacheStage, served_from_cache
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(content: str = "hello", *, temperature: float | None = 0.0) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=(
            Message(role="system", content="You are terse."),
            Message(role="user", content=content),
        ),
        temperature=temperature,
    )


def _response(**overrides: object) -> LLMResponse:
    defaults: dict[str, object] = {
        "content": "answer",
        "input_tokens": 100,
        "output_tokens": 20,
        "model": "gpt-4o",
        "finish_reason": "stop",
    }
    defaults.update(overrides)
    return LLMResponse(**defaults)  # type: ignore[arg-type]


class TestExactCacheStage:
    def test_a_miss_declines(self) -> None:
        stage = ExactCacheStage()
        request = _request()

        result = stage.before(request, _ctx())

        assert not result.short_circuited
        assert result.request is request

    def test_sampled_requests_are_never_cached(self) -> None:
        stage = ExactCacheStage()
        request = _request(temperature=0.9)

        result = stage.before(request, _ctx())

        assert not result.short_circuited

    def test_a_none_temperature_is_not_treated_as_deterministic(self) -> None:
        """The provider default is not zero -- None must not qualify."""
        stage = ExactCacheStage()

        result = stage.before(_request(temperature=None), _ctx())

        assert not result.short_circuited

    def test_a_stored_response_is_served_on_the_next_matching_request(self) -> None:
        stage = ExactCacheStage()
        request = _request()
        ctx = _ctx()
        stage.before(request, ctx)  # populates ctx.scratch with the cache key
        response = _response()

        stage.after(request, response, ctx)
        hit = stage.before(request, _ctx())

        assert hit.short_circuited
        assert hit.response is not None
        assert hit.response.content == "answer"
        assert hit.response.served_from == "exact_cache"
        assert hit.saved_input_tokens == response.input_tokens
        assert hit.saved_output_tokens == response.output_tokens
        assert hit.note == "exact hit"

    def test_a_truncated_stored_reply_is_never_served(self) -> None:
        """finish_reason='length' means a caller who allowed more output
        would silently be capped at whatever ceiling applied the first time.
        """
        stage = ExactCacheStage()
        request = _request()
        ctx = _ctx()
        stage.before(request, ctx)
        stage.after(request, _response(finish_reason="length"), ctx)

        result = stage.before(request, _ctx())

        assert not result.short_circuited

    def test_saved_input_falls_back_to_the_estimator_when_the_stored_response_has_none(
        self,
    ) -> None:
        stage = ExactCacheStage()
        request = _request()
        ctx = _ctx()
        stage.before(request, ctx)
        stage.after(request, _response(input_tokens=0), ctx)

        hit = stage.before(request, _ctx())

        assert hit.saved_input_tokens > 0

    def test_after_does_not_re_store_a_response_that_already_came_from_a_cache(self) -> None:
        """A response with served_from set already came from some cache;
        writing it again would be indistinguishable from storing it fresh
        and is the wrong provenance to persist under this request's key.
        """
        stage = ExactCacheStage()
        request = _request()
        ctx = _ctx()
        stage.before(request, ctx)
        already_cached = served_from_cache(_response(), "some_other_stage")

        stage.after(request, already_cached, ctx)

        # Nothing was stored: a fresh lookup still misses.
        assert not stage.before(request, _ctx()).short_circuited

    def test_after_with_no_matching_before_is_a_safe_no_op(self) -> None:
        stage = ExactCacheStage()
        ctx = _ctx()  # scratch never populated by a before() call

        stage.after(_request(), _response(), ctx)  # must not raise


class TestServedFromCache:
    def test_zeroes_token_counts_and_records_provenance(self) -> None:
        original = _response(input_tokens=100, output_tokens=20, cached_input_tokens=10)

        marked = served_from_cache(original, "exact_cache")

        assert marked.input_tokens == 0
        assert marked.output_tokens == 0
        assert marked.cached_input_tokens == 0
        assert marked.served_from == "exact_cache"
        assert marked.content == original.content


class TestPrefixCacheStage:
    def test_a_long_system_prompt_gets_marked(self) -> None:
        stage = PrefixCacheStage()
        request = LLMRequest(
            model="gpt-4o",
            messages=(
                Message(role="system", content="x" * 5000),
                Message(role="user", content="hi"),
            ),
            temperature=0.0,
        )

        result = stage.before(request, _ctx())

        assert not result.short_circuited
        assert result.saved_input_tokens == 0  # never claims avoided tokens
        assert result.request.messages[0].cacheable is True
        assert "prefix marked" in result.note

    def test_a_short_system_prompt_is_below_the_provider_floor(self) -> None:
        stage = PrefixCacheStage()
        request = LLMRequest(
            model="gpt-4o",
            messages=(Message(role="system", content="short"), Message(role="user", content="hi")),
            temperature=0.0,
        )

        result = stage.before(request, _ctx())

        assert not result.short_circuited
        assert result.note == ""

    def test_no_messages_at_all_declines(self) -> None:
        stage = PrefixCacheStage()
        request = LLMRequest(model="gpt-4o", messages=(), temperature=0.0)

        result = stage.before(request, _ctx())

        assert not result.short_circuited

    def test_multiple_system_messages_all_qualify_unconditionally(self) -> None:
        stage = PrefixCacheStage()
        request = LLMRequest(
            model="gpt-4o",
            messages=(
                Message(role="system", content="x" * 5000),
                Message(role="system", content="a short rule"),
                Message(role="user", content="hi"),
            ),
            temperature=0.0,
        )

        result = stage.before(request, _ctx())

        assert result.request.messages[1].cacheable is True

    def test_growing_history_extends_the_prefix_but_holds_back_the_last_exchange(self) -> None:
        stage = PrefixCacheStage()
        messages = [Message(role="system", content="x" * 5000)]
        for turn in range(6):
            messages.append(Message(role="user", content=f"q{turn}"))
            messages.append(Message(role="assistant", content=f"a{turn}"))
        request = LLMRequest(model="gpt-4o", messages=tuple(messages), temperature=0.0)

        result = stage.before(request, _ctx())

        cacheable = [i for i, m in enumerate(result.request.messages) if m.cacheable]
        assert len(cacheable) == 1
        marked_index = cacheable[0]
        assert marked_index > 0, "marking should extend past just the system message"
        assert marked_index < len(messages) - 2, "the last exchange must never be marked"
