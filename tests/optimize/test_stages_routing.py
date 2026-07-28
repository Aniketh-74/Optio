"""RouteModelsStage: sending short, simple requests to a cheaper model."""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.routing import MAX_ROUTABLE_TOKENS, RouteModelsStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(*, cheap_model: str | None = "gpt-4o-mini") -> StageContext:
    config = OptimizeConfig(route_models=cheap_model is not None, cheap_model=cheap_model)
    return StageContext(config=config, counter=HeuristicCounter())


def _request(**overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": "gpt-4o",
        "messages": (Message(role="user", content="hi"),),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


class TestRouteModelsStage:
    def test_routes_a_short_simple_request(self) -> None:
        stage = RouteModelsStage()

        result = stage.before(_request(), _ctx())

        assert result.request.model == "gpt-4o-mini"
        assert "routed" in result.note

    def test_declines_when_no_cheap_model_is_configured(self) -> None:
        stage = RouteModelsStage()

        result = stage.before(_request(), _ctx(cheap_model=None))

        assert result.request.model == "gpt-4o"
        assert result.note == ""

    def test_declines_when_already_on_the_cheap_model(self) -> None:
        stage = RouteModelsStage()
        request = _request(model="gpt-4o-mini")

        result = stage.before(request, _ctx())

        assert result.note == ""

    def test_declines_when_tools_are_present(self) -> None:
        """Tool selection is exactly where a weaker model degrades first."""
        stage = RouteModelsStage()
        request = _request(tools=({"name": "search"},))

        result = stage.before(request, _ctx())

        assert result.request.model == "gpt-4o"
        assert result.note == ""

    def test_declines_when_a_response_format_is_present(self) -> None:
        """Structured extraction needs more precision than routing risks."""
        stage = RouteModelsStage()
        request = _request(response_format={"type": "json_object"})

        result = stage.before(request, _ctx())

        assert result.request.model == "gpt-4o"
        assert result.note == ""

    def test_declines_a_long_prompt(self) -> None:
        stage = RouteModelsStage()
        # Heuristic counter runs ~4 chars/token; comfortably clears the ceiling.
        long_content = "word " * (MAX_ROUTABLE_TOKENS * 2)
        request = _request(messages=(Message(role="user", content=long_content),))

        result = stage.before(request, _ctx())

        assert result.request.model == "gpt-4o"
        assert result.note == ""

    def test_a_prompt_right_at_the_floor_still_routes(self) -> None:
        stage = RouteModelsStage()
        # Sized to land comfortably under the token ceiling.
        request = _request(messages=(Message(role="user", content="short question"),))

        result = stage.before(request, _ctx())

        assert result.request.model == "gpt-4o-mini"

    def test_never_changes_messages_or_temperature(self) -> None:
        stage = RouteModelsStage()
        request = _request()

        result = stage.before(request, _ctx())

        assert result.request.messages == request.messages
        assert result.request.temperature == request.temperature

    def test_the_note_names_both_models_and_the_estimate(self) -> None:
        stage = RouteModelsStage()

        result = stage.before(_request(), _ctx())

        assert "gpt-4o -> gpt-4o-mini" in result.note
        assert "prompt tokens" in result.note
