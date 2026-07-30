"""Cascade routing: call cheap, verify, escalate on failure (ADR-023).

Cascade is not a stage, so these tests exercise it where it lives -- around the
provider call. Two layers: :class:`CascadeRouter` directly, for the control flow
(eligible/ineligible, accept/reject, fail-open); and through ``Optimizer.call``,
for the property the design turns on -- that a *rejected* cheap answer never
reaches the cache.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from optio_optimize import LLMRequest, LLMResponse, Message, Optimizer
from optio_optimize.cascade import (
    CascadeRouter,
    CascadeStats,
    ModelJudge,
    Verifier,
    default_verifier,
)
from optio_optimize.config import OptimizeConfig
from optio_optimize.errors import OptimizeConfigError
from optio_optimize.stages.routing import MAX_ROUTABLE_TOKENS

pytestmark = pytest.mark.optimize

EXPENSIVE = "gpt-4o"
CHEAP = "gpt-4o-mini"


def _request(text: str = "hi", *, model: str = EXPENSIVE, **overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": model,
        "messages": (Message(role="user", content=text),),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


def _response(model: str, content: str = "an answer", finish: str | None = "stop") -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=50,
        output_tokens=10,
        model=model,
        finish_reason=finish,
    )


class _Provider:
    """A fake provider that records every request and answers per model.

    Answers whatever ``responses[model]`` says, defaulting to a plain stub, and
    keeps the models it was called with in order -- which is the whole
    observable of a cascade: did the cheap model get asked, and did the
    expensive one get asked after it.
    """

    def __init__(self, responses: dict[str, LLMResponse] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[str] = []

    def __call__(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request.model)
        return self.responses.get(request.model, _response(request.model))


def _router(
    verify: Verifier = default_verifier,
    cheap: str | None = CHEAP,
    *,
    structured: bool = False,
    tools: bool = False,
    max_tokens: int | None = None,
) -> CascadeRouter:
    config = OptimizeConfig(
        cascade_routing=cheap is not None,
        cascade_structured_output=structured,
        cascade_tools=tools,
        cascade_max_tokens=max_tokens,
        cheap_model=cheap,
        # keep the pipeline out of these unit tests: only the wrapper is exercised
        exact_cache=False,
        prefix_cache=False,
    )
    return CascadeRouter(config, verify=verify)


_TOOLS = ({"type": "function", "function": {"name": "search", "parameters": {}}},)
_TOOLS_REQ = (
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
)


def _tool_call_response(name: str, arguments: str = "{}", model: str = CHEAP) -> LLMResponse:
    """A response proposing one tool call, via the extra['tool_calls'] convention."""
    return LLMResponse(
        content="",
        input_tokens=50,
        output_tokens=10,
        model=model,
        finish_reason="tool_calls",
        extra={
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]
        },
    )


class TestEligibility:
    def test_ineligible_request_is_a_passthrough(self) -> None:
        """A prompt over the ceiling never gets a cheap attempt."""
        provider = _Provider()
        router = _router()
        long_prompt = "word " * (MAX_ROUTABLE_TOKENS * 2)

        response = router.wrap(provider)(_request(long_prompt))

        assert provider.calls == [EXPENSIVE]
        assert response.model == EXPENSIVE
        assert router.stats == CascadeStats(skipped=1)

    def test_tools_present_is_a_passthrough(self) -> None:
        provider = _Provider()
        router = _router()

        router.wrap(provider)(_request(tools=({"name": "search"},)))

        assert provider.calls == [EXPENSIVE]
        assert router.stats.skipped == 1
        assert router.stats.attempted == 0

    def test_already_on_the_cheap_model_is_a_passthrough(self) -> None:
        provider = _Provider()
        router = _router()

        router.wrap(provider)(_request(model=CHEAP))

        assert provider.calls == [CHEAP]
        assert router.stats.attempted == 0


class TestCascadeFlow:
    def test_accepted_cheap_answer_is_returned_without_escalating(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, "329")})
        router = _router()

        response = router.wrap(provider)(_request())

        assert provider.calls == [CHEAP]  # expensive model never called
        assert response.model == CHEAP
        assert response.content == "329"
        assert router.stats.attempted == 1
        assert router.stats.cheap_passed == 1
        assert router.stats.escalated == 0

    def test_empty_cheap_answer_escalates(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, ""), EXPENSIVE: _response(EXPENSIVE, "319")})
        router = _router()

        response = router.wrap(provider)(_request())

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert response.model == EXPENSIVE
        assert response.content == "319"
        assert router.stats.attempted == 1
        assert router.stats.escalated == 1
        assert router.stats.cheap_passed == 0

    def test_truncated_cheap_answer_escalates(self) -> None:
        """finish_reason == 'length' means the answer was cut off; escalate."""
        provider = _Provider(
            {
                CHEAP: _response(CHEAP, "the answer is 3", finish="length"),
                EXPENSIVE: _response(EXPENSIVE, "319"),
            }
        )
        router = _router()

        response = router.wrap(provider)(_request())

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert response.model == EXPENSIVE

    def test_escalation_re_sends_the_original_model_not_a_mutated_request(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, "")})
        router = _router()

        router.wrap(provider)(_request())

        # cheap first on the cheap model, then the ORIGINAL expensive model
        assert provider.calls == [CHEAP, EXPENSIVE]


class TestVerifier:
    def test_caller_verifier_can_reject_a_structurally_fine_answer(self) -> None:
        """The motivating case: a fluent, complete, wrong answer."""
        provider = _Provider(
            {CHEAP: _response(CHEAP, "329"), EXPENSIVE: _response(EXPENSIVE, "319")}
        )
        # A verifier that happens to know the right answer.
        router = _router(verify=lambda _req, resp: resp.content == "319")

        response = router.wrap(provider)(_request())

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert response.content == "319"
        assert router.stats.escalated == 1

    def test_caller_verifier_can_accept(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, "anything")})
        router = _router(verify=lambda _req, _resp: True)

        response = router.wrap(provider)(_request())

        assert provider.calls == [CHEAP]
        assert response.model == CHEAP

    def test_a_raising_verifier_escalates_rather_than_breaking_the_call(self) -> None:
        def boom(_req: LLMRequest, _resp: LLMResponse) -> bool:
            raise RuntimeError("verifier bug")

        provider = _Provider({CHEAP: _response(CHEAP), EXPENSIVE: _response(EXPENSIVE, "safe")})
        router = _router(verify=boom)

        response = router.wrap(provider)(_request())

        assert response.content == "safe"
        assert router.stats.escalated == 1

    def test_default_verifier_accepts_a_normal_answer(self) -> None:
        assert default_verifier(_request(), _response(CHEAP, "a real answer")) is True

    def test_default_verifier_rejects_empty_and_truncated(self) -> None:
        assert default_verifier(_request(), _response(CHEAP, "   ")) is False
        assert default_verifier(_request(), _response(CHEAP, "cut", finish="length")) is False


_JSON_OBJECT = {"type": "json_object"}


def _schema(required: list[str]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"schema": {"required": required}}}


class TestStructuredOutput:
    """ADR-023 step 1: cascade may take response_format requests, using the
    requested JSON shape as its verifier."""

    def test_response_format_is_skipped_without_the_flag(self) -> None:
        provider = _Provider()
        router = _router()  # cascade_structured_output off

        router.wrap(provider)(_request(response_format=_JSON_OBJECT))

        assert provider.calls == [EXPENSIVE]  # static behaviour: never attempted
        assert router.stats == CascadeStats(skipped=1)

    def test_response_format_is_attempted_with_the_flag(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, '{"answer": 42}')})
        router = _router(structured=True)

        response = router.wrap(provider)(_request(response_format=_JSON_OBJECT))

        assert provider.calls == [CHEAP]
        assert response.model == CHEAP
        assert router.stats.cheap_passed == 1

    def test_non_json_cheap_answer_escalates(self) -> None:
        provider = _Provider(
            {
                CHEAP: _response(CHEAP, "here is your answer: 42"),  # prose, not JSON
                EXPENSIVE: _response(EXPENSIVE, '{"answer": 42}'),
            }
        )
        router = _router(structured=True)

        response = router.wrap(provider)(_request(response_format=_JSON_OBJECT))

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert response.content == '{"answer": 42}'
        assert router.stats.escalated == 1

    def test_missing_required_key_escalates(self) -> None:
        provider = _Provider(
            {
                CHEAP: _response(CHEAP, '{"name": "Ada"}'),  # missing "age"
                EXPENSIVE: _response(EXPENSIVE, '{"name": "Ada", "age": 36}'),
            }
        )
        router = _router(structured=True)

        response = router.wrap(provider)(_request(response_format=_schema(["name", "age"])))

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert response.content == '{"name": "Ada", "age": 36}'  # the escalated answer
        assert router.stats.escalated == 1

    def test_conforming_json_with_all_required_keys_is_accepted(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, '{"name": "Ada", "age": 36}')})
        router = _router(structured=True)

        response = router.wrap(provider)(_request(response_format=_schema(["name", "age"])))

        assert provider.calls == [CHEAP]
        assert response.model == CHEAP


class TestDefaultVerifierConformance:
    def test_valid_json_object_accepted(self) -> None:
        req = _request(response_format=_JSON_OBJECT)
        assert default_verifier(req, _response(CHEAP, '{"a": 1}')) is True

    def test_invalid_json_rejected(self) -> None:
        req = _request(response_format=_JSON_OBJECT)
        assert default_verifier(req, _response(CHEAP, "not json")) is False

    def test_missing_required_key_rejected(self) -> None:
        req = _request(response_format=_schema(["x"]))
        assert default_verifier(req, _response(CHEAP, '{"y": 1}')) is False

    def test_no_response_format_does_not_require_json(self) -> None:
        # a plain-text answer to a plain request is fine even though it is not JSON
        assert default_verifier(_request(), _response(CHEAP, "plain text answer")) is True

    def test_known_tool_call_accepted(self) -> None:
        req = _request(tools=_TOOLS)
        assert default_verifier(req, _tool_call_response("search", '{"q": 1}')) is True

    def test_unknown_tool_call_rejected(self) -> None:
        req = _request(tools=_TOOLS)
        assert default_verifier(req, _tool_call_response("rm_rf", "{}")) is False

    def test_malformed_tool_arguments_rejected(self) -> None:
        req = _request(tools=_TOOLS)
        assert default_verifier(req, _tool_call_response("search", "{oops")) is False

    def test_missing_required_tool_param_rejected(self) -> None:
        # ADR-023 improvement #4: valid JSON, known tool, but the required
        # "query" argument is absent.
        req = _request(tools=_TOOLS_REQ)
        assert default_verifier(req, _tool_call_response("search", "{}")) is False

    def test_present_required_tool_param_accepted(self) -> None:
        req = _request(tools=_TOOLS_REQ)
        assert default_verifier(req, _tool_call_response("search", '{"query": "cats"}')) is True


class TestCostAccounting:
    """ADR-023 improvement #1: cascade's real economics, measured and honest."""

    def test_accepted_cheap_credits_a_saving_and_no_waste(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, "fine")})
        router = _router()
        router.wrap(provider)(_request())

        cost = router.stats.cost_summary(EXPENSIVE, CHEAP)
        assert cost is not None
        assert cost.escalation_waste_usd == 0.0  # nothing was wasted
        assert cost.net_saving_usd > 0  # cheaper than all-expensive
        assert cost.escalation_spend_usd == 0.0

    def test_escalation_is_a_measured_net_loss_not_hidden(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, ""), EXPENSIVE: _response(EXPENSIVE, "real")})
        router = _router()
        router.wrap(provider)(_request())

        cost = router.stats.cost_summary(EXPENSIVE, CHEAP)
        assert cost is not None
        assert cost.escalation_waste_usd > 0  # the rejected cheap attempt cost money
        assert cost.net_saving_usd < 0  # and it made this request cost MORE
        assert cost.total_spend_usd > cost.all_expensive_baseline_usd

    def test_unpriced_model_returns_none(self) -> None:
        router = _router()
        router.wrap(_Provider({CHEAP: _response(CHEAP, "x")}))(_request())
        assert router.stats.cost_summary("no-such-model", CHEAP) is None

    def test_token_accounting_splits_accepted_from_wasted(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, ""), EXPENSIVE: _response(EXPENSIVE, "real")})
        router = _router()
        router.wrap(provider)(_request())  # escalates
        s = router.stats
        assert s.cheap_input_tokens == 50 and s.cheap_output_tokens == 10  # the wasted attempt
        assert s.accepted_cheap_input_tokens == 0  # nothing accepted
        assert s.escalated_input_tokens == 50  # expensive call billed


class TestVerifierCost:
    """ADR-023 improvement #2: a model-based verifier's cost is measured."""

    def test_pop_cost_hook_is_folded_into_stats(self) -> None:
        class CostingVerifier:
            def __call__(self, req: LLMRequest, resp: LLMResponse) -> bool:
                return True

            def pop_cost(self) -> tuple[int, int, float]:
                return 12, 3, 0.0025

        provider = _Provider({CHEAP: _response(CHEAP, "ok")})
        router = _router(verify=CostingVerifier())
        router.wrap(provider)(_request())

        assert router.stats.verifier_input_tokens == 12
        assert router.stats.verifier_output_tokens == 3
        assert router.stats.verifier_usd == 0.0025

    def test_model_judge_grades_and_reports_its_cost(self) -> None:
        def judge_provider(req: LLMRequest) -> LLMResponse:
            return LLMResponse(
                content="PASS",
                input_tokens=40,
                output_tokens=1,
                model=EXPENSIVE,
                finish_reason="stop",
            )

        judge = ModelJudge(provider=judge_provider, model=EXPENSIVE)
        verdict = judge(_request(), _response(CHEAP, "candidate"))
        vin, vout, usd = judge.pop_cost()

        assert verdict is True
        assert (vin, vout) == (40, 1)
        assert usd > 0  # priced from PRICING[gpt-4o]
        assert judge.pop_cost() == (0, 0, 0.0)  # reset after popping


class TestLatencyTracking:
    """ADR-023 improvement #3: cascade's extra call time is visible."""

    def test_phases_are_timed_and_total_is_their_sum(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, ""), EXPENSIVE: _response(EXPENSIVE, "real")})
        router = _router()
        router.wrap(provider)(_request())  # cheap + verify + escalate all run
        s = router.stats
        assert s.cheap_ms >= 0.0 and s.verifier_ms >= 0.0 and s.escalation_ms >= 0.0
        assert s.escalation_ms > 0.0  # an escalation call really happened
        assert s.total_latency_ms == s.cheap_ms + s.verifier_ms + s.escalation_ms


class TestToolRequiredArgsIntegration:
    """#4 through the wrapper, not just the verifier in isolation."""

    def test_missing_required_arg_escalates(self) -> None:
        provider = _Provider(
            {
                CHEAP: _tool_call_response("search", "{}"),  # missing required "query"
                EXPENSIVE: _tool_call_response("search", '{"query": "x"}', model=EXPENSIVE),
            }
        )
        router = _router(tools=True)
        router.wrap(provider)(_request(tools=_TOOLS_REQ))
        assert provider.calls == [CHEAP, EXPENSIVE]
        assert router.stats.escalated == 1


class TestLengthCeiling:
    """ADR-023 step 2: cascade may raise the 500-token attempt ceiling."""

    # ~2000 tokens on any counter: wide margins so tiktoken vs heuristic is moot.
    LONG = "word " * 2000

    def test_a_long_prompt_is_skipped_at_the_default_ceiling(self) -> None:
        provider = _Provider()
        router = _router()  # cascade_max_tokens unset -> default 500

        router.wrap(provider)(_request(self.LONG))

        assert provider.calls == [EXPENSIVE]
        assert router.stats.skipped == 1

    def test_a_raised_ceiling_attempts_a_longer_prompt(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, "handled")})
        router = _router(max_tokens=5000)

        response = router.wrap(provider)(_request(self.LONG))

        assert provider.calls == [CHEAP]
        assert response.model == CHEAP
        assert router.stats.cheap_passed == 1

    def test_still_skips_beyond_the_raised_ceiling(self) -> None:
        provider = _Provider()
        router = _router(max_tokens=1000)  # LONG (~2000) still over

        router.wrap(provider)(_request(self.LONG))

        assert provider.calls == [EXPENSIVE]
        assert router.stats.skipped == 1


class TestToolRequests:
    """ADR-023 step 3: cascade may take tool requests, vetting the proposed call."""

    def test_tool_request_is_skipped_without_the_flag(self) -> None:
        provider = _Provider()
        router = _router()  # cascade_tools off

        router.wrap(provider)(_request(tools=_TOOLS))

        assert provider.calls == [EXPENSIVE]
        assert router.stats == CascadeStats(skipped=1)

    def test_a_valid_known_tool_call_is_accepted(self) -> None:
        provider = _Provider({CHEAP: _tool_call_response("search", '{"q": "cats"}')})
        router = _router(tools=True)

        response = router.wrap(provider)(_request(tools=_TOOLS))

        assert provider.calls == [CHEAP]
        assert response.model == CHEAP
        assert router.stats.cheap_passed == 1

    def test_an_unknown_tool_name_escalates(self) -> None:
        provider = _Provider(
            {
                CHEAP: _tool_call_response("delete_everything", "{}"),  # not in _TOOLS
                EXPENSIVE: _tool_call_response("search", "{}", model=EXPENSIVE),
            }
        )
        router = _router(tools=True)

        router.wrap(provider)(_request(tools=_TOOLS))

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert router.stats.escalated == 1

    def test_malformed_tool_arguments_escalate(self) -> None:
        provider = _Provider(
            {
                CHEAP: _tool_call_response("search", "{not valid json"),
                EXPENSIVE: _tool_call_response("search", "{}", model=EXPENSIVE),
            }
        )
        router = _router(tools=True)

        router.wrap(provider)(_request(tools=_TOOLS))

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert router.stats.escalated == 1

    def test_a_text_answer_to_a_tool_request_is_accepted(self) -> None:
        """A model may legitimately answer in text rather than call a tool."""
        provider = _Provider({CHEAP: _response(CHEAP, "No tool needed; the answer is 4.")})
        router = _router(tools=True)

        response = router.wrap(provider)(_request(tools=_TOOLS))

        assert provider.calls == [CHEAP]
        assert response.model == CHEAP

    def test_empty_answer_with_no_tool_call_escalates(self) -> None:
        provider = _Provider(
            {CHEAP: _response(CHEAP, ""), EXPENSIVE: _tool_call_response("search", model=EXPENSIVE)}
        )
        router = _router(tools=True)

        router.wrap(provider)(_request(tools=_TOOLS))

        assert provider.calls == [CHEAP, EXPENSIVE]
        assert router.stats.escalated == 1


class TestFailOpen:
    def test_a_failing_cheap_call_escalates_to_the_expensive_model(self) -> None:
        class FlakyCheap(_Provider):
            def __call__(self, request: LLMRequest) -> LLMResponse:
                self.calls.append(request.model)
                if request.model == CHEAP:
                    raise ConnectionError("cheap endpoint down")
                return _response(EXPENSIVE, "still answered")

        provider = FlakyCheap()
        router = _router()

        response = router.wrap(provider)(_request())

        assert response.content == "still answered"
        assert provider.calls == [CHEAP, EXPENSIVE]
        assert router.stats.escalated == 1


class TestStats:
    def test_escalation_rate_is_none_before_any_attempt(self) -> None:
        assert CascadeStats().escalation_rate is None

    def test_escalation_rate_counts_only_attempts_not_skips(self) -> None:
        stats = CascadeStats(skipped=10, attempted=4, cheap_passed=3, escalated=1)
        assert stats.escalation_rate == 0.25


class TestConfigValidation:
    def test_cascade_without_a_cheap_model_is_rejected(self) -> None:
        with pytest.raises(OptimizeConfigError, match="cheap_model is unset"):
            OptimizeConfig(cascade_routing=True)

    def test_cascade_and_static_routing_together_is_rejected(self) -> None:
        with pytest.raises(OptimizeConfigError, match="both retarget"):
            OptimizeConfig(cascade_routing=True, route_models=True, cheap_model=CHEAP)

    def test_structured_output_requires_cascade_routing(self) -> None:
        with pytest.raises(OptimizeConfigError, match="cascade_routing is off"):
            OptimizeConfig(cascade_structured_output=True, cheap_model=CHEAP)

    def test_cascade_max_tokens_requires_cascade_routing(self) -> None:
        with pytest.raises(OptimizeConfigError, match="cascade_routing is off"):
            OptimizeConfig(cascade_max_tokens=2000, cheap_model=CHEAP)

    def test_cascade_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(OptimizeConfigError, match="must be positive"):
            OptimizeConfig(cascade_routing=True, cheap_model=CHEAP, cascade_max_tokens=0)

    def test_cascade_tools_requires_cascade_routing(self) -> None:
        with pytest.raises(OptimizeConfigError, match="cascade_routing is off"):
            OptimizeConfig(cascade_tools=True, cheap_model=CHEAP)


class TestOptimizerIntegration:
    def test_cascade_stats_is_none_when_off(self) -> None:
        assert Optimizer().cascade_stats is None

    def test_accepted_cheap_answer_flows_through_the_optimizer(self) -> None:
        provider = _Provider({CHEAP: _response(CHEAP, "cheap and fine")})
        opt = Optimizer(cascade_routing=True, cheap_model=CHEAP)

        response = opt.call(_request(), provider)

        assert response.content == "cheap and fine"
        assert opt.cascade_stats is not None
        assert opt.cascade_stats.cheap_passed == 1

    def test_a_rejected_cheap_answer_is_never_cached(self) -> None:
        """The property ADR-023 decision 2 turns on.

        With exact_cache on, a rejected cheap answer must not be written under
        the request's key and served later as a hit. The accepted (escalated)
        answer is what gets cached; a second identical request returns *that*,
        and no second cheap attempt is even made.
        """
        provider = _Provider(
            {CHEAP: _response(CHEAP, "wrong-and-cheap"), EXPENSIVE: _response(EXPENSIVE, "correct")}
        )
        # verifier rejects the cheap answer, forcing escalation on the first call
        opt = Optimizer(
            cascade_routing=True,
            cheap_model=CHEAP,
            cascade_verifier=lambda _req, resp: resp.content != "wrong-and-cheap",
        )
        request = _request("what is 17 times 24, minus 89?")

        first = opt.call(request, provider)
        second = opt.call(request, provider)

        assert first.content == "correct"
        # the cache served the second call: the ACCEPTED answer, never the rejected cheap one
        assert second.content == "correct"
        assert provider.calls == [CHEAP, EXPENSIVE]  # no cheap attempt on the cached second call
        assert opt.cascade_stats is not None
        assert opt.cascade_stats.attempted == 1


class TestAsyncCascade:
    """Async twin (awrap). The repo runs async tests via asyncio.run, not a
    pytest plugin (see test_pipeline_async), so each test drives its own loop."""

    def test_async_accepts_cheap_answer(self) -> None:
        calls: list[str] = []

        async def provider(request: LLMRequest) -> LLMResponse:
            calls.append(request.model)
            return _response(request.model, "ok")

        router = _router()

        async def go() -> LLMResponse:
            # An inner coroutine rather than awaiting the wrapper's return value
            # directly: awrap yields an Awaitable, and asyncio.run wants a
            # Coroutine. Same shape test_fan_out.py uses for the same reason.
            return await router.awrap(provider)(_request())

        response = asyncio.run(go())

        assert calls == [CHEAP]
        assert response.model == CHEAP
        assert router.stats.cheap_passed == 1

    def test_async_escalates_on_rejection(self) -> None:
        calls: list[str] = []

        async def provider(request: LLMRequest) -> LLMResponse:
            calls.append(request.model)
            content = "" if request.model == CHEAP else "escalated answer"
            return _response(request.model, content)

        router = _router()

        async def go() -> LLMResponse:
            return await router.awrap(provider)(_request())

        response = asyncio.run(go())

        assert calls == [CHEAP, EXPENSIVE]
        assert response.content == "escalated answer"
        assert router.stats.escalated == 1
