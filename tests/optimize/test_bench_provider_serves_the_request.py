"""A provider serves the request it is given (ADR-035).

``AnthropicProvider`` and ``OpenAIProvider`` both sent ``model=self.model`` and
never read ``request.model``. Invisible on an ordinary run -- every workload is
built at the provider's own model, so the two agree -- and load-bearing the
moment a stage changes it, which is exactly what the two model-routing
techniques do:

* ``route_models`` rewrites ``request.model`` to ``config.cheap_model``;
* ``cascade_routing`` sends ``replace(request, model=cheap_model)`` first.

Routed through these providers **both calls would go to the same model** and the
reported saving would be arithmetic over two identical calls. Cascade's first
live run had to be written as a standalone script with its own provider closure
for precisely this reason (ADR-034).
"""

from __future__ import annotations

from typing import Any

import pytest

from optio_optimize.bench.providers import AnthropicProvider, OpenAIProvider, SpendGuard
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


class _AnthropicUsage:
    input_tokens = 100
    output_tokens = 20
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    cache_creation = None


class _AnthropicReply:
    def __init__(self, model: str) -> None:
        self.usage = _AnthropicUsage()
        self.content = [type("B", (), {"type": "text", "text": "ok"})()]
        self.model = model
        self.stop_reason = "end_turn"


class _AnthropicMessages:
    def __init__(self, seen: list[str]) -> None:
        self._seen = seen

    def create(self, **kwargs: Any) -> _AnthropicReply:
        self._seen.append(kwargs["model"])
        return _AnthropicReply(kwargs["model"])


class _AnthropicClient:
    def __init__(self, seen: list[str]) -> None:
        self.messages = _AnthropicMessages(seen)


class _OpenAIMessage:
    content = "ok"
    tool_calls = None


class _OpenAIChoice:
    def __init__(self) -> None:
        self.message = _OpenAIMessage()
        self.finish_reason = "stop"


class _OpenAIUsage:
    prompt_tokens = 100
    completion_tokens = 20
    prompt_tokens_details = None


class _OpenAIReply:
    def __init__(self, model: str) -> None:
        self.choices = [_OpenAIChoice()]
        self.usage = _OpenAIUsage()
        self.model = model


class _OpenAICompletions:
    def __init__(self, seen: list[str]) -> None:
        self._seen = seen

    def create(self, **kwargs: Any) -> _OpenAIReply:
        self._seen.append(kwargs["model"])
        return _OpenAIReply(kwargs["model"])


class _OpenAIClient:
    def __init__(self, seen: list[str]) -> None:
        self.chat = type("C", (), {"completions": _OpenAICompletions(seen)})()


def _anthropic(seen: list[str], model: str = "claude-sonnet-4-5") -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = _AnthropicClient(seen)  # type: ignore[assignment]
    provider.model = model
    provider.guard = SpendGuard(10.0)
    return provider


def _openai(seen: list[str], model: str = "gpt-4o") -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = _OpenAIClient(seen)  # type: ignore[assignment]
    provider.model = model
    provider.guard = SpendGuard(10.0)
    return provider


def _request(model: str) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(Message(role="user", content="hello"),),
        temperature=0.0,
    )


class TestTheRequestedModelIsTheOneCalled:
    def test_anthropic_serves_a_retargeted_request(self) -> None:
        """The routing case: a stage rewrote ``model`` and it must be honoured."""
        seen: list[str] = []

        _anthropic(seen)(_request("claude-haiku-4-5"))

        assert seen == ["claude-haiku-4-5"]

    def test_openai_serves_a_retargeted_request(self) -> None:
        seen: list[str] = []

        _openai(seen)(_request("gpt-4o-mini"))

        assert seen == ["gpt-4o-mini"]

    def test_a_cascade_shaped_pair_hits_two_different_models(self) -> None:
        """The defect, stated as the behaviour it broke.

        Cheap attempt then escalation. Before this, both went to the provider's
        own model and the measured "saving" compared a call with itself.
        """
        seen: list[str] = []
        provider = _anthropic(seen)

        provider(_request("claude-haiku-4-5"))
        provider(_request("claude-sonnet-4-5"))

        assert seen == ["claude-haiku-4-5", "claude-sonnet-4-5"]


class TestNothingMovesForOrdinaryRuns:
    """Every workload is built at the provider's model, so this is a no-op there.

    Asserted rather than assumed, because "no existing number moves" is the
    claim that makes this change safe to land.
    """

    def test_anthropic_matches_when_they_agree(self) -> None:
        seen: list[str] = []

        _anthropic(seen, "claude-sonnet-4-5")(_request("claude-sonnet-4-5"))

        assert seen == ["claude-sonnet-4-5"]

    def test_openai_matches_when_they_agree(self) -> None:
        seen: list[str] = []

        _openai(seen, "gpt-4o")(_request("gpt-4o"))

        assert seen == ["gpt-4o"]


class TestSpendIsChargedAgainstWhatWasServed:
    def test_a_cheap_call_is_not_billed_at_the_expensive_rate(self) -> None:
        """The guard tracking the wrong model is how a run overruns its cap.

        Haiku is a third of Sonnet on input and output, so a routed call priced
        against the provider's model overstates spend threefold.
        """
        seen: list[str] = []
        provider = _anthropic(seen, "claude-sonnet-4-5")

        provider(_request("claude-haiku-4-5"))
        cheap_spend = provider.guard.spent_usd

        provider.guard.spent_usd = 0.0
        provider(_request("claude-sonnet-4-5"))
        expensive_spend = provider.guard.spent_usd

        assert cheap_spend < expensive_spend

    def test_the_response_reports_the_model_that_served_it(self) -> None:
        seen: list[str] = []

        response = _anthropic(seen)(_request("claude-haiku-4-5"))

        assert response.model == "claude-haiku-4-5"

    def test_what_the_api_says_it_served_wins_over_what_was_asked(self) -> None:
        """Ground truth is the response, not the request.

        A gateway or proxy can serve something other than what was asked for.
        Charging the request's model would then bill a substitution at the
        price of the model that was never run -- and a dated id echoed back
        for an alias is the benign version of the same divergence.
        """
        seen: list[str] = []
        provider = _anthropic(seen, "claude-sonnet-4-5")
        provider._client = _SubstitutingClient(seen, serves="claude-haiku-4-5")  # type: ignore[assignment]

        provider(_request("claude-opus-5"))
        substituted = provider.guard.spent_usd

        provider.guard.spent_usd = 0.0
        provider._client = _AnthropicClient(seen)  # type: ignore[assignment]
        provider(_request("claude-opus-5"))
        as_asked = provider.guard.spent_usd

        assert substituted < as_asked


class TestTheGuardEstimatesAgainstTheRequestedModel:
    """The pre-call estimate decides whether a call happens at all.

    Estimating against the provider's model would block an affordable cheap
    call, or -- worse -- admit an expensive one under a cap sized for the cheap
    model, which is how a routed run silently overruns.
    """

    def test_a_cheap_call_is_admitted_under_a_cap_that_blocks_the_expensive_one(
        self,
    ) -> None:
        seen: list[str] = []
        provider = _anthropic(seen, "claude-opus-5")
        provider.guard = SpendGuard(0.010)  # between Haiku's ~$0.0051 and Opus 5's ~$0.0257

        provider(_request("claude-haiku-4-5"))

        assert seen == ["claude-haiku-4-5"]

    def test_the_expensive_call_is_still_blocked_by_that_cap(self) -> None:
        seen: list[str] = []
        provider = _anthropic(seen, "claude-haiku-4-5")
        provider.guard = SpendGuard(0.010)  # between Haiku's ~$0.0051 and Opus 5's ~$0.0257

        with pytest.raises(RuntimeError, match="spend cap"):
            provider(_request("claude-opus-5"))

        assert seen == []


class _SubstitutingClient:
    """A gateway that serves something other than what was asked for."""

    def __init__(self, seen: list[str], *, serves: str) -> None:
        self.messages = _SubstitutingMessages(seen, serves)


class _SubstitutingMessages:
    def __init__(self, seen: list[str], serves: str) -> None:
        self._seen = seen
        self._serves = serves

    def create(self, **kwargs: Any) -> _AnthropicReply:
        self._seen.append(kwargs["model"])
        return _AnthropicReply(self._serves)
