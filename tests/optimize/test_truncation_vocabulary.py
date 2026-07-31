"""Truncation is a question, not a string comparison (ADR-033).

The first live cascade run carried a request with ``max_tokens=16`` and a prompt
that cannot be answered in sixteen tokens. It was the *guaranteed* escalation:
``default_verifier``'s first check rejects a truncated answer. It did not
escalate, and the chopped-off answer was returned as final::

    raw stop_reason      : 'max_tokens'
    our finish_reason    : 'max_tokens'
    verifier checks      : finish_reason == 'length'
    => fires on Anthropic? False

**Anthropic reports ``max_tokens``; OpenAI reports ``length``.** Every truncation
check compared against ``"length"`` only, so on Anthropic all of them were dead
code -- and the whole suite missed it because every fixture exercising
truncation had been written with OpenAI's spelling.

Two call sites matter, and the second is the serious one. ``ExactCacheStage``
refuses to serve a truncated stored response, because ``request_key``
deliberately omits ``max_tokens`` so calls with different ceilings share an
entry. With the guard inert, one ``max_tokens=16`` request poisons the entry for
every later caller who allowed more -- in a stage that is ``Fidelity.IDENTICAL``,
lossless, and on by default.
"""

from __future__ import annotations

import pytest

from optio_optimize.cache import MemoryCache, request_key
from optio_optimize.cascade import default_verifier
from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.caching import ExactCacheStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize

#: Every spelling a provider uses for "the model ran out of room".
TRUNCATED = ["length", "max_tokens", "max_output_tokens"]
#: Reasons that mean the answer finished on its own terms.
COMPLETE = ["stop", "end_turn", "tool_use", "stop_sequence", None]


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(**overrides: object) -> LLMRequest:
    base: dict[str, object] = {
        "model": "claude-sonnet-4-5",
        "messages": (Message(role="user", content="Explain the revolution in detail."),),
        "temperature": 0.0,
    }
    base.update(overrides)
    return LLMRequest(**base)  # type: ignore[arg-type]


def _response(reason: str | None, content: str = "The revolution began") -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=40,
        output_tokens=16,
        model="claude-sonnet-4-5",
        finish_reason=reason,
    )


class TestTheResponseKnowsItWasCut:
    @pytest.mark.parametrize("reason", TRUNCATED)
    def test_every_providers_spelling_counts(self, reason: str) -> None:
        assert _response(reason).was_truncated is True

    @pytest.mark.parametrize("reason", COMPLETE)
    def test_a_completed_answer_does_not(self, reason: str | None) -> None:
        assert _response(reason).was_truncated is False

    def test_an_unknown_reason_is_not_truncated(self) -> None:
        """Fails the safe way.

        Treating an unrecognised reason as truncated would make the exact cache
        refuse to serve anything from a provider whose vocabulary is not listed
        -- a silently disabled cache instead of a visible gap.
        """
        assert _response("some_new_reason").was_truncated is False

    def test_the_raw_reason_is_preserved(self) -> None:
        """ADR-033 decision 2: answer the question without discarding evidence."""
        assert _response("max_tokens").finish_reason == "max_tokens"


class TestCascadeEscalatesOnTruncation:
    @pytest.mark.parametrize("reason", TRUNCATED)
    def test_a_truncated_cheap_answer_is_rejected(self, reason: str) -> None:
        assert default_verifier(_request(), _response(reason)) is False

    def test_the_anthropic_case_is_the_one_that_was_broken(self) -> None:
        """The live run's `truncated` request, as an assertion."""
        assert default_verifier(_request(), _response("max_tokens")) is False

    @pytest.mark.parametrize("reason", COMPLETE)
    def test_a_complete_cheap_answer_is_still_accepted(self, reason: str | None) -> None:
        assert default_verifier(_request(), _response(reason)) is True


class TestTheExactCacheRefusesTruncatedEntries:
    @pytest.mark.parametrize("reason", TRUNCATED)
    def test_a_truncated_entry_is_not_served(self, reason: str) -> None:
        """The serious one: a lossless, on-by-default stage serving a cut answer."""
        backend = MemoryCache()
        request = _request()
        backend.put(request_key(request), _response(reason))
        stage = ExactCacheStage(backend=backend)

        result = stage.before(request, _ctx())

        assert result.response is None

    def test_a_low_ceiling_call_cannot_poison_a_generous_one(self) -> None:
        """The end-to-end shape of the bug.

        ``request_key`` omits ``max_tokens`` on purpose, so these two share an
        entry. That is only sound while the truncation guard works.
        """
        backend = MemoryCache()
        stingy = _request(max_tokens=16)
        generous = _request(max_tokens=4096)
        assert request_key(stingy) == request_key(generous)
        backend.put(request_key(stingy), _response("max_tokens"))

        result = ExactCacheStage(backend=backend).before(generous, _ctx())

        assert result.response is None

    @pytest.mark.parametrize("reason", COMPLETE)
    def test_a_complete_entry_is_still_served(self, reason: str | None) -> None:
        backend = MemoryCache()
        request = _request()
        backend.put(request_key(request), _response(reason))

        result = ExactCacheStage(backend=backend).before(request, _ctx())

        assert result.response is not None

    def test_a_served_entry_keeps_its_finish_reason(self) -> None:
        """ADR-033 decision 3.

        ``served_from_cache`` zeroes token counts because they describe the
        original *call*. The finish reason describes the *answer*, which is the
        thing being reused -- zeroing it would silently re-open this hole.
        """
        backend = MemoryCache()
        request = _request()
        backend.put(request_key(request), _response("end_turn"))

        result = ExactCacheStage(backend=backend).before(request, _ctx())

        assert result.response is not None
        assert result.response.finish_reason == "end_turn"
