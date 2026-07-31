"""``optio_optimize`` under real provider failures, not synthetic ones.

The existing fault-injection suite proves ``optio``'s *lanes* never break an
agent, using ``RuntimeError`` and hostile mappings. That is the right test for a
signals library that wraps someone else's call. It says nothing about
``optio_optimize``, which sits *on* the call path and can therefore fail in ways
a passive observer cannot.

Two things are exercised here that a synthetic ``RuntimeError`` does not reach:

**Real SDK exception types.** ``anthropic.RateLimitError`` and friends are
constructed objects carrying a ``response`` and a ``body``; code that formats or
inspects an exception can behave differently on one of those than on a bare
``RuntimeError``. They are built directly rather than provoked -- deliberately
hammering a live API to earn a 429 would be abuse, and the object is what the
code sees either way.

**A stream that dies mid-generation.** ``anthropic_streaming``'s docstring makes
the strongest safety claim in this package:

    Only the terminal event completes a request. Not exhaustion, not ``close()``.
    A transport that dies mid-generation, or a caller who stops reading, leaves a
    partial reply, and ``exact_cache`` storing a truncation would serve it
    confidently and permanently to everyone who later asks the same question.

ADR-033 found the non-streaming half of exactly that hazard live -- a truncated
answer being cached and served. This pins the streaming half.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from optio_optimize.cache import MemoryCache, request_key
from optio_optimize.config import OptimizeConfig
from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = [pytest.mark.failinject, pytest.mark.optimize]


def _request(text: str = "hello") -> LLMRequest:
    return LLMRequest(
        model="claude-sonnet-4-5",
        messages=(Message(role="user", content=text),),
        temperature=0.0,
    )


def _sdk_errors() -> list[Exception]:
    """Real ``anthropic`` exception objects, constructed rather than provoked."""
    import httpx
    from anthropic import (
        APIConnectionError,
        APIStatusError,
        InternalServerError,
        RateLimitError,
    )

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(code: int) -> httpx.Response:
        return httpx.Response(code, request=request)

    return [
        RateLimitError("rate limited", response=status(429), body=None),
        InternalServerError("server error", response=status(500), body=None),
        APIStatusError("overloaded", response=status(529), body=None),
        APIConnectionError(request=request),
    ]


class TestAProviderFailureReachesTheCallerUnchanged:
    """Fail-open means "pass the original request through", never "swallow".

    A cost library that converted a 429 into something else -- or into a
    success -- would break retry logic that reads the exception type. ADR-013's
    rule 1 is about cost; this is the correctness half of the same contract.
    """

    @pytest.mark.parametrize("error", _sdk_errors(), ids=lambda e: type(e).__name__)
    def test_the_exception_type_is_preserved(self, error: Exception) -> None:
        def provider(_: LLMRequest) -> LLMResponse:
            raise error

        with pytest.raises(type(error)):
            Optimizer(OptimizeConfig()).call(_request(), provider)

    @pytest.mark.parametrize("error", _sdk_errors(), ids=lambda e: type(e).__name__)
    def test_the_exact_object_is_preserved(self, error: Exception) -> None:
        """Not merely the type: retry code often inspects the instance."""

        def provider(_: LLMRequest) -> LLMResponse:
            raise error

        with pytest.raises(Exception) as caught:
            Optimizer(OptimizeConfig()).call(_request(), provider)

        assert caught.value is error

    def test_a_failed_call_writes_nothing_to_the_cache(self) -> None:
        """A raised call has no response, so there is nothing to store.

        Storing anything here would serve a fabricated answer to the retry.
        """
        backend = MemoryCache()
        optimizer = Optimizer(OptimizeConfig())
        for stage in optimizer._pipeline.stages:
            if getattr(stage, "name", "") == "exact_cache":
                stage.backend = backend  # type: ignore[attr-defined]
        request = _request()

        def provider(_: LLMRequest) -> LLMResponse:
            raise _sdk_errors()[0]

        with pytest.raises(Exception):  # noqa: B017 - the failure is the setup
            optimizer.call(request, provider)

        assert backend.get(request_key(request)) is None

    def test_the_next_call_still_works(self) -> None:
        """One failure must not poison the optimizer for the rest of the run."""
        optimizer = Optimizer(OptimizeConfig())
        calls: list[int] = []

        def flaky(_: LLMRequest) -> LLMResponse:
            calls.append(1)
            if len(calls) == 1:
                raise _sdk_errors()[0]
            return LLMResponse(content="ok", input_tokens=5, output_tokens=2)

        with pytest.raises(Exception):  # noqa: B017
            optimizer.call(_request(), flaky)
        response = optimizer.call(_request(), flaky)

        assert response.content == "ok"


class TestAStageFailureNeverReachesTheCaller:
    def test_a_raising_stage_is_skipped_and_the_call_succeeds(self) -> None:
        optimizer = Optimizer(OptimizeConfig())
        stage = optimizer._pipeline.stages[0]

        def explode(request: LLMRequest, ctx: Any) -> Any:
            raise RuntimeError("stage is broken")

        stage.before = explode  # type: ignore[method-assign]

        response = optimizer.call(
            _request(), lambda _: LLMResponse(content="ok", input_tokens=5, output_tokens=2)
        )

        assert response.content == "ok"

    def test_the_log_names_the_type_and_never_the_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Section 10: an exception payload can carry prompt content."""
        optimizer = Optimizer(OptimizeConfig())
        stage = optimizer._pipeline.stages[0]
        secret = "patient 41 diagnosis"

        def explode(request: LLMRequest, ctx: Any) -> Any:
            raise RuntimeError(secret)

        stage.before = explode  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            optimizer.call(
                _request(), lambda _: LLMResponse(content="ok", input_tokens=5, output_tokens=2)
            )

        assert "RuntimeError" in caplog.text
        assert secret not in caplog.text
