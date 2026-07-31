"""A workload must be built for the model the run actually calls.

``Workload.requests()` called ``self.build()`` with no arguments, so every
builder fell back to its ``model: str = "gpt-4o"`` default. ``--model`` changed
which API the *provider* called and which row priced the result, and never
reached ``LLMRequest.model``.

Every stage that branches on the model therefore read ``gpt-4o`` on every live
Anthropic run:

* ``min_prefix_tokens_for(request.model)`` returned the unknown-model fallback
  of 1,024 -- so ADR-027's per-model cacheable floor, whose entire purpose is
  that Haiku 4.5 needs 4,096, **was inert inside the benchmark built to validate
  it**. A live Haiku run placed the breakpoint anyway and the provider discarded
  it: ``reads 0  writes 0``, exactly the failure ADR-027 describes, on the model
  it describes it for.
* ``TrimHistoryStage._risk_threshold(request.model)`` priced the output-token
  risk of trimming at gpt-4o's ratio rather than the served model's, which is
  ADR-026's gate deciding against the wrong prices.

It passed unnoticed because gpt-4o's fallback floor (1,024) happens to equal
Sonnet 4.5's real one, so the one model where the bug is invisible is the one
that made ``prefix_cache`` look like it worked.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.providers import SimulatedProvider
from optio_optimize.bench.workloads import WORKLOADS
from optio_optimize.types import LLMRequest, LLMResponse

pytestmark = pytest.mark.optimize


class TestTheRequestedModelReachesTheRequest:
    @pytest.mark.parametrize("name", sorted(WORKLOADS))
    def test_every_workload_builds_for_the_model_it_is_given(self, name: str) -> None:
        requests = WORKLOADS[name].requests("claude-haiku-4-5")

        assert requests
        assert {r.model for r in requests} == {"claude-haiku-4-5"}

    @pytest.mark.parametrize("name", sorted(WORKLOADS))
    def test_no_workload_silently_keeps_its_default(self, name: str) -> None:
        """The failure was silent, so this asserts the absence directly."""
        requests = WORKLOADS[name].requests("claude-opus-4-5")

        assert "gpt-4o" not in {r.model for r in requests}

    def test_the_default_is_still_available(self) -> None:
        requests = WORKLOADS["multi_turn_chat"].requests()

        assert {r.model for r in requests} == {"gpt-4o"}


class TestTheFloorFollowsTheServedModel:
    def test_a_haiku_run_gets_haikus_floor(self) -> None:
        """The consequence that made this worth finding.

        `multi_turn_chat` carries roughly 1,400-1,800 tokens. Under gpt-4o's
        fallback floor of 1,024 that clears and gets a breakpoint; under Haiku
        4.5's real 4,096 it does not, and the provider discards the marker it
        was sent.
        """
        from optio_optimize.stages.caching import min_prefix_tokens_for

        requests = WORKLOADS["multi_turn_chat"].requests("claude-haiku-4-5")

        assert min_prefix_tokens_for(requests[0].model) == 4_096

    def test_a_sonnet_run_gets_sonnets_floor(self) -> None:
        from optio_optimize.stages.caching import min_prefix_tokens_for

        requests = WORKLOADS["multi_turn_chat"].requests("claude-sonnet-4-5")

        assert min_prefix_tokens_for(requests[0].model) == 1_024


class TestTheHarnessPassesTheProvidersModel:
    def test_compare_builds_for_the_model_it_prices(self) -> None:
        """The two must not be able to drift apart.

        ``compare`` already took ``model`` for pricing. It priced against the
        served model while handing the stages a request built for another one,
        and nothing connected the two.
        """
        from optio_optimize.bench.harness import compare
        from optio_optimize.bench.providers import SimulatedProvider

        provider = SimulatedProvider(model="claude-haiku-4-5")
        result = compare(WORKLOADS["unique_questions"], provider, model="claude-haiku-4-5")

        assert result.model == "claude-haiku-4-5"

    def test_the_stages_saw_that_model(self) -> None:
        """What reaches the provider is what the stages were handed."""
        from optio_optimize.bench.harness import run_arm
        from optio_optimize.bench.providers import SimulatedProvider
        from optio_optimize.config import OptimizeConfig

        seen: list[str] = []
        provider = _Recording(SimulatedProvider(model="claude-haiku-4-5"), seen)

        run_arm(
            "opt",
            WORKLOADS["unique_questions"].requests("claude-haiku-4-5"),
            provider,
            OptimizeConfig(),
        )

        assert set(seen) == {"claude-haiku-4-5"}


class _Recording:
    """A provider that notes the model on every request it is handed."""

    def __init__(self, inner: SimulatedProvider, seen: list[str]) -> None:
        self._inner = inner
        self._seen = seen

    @property
    def is_live(self) -> bool:
        return False

    @property
    def models_latency(self) -> bool:
        return False

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def label(self) -> str:
        return "recording"

    def reset(self) -> None:
        self._inner.reset()

    def __call__(self, request: LLMRequest) -> LLMResponse:
        self._seen.append(request.model)
        return self._inner(request)
