"""A workload whose stable prefix clears every model's cacheable floor.

The suite's largest *stable* prefix was ~1,400 tokens, and that made it
structurally unable to measure its own best stage on half the models it
supports. ``prefix_cache`` needs a stable prefix above the model's minimum, and
those minimums run to **4,096 on Haiku 4.5, Opus 4.6 and Opus 4.5** (ADR-027).
``mcp_agent``'s 11,799-token prompt looked like the exception until the decline
reason showed its stable head is ~1,387 tokens; the rest is tool results that
change every step.

So the one lossless stage worth 74.5% on `multi_turn_chat`/Sonnet 4.5 could not
be demonstrated at all on a 4,096-floor model, and the suite reported that as
"reads 0" rather than as a gap in its own coverage.

This is representativeness, not benchmark gaming. Production agents carry
several thousand tokens of system prompt and tool schema on every single turn --
that is precisely the traffic this library targets, and precisely the traffic
the suite had no workload for. Added as a *new* workload rather than by growing
the existing ones, so every historical number in
``docs/optimize-benchmarks.md`` stays comparable.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.workloads import WORKLOADS
from optio_optimize.stages.caching import MIN_PREFIX_TOKENS_BY_MODEL, min_prefix_tokens_for
from optio_optimize.tokens import count_request, default_counter

pytestmark = pytest.mark.optimize

WORKLOAD = "large_system_agent"


def _stable_prefix_tokens(model: str) -> int:
    """Tokens shared by every request in the workload, from the front.

    The figure that decides whether a breakpoint can pay: a prefix cache matches
    from the start of the prompt and stops at the first difference.
    """
    requests = WORKLOADS[WORKLOAD].requests(model)
    counter = default_counter()
    first = requests[0].messages
    shared = 0
    for index, message in enumerate(first):
        if any(
            index >= len(other.messages) or other.messages[index] != message for other in requests
        ):
            break
        shared += counter.count_text(message.content, model)
    tools = count_request(requests[0], counter) - sum(
        counter.count_text(m.content, model) for m in first
    )
    return shared + max(0, tools)


class TestTheWorkloadExists:
    def test_it_is_registered(self) -> None:
        assert WORKLOAD in WORKLOADS

    def test_it_declares_what_it_is_for(self) -> None:
        workload = WORKLOADS[WORKLOAD]

        assert "prefix_cache" in workload.tags
        assert workload.expectation

    def test_it_builds_for_the_model_it_is_given(self) -> None:
        requests = WORKLOADS[WORKLOAD].requests("claude-haiku-4-5")

        assert {r.model for r in requests} == {"claude-haiku-4-5"}


class TestTheStablePrefixClearsEveryFloor:
    @pytest.mark.parametrize("model", sorted(MIN_PREFIX_TOKENS_BY_MODEL))
    def test_it_clears_this_models_minimum(self, model: str) -> None:
        """Every published Anthropic floor, including the 4,096 tier.

        This is the property the workload exists for. If it stops holding, the
        suite has silently lost its only coverage of ``prefix_cache`` on
        Haiku-class models.
        """
        assert _stable_prefix_tokens(model) > min_prefix_tokens_for(model)

    def test_it_clears_the_highest_floor_with_margin(self) -> None:
        """Not merely over the line.

        A workload sitting just above 4,096 would fall below it again on the
        first wording change, and the failure mode is silent: the report says
        "reads 0" and nothing says why.
        """
        assert _stable_prefix_tokens("claude-haiku-4-5") > 4_096 * 1.2


class TestThePrefixIsActuallyStable:
    def test_every_request_shares_the_system_prompt(self) -> None:
        requests = WORKLOADS[WORKLOAD].requests("claude-haiku-4-5")
        heads = {r.messages[0].content for r in requests}

        assert len(heads) == 1

    def test_every_request_carries_the_same_tools(self) -> None:
        """Tools are part of the cached prefix and sit ahead of the messages.

        A toolset that varies per call defeats the cache exactly as a timestamp
        above the instructions does -- the failure `timestamped_agent` exists to
        price.
        """
        requests = WORKLOADS[WORKLOAD].requests("claude-haiku-4-5")

        assert len({repr(r.tools) for r in requests}) == 1

    def test_the_conversation_still_grows(self) -> None:
        """Otherwise this is `retry_storm` and `exact_cache` takes the credit."""
        requests = WORKLOADS[WORKLOAD].requests("claude-haiku-4-5")

        assert len(requests[-1].messages) > len(requests[0].messages)

    def test_no_two_requests_are_identical(self) -> None:
        from optio_optimize.cache import request_key

        requests = WORKLOADS[WORKLOAD].requests("claude-haiku-4-5")

        assert len({request_key(r) for r in requests}) == len(requests)


class TestTheStageAcceptsIt:
    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-opus-4-5", "claude-sonnet-4-5"])
    def test_a_breakpoint_is_placed(self, model: str) -> None:
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.caching import PrefixCacheStage

        stage = PrefixCacheStage()
        requests = WORKLOADS[WORKLOAD].requests(model)

        result = stage.before(
            requests[0], StageContext(config=OptimizeConfig(), counter=default_counter())
        )

        assert any(m.cacheable for m in result.request.messages), stage.last_decline_reason
