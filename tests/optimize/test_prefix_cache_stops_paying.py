"""Stop writing when the provider says the writes are not being read (ADR-030 amendment).

The digest guard landed and the live run still came back negative::

    ### timestamped_agent          before      after
        cost reduction            -23.5%      -17.1%
        provider cache writes     20,333      14,795

Correct behaviour, arriving too late. ``unstable_prefix`` needs
``MIN_OBSERVATIONS = 10`` before it will say anything, and that threshold is
right for *reporting a finding to a person* -- "three requests with three
distinct system prompts is equally consistent with a bug and with an agent that
has only just started". On a twelve-request workload it means ten requests pay
the 1.25x write premium before the guard fires.

The stronger signal was there all along and is not a proxy at all: **the
provider reports what it actually served.** ``cache_read_input_tokens`` comes
back on every response. A marked request that produces a write and no read is
the outcome itself, not an inference from prompt digests, and it converges in
about three requests instead of ten.

It is also correct in a case the digest check cannot see: a prefix that is
byte-stable but always expires before reuse writes at 1.25x forever and reads
nothing. Declining is right there too, and the digests look perfect.

The asymmetry that sets the threshold: a wasted write costs 0.25x base per
prefix token (1.25x paid instead of 1.0x), while a *missed* cache costs 0.9x
(1.0x paid instead of 0.1x). Declining wrongly is ~3.6x worse per request than
writing wrongly, so this waits for repeated, unambiguous evidence rather than
reacting to one miss -- and a single read resets it.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.caching import PrefixCacheStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(head: str = "H") -> LLMRequest:
    return LLMRequest(
        model="claude-sonnet-4-5",
        messages=(
            Message(role="system", content=head + " policy " + ("word " * 1_400)),
            Message(role="user", content="q"),
        ),
        temperature=0.0,
    )


def _response(*, read: int, written: int) -> LLMResponse:
    return LLMResponse(
        content="ok",
        input_tokens=2_000,
        output_tokens=20,
        cached_input_tokens=read,
        cache_write_tokens=written,
        model="claude-sonnet-4-5",
    )


def _turn(stage: PrefixCacheStage, response: LLMResponse, head: str = "H") -> bool:
    """One request/response cycle. Returns whether the request was marked."""
    ctx = _ctx()
    request = _request(head)
    result = stage.before(request, ctx)
    marked = any(m.cacheable for m in result.request.messages)
    stage.after(result.request, response, ctx)
    return marked


class TestRepeatedUnreadWritesStopTheStage:
    def test_it_gives_up_after_repeated_unrewarded_writes(self) -> None:
        """The `timestamped_agent` case, measured rather than inferred."""
        stage = PrefixCacheStage()

        marked = [_turn(stage, _response(read=0, written=1_500)) for _ in range(6)]

        assert marked[0] is True
        assert marked[-1] is False

    def test_it_converges_in_far_fewer_than_ten_requests(self) -> None:
        """The whole point of the amendment.

        Ten requests of premium on a twelve-request workload is why the digest
        guard alone still measured -17.1%.
        """
        stage = PrefixCacheStage()

        marked = [_turn(stage, _response(read=0, written=1_500)) for _ in range(10)]

        assert marked.count(True) <= 4

    def test_it_says_why(self) -> None:
        stage = PrefixCacheStage()

        for _ in range(6):
            _turn(stage, _response(read=0, written=1_500))

        assert "read" in stage.last_decline_reason.lower()


class TestASingleReadIsProofItWorks:
    def test_a_read_keeps_the_stage_marking(self) -> None:
        stage = PrefixCacheStage()

        marked = [_turn(stage, _response(read=1_500, written=0)) for _ in range(12)]

        assert all(marked)

    def test_the_first_write_is_expected_and_forgiven(self) -> None:
        """Nothing can be read from a cache that was never written.

        Call one always writes and reads nothing. Treating that as evidence
        would disable the stage on every conversation's opening turn.
        """
        stage = PrefixCacheStage()

        first = _turn(stage, _response(read=0, written=1_500))
        second = _turn(stage, _response(read=1_500, written=0))

        assert (first, second) == (True, True)

    def test_one_read_resets_the_run(self) -> None:
        """A read is direct proof the prefix is cacheable.

        The counter is consecutive-unrewarded, not lifetime-unrewarded: an
        agent whose traffic goes quiet past the TTL and then resumes must not
        accumulate its way to a permanent decline.
        """
        stage = PrefixCacheStage()

        _turn(stage, _response(read=0, written=1_500))
        _turn(stage, _response(read=0, written=1_500))
        _turn(stage, _response(read=900, written=0))
        after_reset = [_turn(stage, _response(read=0, written=1_500)) for _ in range(2)]

        assert all(after_reset)


class TestOnlyMarkedRequestsCount:
    def test_an_unmarked_request_is_not_evidence(self) -> None:
        """A request we never marked cannot tell us anything about marking.

        Below-floor requests produce reads 0 / writes 0 all day, and counting
        them would disable the stage on workloads it had never even tried.
        """
        stage = PrefixCacheStage()
        small = LLMRequest(
            model="claude-sonnet-4-5",
            messages=(Message(role="system", content="tiny"), Message(role="user", content="q")),
            temperature=0.0,
        )

        for _ in range(8):
            ctx = _ctx()
            result = stage.before(small, ctx)
            stage.after(result.request, _response(read=0, written=0), ctx)

        assert any(m.cacheable for m in stage.before(_request(), _ctx()).request.messages)

    def test_a_response_without_usage_is_not_evidence(self) -> None:
        """Providers that report no cache fields must not disable the stage.

        OpenAI caches automatically and reports no write count; reading that
        silence as "the write was wasted" would turn the marker off for a
        provider where it was never the mechanism in the first place.
        """
        stage = PrefixCacheStage()

        marked = [_turn(stage, _response(read=0, written=0)) for _ in range(8)]

        assert all(marked)
