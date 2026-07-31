"""A breakpoint nobody reads is pure cost (ADR-030).

The full live suite on `claude-sonnet-4-5`, 2026-07-31, returned ten workloads
between 75% and 97% cost reduction, and one going the other way::

    ### timestamped_agent
        cost            $0.06496 -> $0.08021   -23.5%
        provider cache  reads 0   writes 20,333
        output quality  100.0% identical

20,333 prompt tokens written into the provider's cache at the 1.25x premium and
read back **zero times**. The arms differ, the effect is one-directional, and it
reproduces -- a straight ADR-013 rule 1 violation caused by the stage this
package calls its largest lossless win.

`detect_unstable_prefix` had already reported the cause, on that very run,
before the first workload: "the system prompt differed on 100% of the last 10
requests, so no provider prefix cache can hit". The detector knew. The writer
never asked.

The same stage was also declining breakpoints that would pay, because it
measured the prefix over `request.messages` alone and ignored `request.tools` --
schemas Anthropic caches *ahead of* the system prompt. On `large_system_agent`
it reported "~1715 tokens" for a 5,186-token prefix and declined on the
4,096-token tier.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.caching import PrefixCacheStage
from optio_optimize.stages.diagnostics import MIN_OBSERVATIONS, UnstablePrefixStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(*, head: str, model: str = "claude-sonnet-4-5", tools: int = 0) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(
            Message(role="system", content=head + " policy clause " + ("word " * 1_400)),
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ),
        tools=tuple(
            {
                "type": "function",
                "function": {
                    "name": f"tool_{n}",
                    "description": "does a thing " * 40,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for n in range(tools)
        ),
        temperature=0.0,
    )


def _run(stage: PrefixCacheStage, detector: UnstablePrefixStage, request: LLMRequest) -> bool:
    """Feed one request through the detector then the stage; was it marked?"""
    ctx = _ctx()
    detector.before(request, ctx)
    result = stage.before(request, ctx)
    return any(m.cacheable for m in result.request.messages)


class TestAnUnstablePrefixIsNotWritten:
    def test_a_prefix_that_never_repeats_stops_being_marked(self) -> None:
        """The -23.5% case.

        Every turn carries a fresh timestamp above the instructions, so no
        write can ever be read. After the detector's window fills, the stage
        must stop paying the write premium.
        """
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()

        marked = [
            _run(stage, detector, _request(head=f"Current time: 09:{n:02d}:31Z"))
            for n in range(MIN_OBSERVATIONS + 4)
        ]

        assert marked[-1] is False

    def test_it_says_why(self) -> None:
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()

        for n in range(MIN_OBSERVATIONS + 4):
            _run(stage, detector, _request(head=f"Current time: 09:{n:02d}:31Z"))

        assert "unstable" in stage.last_decline_reason.lower()

    def test_a_stable_prefix_is_still_marked(self) -> None:
        """The default is to cache. Declining is the exception."""
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()

        marked = [
            _run(stage, detector, _request(head="Fixed header"))
            for _ in range(MIN_OBSERVATIONS + 4)
        ]

        assert all(marked)

    def test_silence_is_not_evidence(self) -> None:
        """Before the window fills, behaviour is exactly what it was.

        A stage that declined on one or two observations would refuse to cache
        the opening turns of every conversation -- the turns where the prefix
        cache pays most.
        """
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()

        marked = [
            _run(stage, detector, _request(head=f"Current time: 09:{n:02d}:31Z"))
            for n in range(MIN_OBSERVATIONS - 1)
        ]

        assert all(marked)

    def test_an_occasionally_changing_prefix_still_caches(self) -> None:
        """One read recovers several writes at 1.25x against 0.1x.

        A multi-tenant service legitimately alternates between a handful of
        system prompts and each one caches separately. Only a prefix that is
        never the same twice is a pure loss.
        """
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()

        marked = [
            _run(stage, detector, _request(head=f"Tenant {n % 3}"))
            for n in range(MIN_OBSERVATIONS + 8)
        ]

        assert all(marked)


class TestToolSchemasCountTowardThePrefix:
    def test_tools_lift_a_prefix_over_the_floor(self) -> None:
        """The `large_system_agent` case.

        Anthropic caches tools -> system -> messages, so a breakpoint in the
        system block caches every schema ahead of it. Counting only messages
        understates every tool-carrying request.
        """
        stage = PrefixCacheStage()
        below = stage.before(_request(head="H", model="claude-haiku-4-5"), _ctx())
        with_tools = stage.before(_request(head="H", model="claude-haiku-4-5", tools=40), _ctx())

        assert not any(m.cacheable for m in below.request.messages)
        assert any(m.cacheable for m in with_tools.request.messages)

    def test_the_decline_reason_counts_tools_too(self) -> None:
        """ADR-027 made "no cache reads" diagnosable; the figure must stay true."""
        stage = PrefixCacheStage()

        stage.before(_request(head="H", model="claude-haiku-4-5", tools=4), _ctx())
        small = stage.last_decline_reason

        stage.before(_request(head="H", model="claude-haiku-4-5", tools=20), _ctx())
        larger = stage.last_decline_reason

        assert small != larger

    def test_a_request_without_tools_is_unchanged(self) -> None:
        """No tools, no difference -- this must not move existing behaviour."""
        stage = PrefixCacheStage()

        result = stage.before(_request(head="H", model="claude-sonnet-4-5"), _ctx())

        assert any(m.cacheable for m in result.request.messages)


class TestTheTwoRulesCompose:
    def test_tools_cannot_resurrect_an_unwritable_prefix(self) -> None:
        """Instability is checked before size.

        Otherwise counting tools would push an unstable prefix back over the
        floor and reinstate exactly the write nobody reads.
        """
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()

        marked = [
            _run(stage, detector, _request(head=f"Current time: 09:{n:02d}:31Z", tools=40))
            for n in range(MIN_OBSERVATIONS + 4)
        ]

        assert marked[-1] is False

    def test_no_saving_is_booked_when_declined_for_instability(self) -> None:
        # ADR-024: a stage may not report work that bought nothing.
        stage, detector = PrefixCacheStage(), UnstablePrefixStage()
        ctx = _ctx()

        for n in range(MIN_OBSERVATIONS + 4):
            request = _request(head=f"Current time: 09:{n:02d}:31Z")
            detector.before(request, ctx)
            result = stage.before(request, ctx)

        assert result.saved_input_tokens == 0
        assert result.note == ""
