"""The unstable-prefix detector: what it catches, and what it must not claim.

A diagnostic's failure mode is not being wrong about the world, it is being
confidently wrong to a person who then goes looking. A false
``unstable_system_prompt`` sends someone hunting for a timestamp in a prompt
that has none; a false ``unstable_tool_order`` tells them to sort a list that is
already sorted and is genuinely carrying different tools. So most of these tests
are about the detector staying quiet.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import Fidelity, StageContext
from optio_optimize.stages.diagnostics import (
    MIN_OBSERVATIONS,
    PrefixFinding,
    UnstablePrefixStage,
)
from optio_optimize.tokens import default_counter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize

_STABLE_SYSTEM = "You are a careful assistant. Answer only from the given context."


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=default_counter())


def _tool(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": f"Does {name}."}}


def _run(
    stage: UnstablePrefixStage,
    count: int,
    *,
    system: str | None = _STABLE_SYSTEM,
    system_varies: bool = False,
    tools: list[tuple[dict[str, Any], ...]] | None = None,
) -> list[PrefixFinding]:
    """Feed ``count`` requests through the stage and return its findings."""
    ctx = _ctx()
    for n in range(count):
        messages: tuple[Message, ...] = ()
        if system is not None:
            text = f"Current time: 2026-07-29T10:{n:02d}:00Z\n{system}" if system_varies else system
            messages = (Message(role="system", content=text),)
        messages = (*messages, Message(role="user", content=f"question {n}"))
        request = LLMRequest(
            model="gpt-4o",
            messages=messages,
            tools=tools[n % len(tools)] if tools else (),
            temperature=0.0,
        )
        stage.before(request, ctx)
    return stage.findings


class TestItCatchesTheBugItExistsFor:
    def test_a_timestamped_system_prompt_is_reported(self) -> None:
        findings = _run(UnstablePrefixStage(), 20, system_varies=True)

        assert [f.kind for f in findings] == ["unstable_system_prompt"]

    def test_the_report_says_what_to_do_about_it(self) -> None:
        # A finding nobody can act on is noise. The fix for this one is
        # specific: move the varying part below the stable block.
        findings = _run(UnstablePrefixStage(), 20, system_varies=True)

        assert "below" in findings[0].detail
        assert "timestamp" in findings[0].detail

    def test_an_unstable_tool_order_is_reported(self) -> None:
        # The same three tools, permuted. Every request carries identical
        # capability and none of it caches.
        a, b, c = _tool("alpha"), _tool("beta"), _tool("gamma")
        findings = _run(
            UnstablePrefixStage(),
            20,
            tools=[(a, b, c), (b, c, a), (c, a, b), (a, c, b), (b, a, c)],
        )

        assert "unstable_tool_order" in [f.kind for f in findings]


class TestItStaysQuietWhenItShould:
    def test_a_stable_prompt_produces_no_finding(self) -> None:
        assert _run(UnstablePrefixStage(), 50) == []

    def test_a_young_process_is_not_accused(self) -> None:
        # Three requests with three distinct prompts is equally consistent with
        # a bug and with an agent that has only just started.
        assert _run(UnstablePrefixStage(), MIN_OBSERVATIONS - 1, system_varies=True) == []

    def test_a_handful_of_distinct_prompts_is_not_a_bug(self) -> None:
        # Several features or tenants behind one optimizer legitimately send a
        # few different system prompts. Each caches separately; nothing is
        # broken. Only a prompt that is *never* the same twice is the failure.
        stage = UnstablePrefixStage()
        ctx = _ctx()
        for n in range(40):
            request = LLMRequest(
                model="gpt-4o",
                messages=(
                    Message(role="system", content=f"You are assistant number {n % 4}."),
                    Message(role="user", content=f"q{n}"),
                ),
                temperature=0.0,
            )
            stage.before(request, ctx)

        assert stage.findings == []

    def test_genuinely_different_tools_are_not_an_ordering_bug(self) -> None:
        """The discriminator the tool finding depends on.

        Sending different tools per request is a legitimate design -- per-tenant
        tool sets, feature-gated capabilities. Telling that caller to "sort your
        list" would be wrong advice about a real cost they may have chosen. The
        stage separates the cases by comparing each request against its own
        sorted form: when sorting does not stabilize it, the tools themselves
        differ and this is not an ordering problem.
        """
        stage = UnstablePrefixStage()
        ctx = _ctx()
        for n in range(30):
            request = LLMRequest(
                model="gpt-4o",
                messages=(Message(role="user", content=f"q{n}"),),
                tools=(_tool(f"tool_{n}"), _tool(f"other_{n}")),
                temperature=0.0,
            )
            stage.before(request, ctx)

        assert [f.kind for f in stage.findings] == []

    def test_requests_with_no_system_prompt_are_not_accused(self) -> None:
        assert _run(UnstablePrefixStage(), 30, system=None) == []


class TestItChangesNothing:
    def test_the_request_passes_through_untouched(self) -> None:
        stage = UnstablePrefixStage()
        request = LLMRequest(
            model="gpt-4o",
            messages=(Message(role="system", content=_STABLE_SYSTEM),),
            tools=(_tool("alpha"),),
            temperature=0.0,
        )

        result = stage.before(request, _ctx())

        assert result.request is request
        assert result.response is None
        assert result.saved_input_tokens == 0
        assert result.saved_output_tokens == 0

    def test_it_never_claims_to_have_done_something(self) -> None:
        # A note is how a stage tells the report it fired. This one never
        # fires -- its output is the finding, not a saving, and appearing in a
        # savings table with 0 tokens would read as a stage that failed.
        stage = UnstablePrefixStage()
        _run(stage, 20, system_varies=True)

        assert stage.before(_request_with_time(99), _ctx()).note == ""

    def test_it_is_trivially_identical_fidelity(self) -> None:
        assert UnstablePrefixStage().fidelity is Fidelity.IDENTICAL
        assert not UnstablePrefixStage().lossy


class TestItRespectsTheContentRule:
    def test_no_prompt_content_reaches_a_finding(self) -> None:
        """§10 outlives the package boundary: findings carry shape, not text.

        The prompt here contains a distinctive secret. It is what makes the
        prefix unstable, so it is exactly the string a naive implementation
        would quote back while explaining the problem.
        """
        stage = UnstablePrefixStage()
        ctx = _ctx()
        for n in range(20):
            request = LLMRequest(
                model="gpt-4o",
                messages=(
                    Message(role="system", content=f"session-token-hunter2-{n} You are helpful."),
                    Message(role="user", content="q"),
                ),
                temperature=0.0,
            )
            stage.before(request, ctx)

        rendered = " ".join(f.detail for f in stage.findings)
        assert stage.findings, "expected a finding, or this test proves nothing"
        assert "hunter2" not in rendered
        assert "You are helpful" not in rendered

    def test_the_log_line_carries_no_content_either(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="optio_optimize"):
            stage = UnstablePrefixStage()
            ctx = _ctx()
            for n in range(20):
                stage.before(
                    LLMRequest(
                        model="gpt-4o",
                        messages=(
                            Message(role="system", content=f"secret-{n} instructions"),
                            Message(role="user", content="q"),
                        ),
                        temperature=0.0,
                    ),
                    ctx,
                )

        assert "secret-" not in caplog.text


class TestItWarnsOnceNotPerRequest:
    def test_a_repeated_warning_would_be_filtered_out(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 200 requests through a hot loop must not produce 190 identical
        # warnings; that is a log people learn to ignore.
        with caplog.at_level(logging.WARNING, logger="optio_optimize"):
            _run(UnstablePrefixStage(), 200, system_varies=True)

        assert caplog.text.count("system prompt differed") == 1


class TestTheWorkloadPairPricesTheBug:
    """``timestamped_agent`` and ``multi_turn_chat`` differ by one line.

    That line is a clock above the system prompt instead of below it. Measured
    against the simulator's automatic-prefix-cache model, the clean version
    gets 16,128 of 19,050 prompt tokens served from the provider's cache and
    the timestamped one gets **zero** -- which is the entire cost of the bug,
    priced rather than asserted.
    """

    def test_the_detector_separates_the_pair(self) -> None:
        from optio_optimize import LLMResponse, Optimizer
        from optio_optimize.bench import WORKLOADS

        def _reply(request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                content="x",
                input_tokens=100,
                output_tokens=10,
                model=request.model,
                finish_reason="stop",
            )

        verdicts = {}
        for name in ("multi_turn_chat", "timestamped_agent"):
            optimizer = Optimizer(OptimizeConfig())
            for request in WORKLOADS[name].requests():
                optimizer.call(request, _reply)
            verdicts[name] = [f.kind for f in optimizer.findings]

        assert verdicts["timestamped_agent"] == ["unstable_system_prompt"]
        assert verdicts["multi_turn_chat"] == [], (
            "a growing conversation is not an unstable prefix: the system prompt is "
            "identical every turn and only the history after it grows"
        )

    def test_the_provider_cache_gets_nothing_from_the_broken_one(self) -> None:
        from optio_optimize.bench import WORKLOADS, SimulatedProvider
        from optio_optimize.bench.harness import run_arm

        cached = {}
        for name in ("multi_turn_chat", "timestamped_agent"):
            arm, _ = run_arm(
                name,
                WORKLOADS[name].requests(),
                SimulatedProvider(prefix_cache_style="automatic"),
                OptimizeConfig(enabled=False),
            )
            cached[name] = arm.cached_input_tokens

        assert cached["timestamped_agent"] == 0
        assert cached["multi_turn_chat"] > 10_000


def _request_with_time(n: int) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=(
            Message(role="system", content=f"Current time: {n}\n{_STABLE_SYSTEM}"),
            Message(role="user", content="q"),
        ),
        temperature=0.0,
    )
