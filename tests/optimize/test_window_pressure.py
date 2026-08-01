"""``context_limit`` finally reads something (ADR-037).

The field was accepted, validated, documented and read by nothing; so was
:func:`~optio_optimize.tokens.fits_in_window`. This stage is the first
production caller of both.

What it reports is bounded by what the provider actually enforces, and that was
measured before any of this was written::

    prompt 217,554                       -> 400  prompt is too long: > 200000
    prompt 158,965 + max_tokens 21,000   -> ACCEPTED, generated normally

So the window binds the **prompt**, and the sum of prompt and reply ceiling is
not a limit at all. This stage therefore says nothing about ``max_tokens``, and
nothing here lowers one on account of window pressure -- there is no rejection
to prevent, and a lower ceiling would truncate a reply the provider was willing
to generate.

Like ``detect_unstable_prefix``, it changes nothing. The fix for a prompt that
will not fit lives in the caller's conversation management, and a stage that
"helpfully" trimmed it would be an ``ALTERED`` change ADR-015 has no evidence
for.
"""

from __future__ import annotations

import logging

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import Fidelity, StageContext
from optio_optimize.stages.diagnostics import WindowPressureStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(**config: object) -> StageContext:
    return StageContext(config=OptimizeConfig(**config), counter=HeuristicCounter())  # type: ignore[arg-type]


def _request(model: str, *, words: int) -> LLMRequest:
    """A request whose prompt is roughly ``words`` words of ordinary prose."""
    return LLMRequest(
        model=model,
        messages=(Message(role="user", content=" ".join(["reconcile"] * words)),),
        temperature=0.0,
    )


class TestItReportsAPromptThatCannotFit:
    def test_a_prompt_over_the_window_is_reported(self) -> None:
        stage = WindowPressureStage()

        stage.before(_request("claude-haiku-4-5", words=400_000), _ctx())

        kinds = [f.kind for f in stage.findings]
        assert "prompt_exceeds_context_window" in kinds

    def test_the_finding_names_the_limit_and_carries_no_prompt_content(self) -> None:
        """Section 10 applies to this package too.

        The detail must be actionable without quoting a single token of what the
        caller sent -- the same bar ``detect_unstable_prefix`` meets by
        describing the *shape* of a problem rather than its text.
        """
        stage = WindowPressureStage()
        request = _request("claude-haiku-4-5", words=400_000)

        stage.before(request, _ctx())

        detail = stage.findings[0].detail
        assert "200,000" in detail or "200000" in detail
        assert "reconcile" not in detail

    def test_a_request_under_pressure_is_reported_before_it_fails(self) -> None:
        """The point of the diagnostic: warn while the caller can still act.

        65,000 words is ~162,500 tokens, which the 1.15 margin inflates past
        90% of 200,000 but not past 200,000 itself. That band exists only
        because both thresholds go through ``fits_in_window``; comparing raw
        tokens against ``limit * PRESSURE_RATIO`` made it empty, and this test
        is what found that.
        """
        stage = WindowPressureStage()

        stage.before(_request("claude-haiku-4-5", words=65_000), _ctx())

        kinds = [f.kind for f in stage.findings]
        assert "prompt_near_context_window" in kinds
        assert "prompt_exceeds_context_window" not in kinds

    def test_an_ordinary_request_produces_no_finding(self) -> None:
        stage = WindowPressureStage()

        stage.before(_request("claude-haiku-4-5", words=50), _ctx())

        assert stage.findings == []

    def test_each_finding_is_logged_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning repeating through a hot loop is one people filter out."""
        stage = WindowPressureStage()
        ctx = _ctx()
        with caplog.at_level(logging.WARNING, logger="optio_optimize"):
            for _ in range(5):
                stage.before(_request("claude-haiku-4-5", words=400_000), ctx)

        assert len(stage.findings) == 1
        assert len(caplog.records) == 1


class TestWhereTheLimitComesFrom:
    def test_the_callers_context_limit_wins_over_the_table(self) -> None:
        stage = WindowPressureStage()

        stage.before(_request("claude-haiku-4-5", words=8_000), _ctx(context_limit=1_000))

        assert [f.kind for f in stage.findings] == ["prompt_exceeds_context_window"]

    def test_the_table_answers_when_the_caller_has_not(self) -> None:
        stage = WindowPressureStage()

        stage.before(_request("claude-haiku-4-5", words=400_000), _ctx())

        assert stage.findings

    def test_an_unmeasured_model_gets_no_opinion(self) -> None:
        """Seven Anthropic models sit in exactly this position (ADR-037).

        Their window is known only to exceed 217,554, which is not a number.
        Silence is the correct output; a guessed window would make this stage
        wrong on the newest and largest models in the table.
        """
        stage = WindowPressureStage()

        stage.before(_request("claude-opus-5", words=400_000), _ctx())

        assert stage.findings == []

    def test_an_unknown_model_with_an_explicit_limit_is_still_checked(self) -> None:
        """The caller knows their window even when this package does not."""
        stage = WindowPressureStage()

        stage.before(_request("some-future-model", words=8_000), _ctx(context_limit=1_000))

        assert stage.findings


class TestItChangesNothing:
    def test_the_request_passes_through_untouched(self) -> None:
        stage = WindowPressureStage()
        request = _request("claude-haiku-4-5", words=400_000)

        result = stage.before(request, _ctx())

        assert result.request is request
        assert not result.short_circuited

    def test_it_claims_no_saving_and_leaves_no_note(self) -> None:
        """A note marks a stage as having *done* something, and this one never does."""
        stage = WindowPressureStage()

        result = stage.before(_request("claude-haiku-4-5", words=400_000), _ctx())

        assert result.saved_input_tokens == 0
        assert result.saved_output_tokens == 0
        assert result.note == ""

    def test_it_is_identical_fidelity(self) -> None:
        assert WindowPressureStage.fidelity is Fidelity.IDENTICAL

    def test_it_never_touches_max_tokens(self) -> None:
        """The measured reason: ``prompt + max_tokens`` over the window is accepted.

        158,965 + 21,000 against a 200,000 window generated normally. Lowering a
        ceiling here would pay a real truncation for an error that does not
        occur.
        """
        stage = WindowPressureStage()
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content=" ".join(["reconcile"] * 125_000)),),
            max_tokens=60_000,
            temperature=0.0,
        )

        result = stage.before(request, _ctx())

        assert result.request.max_tokens == 60_000


class TestTheCheapGuardCannotSkipARealFinding:
    """The guard skips work; it must never skip a finding.

    It exists because counting a 2.7 MB conversation costs 152 ms and this stage
    saves nothing. Its correctness rests on characters being an upper bound on
    tokens -- true for message text, and **false** for the two things
    ``count_request`` also counts:

    * **tool schemas**, which live in ``request.tools`` and contribute nothing to
      any message's ``content``;
    * **images**, which contribute ~1,600 tokens each while ``Message.content``
      holds only extracted text (ADR-022).

    Either one lets a request that will certainly be rejected slip past a guard
    that was written as though messages were the whole prompt -- the same
    flattering assumption that cost $7.60 in the probe that produced ADR-037.
    """

    def test_a_prompt_that_is_mostly_tool_schemas_is_still_measured(self) -> None:
        stage = WindowPressureStage()
        tools = tuple(
            {
                "type": "function",
                "function": {
                    "name": f"tool_{n}",
                    "description": "reconcile " * 400,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for n in range(200)
        )
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="go"),),
            tools=tools,
            temperature=0.0,
        )

        stage.before(request, _ctx(context_limit=1_000))

        assert [f.kind for f in stage.findings] == ["prompt_exceeds_context_window"]

    def test_many_small_messages_add_up(self) -> None:
        """A long conversation is many small turns, not one big one.

        The shape the bound is most likely to be written wrongly for, and the
        shape real agent traffic actually has.
        """
        stage = WindowPressureStage()
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=tuple(Message(role="user", content="reconcile " * 20) for _ in range(200)),
            temperature=0.0,
        )

        stage.before(request, _ctx(context_limit=1_000))

        assert stage.findings

    def test_an_ordinary_request_never_reaches_the_tokenizer(self) -> None:
        """The guard's whole purpose, pinned without a timing assertion.

        Counting an 81-turn conversation costs 152 ms and this stage saves
        nothing, so on ordinary traffic it must not count at all. Asserting on
        elapsed time would be flaky; asserting the counter was never called is
        exact.
        """

        class _RecordingCounter:
            is_exact = False

            def __init__(self) -> None:
                self.calls = 0

            def count_text(self, text: str, model: str = "") -> int:
                self.calls += 1
                return len(text) // 4

        counter = _RecordingCounter()
        ctx = StageContext(config=OptimizeConfig(), counter=counter)
        stage = WindowPressureStage()

        stage.before(_request("claude-haiku-4-5", words=50), ctx)

        assert counter.calls == 0

    def test_an_image_heavy_prompt_is_still_measured(self) -> None:
        """``Message.content`` holds extracted text, so an image is ~0 characters.

        A vision request used to count 8 tokens against a real ~1,535 for
        exactly this reason (ADR-022).
        """
        stage = WindowPressureStage()
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=tuple(
                Message(
                    role="user",
                    content="see",
                    extra={
                        "_raw": {
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "",
                                    },
                                }
                            ]
                        }
                    },
                )
                for _ in range(30)
            ),
            temperature=0.0,
        )

        stage.before(request, _ctx(context_limit=1_000))

        assert stage.findings


class TestItRunsAfterTheStagesThatShrinkThePrompt:
    """Placement is load-bearing here, and the first draft had it wrong.

    ``detect_unstable_prefix`` runs first because it diagnoses how the caller
    assembles their prompt. This one answers *will the provider reject what we
    are about to send*, which is a fact about the final request -- so warning
    about a prompt ``trim_history`` has already cut back is warning about a
    rejection that will not happen.

    Placed first it was also actively harmful: counting an untrimmed 2.7 MB
    conversation cost **152 ms against the 100 ms budget**, so eight stages were
    skipped and the trim that would have fixed that very request never ran.
    """

    def test_it_comes_after_trim_history(self) -> None:
        from optio_optimize.stages import build_stages

        names = [s.name for s in build_stages(OptimizeConfig())]

        assert names.index("window_pressure") > names.index("trim_history")

    def test_it_comes_after_prefix_cache_too(self) -> None:
        """Which makes it last, since prefix_cache must see the final list."""
        from optio_optimize.stages import build_stages

        names = [s.name for s in build_stages(OptimizeConfig())]

        assert names[-1] == "window_pressure"

    def test_it_is_built_when_the_flag_is_on_and_absent_when_off(self) -> None:
        from optio_optimize.stages import build_stages

        on = [s.name for s in build_stages(OptimizeConfig(detect_window_pressure=True))]
        off = [s.name for s in build_stages(OptimizeConfig(detect_window_pressure=False))]

        assert "window_pressure" in on
        assert "window_pressure" not in off

    def test_a_huge_prompt_still_gets_trimmed(self) -> None:
        """The regression this placement exists to prevent, end to end.

        A diagnostic that saves nothing must never consume the budget the
        saving stages need.
        """
        from optio_optimize.pipeline import Pipeline
        from optio_optimize.stages import build_stages

        config = OptimizeConfig()
        pipeline = Pipeline(config=config, stages=build_stages(config))
        messages = tuple(
            Message(role="user" if i % 2 == 0 else "assistant", content=f"q{i} " + "detail " * 4000)
            for i in range(81)
        )
        request = LLMRequest(model="claude-haiku-4-5", messages=messages, temperature=0.0)

        prepared = pipeline.prepare(request)

        assert len(prepared.request.messages) < 81


class TestTheSafetyMarginIsAppliedInTheRightDirection:
    def test_an_inexact_counter_is_treated_as_possibly_undercounting(self) -> None:
        """``fits_in_window`` inflates an estimate rather than trusting it.

        Being wrong optimistically means a provider rejection the user sees as a
        crash; being wrong pessimistically means one warning that was not
        strictly needed. This stage inherits that asymmetry rather than
        re-deciding it, which is the whole reason the function exists.
        """
        stage = WindowPressureStage()
        ctx = _ctx()
        assert not ctx.counter.is_exact

        # Sized to sit between the raw count and the count times 1.15, so it
        # only reports when the margin is applied.
        from optio_optimize.tokens import count_request

        request = _request("claude-haiku-4-5", words=60_000)
        raw = count_request(request, ctx.counter)
        ctx = _ctx(context_limit=int(raw * 1.08))

        stage.before(request, ctx)

        assert [f.kind for f in stage.findings] == ["prompt_exceeds_context_window"]
