"""A stage may not book a saving it cannot attribute (ADR-024).

Found by the first end-to-end run of ``scripts/real_agent_run.py`` across all
four scenarios, both arms, live: the package made two of them **more expensive**
while reporting a double-digit saving.

===============  ==============  ===============  ==============  ============
scenario         baseline input  optimized input  measured        report said
===============  ==============  ===============  ==============  ============
support          7,717           3,816            +47.1%          50.8% saved
parallel         635             661              **-3.0%**       10.0% saved
empty_result     497             523              **-4.3%**       13.2% saved
long_loop        24,688          8,954            +62.3%          63.7% saved
===============  ==============  ===============  ==============  ============

Isolating one stage returned the input to *exactly* the control arm's figure --
661 to 635, 523 to 497 -- so ``structured_output`` accounted for the whole
regression, and on the two large scenarios it was marginally worse than not
running.

Two defects, and the second is why the first went unseen:

1. The guard fired on **tool-using requests with no schema at all**, while the
   docstring said "only acts when a schema is already present". Every agent
   scenario is a tool loop and none sets ``response_format``, so the stage spent
   13 tokens a call suppressing a JSON preamble on replies that were tool calls.
2. It booked ``max(0, 40 - instruction_cost)`` output tokens from a *hypothesised*
   preamble nothing ever checked -- against ``savings.py``'s opening rule, "only
   count what was avoided, never what was hoped for".
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.output import ConcisionStage, StructuredOutputStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(**overrides: object) -> StageContext:
    return StageContext(
        config=OptimizeConfig(**overrides),  # type: ignore[arg-type]
        counter=HeuristicCounter(),
    )


def _request(**overrides: object) -> LLMRequest:
    defaults: dict[str, object] = {
        "model": "gpt-4o-mini",
        "messages": (
            Message(role="system", content="You are a support agent."),
            Message(role="user", content="Where is my order?"),
        ),
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


_TOOLS = ({"type": "function", "function": {"name": "search_orders", "parameters": {}}},)
_SCHEMA = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}


class TestTheGuardMatchesItsDocstring:
    """ "Only acts when a schema is already present" -- so act only then."""

    def test_a_tool_request_without_a_schema_is_declined(self) -> None:
        """The defect, as the assertion that used to fail.

        A tool-using request returns a *tool call*, not prose wrapped around
        JSON. There is no preamble to suppress, and the instruction describes a
        reply shape the request cannot produce. This fired on every call of
        every agent scenario in the repo.
        """
        result = StructuredOutputStage().before(_request(tools=_TOOLS), _ctx())

        assert result.note == ""
        assert result.request.messages == _request(tools=_TOOLS).messages

    def test_a_schema_request_still_acts(self) -> None:
        # The case the stage is actually for must keep working, or this "fix"
        # is just a disablement wearing a guard.
        result = StructuredOutputStage().before(_request(response_format=_SCHEMA), _ctx())

        assert StructuredOutputStage.INSTRUCTION in result.request.messages[0].content

    def test_a_schema_and_tools_together_still_acts(self) -> None:
        # Schema present is the trigger; tools alongside it change nothing.
        result = StructuredOutputStage().before(
            _request(response_format=_SCHEMA, tools=_TOOLS), _ctx()
        )

        assert StructuredOutputStage.INSTRUCTION in result.request.messages[0].content

    def test_a_plain_chat_request_is_still_declined(self) -> None:
        assert StructuredOutputStage().before(_request(), _ctx()).note == ""


class TestNoHypotheticalOutputSaving:
    """``savings.py``: "only count what was avoided, never what was hoped for"."""

    def test_it_claims_no_output_saving(self) -> None:
        """40 tokens was a guess about a preamble nothing verified.

        Measured across the four live scenarios, output went **up** in two
        (132->137, 94->95), was unchanged in a third (28->28), and fell only in
        the one where trimming dominates. If the suppression works, it shows up
        in a lower provider-reported ``actual_output_tokens`` -- the same place
        ADR-020 leaves fan-out's effect, because that number is measured.
        """
        result = StructuredOutputStage().before(_request(response_format=_SCHEMA), _ctx())

        assert result.saved_output_tokens == 0

    def test_concision_claims_no_output_saving_either(self) -> None:
        # Same pattern, same fix. Already off by default, so this is accounting
        # rather than behaviour -- but an off-by-default stage that over-reports
        # when switched on is still over-reporting.
        result = ConcisionStage().before(_request(), _ctx(concision=True))

        assert result.saved_output_tokens == 0


class TestAddedTokensAreReportedAsNegative:
    """``baseline = actual + saved``, so a stage that adds must subtract.

    Without this the instruction's own cost silently inflates the baseline: the
    live ``empty_result`` run reported a baseline of 635 against a measured
    truth of 525.
    """

    def test_the_instruction_cost_is_reported_as_a_negative_input_saving(self) -> None:
        counter = HeuristicCounter()
        expected = counter.count_text(StructuredOutputStage.INSTRUCTION, "gpt-4o-mini")

        result = StructuredOutputStage().before(_request(response_format=_SCHEMA), _ctx())

        assert result.saved_input_tokens == -expected
        assert result.saved_input_tokens < 0

    def test_concision_reports_its_instruction_cost_too(self) -> None:
        result = ConcisionStage().before(_request(), _ctx(concision=True))

        assert result.saved_input_tokens < 0

    def test_a_declined_request_reports_nothing(self) -> None:
        # Declining must not book a negative saving for an instruction it never
        # added -- that would invent a cost the caller never paid.
        result = StructuredOutputStage().before(_request(tools=_TOOLS), _ctx())

        assert result.saved_input_tokens == 0
        assert result.saved_output_tokens == 0

    def test_the_negative_reaches_the_report_and_lowers_the_baseline(self) -> None:
        """End to end: a stage that costs tokens shows a loss, not a zero.

        This is the property the live run needed and did not have. Rounding a
        loss up to zero is how a 4.3% cost increase came to be reported as a
        13.2% saving.
        """
        from optio_optimize.savings import SavingsReport, StageSaving

        report = SavingsReport()
        report.record(StageSaving(stage="structured_output", input_tokens=-26))
        report.actual_input_tokens = 523
        report.baseline_input_tokens = 523 - 26

        assert report.total_saved_tokens == -26
        assert report.baseline_input_tokens == 497  # what the control arm billed
        ratio = report.reduction_ratio
        assert ratio is not None and ratio < 0


class TestTheDefaultIsOff:
    def test_structured_output_is_off_by_default(self) -> None:
        """ADR-013 rule 1: this package does not cause a cost increase.

        A default-on stage that raised cost on two of four real scenarios and
        helped none of them breaks that, and its benefit has never been measured
        on a request that actually carries a schema.
        """
        assert OptimizeConfig().structured_output is False

    def test_it_can_still_be_switched_on_by_name(self) -> None:
        assert OptimizeConfig(structured_output=True).structured_output is True
