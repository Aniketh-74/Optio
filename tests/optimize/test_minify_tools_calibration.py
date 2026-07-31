"""Tool-schema calibration is per-vendor, and ours was OpenAI's (ADR-036).

``ANNOTATION_STRIP_CALIBRATION = 0.37`` scales ``minify_tools``' raw JSON token
delta into a claimed saving. It was fitted against ``gpt-4o-mini`` and applied
to every vendor.

Measured against Anthropic's exact, free ``count_tokens`` across five tool
counts and three models::

    tools   real   raw json   claimed   real/raw
        1     89         69        25      1.290
        3    267        207        76      1.290
        5    445        345       127      1.290
       10    890        690       255      1.290
       20  1,780      1,380       510      1.290

    claude-haiku-4-5    1.290
    claude-sonnet-4-5   1.290
    claude-opus-4-5     1.290

**1.29 everywhere** -- a property of the API's tool rendering, not of a model.
Against 0.37, ``minify_tools`` understated its own saving by **71.4%** on
Anthropic: 993 claimed against 3,471 the provider actually stopped billing.

The vendors differ in *direction*, not just magnitude: OpenAI bills less than the
raw JSON tokenizes to, Anthropic bills more, because it re-renders the schema.
No single constant can be right for both.
"""

from __future__ import annotations

import pytest

from optio_optimize.stages.tools import (
    ANNOTATION_STRIP_CALIBRATION,
    ANNOTATION_STRIP_CALIBRATION_BY_MODEL,
    annotation_strip_calibration_for,
)

pytestmark = pytest.mark.optimize


class TestTheRatioFollowsTheVendor:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-opus-5",
            "claude-haiku-4-5-20251001",
        ],
    )
    def test_anthropic_models_use_the_measured_anthropic_ratio(self, model: str) -> None:
        assert annotation_strip_calibration_for(model) == pytest.approx(1.29)

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    def test_openai_models_keep_the_ratio_fitted_against_them(self, model: str) -> None:
        assert annotation_strip_calibration_for(model) == pytest.approx(0.37)

    def test_the_two_vendors_disagree_in_direction_not_just_size(self) -> None:
        """OpenAI bills less than the raw JSON; Anthropic bills more.

        The reason one constant cannot serve both, stated as an assertion so a
        future edit that averages them fails here.
        """
        openai = annotation_strip_calibration_for("gpt-4o-mini")
        anthropic = annotation_strip_calibration_for("claude-sonnet-4-5")

        assert openai < 1.0 < anthropic


class TestAnUnknownVendorCannotBeFlattered:
    def test_it_falls_back_to_the_lowest_measured_ratio(self) -> None:
        """ADR-036 decision 2, and ADR-027's reasoning about floors.

        Not the mean and not the higher: when the table cannot answer, fail
        toward the number that cannot overstate.
        """
        fallback = annotation_strip_calibration_for("some-future-model")

        assert fallback == ANNOTATION_STRIP_CALIBRATION
        assert fallback == min(ANNOTATION_STRIP_CALIBRATION_BY_MODEL.values())

    def test_an_empty_model_is_safe(self) -> None:
        assert annotation_strip_calibration_for("") == ANNOTATION_STRIP_CALIBRATION


class TestTheClaimedSavingMovesWithTheVendor:
    def test_anthropic_claims_more_than_openai_for_the_same_schemas(self) -> None:
        """The point of the change, measured through the stage itself."""
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.tools import MinifyToolsStage
        from optio_optimize.tokens import HeuristicCounter
        from optio_optimize.types import LLMRequest, Message

        tools = tuple(
            {
                "type": "function",
                "function": {
                    "name": f"tool_{n}",
                    "description": "does a thing",
                    "parameters": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "title": f"Tool {n} Arguments",
                        "$comment": "generated; do not edit",
                        "type": "object",
                        "properties": {"a": {"title": "A", "type": "string"}},
                    },
                },
            }
            for n in range(10)
        )
        ctx = StageContext(config=OptimizeConfig(), counter=HeuristicCounter())

        def claimed(model: str) -> int:
            request = LLMRequest(
                model=model,
                messages=(Message(role="user", content="go"),),
                tools=tools,
                temperature=0.0,
            )
            return MinifyToolsStage().before(request, ctx).saved_input_tokens

        assert claimed("claude-sonnet-4-5") > claimed("gpt-4o-mini")

    def test_the_ratio_between_them_matches_the_measurement(self) -> None:
        """1.29 / 0.37 = 3.49x, which is the 71.4% understatement inverted."""
        anthropic = annotation_strip_calibration_for("claude-sonnet-4-5")
        openai = annotation_strip_calibration_for("gpt-4o-mini")

        assert anthropic / openai == pytest.approx(3.49, abs=0.02)
