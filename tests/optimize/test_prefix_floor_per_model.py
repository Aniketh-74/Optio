"""The cacheable prefix floor is per-model (ADR-027).

Anthropic's minimum cacheable prefix spans a factor of eight, and one constant
was wrong in both directions: too high for Opus 5, where it declined a
breakpoint that would have worked, and too low for four models, where it placed
one the provider silently discards **while the note reported success**.

============================================ =========
model                                        minimum
============================================ =========
Opus 5                                       512
Opus 4.8, Sonnet 5, Sonnet 4.6 / 4.5         1,024
Opus 4.7, Haiku 3.5                          2,048
**Haiku 4.5**, Opus 4.6 / 4.5                **4,096**
============================================ =========

What forced the issue: the benchmark now defaults to ``claude-haiku-4-5``, whose
floor is 4,096, and **eleven of its twelve workloads have prompts below that**.
The full live run reported zero provider cache reads on every one of them and
nothing said why -- which reads like a broken stage rather than a prompt that
was never eligible.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.caching import (
    MIN_PREFIX_TOKENS,
    PrefixCacheStage,
    min_prefix_tokens_for,
)
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(model: str, *, prompt_tokens: int) -> LLMRequest:
    """A conversation whose stable prefix is roughly ``prompt_tokens`` long."""
    system = "policy clause " + ("word " * max(1, prompt_tokens))
    return LLMRequest(
        model=model,
        messages=(
            Message(role="system", content=system),
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ),
        temperature=0.0,
    )


class TestTheFloorFollowsTheModel:
    @pytest.mark.parametrize(
        ("model", "floor"),
        [
            ("claude-opus-5", 512),
            ("claude-sonnet-5", 1_024),
            ("claude-sonnet-4-6", 1_024),
            ("claude-opus-4-8", 1_024),
            ("claude-opus-4-7", 2_048),
            ("claude-haiku-4-5", 4_096),
            ("claude-opus-4-6", 4_096),
        ],
    )
    def test_each_model_gets_its_published_minimum(self, model: str, floor: int) -> None:
        assert min_prefix_tokens_for(model) == floor

    def test_a_dated_id_resolves_through_its_alias(self) -> None:
        # The API reports the dated id back on every response, so a table that
        # only knows the alias returns the wrong floor for half the lookups.
        assert min_prefix_tokens_for("claude-haiku-4-5-20251001") == 4_096
        assert min_prefix_tokens_for("claude-sonnet-4-5-20250929") == 1_024

    def test_an_unknown_model_keeps_todays_constant(self) -> None:
        """Not the lowest floor and not the highest.

        The lowest would place markers four known models discard; the highest
        would decline breakpoints that work on most. Today's value means an
        unrecognized name behaves exactly as it did before this ADR, which is
        the property ADR-016 actually asks for.
        """
        assert min_prefix_tokens_for("some-future-model") == MIN_PREFIX_TOKENS
        assert min_prefix_tokens_for("gpt-4o") == MIN_PREFIX_TOKENS


class TestOpus5GainsCachingItWasDenied:
    def test_a_700_token_prefix_is_marked_on_opus_5(self) -> None:
        # Between 512 and 1,024: eligible on Opus 5 and refused by the old
        # constant. The one direction of this change that adds a saving.
        result = PrefixCacheStage().before(_request("claude-opus-5", prompt_tokens=700), _ctx())

        assert any(m.cacheable for m in result.request.messages)

    def test_the_same_prefix_is_refused_on_haiku_4_5(self) -> None:
        result = PrefixCacheStage().before(_request("claude-haiku-4-5", prompt_tokens=700), _ctx())

        assert not any(m.cacheable for m in result.request.messages)


class TestABelowFloorPromptIsDeclinedAndExplained:
    def test_a_sub_floor_prompt_is_not_marked_on_haiku(self) -> None:
        """The eleven-of-twelve case, as an assertion.

        A 2,500-token prompt clears the old 1,024 constant and misses Haiku
        4.5's real 4,096, so the marker was placed, reported, sent -- and
        discarded by the provider.
        """
        result = PrefixCacheStage().before(
            _request("claude-haiku-4-5", prompt_tokens=2_500), _ctx()
        )

        assert not any(m.cacheable for m in result.request.messages)

    def test_the_decline_names_the_floor_and_the_model(self) -> None:
        """ "No cache reads" must be diagnosable from the report.

        Silence here is what made the live run read as a broken stage rather
        than a prompt that was never eligible.
        """
        stage = PrefixCacheStage()

        stage.before(_request("claude-haiku-4-5", prompt_tokens=2_500), _ctx())

        assert "4096" in stage.last_decline_reason.replace(",", "")
        assert "claude-haiku-4-5" in stage.last_decline_reason

    def test_a_prompt_over_the_floor_is_still_marked(self) -> None:
        # The fix must not disable the stage on the workloads where it works.
        result = PrefixCacheStage().before(
            _request("claude-haiku-4-5", prompt_tokens=5_000), _ctx()
        )

        assert any(m.cacheable for m in result.request.messages)

    def test_no_saving_is_booked_for_a_declined_prefix(self) -> None:
        # ADR-024: a stage may not report work that bought nothing.
        result = PrefixCacheStage().before(
            _request("claude-haiku-4-5", prompt_tokens=2_500), _ctx()
        )

        assert result.saved_input_tokens == 0
        assert result.note == ""
