"""CompressPromptStage: dropping sentences near-duplicate to one already kept."""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.compress import CompressPromptStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(content: str) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=(Message(role="user", content=content),),
        temperature=0.0,
    )


SENT_A = "The quarterly revenue increased significantly this period."
SENT_A_NEAR = "The quarterly revenue increased significantly this quarter."
SENT_B = "Operating costs remained flat year over year."
QUESTION = "What drove the change?"


class TestCompressPromptStage:
    def test_a_near_duplicate_sentence_is_dropped(self) -> None:
        stage = CompressPromptStage()
        content = " ".join([SENT_A, SENT_A_NEAR, QUESTION])

        result = stage.before(_request(content), _ctx())

        assert not result.short_circuited
        assert SENT_A in result.request.messages[0].content
        assert SENT_A_NEAR not in result.request.messages[0].content
        assert result.saved_input_tokens > 0
        assert result.note == "near-duplicate sentences dropped"

    def test_genuinely_different_sentences_all_survive(self) -> None:
        stage = CompressPromptStage()
        content = " ".join([SENT_A, SENT_B, QUESTION])

        result = stage.before(_request(content), _ctx())

        assert not result.short_circuited
        assert result.request.messages[0].content == content
        assert result.saved_input_tokens == 0

    def test_the_final_sentence_is_never_dropped_even_if_it_repeats(self) -> None:
        stage = CompressPromptStage()
        content = " ".join([SENT_A, SENT_B, SENT_A_NEAR])  # tail repeats SENT_A

        result = stage.before(_request(content), _ctx())

        assert result.request.messages[0].content.endswith(SENT_A_NEAR)

    def test_too_few_sentences_to_score_is_untouched(self) -> None:
        stage = CompressPromptStage()
        content = f"{SENT_A} {QUESTION}"  # one body sentence, one tail

        result = stage.before(_request(content), _ctx())

        assert result.request.messages[0].content == content
        assert result.saved_input_tokens == 0

    def test_order_of_surviving_sentences_is_preserved(self) -> None:
        stage = CompressPromptStage()
        content = " ".join([SENT_B, SENT_A, SENT_A_NEAR, QUESTION])

        result = stage.before(_request(content), _ctx())

        text = result.request.messages[0].content
        assert text.index(SENT_B) < text.index(SENT_A)
