"""DeduplicateStage and PruneRetrievalStage: trimming redundant retrieved context.

``rag_queries`` (bench/workloads.py) is the workload these exist for: context
assembled from overlapping retrieved chunks, joined with blank lines the same
way this module's stages split them back apart.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.retrieval import DeduplicateStage, PruneRetrievalStage
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize

CHUNK_A = "Quarterly revenue increased, driven by enterprise subscription growth."
CHUNK_B = "Operating expenses rose more slowly than revenue in the period."
CHUNK_C = "The weather in the reporting period was unseasonably mild."


def _ctx() -> StageContext:
    return StageContext(config=OptimizeConfig(), counter=HeuristicCounter())


def _request(content: str) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=(Message(role="user", content=content),),
        temperature=0.0,
    )


class TestDeduplicateRemovesExactRepeats:
    def test_a_repeated_block_is_dropped(self) -> None:
        stage = DeduplicateStage()
        content = "\n\n".join([CHUNK_A, CHUNK_B, CHUNK_A, "Question: what drove revenue?"])

        result = stage.before(_request(content), _ctx())

        blocks = result.request.messages[0].content.split("\n\n")
        assert blocks == [CHUNK_A, CHUNK_B, "Question: what drove revenue?"]
        assert result.saved_input_tokens > 0

    def test_the_final_block_survives_even_if_it_repeats_earlier_content(self) -> None:
        stage = DeduplicateStage()
        content = "\n\n".join([CHUNK_A, CHUNK_B, CHUNK_A])  # tail repeats the first block

        result = stage.before(_request(content), _ctx())

        blocks = result.request.messages[0].content.split("\n\n")
        assert blocks[-1] == CHUNK_A
        assert blocks.count(CHUNK_A) == 2, "the tail's repeat must survive removal"

    def test_no_duplicates_means_no_change(self) -> None:
        stage = DeduplicateStage()
        content = "\n\n".join([CHUNK_A, CHUNK_B, CHUNK_C])

        result = stage.before(_request(content), _ctx())

        assert result.request.messages[0].content == content
        assert result.saved_input_tokens == 0
        assert not result.short_circuited

    def test_a_single_block_message_is_untouched(self) -> None:
        stage = DeduplicateStage()

        result = stage.before(_request("just one block"), _ctx())

        assert result.request.messages[0].content == "just one block"
        assert result.saved_input_tokens == 0

    def test_order_is_preserved(self) -> None:
        stage = DeduplicateStage()
        content = "\n\n".join([CHUNK_A, CHUNK_B, CHUNK_A, CHUNK_C, "Question: ..."])

        result = stage.before(_request(content), _ctx())

        blocks = result.request.messages[0].content.split("\n\n")
        assert blocks == [CHUNK_A, CHUNK_B, CHUNK_C, "Question: ..."]


class TestPruneRetrievalDropsLowOverlapBlocks:
    def test_an_unrelated_block_is_dropped(self) -> None:
        stage = PruneRetrievalStage()
        content = "\n\n".join(
            [CHUNK_A, CHUNK_B, CHUNK_C, "Question: what drove revenue this quarter?"]
        )

        result = stage.before(_request(content), _ctx())

        blocks = result.request.messages[0].content.split("\n\n")
        assert CHUNK_C not in blocks
        assert CHUNK_A in blocks
        assert CHUNK_B in blocks
        assert result.saved_input_tokens > 0

    def test_at_least_one_block_always_survives(self) -> None:
        stage = PruneRetrievalStage()
        content = "\n\n".join([CHUNK_C, "another unrelated aside", "Question: what drove revenue?"])

        result = stage.before(_request(content), _ctx())

        blocks = result.request.messages[0].content.split("\n\n")
        # The question survives plus at least one context block, never zero.
        assert len(blocks) >= 2

    def test_the_final_block_is_never_scored_or_dropped(self) -> None:
        stage = PruneRetrievalStage()
        content = "\n\n".join([CHUNK_A, CHUNK_B, "totally unrelated question text"])

        result = stage.before(_request(content), _ctx())

        blocks = result.request.messages[0].content.split("\n\n")
        assert blocks[-1] == "totally unrelated question text"

    def test_too_few_blocks_to_score_is_untouched(self) -> None:
        stage = PruneRetrievalStage()
        content = "\n\n".join([CHUNK_A, "Question: what drove revenue?"])

        result = stage.before(_request(content), _ctx())

        assert result.request.messages[0].content == content
        assert result.saved_input_tokens == 0

    def test_a_question_with_no_recognizable_words_leaves_the_message_untouched(self) -> None:
        stage = PruneRetrievalStage()
        content = "\n\n".join([CHUNK_A, CHUNK_B, CHUNK_C, "???"])

        result = stage.before(_request(content), _ctx())

        assert result.request.messages[0].content == content


class TestItIntegratesWithThePipeline:
    def test_overlapping_rag_context_shrinks_without_breaking_the_call(self) -> None:
        """The shape rag_queries exercises: repeated and low-value chunks per call."""
        from optio_optimize import LLMResponse, Optimizer

        optimizer = Optimizer(
            deduplicate=True,
            prune_retrieval=True,
            trim_history=False,
            exact_cache=False,
            prefix_cache=False,
        )
        seen_lengths: list[int] = []

        def provider(request: LLMRequest) -> LLMResponse:
            seen_lengths.append(len(request.messages[0].content))
            return LLMResponse(content="ok", input_tokens=100, output_tokens=5, model=request.model)

        context = "\n\n".join([CHUNK_A, CHUNK_B, CHUNK_A, CHUNK_C, "Question: what drove revenue?"])
        request = LLMRequest(
            model="gpt-4o",
            messages=(Message(role="user", content=context),),
            temperature=0.0,
        )

        response = optimizer.call(request, provider)

        assert response.content == "ok"
        assert seen_lengths[0] < len(context)
        assert optimizer.report.total_saved_tokens > 0


class TestReorderContextPutsTheBestBlocksAtTheEdges:
    """Positional attention is U-shaped; this stage exploits that and nothing else.

    The one stage in the package that saves no tokens. Its tests are therefore
    about arrangement and about the guarantee that it stays a rearrangement --
    the same blocks, none added, none dropped.
    """

    def test_the_strongest_block_goes_first_and_the_next_goes_last(self) -> None:
        from optio_optimize.stages.retrieval import ReorderContextStage

        question = "Question: what drove revenue growth this quarter?"
        strong = "Revenue growth this quarter was driven by enterprise subscriptions."
        medium = "Revenue and costs are reported quarterly."
        weak = "The parking garage closes on Tuesday for resurfacing."
        content = "\n\n".join([weak, medium, strong, question])

        result = ReorderContextStage().before(_request(content), _ctx())
        blocks = result.request.messages[0].content.split("\n\n")

        assert blocks[0] == strong, "the best block must lead"
        assert blocks[-1] == question, "the question stays at the tail"
        assert blocks[-2] == medium, "the second-best block goes to the far edge"
        assert blocks[1] == weak, "the weakest block belongs in the middle"

    def test_it_is_a_rearrangement_and_never_a_removal(self) -> None:
        from optio_optimize.stages.retrieval import ReorderContextStage

        question = "Question: what drove revenue?"
        blocks = [f"block {n} revenue costs" for n in range(6)]
        content = "\n\n".join([*blocks, question])

        result = ReorderContextStage().before(_request(content), _ctx())
        out = result.request.messages[0].content.split("\n\n")

        assert sorted(out) == sorted([*blocks, question])
        assert result.saved_input_tokens == 0, "this stage saves no tokens, by design"

    def test_an_already_arranged_message_is_left_alone(self) -> None:
        from optio_optimize.stages.retrieval import ReorderContextStage

        stage = ReorderContextStage()
        content = "\n\n".join(["revenue grew", "parking closed", "Question: revenue?"])

        once = stage.before(_request(content), _ctx()).request
        twice = stage.before(once, _ctx())

        assert twice.note == "", "reordering an arranged message is churn, not work"

    def test_too_few_blocks_to_rank_is_a_decline(self) -> None:
        from optio_optimize.stages.retrieval import ReorderContextStage

        result = ReorderContextStage().before(_request("just one block"), _ctx())
        assert result.note == ""

    def test_it_reshapes_without_claiming_to_alter_content(self) -> None:
        from optio_optimize.stages.base import Fidelity
        from optio_optimize.stages.retrieval import ReorderContextStage

        assert ReorderContextStage().fidelity is Fidelity.SHAPED
        assert not ReorderContextStage().lossy

    @pytest.mark.parametrize(
        ("ranked", "expected"),
        [
            (["a"], ["a"]),
            (["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "c", "b"]),
            (["a", "b", "c", "d"], ["a", "c", "d", "b"]),
            (["a", "b", "c", "d", "e"], ["a", "c", "e", "d", "b"]),
        ],
    )
    def test_the_edge_first_arrangement(self, ranked: list[str], expected: list[str]) -> None:
        from optio_optimize.stages.retrieval import _edges_first

        assert _edges_first(ranked) == expected
