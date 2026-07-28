"""Deduplicating and pruning retrieved context within a single request.

Both stages work on the same unit: a message's content split on blank lines,
the convention most retrieval code already uses to join chunks into a prompt
(and the one this package's own benchmark uses -- see
``bench/workloads.py:_rag_queries``). Neither stage looks across messages or
across requests; scope stays inside one request, same as every other stage
here.

The last block of a message is never a candidate for removal. By the same
convention :class:`~optio_optimize.stages.caching.PrefixCacheStage` uses for
"the newest message", the final paragraph is read as the instruction or
question, not supporting context -- dropping it would delete the actual ask
rather than its evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio_optimize.similarity import overlap_ratio, words
from optio_optimize.stages.base import Fidelity, Stage, StageResult

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest, Message

#: How message content is expected to separate distinct chunks. Not
#: configurable: it is the shape ``"\n\n".join(chunks)`` produces, which is
#: how retrieval code overwhelmingly already assembles a context block.
_BLOCK_SEP = "\n\n"


def _blocks(content: str) -> list[str]:
    """Split message content into candidate context blocks."""
    return content.split(_BLOCK_SEP)


class DeduplicateStage(Stage):
    """Drop context blocks that repeat, byte for byte, earlier in the message.

    Retrieval commonly returns the same passage more than once: two
    sub-queries land on the same chunk, or a re-ranker's windows overlap. This
    keeps the first occurrence of each block and removes every exact repeat
    that follows it within the same message.

    ``SHAPED``, not ``IDENTICAL``: no unique information is removed -- a
    repeated block carries nothing its first occurrence did not already
    provide -- but the prompt sent to the model is still different text, and
    this package does not claim byte-identical output for a transform it has
    not proven that strictly. (``structured_output`` was marked lossless
    under the same reasoning once; the live A/B benchmark showed it was not.
    Claiming ``IDENTICAL`` here would repeat that mistake without a benchmark
    to catch it.)
    """

    fidelity = Fidelity.SHAPED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "deduplicate"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Remove exact-duplicate blocks from each message, independently."""
        changed = False
        saved = 0
        new_messages: list[Message] = []

        for message in request.messages:
            blocks = _blocks(message.content)
            if len(blocks) < 2:
                new_messages.append(message)
                continue

            *body, tail = blocks
            seen: set[str] = set()
            kept: list[str] = []
            for block in body:
                key = block.strip()
                if key and key in seen:
                    saved += ctx.counter.count_text(block, request.model)
                    changed = True
                    continue
                if key:
                    seen.add(key)
                kept.append(block)

            if len(kept) == len(body):
                new_messages.append(message)
            else:
                new_messages.append(message.with_content(_BLOCK_SEP.join([*kept, tail])))

        if not changed:
            return self.declines(request)
        return StageResult(
            request=request.with_messages(tuple(new_messages)),
            saved_input_tokens=saved,
            note="exact-duplicate context blocks removed",
        )


#: Blocks (excluding the protected tail) required before scoring is worth it.
#: Below this there is nothing meaningful to rank against.
MIN_BLOCKS_TO_SCORE = 2

#: Share of a block's distinct words that must also appear in the query for
#: the block to survive. Low on purpose: word overlap between a short
#: question and a full passage is small even for the passage that answers it,
#: so a strict threshold would prune the block someone actually wanted. This
#: only removes blocks that share almost no vocabulary with what was asked.
MIN_RELEVANCE = 0.05

#: Always keep at least this many blocks. Zero context is a worse failure than
#: a slightly bloated prompt -- the same asymmetry that sets
#: :data:`~optio_optimize.stages.output.FLOOR_TOKENS`.
MIN_KEPT_BLOCKS = 1


class PruneRetrievalStage(Stage):
    """Drop retrieved chunks that share almost no vocabulary with the question.

    A cheap stand-in for a reranker: no embeddings, no network call, just word
    overlap between each context block and the message's final block (the
    question, by the module's convention). A block scoring below
    :data:`MIN_RELEVANCE` is judged unlikely to be something a real reranker
    would have kept either, and is dropped -- unless doing so would leave
    fewer than :data:`MIN_KEPT_BLOCKS`, in which case the highest-scoring
    blocks are kept regardless of whether they cleared the threshold.

    Deliberately conservative. This is not semantic search -- two chunks about
    the same topic phrased differently will not overlap lexically and both
    survive -- so it catches only the clearest waste: a chunk retrieval
    attached that shares no vocabulary with what was actually asked. A
    stricter threshold would need the eval-suite gate ADR-013 requires for
    stages that can misjudge relevance in the direction that costs someone a
    right answer.
    """

    fidelity = Fidelity.SHAPED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "prune_retrieval"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Drop low-overlap context blocks, keeping at least one."""
        changed = False
        saved = 0
        new_messages: list[Message] = []

        for message in request.messages:
            blocks = _blocks(message.content)
            if len(blocks) < MIN_BLOCKS_TO_SCORE + 1:
                new_messages.append(message)
                continue

            *body, tail = blocks
            query_words = words(tail)
            if not query_words:
                new_messages.append(message)
                continue

            scored = [(block, overlap_ratio(block, query_words)) for block in body]
            survivors = sum(1 for _, score in scored if score >= MIN_RELEVANCE)
            keep_count = max(MIN_KEPT_BLOCKS, survivors)
            ranked = sorted(range(len(scored)), key=lambda i: scored[i][1], reverse=True)
            keep_idx = set(ranked[:keep_count])

            kept: list[str] = []
            for i, (block, _score) in enumerate(scored):
                if i in keep_idx:
                    kept.append(block)
                else:
                    saved += ctx.counter.count_text(block, request.model)
                    changed = True

            if len(kept) == len(body):
                new_messages.append(message)
            else:
                new_messages.append(message.with_content(_BLOCK_SEP.join([*kept, tail])))

        if not changed:
            return self.declines(request)
        return StageResult(
            request=request.with_messages(tuple(new_messages)),
            saved_input_tokens=saved,
            note="low-relevance context blocks dropped",
        )
