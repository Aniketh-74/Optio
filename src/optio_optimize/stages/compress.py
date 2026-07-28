"""Drop sentences whose meaning a kept sentence already covers.

Where ``deduplicate`` (``stages/retrieval.py``) removes blocks that repeat
*byte for byte*, this removes ones that merely say the same thing again --
paraphrased restatement, a summary sentence that repeats what the paragraph
before it already said. The stronger claim is why this is ``ALTERED`` and
``deduplicate`` is only ``SHAPED``: exact repeats provably carry no new
information, but "near-duplicate" is a lexical-similarity judgment call, and
a judgment call can be wrong.

**What six live workloads found (2026-07-29, ADR-015).** On four of them this
stage is close to free: ``fan_out`` gave up 82.9% of its input tokens for
byte-identical output on every response, ``tool_calling_chat`` 68.9%, both
against a measured zero divergence floor. On two synthetic RAG workloads it
caused a real regression, and the shape of it is worth knowing because it is
not the one the risk model predicted.

The risk model here has always been "a false near-duplicate judgment erases a
fact that was never restated". That is not what happened. Nothing was erased:
every distinct sentence survived. What the stage removed was *repetition* --
a system prompt stating ``"If the context does not contain the answer, say
exactly: INSUFFICIENT CONTEXT. Never speculate."`` nine times collapsed to
stating it once, which is information-preserving by any reasonable definition.
The model then stopped honouring it, answering 6 of 10 questions it had
correctly refused, with an attribution the context did not support.

So the failure mode this stage actually carries, in addition to the one
already documented, is: **an instruction whose force came from being repeated
loses that force, and only requests that exercise that instruction reveal
it.** The same 9-to-1 collapse on ``fan_out`` and ``tool_calling_chat`` was
harmless because their tasks never reach a conditional-refusal branch. A
caller whose system prompt leans on repetition for emphasis -- a common
prompt-engineering habit -- is the one at risk here, and no token-count or
cost metric can see it. See ``docs/optimize-benchmarks.md``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from optio_optimize.similarity import jaccard
from optio_optimize.stages.base import Fidelity, Stage, StageResult

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest, Message

#: How a message's content is split into candidate units. Deliberately
#: simple -- a real sentence tokenizer is a dependency this package does not
#: otherwise need, and getting a boundary wrong here only makes compression
#: slightly less aggressive, never incorrect.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Similarity above which a later sentence is judged to repeat an earlier
#: one's meaning. High on purpose: lexical Jaccard similarity between two
#: genuinely different sentences on the same topic is already low, so a
#: lenient threshold would drop sentences that still say something new --
#: the mirror image of the false-positive risk
#: :data:`~optio_optimize.stages.retrieval.MIN_RELEVANCE` is calibrated
#: against on the opposite side.
NEAR_DUPLICATE_THRESHOLD = 0.6

#: Sentences (excluding the protected tail) required before scoring is worth
#: it. Below this there is nothing meaningful to compare against.
MIN_SENTENCES_TO_SCORE = 2


class CompressPromptStage(Stage):
    """Drop sentences near-duplicate to one already kept, per message.

    The final sentence of a message is never a candidate for removal -- by
    the same convention every other stage here uses for "the newest
    message", it is read as the question or instruction, not supporting
    context.

    ``Fidelity.ALTERED``: a sentence dropped for being "near-duplicate" is a
    lexical-similarity judgment, not a guarantee that no information was
    lost -- two sentences can share most of their words while differing in
    the one that matters (a date, a negation, a name). Off by default
    (ADR-013); the eval gate's ``FactPreservationCase`` exists to catch that
    failure on the cases it covers, not to prove it can never happen.
    """

    fidelity = Fidelity.ALTERED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "compress_prompt"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Drop near-duplicate sentences from each message, independently."""
        changed = False
        saved = 0
        new_messages: list[Message] = []

        for message in request.messages:
            sentences = _split_sentences(message.content)
            if len(sentences) < MIN_SENTENCES_TO_SCORE + 1:
                new_messages.append(message)
                continue

            *body, tail = sentences
            kept: list[str] = []
            for sentence in body:
                if any(jaccard(sentence, k) >= NEAR_DUPLICATE_THRESHOLD for k in kept):
                    saved += ctx.counter.count_text(sentence, request.model)
                    changed = True
                    continue
                kept.append(sentence)

            if len(kept) == len(body):
                new_messages.append(message)
            else:
                new_messages.append(message.with_content(" ".join([*kept, tail])))

        if not changed:
            return self.declines(request)
        return StageResult(
            request=request.with_messages(tuple(new_messages)),
            saved_input_tokens=saved,
            note="near-duplicate sentences dropped",
        )


def _split_sentences(content: str) -> list[str]:
    """Split message content into sentence-ish units."""
    return [s for s in _SENTENCE_SPLIT.split(content.strip()) if s]
