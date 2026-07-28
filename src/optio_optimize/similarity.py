"""Cheap lexical similarity -- the zero-dependency default across this package.

No embeddings, no network call. Several stages need an answer to "how similar
are these two texts", and the one here is word-set overlap, not cosine
distance over a vector index. That is a real ceiling: two texts about the
same thing phrased differently score low here even though a human (or an
embedding model) would call them close. Accepted deliberately -- it costs
nothing, needs no new dependency, and never performs network I/O on the hot
path, the same constraint optio's core holds for the identical reason (§4.1).

Every stage that uses a similarity function here takes one as a parameter
rather than calling this module directly (:class:`~optio_optimize.stages.
semantic_cache.SemanticCacheStage`'s ``similarity_fn`` is the clearest
example) -- swapping in a real embedding metric is a constructor argument,
not a rewrite.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    """Return the lowercased word set of ``text``."""
    return set(_WORD.findall(text.lower()))


def overlap_ratio(text: str, reference_words: set[str]) -> float:
    """Fraction of ``text``'s distinct words that also appear in ``reference_words``.

    Asymmetric by design: "how much of this text is covered by that
    vocabulary" is what :class:`~optio_optimize.stages.retrieval.
    PruneRetrievalStage` needs -- how much of a chunk is relevant to the
    question -- not "how similar are these two texts overall".

    Args:
        text: The text being scored.
        reference_words: The vocabulary to check coverage against.

    Returns:
        ``0.0`` for empty ``text``; otherwise the fraction of ``text``'s
        distinct words present in ``reference_words``.
    """
    text_words = words(text)
    if not text_words:
        return 0.0
    return len(text_words & reference_words) / len(text_words)


def jaccard(a: str, b: str) -> float:
    """Symmetric similarity: shared words over the union of both vocabularies.

    The default metric for "are these two texts close enough to treat as the
    same": :class:`~optio_optimize.stages.compress.CompressPromptStage`'s
    near-duplicate-sentence check, and :class:`~optio_optimize.stages.
    semantic_cache.SemanticCacheStage`'s default match function.

    Args:
        a: First text.
        b: Second text.

    Returns:
        ``1.0`` when both are empty (nothing to disagree about), ``0.0``
        when exactly one is, otherwise the Jaccard index of their word sets.
    """
    a_words, b_words = words(a), words(b)
    if not a_words and not b_words:
        return 1.0
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)
