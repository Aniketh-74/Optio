"""The shared lexical similarity helpers: words, overlap_ratio, jaccard."""

from __future__ import annotations

import pytest

from optio_optimize.similarity import jaccard, overlap_ratio, words

pytestmark = pytest.mark.optimize


class TestWords:
    def test_lowercases_and_tokenizes(self) -> None:
        assert words("Revenue Grew") == {"revenue", "grew"}

    def test_punctuation_is_not_a_word(self) -> None:
        assert words("hello, world!") == {"hello", "world"}

    def test_empty_text_has_no_words(self) -> None:
        assert words("") == set()


class TestOverlapRatio:
    def test_full_coverage_scores_one(self) -> None:
        assert overlap_ratio("revenue grew", {"revenue", "grew", "extra"}) == 1.0

    def test_no_shared_words_scores_zero(self) -> None:
        assert overlap_ratio("apples oranges", {"revenue", "grew"}) == 0.0

    def test_empty_text_scores_zero_not_undefined(self) -> None:
        assert overlap_ratio("", {"revenue"}) == 0.0

    def test_is_asymmetric(self) -> None:
        # "how much of TEXT is covered by REFERENCE" -- reversing which side
        # is which changes the answer, unlike jaccard.
        text_score = overlap_ratio("a b c", {"a"})
        reference_score = overlap_ratio("a", {"a", "b", "c"})
        assert text_score != reference_score
        assert text_score == pytest.approx(1 / 3)
        assert reference_score == 1.0


class TestJaccard:
    def test_identical_text_scores_one(self) -> None:
        assert jaccard("revenue grew", "revenue grew") == 1.0

    def test_disjoint_text_scores_zero(self) -> None:
        assert jaccard("apples oranges", "revenue grew") == 0.0

    def test_partial_overlap(self) -> None:
        # {a,b} vs {b,c}: intersection 1, union 3.
        assert jaccard("a b", "b c") == pytest.approx(1 / 3)

    def test_both_empty_scores_one_not_zero(self) -> None:
        """Nothing to disagree about -- treated as trivially identical."""
        assert jaccard("", "") == 1.0

    def test_one_empty_scores_zero(self) -> None:
        assert jaccard("", "something") == 0.0

    def test_is_symmetric(self) -> None:
        assert jaccard("a b c", "b c d") == jaccard("b c d", "a b c")
