"""Two provider limits, measured rather than transcribed (ADR-037).

A model has a **context window** that binds the prompt and a **maximum output**
that binds the reply, and they are not the same number, not derivable from each
other, and only one of them was ever modelled here.

Read out of the provider's own 400s on 2026-08-01::

    model                 context window   max output
    claude-opus-4-5              200,000       64,000
    claude-opus-4-1              200,000       32,000     <- quarter of some caps
    claude-sonnet-4-5            200,000       64,000
    claude-haiku-4-5             200,000       64,000
    claude-opus-5              > 217,554      128,000
    claude-sonnet-5            > 217,554      128,000

The seven ``> 217,554`` rows are a **lower bound, not a window**: those models
accepted a 217,554-token prompt rather than rejecting it. ADR-037 records them
as unknown, because "larger than we probed" is not a number and ADR-029 exists
because inferring one reported a $10 bill as $30.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import (
    CONTEXT_WINDOW,
    MAX_OUTPUT_TOKENS,
    context_window_for,
    max_output_tokens_for,
)

pytestmark = pytest.mark.optimize


class TestTheWindowIsWhatTheProviderSaid:
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-5", "claude-opus-4-1", "claude-sonnet-4-5", "claude-haiku-4-5"],
    )
    def test_the_four_measured_windows_are_200k(self, model: str) -> None:
        assert context_window_for(model) == 200_000

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        ],
    )
    def test_a_window_known_only_as_a_lower_bound_is_not_a_window(self, model: str) -> None:
        """These models *accepted* the 217,554-token probe.

        That establishes ``window > 217,554`` and nothing more. Recording
        1,000,000 here would be the inference ADR-029 forbids, and it would make
        the diagnostic silent on exactly the requests it exists to catch.
        """
        assert context_window_for(model) is None
        assert model not in CONTEXT_WINDOW

    def test_an_unknown_model_gets_no_opinion(self) -> None:
        assert context_window_for("some-future-model") is None
        assert context_window_for("") is None

    def test_a_dated_id_resolves_to_its_alias(self) -> None:
        """The API reports a dated id back on every response (ADR-029)."""
        assert context_window_for("claude-haiku-4-5-20251001") == 200_000

    def test_a_newer_generation_never_inherits_an_older_window(self) -> None:
        """``claude-opus-4-1`` must not answer for ``claude-opus-4-8``."""
        assert context_window_for("claude-opus-4-8") is None

    @pytest.mark.parametrize("model", ["claude-opus-4-10", "claude-haiku-4-52", "claude-opus-4-5x"])
    def test_a_bare_prefix_match_is_not_the_same_model(self, model: str) -> None:
        """``claude-opus-4-10`` starts with ``claude-opus-4-1`` and is not it.

        The discriminating case for the suffix rule, and a plausible one: a
        two-digit successor to a one-digit model. Without the rule
        ``claude-opus-4-10`` silently inherits ``claude-opus-4-1``'s figures --
        including its 32,000 output cap, the lowest in the table, which is the
        direction that produces a 400 rather than a missed optimization.
        """
        assert context_window_for(model) is None
        assert max_output_tokens_for(model) is None


class TestTheOutputCapIsADifferentLimit:
    @pytest.mark.parametrize(
        ("model", "cap"),
        [
            ("claude-opus-4-5", 64_000),
            ("claude-opus-4-1", 32_000),
            ("claude-sonnet-4-5", 64_000),
            ("claude-haiku-4-5", 64_000),
            ("claude-opus-5", 128_000),
            ("claude-sonnet-5", 128_000),
            ("claude-fable-5", 128_000),
        ],
    )
    def test_the_measured_caps(self, model: str, cap: int) -> None:
        assert max_output_tokens_for(model) == cap

    def test_the_cap_is_not_a_fraction_of_the_window(self) -> None:
        """Three models share a 200,000 window with different caps.

        Stated as an assertion so a future edit that derives one from the other
        fails here rather than in a user's 400.
        """
        assert context_window_for("claude-opus-4-1") == context_window_for("claude-opus-4-5")
        assert max_output_tokens_for("claude-opus-4-1") != max_output_tokens_for("claude-opus-4-5")

    def test_an_unknown_model_gets_no_opinion(self) -> None:
        assert max_output_tokens_for("some-future-model") is None
        assert max_output_tokens_for("") is None

    def test_a_dated_id_resolves_to_its_alias(self) -> None:
        assert max_output_tokens_for("claude-sonnet-4-5-20250929") == 64_000


class TestTheTablesAgreeWithWhatIsPriced:
    def test_every_capped_model_is_one_this_package_prices(self) -> None:
        """A limit for a model nothing can price is a row nobody reads."""
        from optio_optimize.config import PRICING

        assert set(MAX_OUTPUT_TOKENS) <= set(PRICING)
        assert set(CONTEXT_WINDOW) <= set(PRICING)

    def test_no_window_is_smaller_than_its_own_output_cap(self) -> None:
        """A reply that cannot fit the window it is generated into is incoherent."""
        for model, window in CONTEXT_WINDOW.items():
            cap = MAX_OUTPUT_TOKENS.get(model)
            if cap is not None:
                assert cap.tokens < window.tokens, model
