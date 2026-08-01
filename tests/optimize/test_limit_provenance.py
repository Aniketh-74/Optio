"""Where a limit came from is part of the limit (ADR-041).

``CONTEXT_WINDOW`` and ``MAX_OUTPUT_TOKENS`` carried 15 Anthropic models and
nothing else, and the docstrings explained why: every value was read out of a
provider's own 400 rather than off a documentation page. That rule is a good
one and it produced a bad outcome -- **coverage became a function of whose API
key the maintainer happened to hold.** For a library that means to work against
every vendor, that is the wrong thing for coverage to depend on.

The rule was applied at one bar where two are needed. There are two ways to know
a limit and they are not equally strong:

*measured*
    This project sent a request and read the answer. Costs a key and some money.

*published*
    The vendor documents it and we have not observed it. Costs nothing, can go
    stale, and is **not a guess** -- it is a citable claim with a URL and a date.

The tables had a slot for the first and for absence, and none for the second, so
a documented-but-unobserved figure had nowhere to live and was filed next to
genuine unknowns. That single missing slot is what made this package look
Anthropic-only.

What does *not* change: nothing may be entered without saying where it came
from. The type makes that structural rather than a convention -- a bare int is
no longer a valid table entry.
"""

from __future__ import annotations

import re

import pytest

from optio_optimize.config import (
    CONTEXT_WINDOW,
    MAX_OUTPUT_TOKENS,
    Evidence,
    context_window_for,
    context_window_provenance,
    max_output_tokens_for,
    max_output_tokens_provenance,
)

pytestmark = pytest.mark.optimize


class TestALimitCarriesItsEvidence:
    def test_a_measured_window_says_so(self) -> None:
        assert context_window_provenance("claude-haiku-4-5") is Evidence.MEASURED

    def test_a_measured_cap_says_so(self) -> None:
        assert max_output_tokens_provenance("claude-haiku-4-5") is Evidence.MEASURED

    def test_an_absent_model_has_no_evidence_rather_than_weak_evidence(self) -> None:
        """Seven Anthropic models sit here: a probe established only that their
        window exceeds 217,554, which is not a number (ADR-037). ``None`` is the
        honest answer and must stay distinguishable from a published one."""
        assert context_window_provenance("claude-opus-5") is None
        assert context_window_for("claude-opus-5") is None


class TestTheTablesAreNoLongerOneVendor:
    """The point of the exercise. Both tables held 15 Anthropic models and
    nothing else -- not because the package is Anthropic-only, but because
    measurement needs a key and Anthropic's was the key to hand.

    These figures are OpenAI's own, read off their model pages on the date
    recorded beside them. Nobody billed anything to learn them.
    """

    def test_openai_windows_are_carried_and_marked_published(self) -> None:
        assert context_window_for("gpt-4o") == 128_000
        assert context_window_provenance("gpt-4o") is Evidence.PUBLISHED

    def test_openai_caps_are_carried_and_marked_published(self) -> None:
        assert max_output_tokens_for("gpt-4o") == 16_384
        assert max_output_tokens_for("gpt-4o-mini") == 16_384
        assert max_output_tokens_provenance("gpt-4o") is Evidence.PUBLISHED
        assert max_output_tokens_provenance("gpt-4o-mini") is Evidence.PUBLISHED

    def test_the_openai_cap_is_far_below_its_window(self) -> None:
        """16,384 against 128,000 -- the two limits are independent (ADR-037),
        and this is the widest gap in either table. A cap inferred from a window
        would be wrong here by a factor of nearly eight."""
        window = context_window_for("gpt-4o")
        cap = max_output_tokens_for("gpt-4o")
        assert window is not None and cap is not None
        assert cap * 7 < window

    def test_published_is_never_mistaken_for_measured(self) -> None:
        """The distinction has to survive the lookup, or it buys nothing.

        A caller deciding whether to trust a limit enough to act on it needs the
        two apart -- which is the whole reason the slot was added.
        """
        assert context_window_provenance("claude-haiku-4-5") is Evidence.MEASURED
        assert context_window_provenance("gpt-4o") is Evidence.PUBLISHED

    def test_more_than_one_vendor_now_has_a_window(self) -> None:
        vendors = {model.split("-")[0] for model in CONTEXT_WINDOW}

        assert len(vendors) > 1


class TestTheNumbersStillReadTheSame:
    """The lookup contract is unchanged; only what backs it is richer.

    Every caller of these two functions predates this change, and a migration
    that quietly altered a limit would be indistinguishable from the drift the
    provenance is meant to expose.
    """

    def test_the_measured_windows_are_unchanged(self) -> None:
        assert context_window_for("claude-haiku-4-5") == 200_000
        assert context_window_for("claude-opus-4-1") == 200_000

    def test_the_measured_caps_are_unchanged(self) -> None:
        assert max_output_tokens_for("claude-opus-4-1") == 32_000
        assert max_output_tokens_for("claude-haiku-4-5") == 64_000
        assert max_output_tokens_for("claude-opus-5") == 128_000

    def test_generation_boundaries_still_do_not_leak(self) -> None:
        """ADR-029's rule, restated because this change touches the lookup.

        ``claude-opus-4-1`` caps at 32,000 and ``claude-opus-4-5`` at 64,000, so
        inheriting across that boundary hands back double the real limit and
        produces the 400 the lookup exists to avoid.
        """
        assert max_output_tokens_for("claude-opus-4-1-20250805") == 32_000
        assert context_window_for("claude-opus-4") is None

    def test_a_longer_version_number_is_a_different_model(self) -> None:
        """The case a bare ``startswith`` gets wrong, and the reason
        ``_SAME_MODEL_SUFFIX`` requires four digits or a ``v`` tag.

        ``claude-opus-4-10`` starts with ``claude-opus-4-1``, so prefix matching
        alone would hand a hypothetical tenth release the first one's 32,000 cap
        -- the generation-boundary leak ADR-029 exists to prevent, in the one
        shape a date-suffix test cannot reach.
        """
        assert max_output_tokens_for("claude-opus-4-10") is None

    def test_an_unknown_model_is_still_silent(self) -> None:
        assert context_window_for("some-future-model") is None
        assert max_output_tokens_for("") is None


class TestNothingEntersWithoutASource:
    """The structural half. A convention that says "cite your source" is a
    convention someone forgets under deadline; a constructor that will not build
    without one is not.
    """

    @pytest.mark.parametrize("table", [CONTEXT_WINDOW, MAX_OUTPUT_TOKENS])
    def test_every_entry_names_where_it_came_from(self, table: dict[str, object]) -> None:
        for model, limit in table.items():
            assert getattr(limit, "source", ""), f"{model} has no source"

    @pytest.mark.parametrize("table", [CONTEXT_WINDOW, MAX_OUTPUT_TOKENS])
    def test_every_entry_is_dated(self, table: dict[str, object]) -> None:
        """A published figure with no date cannot be told from a current one.

        Staleness has to be visible: this is the same reason a recording carries
        the date it was made (ADR-039).
        """
        for model, limit in table.items():
            checked = getattr(limit, "checked", "")
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", checked), f"{model}: {checked!r}"

    @pytest.mark.parametrize("table", [CONTEXT_WINDOW, MAX_OUTPUT_TOKENS])
    def test_a_published_entry_cites_a_url(self, table: dict[str, object]) -> None:
        """Measured entries name a probe script; published ones must name a page
        a reader can open and check, because that is the only thing separating
        "the vendor says so" from "someone typed a number"."""
        for model, limit in table.items():
            if getattr(limit, "evidence", None) is Evidence.PUBLISHED:
                assert str(getattr(limit, "source", "")).startswith("http"), model

    @pytest.mark.parametrize("table", [CONTEXT_WINDOW, MAX_OUTPUT_TOKENS])
    def test_a_bare_number_is_not_a_valid_entry(self, table: dict[str, object]) -> None:
        """The point of the type. Before this, a table entry was an ``int``, and
        an ``int`` cannot say where it came from."""
        for model, limit in table.items():
            assert not isinstance(limit, int), f"{model} is a bare int"


class TestTheDiagnosticSaysWhatItIsGoingOn:
    """A warning that a prompt will be rejected is only actionable if the reader
    can tell whether the limit was observed or read off a page. A published
    window that has moved produces a warning about a rejection that will not
    happen, and the reader needs to be able to reach that conclusion themselves.
    """

    def test_a_measured_limit_is_not_hedged(self) -> None:
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.diagnostics import WindowPressureStage
        from optio_optimize.tokens import HeuristicCounter
        from optio_optimize.types import LLMRequest, Message

        stage = WindowPressureStage()
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content=" ".join(["reconcile"] * 400_000)),),
            temperature=0.0,
        )

        stage.before(
            request,
            StageContext(config=OptimizeConfig(), counter=HeuristicCounter()),
        )

        detail = stage.findings[0].detail
        assert "published" not in detail

    def test_the_finding_still_carries_no_prompt_content(self) -> None:
        """Section 10 survives the change."""
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.diagnostics import WindowPressureStage
        from optio_optimize.tokens import HeuristicCounter
        from optio_optimize.types import LLMRequest, Message

        stage = WindowPressureStage()
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content=" ".join(["reconcile"] * 400_000)),),
            temperature=0.0,
        )

        stage.before(
            request,
            StageContext(config=OptimizeConfig(), counter=HeuristicCounter()),
        )

        assert "reconcile" not in stage.findings[0].detail
