"""``python -m optio_optimize.bench``'s ``--stage`` isolation (ADR-015).

The old ``--aggressive`` flag turned on ``semantic_cache`` and
``compress_prompt`` together, so no live result from it could be attributed
to either stage individually -- exactly the defect ADR-015 exists to fix.
``main()`` returns an int exit code rather than raising ``SystemExit``, so
these call it directly.
"""

from __future__ import annotations

import pytest

from optio_optimize.bench.__main__ import _build_live_summarizer, _resolve_cheap_model, main
from optio_optimize.bench.providers import SimulatedProvider

pytestmark = pytest.mark.optimize


class TestStageIsolation:
    def test_a_single_stage_flag_enables_only_that_stage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--stage", "compress_prompt", "--workload", "rag_queries"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "compress_prompt" in out
        assert "semantic_cache" not in out

    def test_omitting_stage_enables_none_of_the_four(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--workload", "unique_questions"])

        out = capsys.readouterr().out
        assert exit_code == 0
        # SHAPED-tier stages (trim_history, dedup, ...) are on by default and
        # also trigger the "may reshape replies" note -- it isn't specific to
        # ALTERED stages. What must be absent is any of the four by name.
        for stage in ("route_models", "compress_prompt", "semantic_cache", "summarize_history"):
            assert stage not in out

    def test_two_stage_flags_can_still_combine_deliberately(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Deliberate combination is still possible -- isolation is the
        # default, not the only option -- but now the caller chose it
        # explicitly rather than inheriting it from one bundled flag.
        exit_code = main(
            ["--stage", "compress_prompt", "--stage", "semantic_cache", "--workload", "rag_queries"]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "compress_prompt" in out
        assert "semantic_cache" in out

    def test_strict_fidelity_with_a_stage_flag_is_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # An ALTERED stage cannot honor "byte-identical output" by
        # definition; this must fail fast rather than silently run and
        # report a meaningless divergence count.
        exit_code = main(["--stage", "compress_prompt", "--strict-fidelity"])

        assert exit_code == 2
        assert "cannot honor" in capsys.readouterr().err


class TestRouteModelsCheapModelResolution:
    def test_defaults_to_the_cheap_counterpart_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # SimulatedProvider defaults to "gpt-4o", which CHEAP_COUNTERPART maps
        # to "gpt-4o-mini" -- no --cheap-model needed.
        exit_code = main(["--stage", "route_models", "--workload", "unique_questions"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "gpt-4o to gpt-4o-mini" in out

    def test_an_explicit_cheap_model_overrides_the_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(
            [
                "--stage",
                "route_models",
                "--cheap-model",
                "gemini-2.0-flash",
                "--workload",
                "unique_questions",
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "gpt-4o to gemini-2.0-flash" in out

    def test_the_result_carries_a_pricing_caveat(self, capsys: pytest.CaptureFixture[str]) -> None:
        # ArmResult prices the whole optimized arm at one flat rate, so a
        # successful route (which would call a cheaper model per-request)
        # is not reflected in the cost number here -- the CLI must say so
        # rather than let the number look more trustworthy than it is.
        main(["--stage", "route_models", "--workload", "unique_questions"])

        out = capsys.readouterr().out
        assert "do not reflect the cheaper model actually being called" in out

    def test_a_model_with_no_known_cheap_counterpart_fails_fast(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # gpt-4o-mini has no entry in CHEAP_COUNTERPART (it is already the
        # cheap end of its own family) -- there is nothing honest to route
        # to. main() always benchmarks SimulatedProvider's default "gpt-4o"
        # model, which does resolve, so this exercises the resolver directly
        # rather than trying to force an unresolvable provider through the CLI.
        assert _resolve_cheap_model(None, "gpt-4o-mini") is None

    def test_an_explicit_cheap_model_always_wins_over_the_table(self) -> None:
        assert _resolve_cheap_model("claude-haiku-4", "gpt-4o") == "claude-haiku-4"

    def test_a_known_model_resolves_from_the_table(self) -> None:
        assert _resolve_cheap_model(None, "gpt-4o") == "gpt-4o-mini"


class TestLiveSummarizerBuilder:
    def test_uses_the_cheap_model_when_one_is_given(self) -> None:
        provider = SimulatedProvider(model="gpt-4o")
        summarizer = _build_live_summarizer(provider, "gpt-4o-mini")

        summary = summarizer("user: what is the budget?\nassistant: $50,000")

        assert isinstance(summary, str)
        assert summary  # SimulatedProvider always returns non-empty content

    def test_falls_back_to_the_providers_own_model_when_none_is_given(self) -> None:
        provider = SimulatedProvider(model="claude-haiku-4")
        summarizer = _build_live_summarizer(provider, None)

        summary = summarizer("user: what is the budget?\nassistant: $50,000")

        assert isinstance(summary, str)
        assert summary


class TestSummarizeHistoryIsolation:
    def test_runs_without_live_and_warns_it_is_plumbing_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--stage", "summarize_history", "--workload", "multi_turn_chat_long"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "summarize_history" in out
        assert "not ADR-015 evidence" in out
