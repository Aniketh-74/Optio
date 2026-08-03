"""``python -m optio_optimize.bench``'s ``--stage`` isolation (ADR-015).

The old ``--aggressive`` flag turned on ``semantic_cache`` and
``compress_prompt`` together, so no live result from it could be attributed
to either stage individually -- exactly the defect ADR-015 exists to fix.
``main()`` returns an int exit code rather than raising ``SystemExit``, so
these call it directly.
"""

from __future__ import annotations

import pathlib

import pytest

from optio_optimize.bench.__main__ import (
    _build_live_judge,
    _build_live_summarizer,
    _resolve_cheap_model,
    main,
)
from optio_optimize.bench.providers import SimulatedProvider
from optio_optimize.types import LLMRequest, LLMResponse

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


class TestRecordingComposesWithTwoProviderAudits:
    def test_the_second_arm_is_built_from_the_recorded_providers_inner(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``--record`` + ``--route-models-audit`` exited 2 on 2026-08-03.

        ``--record`` wraps the provider before the audit builds its cheap arm,
        and ``_same_provider_at`` tried to mirror the *wrapper* --
        ``RecordingProvider(model=..., guard=...)`` is not a constructor it
        has. The audit that cost real money to reach was unreachable with the
        flag that exists to keep what it paid for.
        """
        from optio_optimize.bench.__main__ import _same_provider_at
        from optio_optimize.bench.recording import RecordingProvider

        recording = RecordingProvider(SimulatedProvider(), tmp_path / "run.jsonl")

        second = _same_provider_at(recording, "gpt-4o-mini", None)

        assert second is not None
        assert second.model == "gpt-4o-mini"

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


class TestIsolateFlag:
    """``--stage X`` alone still runs the default-on stages alongside X."""

    def test_stage_without_isolate_leaves_the_default_stages_running(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The confounder --isolate exists for: deduplicate contributes to the
        # same delta, so the measured saving is not compress_prompt's alone.
        exit_code = main(["--stage", "compress_prompt", "--workload", "rag_queries"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "deduplicate" in out

    def test_isolate_turns_every_other_stage_off(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--stage", "compress_prompt", "--isolate", "--workload", "rag_queries"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "isolated run: compress_prompt only" in out
        assert "compress_prompt" in out
        for other in ("deduplicate", "prune_retrieval", "trim_history", "structured_output"):
            assert other not in out

    def test_isolate_also_drops_the_lossless_caches(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # exact_cache resolving a repeat before the isolated stage ever sees
        # it is how the original --aggressive run credited one stage for
        # another's work; retry_storm is the workload where that bites.
        exit_code = main(["--stage", "compress_prompt", "--isolate", "--workload", "retry_storm"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "exact_cache" not in out
        assert "n/a  (0/0 lookups)" in out

    def test_isolate_without_any_stage_is_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--isolate", "--workload", "rag_queries"])

        assert exit_code == 2
        assert "would benchmark an empty pipeline against itself" in capsys.readouterr().err


class TestNondeterminismControl:
    """The floor every divergence number has to be read against."""

    def test_control_runs_both_arms_unoptimized(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--control", "--workload", "unique_questions"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "optimizer off on BOTH arms" in out
        assert "floor: 0/12 differ with identical prompts" in out

    def test_control_says_the_simulator_cannot_measure_this(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A simulated floor of zero is a property of the simulator being a
        # pure function, not evidence that a provider is deterministic.
        main(["--control", "--workload", "rag_queries"])

        assert "always reports 0 divergences" in capsys.readouterr().out

    def test_control_ignores_stage_selection(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--control", "--stage", "compress_prompt", "--workload", "rag_queries"])

        out = capsys.readouterr().out
        assert "compress_prompt" not in out
        assert "floor:" in out


class TestDivergencePairsAreReadable:
    def test_show_divergences_prints_the_pairs(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "--stage",
                "compress_prompt",
                "--isolate",
                "--workload",
                "rag_queries",
                "--show-divergences",
                "2",
            ]
        )

        out = capsys.readouterr().out
        assert out.count("--- divergence ---") == 2
        assert "baseline:" in out
        assert "optimized:" in out

    def test_pairs_are_not_printed_unless_asked(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--stage", "compress_prompt", "--isolate", "--workload", "rag_queries"])

        assert "--- divergence ---" not in capsys.readouterr().out


class TestLiveJudgeBuilder:
    def test_a_verdict_of_equivalent_or_better_is_not_a_regression(self) -> None:
        verdicts = iter(["EQUIVALENT", "BETTER", "WORSE"])

        class _Judging(SimulatedProvider):
            def __call__(self, request: LLMRequest) -> LLMResponse:
                response = super().__call__(request)
                object.__setattr__(response, "content", next(verdicts))
                return response

        judge = _build_live_judge(_Judging())

        assert judge("a", "b") is True
        assert judge("a", "b") is True
        assert judge("a", "b") is False

    def test_the_judge_asks_about_regression_not_similarity(self) -> None:
        # The first version asked "is B equivalent to A" and scored a fuller
        # answer as WORSE, producing a 10/10-regression number that the
        # diverged pairs contradicted. The prompt must offer BETTER.
        captured = {}

        class _Capturing(SimulatedProvider):
            def __call__(self, request: LLMRequest) -> LLMResponse:
                captured["system"] = request.messages[0].content
                return super().__call__(request)

        _build_live_judge(_Capturing())("a", "b")

        assert "REGRESSION" in captured["system"]
        assert "BETTER" in captured["system"]
        assert "Extra detail in B is never WORSE." in captured["system"]
