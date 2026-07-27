"""The cost lane (M2-4).

Covers what the lane does with a span: what it prices, what it refuses to
price, and which signals it emits versus omits. The omissions matter as much as
the emissions -- `docs/signals.md` makes a missing attribute mean *unknown*, so
a lane that fills gaps with zeros would silently defeat every budget policy
written against it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio import semconv
from optio.config import BudgetPolicy, default_config
from optio.lanes.base import Signal
from optio.lanes.cost.lane import CostLane
from optio.lanes.cost.ledger import CostLedger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan


class _StubRun:
    """A minimal RunLike."""

    def __init__(self, run_id: str = "run-1", budget: BudgetPolicy | None = None):
        self.run_id = run_id
        self.budget = budget


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """An in-memory exporter."""
    return InMemorySpanExporter()


MakeSpan = Callable[..., "ReadableSpan"]


@pytest.fixture
def make_span(exporter: InMemorySpanExporter) -> MakeSpan:
    """Build a finished span carrying the given attributes."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    def _make(**attributes: object) -> ReadableSpan:
        before = len(exporter.get_finished_spans())
        with tracer.start_as_current_span("llm") as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)  # type: ignore[arg-type]
        return exporter.get_finished_spans()[before]

    return _make


def _signals(lane_output: Sequence[Signal]) -> dict[str, bool | int | float | str]:
    """Index a lane's signals by name."""
    return {s.name: s.value for s in lane_output}


class TestPricingASpan:
    def test_a_priced_span_emits_actual_cost(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert signals[semconv.RUN_ACTUAL_COST] == pytest.approx(2.50)

    def test_response_model_wins_over_request_model(self, make_span: MakeSpan) -> None:
        # The response reports what actually served the request, which is what
        # was billed -- a request for "gpt-4o" served by a dated snapshot should
        # be priced as what ran.
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_RESPONSE_MODEL: "gpt-4o-mini",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert signals[semconv.RUN_ACTUAL_COST] == pytest.approx(0.15)

    def test_costs_accumulate_across_steps(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        run = _StubRun()

        for _ in range(3):
            span = make_span(
                **{
                    semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                    semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                    semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
                }
            )
            signals = _signals(lane.process_span(span, run))

        assert signals[semconv.RUN_ACTUAL_COST] == pytest.approx(7.50)

    def test_zero_token_span_costs_zero(self, make_span: MakeSpan) -> None:
        # A real zero, which must survive rather than being treated as unknown.
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 0,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert signals[semconv.RUN_ACTUAL_COST] == 0.0


class TestUnpriceableSpans:
    def test_unknown_model_emits_no_cost(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "a-model-we-do-not-know",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 1000,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert semconv.RUN_ACTUAL_COST not in signals

    def test_a_tool_span_without_tokens_emits_no_cost(self, make_span: MakeSpan) -> None:
        # Not every GenAI span is a billable model call.
        lane = CostLane(default_config())
        span = make_span(**{semconv.GEN_AI_TOOL_NAME: "search"})

        signals = _signals(lane.process_span(span, _StubRun()))
        assert semconv.RUN_ACTUAL_COST not in signals

    def test_an_unpriceable_step_surfaces_as_a_leak(self, make_span: MakeSpan) -> None:
        # Left reserved-and-open deliberately: a step whose cost is unknown is a
        # gap in the evidence, and the run should say so rather than hide it.
        lane = CostLane(default_config())
        run = _StubRun()
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "unknown-model",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 1000,
            }
        )
        lane.process_span(span, run)

        assert lane.ledger.close_run(run.run_id).leaked_steps == 1

    def test_negative_token_counts_are_not_priced(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: -100,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: -100,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert semconv.RUN_ACTUAL_COST not in signals


class TestBudgetSignals:
    def test_budget_remaining_is_emitted_when_a_budget_exists(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        run = _StubRun(budget=BudgetPolicy(limit_usd=10.00))
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, run))
        assert signals[semconv.RUN_BUDGET_REMAINING] == pytest.approx(7.50)

    def test_budget_remaining_is_omitted_without_a_budget(self, make_span: MakeSpan) -> None:
        # Absent, not infinite: "no budget" means nobody told us the limit.
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert semconv.RUN_BUDGET_REMAINING not in signals

    def test_budget_remaining_goes_negative_when_overspent(self, make_span: MakeSpan) -> None:
        # Clamping at zero would erase the most actionable fact: by how much.
        lane = CostLane(default_config())
        run = _StubRun(budget=BudgetPolicy(limit_usd=1.00))
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, run))
        assert signals[semconv.RUN_BUDGET_REMAINING] == pytest.approx(-1.50)

    def test_projected_cost_needs_a_step_ceiling(self, make_span: MakeSpan) -> None:
        # Without max_steps there is no finite worst case, so no projection.
        lane = CostLane(default_config())
        run = _StubRun(budget=BudgetPolicy(limit_usd=10.00))
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, run))
        assert semconv.RUN_PROJECTED_COST not in signals

    def test_projected_cost_extrapolates_remaining_steps(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        run = _StubRun(budget=BudgetPolicy(limit_usd=100.00, max_steps=5))
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )

        signals = _signals(lane.process_span(span, run))
        # One step at 2.50, four remaining at the same rate.
        assert signals[semconv.RUN_PROJECTED_COST] == pytest.approx(12.50)


class TestRunEnd:
    def test_run_end_emits_the_final_cost(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        run = _StubRun()
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert signals[semconv.RUN_ACTUAL_COST] == pytest.approx(2.50)

    def test_a_run_with_nothing_priced_reports_no_cost_at_all(self, make_span: MakeSpan) -> None:
        # The highest-stakes omission in the lane. Emitting 0.0 would tell a
        # budget policy the run was free, when the truth is that we could not
        # price it.
        lane = CostLane(default_config())
        run = _StubRun()
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "not-a-real-model",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert semconv.RUN_ACTUAL_COST not in signals

    def test_an_unpriced_run_reports_no_cost_per_task_either(self, make_span: MakeSpan) -> None:
        # Dividing an unpriced run's zero would report perfect efficiency for a
        # run we could not price at all.
        lane = CostLane(default_config())
        run = _StubRun()
        run.successes = 3  # type: ignore[attr-defined]
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "not-a-real-model",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert semconv.RUN_COST_PER_SUCCESSFUL_TASK not in signals

    def test_a_genuinely_free_run_still_reports_zero(self, make_span: MakeSpan) -> None:
        # The other side of the rule: a real zero must survive. Priced-at-zero
        # and could-not-price are different facts.
        lane = CostLane(default_config())
        run = _StubRun()
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 0,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert signals[semconv.RUN_ACTUAL_COST] == 0.0

    def test_cost_per_successful_task_is_omitted_without_a_success_count(
        self, make_span: MakeSpan
    ) -> None:
        # The quality lane owns successes (M5). Treating an unscored run as one
        # success would publish a headline number derived from a guess.
        lane = CostLane(default_config())
        run = _StubRun()
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert semconv.RUN_COST_PER_SUCCESSFUL_TASK not in signals

    def test_cost_per_successful_task_is_emitted_when_successes_are_known(
        self, make_span: MakeSpan
    ) -> None:
        lane = CostLane(default_config())
        run = _StubRun()
        run.successes = 2  # type: ignore[attr-defined]
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert signals[semconv.RUN_COST_PER_SUCCESSFUL_TASK] == pytest.approx(1.25)

    def test_zero_successes_omits_cost_per_task(self, make_span: MakeSpan) -> None:
        # A run that succeeded at nothing has a cost and a failure, both already
        # reported. Dividing by zero would emit infinity.
        lane = CostLane(default_config())
        run = _StubRun()
        run.successes = 0  # type: ignore[attr-defined]
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
            }
        )
        lane.process_span(span, run)

        signals = _signals(lane.on_run_end(run))
        assert semconv.RUN_COST_PER_SUCCESSFUL_TASK not in signals


class TestMalformedAttributes:
    """A framework reporting nonsense must produce no signal, not a bad one."""

    def test_boolean_token_counts_are_not_priced(self, make_span: MakeSpan) -> None:
        # bool subclasses int, so `True` would otherwise price as 1 token.
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                semconv.GEN_AI_USAGE_INPUT_TOKENS: True,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: True,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert semconv.RUN_ACTUAL_COST not in signals

    def test_a_non_string_model_is_not_priced(self, make_span: MakeSpan) -> None:
        lane = CostLane(default_config())
        span = make_span(
            **{
                semconv.GEN_AI_REQUEST_MODEL: 12345,
                semconv.GEN_AI_USAGE_INPUT_TOKENS: 1000,
                semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 500,
            }
        )

        signals = _signals(lane.process_span(span, _StubRun()))
        assert semconv.RUN_ACTUAL_COST not in signals

    def test_a_span_with_no_context_still_gets_a_step_id(self) -> None:
        # Defensive: the ledger keys on step id, so a span without a usable
        # context must still produce a distinct one rather than collide.
        from optio.lanes.cost.lane import _step_id

        class _NoContextSpan:
            def get_span_context(self) -> None:
                return None

        first, second = _NoContextSpan(), _NoContextSpan()
        assert _step_id(first) != _step_id(second)  # type: ignore[arg-type]


class TestIsolation:
    def test_concurrent_runs_do_not_share_cost(self, make_span: MakeSpan) -> None:
        # run_id is the join key; a leak here would merge two customers' costs.
        lane = CostLane(default_config())
        first, second = _StubRun("run-a"), _StubRun("run-b")

        for run in (first, first, second):
            span = make_span(
                **{
                    semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
                    semconv.GEN_AI_USAGE_INPUT_TOKENS: 1_000_000,
                    semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 0,
                }
            )
            lane.process_span(span, run)

        assert lane.ledger.snapshot("run-a").actual == pytest.approx(5.00)
        assert lane.ledger.snapshot("run-b").actual == pytest.approx(2.50)

    def test_a_shared_ledger_can_be_injected(self) -> None:
        ledger = CostLedger()
        lane = CostLane(default_config(), ledger=ledger)
        assert lane.ledger is ledger
