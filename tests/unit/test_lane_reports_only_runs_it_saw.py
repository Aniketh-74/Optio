"""A lane must not report on a run it never observed (ADR-044).

``on_run_end`` fires **every** registered run-end observer, not just the one
whose provider produced the run. Two live tracer providers in one process --
two agents, a test suite, a service that reconfigures tracing -- means two taps,
each with its own :class:`~optio.lanes.cost.ledger.CostLedger`, and both are
asked about every run that ends.

The tap that actually metered the run answers from evidence. The other one has
never heard of the run, so its ledger reports ``reconciled=0, reserved=0,
open_steps=0`` -- which ``_has_cost_evidence`` reads as *nothing was attempted
yet*, the state where a full budget genuinely is available. It then emits
``budget_remaining = <the whole limit>`` for a run that has been spending money
the entire time.

That is precisely the failure `test_an_unknown_model_reports_no_cost_rather_
than_a_free_run` was written for: *"a policy reading `deny if budget_remaining
< 0.50` would never fire, and the runaway agent this library exists to catch
would run unchecked."* The guard was put in the arithmetic; the hole was in who
was allowed to do the arithmetic.

It surfaced when ADR-043 stopped taps being silently lost to ``id()`` reuse.
Before that fix the foreign taps were mostly not installed, so they were not
around to answer. Fixing one silent failure exposed the one it had been hiding.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider

from optio.config import BudgetPolicy, Config
from optio.lanes.cost.lane import CostLane


class _Run:
    """The minimum a lane needs to be asked about a run."""

    def __init__(self, run_id: str, budget: BudgetPolicy | None) -> None:
        self.run_id = run_id
        self.budget = budget
        self.outcomes: list[object] = []


class TestALaneStaysSilentAboutRunsItNeverSaw:
    def test_an_unseen_run_produces_no_signals(self) -> None:
        """The bug, at the smallest scale that shows it."""
        lane = CostLane(Config())

        signals = lane.on_run_end(
            _Run("a-run-this-lane-never-metered", BudgetPolicy(limit_usd=10.0, max_steps=100))
        )

        assert signals == [], f"a lane invented {[s.name for s in signals]} for a run it never saw"

    def test_it_does_not_invent_a_full_budget(self) -> None:
        """Named separately because this is the consequence that matters.

        ``budget_remaining`` equal to the whole limit is not a harmless
        imprecision: it is the one value that guarantees a budget policy never
        fires.
        """
        from optio import semconv

        lane = CostLane(Config())

        signals = lane.on_run_end(_Run("unseen", BudgetPolicy(limit_usd=10.0, max_steps=100)))

        assert semconv.RUN_BUDGET_REMAINING not in {s.name for s in signals}

    def test_a_lane_that_did_meter_the_run_still_reports(self) -> None:
        """The other half. Silence for unseen runs must not silence real ones --
        suppressing both would trade a fabricated number for a missing one."""
        from optio import semconv

        lane = CostLane(Config())
        lane.ledger.reserve("seen", "step-1", 2.50)
        lane.ledger.reconcile("seen", "step-1", 2.50)

        signals = lane.on_run_end(_Run("seen", BudgetPolicy(limit_usd=10.0, max_steps=100)))

        names = {s.name for s in signals}
        assert semconv.RUN_ACTUAL_COST in names
        assert semconv.RUN_BUDGET_REMAINING in names

    def test_a_run_whose_every_step_was_unpriceable_still_reports_nothing(self) -> None:
        """The case the arithmetic guard already handled, kept under the new one.

        Three steps reserved at zero and never reconciled: the lane *did* see
        this run, so the silence has to come from ``_has_cost_evidence`` rather
        than from the new check -- both paths must stay closed.
        """
        from optio import semconv

        lane = CostLane(Config())
        for i in range(3):
            lane.ledger.reserve("unpriceable", f"step-{i}", 0.0)

        signals = lane.on_run_end(_Run("unpriceable", BudgetPolicy(limit_usd=10.0, max_steps=100)))

        names = {s.name for s in signals}
        assert semconv.RUN_ACTUAL_COST not in names
        assert semconv.RUN_BUDGET_REMAINING not in names


class TestRunEndFiringTwiceReportsOnce:
    """Run end can fire more than once (M1-2), and closing is final (ADR-010).

    The second firing must add nothing. Without the ``is_finalised`` guard the
    lane closes an already-closed run and emits the same signals again -- and
    since signals are written as span attributes, a repeat is not merely
    redundant: any consumer counting emissions sees the run's cost twice.
    """

    def test_a_closed_but_not_yet_evicted_run_is_not_reported_again(self) -> None:
        """The state ``is_finalised`` exists for, reached directly.

        ``on_run_end`` closes *and* evicts, so after one firing the run has left
        the ledger entirely and the ``knows`` check above already answers. The
        gap between the two -- closed, state still present -- is only reachable
        by closing without evicting, which is what this does. Otherwise
        ``is_finalised`` is a guard no test can distinguish from its absence,
        and a mutation removing it stays green.
        """
        lane = CostLane(Config())
        lane.ledger.reserve("seen", "step-1", 2.50)
        lane.ledger.reconcile("seen", "step-1", 2.50)
        lane.ledger.close_run("seen")

        signals = lane.on_run_end(_Run("seen", BudgetPolicy(limit_usd=10.0, max_steps=100)))

        assert signals == [], f"a finalised run was re-reported as {[s.name for s in signals]}"

    def test_the_second_firing_is_silent(self) -> None:
        lane = CostLane(Config())
        lane.ledger.reserve("seen", "step-1", 2.50)
        lane.ledger.reconcile("seen", "step-1", 2.50)
        run = _Run("seen", BudgetPolicy(limit_usd=10.0, max_steps=100))

        first = lane.on_run_end(run)
        second = lane.on_run_end(run)

        assert first, "the first firing should report the run"
        assert second == [], f"the second firing re-reported {[s.name for s in second]}"


class TestTwoProvidersInOneProcess:
    """The shape that produced the CI failure, end to end."""

    def test_a_second_providers_tap_does_not_answer_for_the_first(self) -> None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from optio import meter, semconv
        from optio.runtime.installer import install_tap

        config = Config()

        # A second provider, alive for the whole run, whose tap is registered as
        # a run-end observer and never sees a single span of it.
        bystander = TracerProvider()
        install_tap(config, bystander)

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("agent")

        @meter(budget=BudgetPolicy(limit_usd=10.0, max_steps=100), provider=provider)
        def run_agent() -> None:
            for _ in range(3):
                with tracer.start_as_current_span("llm") as span:
                    span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "a-model-from-the-future")
                    span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000_000)

        try:
            run_agent()

            run_span = next(
                s for s in exporter.get_finished_spans() if s.name.startswith("optio.run")
            )
            attributes = run_span.attributes or {}
            assert semconv.RUN_BUDGET_REMAINING not in attributes, (
                "a bystander provider's tap reported a full budget for a run it never metered"
            )
        finally:
            provider.shutdown()
            bystander.shutdown()
