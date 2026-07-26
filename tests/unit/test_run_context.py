"""Run identity and lifecycle.

The idempotent-end tests matter more than they look: run-end is where the ledger
reconcile sweep and quality scoring will attach (M2, M5). If ``end()`` can fire
twice, cost is double-counted -- silently, in production, in the direction that
makes the product's core number wrong (R-TECH-1).
"""

from __future__ import annotations

import contextvars
import threading

import pytest

from agentmeter import Config, RunContext, current_run
from agentmeter.config import BudgetPolicy


class TestIdentity:
    def test_ids_are_unique(self):
        ids = {RunContext().run_id for _ in range(1000)}
        assert len(ids) == 1000

    def test_explicit_id_is_honored(self):
        assert RunContext(run_id="external-123").run_id == "external-123"

    def test_ids_are_unique_across_threads(self):
        ids: list[str] = []
        lock = threading.Lock()

        def make() -> None:
            run_id = RunContext().run_id
            with lock:
                ids.append(run_id)

        threads = [threading.Thread(target=make) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(ids)) == 50


class TestLifecycle:
    def test_new_run_is_not_active(self):
        assert RunContext().is_active is False

    def test_start_activates(self):
        assert RunContext().start().is_active is True

    def test_start_is_idempotent(self):
        run = RunContext().start()
        first_start = run.started_at
        run.start()
        assert run.started_at == first_start

    def test_end_deactivates(self):
        run = RunContext().start()
        run.end()
        assert run.is_active is False
        assert run.ended_at is not None

    def test_end_is_idempotent(self):
        run = RunContext().start()
        run.end()
        first_end = run.ended_at
        run.end()
        run.end()
        assert run.ended_at == first_end

    def test_end_without_start_does_not_raise(self):
        RunContext().end()

    def test_duration_is_none_until_ended(self):
        run = RunContext().start()
        assert run.duration_seconds is None

    def test_duration_is_measured_once_ended(self):
        run = RunContext().start()
        run.end()
        duration: float | None = run.duration_seconds
        assert duration is not None and duration >= 0


class TestContextTracking:
    def test_no_current_run_outside_scope(self):
        assert current_run() is None

    def test_current_run_inside_scope(self):
        with RunContext() as run:
            assert current_run() is run
        assert current_run() is None

    def test_nesting_restores_the_outer_run(self):
        with RunContext() as outer:
            with RunContext() as inner:
                assert current_run() is inner
            assert current_run() is outer
        assert current_run() is None

    def test_context_is_cleared_when_the_agent_raises(self):
        with pytest.raises(ValueError), RunContext():
            raise ValueError("agent error")
        assert current_run() is None

    def test_exit_does_not_suppress_exceptions(self):
        # A truthy __exit__ would silently swallow the agent's exception. The
        # annotation says None, but the annotation is not what runs -- assert on
        # the actual returned value.
        run = RunContext().start()
        returned = RunContext.__exit__(run, ValueError, ValueError("x"), None)
        assert not returned

    def test_end_in_a_different_context_does_not_raise(self):
        # ContextVar.reset() rejects a token from another context. A run started
        # in one context and ended in another (async task boundary, framework
        # run-end callback) must still end cleanly -- run-end is on the agent's
        # path, so raising here would violate ADR-004.
        run = contextvars.copy_context().run(lambda: RunContext().start())
        run.end()
        assert run.is_active is False

    def test_end_in_a_different_context_does_not_leak_the_run(self):
        run = contextvars.copy_context().run(lambda: RunContext().start())
        run.end()
        assert current_run() is None

    def test_run_ended_on_another_thread_completes_cleanly(self):
        # A ContextVar entry can only be cleared from the context that set it, so
        # ending on another thread cannot unset this thread's current run. What
        # must hold is that it does not raise and the run is marked ended --
        # anything stricter is not achievable and would be a false promise.
        run = RunContext().start()
        error: list[BaseException] = []

        def end_it() -> None:
            try:
                run.end()
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=end_it)
        thread.start()
        thread.join()

        assert not error, f"end() raised across threads: {error}"
        assert run.is_active is False


class TestConfiguration:
    def test_budget_string_is_parsed(self):
        run = RunContext(budget="$0.50")
        assert run.budget is not None
        assert run.budget.limit_usd == pytest.approx(0.50)

    def test_budget_is_optional(self):
        assert RunContext().budget is None

    def test_budget_policy_passes_through(self):
        policy = BudgetPolicy(limit_usd=1.0, max_steps=10)
        assert RunContext(budget=policy).budget is policy

    def test_explicit_config_is_used(self):
        config = Config(quality_lane=True)
        assert RunContext(config=config).config is config

    def test_repr_shows_state(self):
        run = RunContext()
        assert "new" in repr(run)
        run.start()
        assert "active" in repr(run)
        run.end()
        assert "ended" in repr(run)
