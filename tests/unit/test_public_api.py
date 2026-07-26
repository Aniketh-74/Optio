"""The frozen public surface (Section 8.1, task M0-4).

These assert the *contract*, not the implementation: signatures, the identity
guarantee, and the setup-time-failure rule. They are what makes breaking the
surface after M1 a visible, ADR-requiring event rather than an accident.
"""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest

import agentmeter
from agentmeter import BudgetPolicy, Config, RunContext, current_run, instrument, meter
from agentmeter.errors import AgentMeterConfigError


class TestSurface:
    def test_documented_names_are_importable(self):
        for name in ("instrument", "meter", "RunContext", "Config", "BudgetPolicy"):
            assert hasattr(agentmeter, name), name

    def test_all_entries_exist(self):
        for name in agentmeter.__all__:
            assert hasattr(agentmeter, name), name

    def test_all_has_no_duplicates(self):
        assert len(agentmeter.__all__) == len(set(agentmeter.__all__))

    def test_all_covers_every_public_name(self):
        # A public name missing from __all__ is an undocumented surface that
        # users will import anyway and we will then be unable to change.
        ignored = {"annotations"}  # __future__ import, not a public export
        public = {
            name
            for name, value in vars(agentmeter).items()
            if not name.startswith("_")
            and name not in ignored
            and not isinstance(value, ModuleType)
        }
        assert public - set(agentmeter.__all__) == set()

    def test_version_is_exposed(self):
        assert isinstance(agentmeter.__version__, str)

    def test_instrument_signature_is_locked(self):
        # Additive keyword-only parameters with defaults are compatible; a
        # *breaking* change to this surface needs an ADR (§16 rule 12). What
        # must not drift is the positional target and the keyword-only-ness of
        # everything else, since that is what callers actually depend on.
        params = inspect.signature(instrument).parameters
        assert list(params) == [
            "target",
            "adapter",
            "config",
            "provider",
            "overrides",
        ]
        assert params["target"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for name in ("adapter", "config", "provider"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[name].default is None

    def test_public_callables_are_annotated(self):
        for fn in (instrument, meter):
            hints = inspect.get_annotations(fn)
            assert "return" in hints, fn.__name__


class _FakeGraph:
    """Stands in for a compiled LangGraph graph.

    ``__module__`` is rewritten below so adapter resolution sees a LangGraph
    object without this test suite depending on LangGraph being installed.
    """

    def invoke(self, *args, **kwargs):
        return None

    def stream(self, *args, **kwargs):
        return iter(())


_FakeGraph.__module__ = "langgraph.graph.state"


class TestInstrument:
    def test_returns_the_same_object(self):
        agent = _FakeGraph()
        assert instrument(agent) is agent

    def test_accepts_an_explicit_adapter(self):
        agent = _FakeGraph()
        assert instrument(agent, adapter="langgraph") is agent

    def test_unknown_adapter_raises_at_setup(self):
        with pytest.raises(AgentMeterConfigError, match="unknown adapter"):
            instrument(_FakeGraph(), adapter="not_a_framework")

    def test_naming_the_wrong_adapter_raises_at_setup(self):
        # All four adapters ship as of M4, so there is no longer a "planned"
        # case to test. The mismatch this replaces it with is the one that
        # actually costs a user: an explicit adapter= that does not fit the
        # object, which must fail rather than instrument nothing.
        with pytest.raises(AgentMeterConfigError, match="cannot instrument"):
            instrument(_FakeGraph(), adapter="crewai")

    def test_unrecognised_target_raises_at_setup(self):
        # Section 6.1: an unknown framework fails loudly at setup. Silently
        # instrumenting nothing would leave the user believing they have
        # coverage they do not have.
        with pytest.raises(AgentMeterConfigError, match="no adapter matches"):
            instrument(object())

    def test_unknown_config_option_raises_at_setup(self):
        with pytest.raises(AgentMeterConfigError, match="unknown config option"):
            instrument(_FakeGraph(), definitely_not_an_option=True)

    def test_overrides_apply(self):
        # Should not raise; the override is a real field.
        assert instrument(_FakeGraph(), quality_lane=True) is not None


class TestMeter:
    def test_bare_decorator_preserves_behavior(self):
        @meter
        def double(x: int) -> int:
            return x * 2

        assert double(21) == 42

    def test_called_decorator_preserves_behavior(self):
        @meter(budget="$0.50")
        def double(x: int) -> int:
            return x * 2

        assert double(21) == 42

    def test_preserves_metadata(self):
        @meter
        def documented(x: int) -> int:
            """Docstring survives."""
            return x

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Docstring survives."

    def test_exceptions_from_the_agent_propagate(self):
        @meter
        def explode() -> None:
            raise ValueError("agent error")

        # agentmeter observes runs; it must never swallow the user's exception.
        with pytest.raises(ValueError, match="agent error"):
            explode()

    def test_run_is_active_inside_and_cleared_after(self):
        seen: list[str | None] = []

        @meter
        def inner() -> None:
            run = current_run()
            seen.append(run.run_id if run else None)

        inner()
        assert seen[0] is not None
        assert current_run() is None

    def test_bad_budget_raises_at_decoration_time(self):
        with pytest.raises(AgentMeterConfigError):
            meter(budget="not a number")


class TestBudgetPolicy:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [("$0.50", 0.50), ("0.50", 0.50), (0.5, 0.5), (2, 2.0), ("$1,000.00", 1000.0)],
    )
    def test_parse_accepts_documented_forms(self, spec, expected):
        assert BudgetPolicy.parse(spec).limit_usd == pytest.approx(expected)

    def test_parse_is_idempotent(self):
        policy = BudgetPolicy(limit_usd=1.0)
        assert BudgetPolicy.parse(policy) is policy

    def test_rejects_unparseable(self):
        with pytest.raises(AgentMeterConfigError):
            BudgetPolicy.parse("free")

    def test_rejects_non_positive(self):
        with pytest.raises(AgentMeterConfigError):
            BudgetPolicy(limit_usd=0)

    def test_is_immutable(self):
        with pytest.raises(FrozenInstanceError):
            BudgetPolicy(limit_usd=1.0).limit_usd = 2.0  # type: ignore[misc]


class TestRunContextSurface:
    def test_usable_as_a_context_manager(self):
        with RunContext(budget="$0.50") as run:
            assert isinstance(run, RunContext)
            assert run.is_active

    def test_config_defaults_are_the_documented_flags(self):
        config = Config()
        assert config.cost_lane is True
        assert config.behavior_lane is True
        # Off by default is an architectural decision, not a preference (ADR-003).
        assert config.quality_lane is False
        assert config.store_backend == "memory"
