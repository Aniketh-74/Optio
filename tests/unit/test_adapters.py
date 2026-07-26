"""Adapter resolution and the LangGraph adapter (M1-4).

Adapters fail **loudly**, unlike everything in ``runtime/``. A framework
mismatch is a setup error, and instrumenting nothing silently would leave the
user believing they have coverage they do not have (Section 4.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from agentmeter.adapters.langgraph import LangGraphAdapter
from agentmeter.config import default_config
from agentmeter.errors import AgentMeterConfigError, UnsupportedFrameworkError
from agentmeter.runtime import installer
from agentmeter.runtime.adapter_base import Adapter
from agentmeter.runtime.adapter_registry import (
    available_adapters,
    load_adapter,
    resolve_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clean_installs() -> Iterator[None]:
    """Forget tracked tap installs between tests."""
    installer.reset_installations()
    yield
    installer.reset_installations()


class _FakeGraph:
    """Stands in for a compiled LangGraph graph."""

    def invoke(self, *args: object, **kwargs: object) -> None:
        return None

    def stream(self, *args: object, **kwargs: object) -> Iterator[None]:
        return iter(())


_FakeGraph.__module__ = "langgraph.graph.state"


class _FakeChain:
    """A LangChain object that is not a graph -- no ``invoke``/``stream``."""

    def batch(self, *args: object, **kwargs: object) -> None:
        return None


_FakeChain.__module__ = "langchain_core.runnables"


class TestRegistry:
    def test_langgraph_is_available(self) -> None:
        assert "langgraph" in available_adapters()

    def test_loading_returns_the_named_adapter(self) -> None:
        # Guards a real bug: a "find the class ending in Adapter" scan matched
        # the imported abstract base before the concrete class.
        adapter = load_adapter("langgraph")
        assert isinstance(adapter, LangGraphAdapter)
        assert type(adapter) is not Adapter

    def test_unknown_adapter_is_rejected(self) -> None:
        with pytest.raises(UnsupportedFrameworkError, match="unknown adapter"):
            load_adapter("not_a_framework")

    @pytest.mark.parametrize("name", ["openai_agents", "crewai", "claude_agent"])
    def test_planned_adapters_say_so(self, name: str) -> None:
        # Distinct from "unknown": these are designed (M4), just not built, and
        # the user's fix is to wait rather than to check their spelling.
        with pytest.raises(UnsupportedFrameworkError, match="not implemented yet"):
            load_adapter(name)

    def test_resolution_finds_the_matching_adapter(self) -> None:
        assert isinstance(resolve_adapter(_FakeGraph()), LangGraphAdapter)

    def test_resolution_failure_names_the_target_and_the_options(self) -> None:
        # "Unsupported framework" without either is a dead end for the user.
        with pytest.raises(UnsupportedFrameworkError) as excinfo:
            resolve_adapter(object())

        message = str(excinfo.value)
        assert "object" in message
        assert "langgraph" in message


class TestLangGraphAdapter:
    def test_matches_a_compiled_graph(self) -> None:
        assert LangGraphAdapter().matches(_FakeGraph()) is True

    def test_rejects_a_plain_object(self) -> None:
        assert LangGraphAdapter().matches(object()) is False

    def test_rejects_a_langchain_object_without_the_graph_interface(self) -> None:
        assert LangGraphAdapter().matches(_FakeChain()) is False

    def test_instrument_returns_the_same_object(self) -> None:
        graph = _FakeGraph()
        assert LangGraphAdapter().instrument(graph, default_config()) is graph

    def test_instrument_rejects_a_mismatched_target(self) -> None:
        with pytest.raises(UnsupportedFrameworkError, match="cannot instrument"):
            LangGraphAdapter().instrument(object(), default_config())

    def test_instrument_installs_the_tap(self) -> None:
        provider = TracerProvider()
        LangGraphAdapter().instrument(_FakeGraph(), default_config(), provider=provider)

        assert installer.installed_tap(provider) is not None


class TestIdentityContract:
    """``instrument()`` must hand back the object it was given (Section 6.7)."""

    def test_an_adapter_that_swaps_the_object_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `agent = instrument(agent)` would otherwise quietly replace the user's
        # agent with something else.
        from agentmeter import api

        class _SwappingAdapter(Adapter):
            name = "swapper"

            def matches(self, target: object) -> bool:
                return True

            def instrument(
                self,
                target: object,
                config: object,
                provider: object = None,
            ) -> object:
                return object()

        monkeypatch.setattr(api, "resolve_adapter", lambda _target: _SwappingAdapter())

        with pytest.raises(AgentMeterConfigError, match="different object"):
            api.instrument(_FakeGraph())


class TestInstaller:
    def test_installing_twice_reuses_one_tap(self) -> None:
        # Two taps on one provider means every span dispatched twice, which in
        # M2 is a doubled cost signal (R-TECH-1). instrument() is easy to call
        # more than once, so this has to hold.
        provider = TracerProvider()
        first = installer.install_tap(default_config(), provider)
        second = installer.install_tap(default_config(), provider)

        assert first is second

    def test_separate_providers_get_separate_taps(self) -> None:
        first = installer.install_tap(default_config(), TracerProvider())
        second = installer.install_tap(default_config(), TracerProvider())

        assert first is not second

    def test_no_sdk_provider_is_not_an_error(self) -> None:
        # A user with no OTel SDK configured records no spans, so there is
        # nothing to tap. A fact worth reporting, not a broken setup.
        class NoOpProvider:
            def get_tracer(self, *args: object, **kwargs: object) -> object:
                return trace.NoOpTracer()

        assert installer.install_tap(default_config(), NoOpProvider()) is None  # type: ignore[arg-type]

    def test_installed_tap_is_none_before_install(self) -> None:
        assert installer.installed_tap(TracerProvider()) is None
