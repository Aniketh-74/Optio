"""Adapter resolution and the four shipped adapters (M1-4, M4-1..3).

Adapters fail **loudly**, unlike everything in ``runtime/``. A framework
mismatch is a setup error, and instrumenting nothing silently would leave the
user believing they have coverage they do not have (Section 4.2).

The fakes below carry the real frameworks' module paths and attribute shapes
rather than importing them. Section 10 keeps framework deps behind extras and
out of core imports, and a test that skips when CrewAI is absent is a test that
never runs in CI. Recognition is a pure function of module path and attributes,
which is exactly what these exercise -- what they cannot catch is a framework
renaming its module, which is R-TECH-3's standing maintenance tax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from agentmeter.adapters.claude_agent import ClaudeAgentAdapter
from agentmeter.adapters.crewai import CrewAIAdapter
from agentmeter.adapters.langgraph import LangGraphAdapter
from agentmeter.adapters.openai_agents import OpenAIAgentsAdapter
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


class _FakeOpenAIAgent:
    """Stands in for an OpenAI Agents SDK ``Agent``."""

    name = "researcher"
    tools: ClassVar[list[object]] = []
    instructions = "do the thing"
    handoffs: ClassVar[list[object]] = []


_FakeOpenAIAgent.__module__ = "agents.agent"


class _FakeCrew:
    """Stands in for a CrewAI ``Crew``."""

    agents: ClassVar[list[object]] = []
    tasks: ClassVar[list[object]] = []


_FakeCrew.__module__ = "crewai.crew"


class _FakeCrewAgent:
    """Stands in for a single CrewAI ``Agent``."""

    role = "researcher"
    goal = "find the thing"


_FakeCrewAgent.__module__ = "crewai.agent"


class _FakeClaudeClient:
    """Stands in for ``ClaudeSDKClient``."""

    async def query(self, *args: object, **kwargs: object) -> None:
        return None

    async def receive_response(self, *args: object, **kwargs: object) -> None:
        return None


_FakeClaudeClient.__module__ = "claude_agent_sdk.client"


class _FakeClaudeOptions:
    """``ClaudeAgentOptions`` -- config, not a client."""

    model = "claude-opus-4"


_FakeClaudeOptions.__module__ = "claude_agent_sdk.types"


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

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("langgraph", LangGraphAdapter),
            ("openai_agents", OpenAIAgentsAdapter),
            ("crewai", CrewAIAdapter),
            ("claude_agent", ClaudeAgentAdapter),
        ],
    )
    def test_every_documented_adapter_loads(self, name: str, expected: type[Adapter]) -> None:
        # These four were listed as planned from M1. Loading each by the name
        # users actually type is the check that the registry entry, the module
        # path, and the class name all agree.
        assert isinstance(load_adapter(name), expected)

    def test_all_four_adapters_are_available(self) -> None:
        assert available_adapters() == ["claude_agent", "crewai", "langgraph", "openai_agents"]

    def test_loading_an_adapter_does_not_import_its_framework(self) -> None:
        # Section 10: framework deps live behind extras and must never be
        # imported at core import time. None of the four frameworks is installed
        # in the dev environment, so a real import here would raise.
        for name in available_adapters():
            load_adapter(name)

    def test_a_planned_but_unbuilt_adapter_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `_PLANNED_ADAPTERS` is empty as of M4, so this branch is unreachable
        # from real input -- but it is the mechanism that gives the *next*
        # planned adapter a useful message. "Planned, not built" and "no such
        # thing" are different problems with different fixes for the user, and
        # deleting the distinction because it is momentarily unused would mean
        # rediscovering the need for it in M5.
        from agentmeter.runtime import adapter_registry

        monkeypatch.setattr(adapter_registry, "_PLANNED_ADAPTERS", frozenset({"future_sdk"}))

        with pytest.raises(UnsupportedFrameworkError, match="not implemented yet"):
            load_adapter("future_sdk")

    def test_a_planned_adapter_is_listed_when_resolution_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentmeter.runtime import adapter_registry

        monkeypatch.setattr(adapter_registry, "_PLANNED_ADAPTERS", frozenset({"future_sdk"}))

        with pytest.raises(UnsupportedFrameworkError, match="planned"):
            resolve_adapter(object())

    def test_nothing_is_advertised_as_planned_today(self) -> None:
        # All four ship. If this fails, the message above needs re-checking.
        message = ""
        try:
            resolve_adapter(object())
        except UnsupportedFrameworkError as error:
            message = str(error)

        assert "planned" not in message, f"empty planned list leaked into: {message}"

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


class TestOpenAIAgentsAdapter:
    def test_matches_an_sdk_agent(self) -> None:
        assert OpenAIAgentsAdapter().matches(_FakeOpenAIAgent()) is True

    def test_rejects_a_plain_object(self) -> None:
        assert OpenAIAgentsAdapter().matches(object()) is False

    def test_rejects_a_foreign_object_named_agent(self) -> None:
        # `agents` is a short, generic top-level module name that a user package
        # could plausibly claim. Module path alone must not be enough.
        class Impostor:
            name = "mine"
            tools: ClassVar[list[object]] = []

        Impostor.__module__ = "myapp.agents"
        assert OpenAIAgentsAdapter().matches(Impostor()) is False

    def test_rejects_an_sdk_object_that_is_not_an_agent(self) -> None:
        class Runner:
            pass

        Runner.__module__ = "agents.run"
        assert OpenAIAgentsAdapter().matches(Runner()) is False

    def test_instrument_returns_the_same_object(self) -> None:
        agent = _FakeOpenAIAgent()
        assert OpenAIAgentsAdapter().instrument(agent, default_config()) is agent

    def test_instrument_rejects_a_mismatched_target(self) -> None:
        with pytest.raises(UnsupportedFrameworkError, match="cannot instrument"):
            OpenAIAgentsAdapter().instrument(object(), default_config())


class TestCrewAIAdapter:
    def test_matches_a_crew(self) -> None:
        assert CrewAIAdapter().matches(_FakeCrew()) is True

    def test_matches_a_single_agent(self) -> None:
        # A one-agent flow is a reasonable thing to pass, and the tap is
        # installed process-wide either way.
        assert CrewAIAdapter().matches(_FakeCrewAgent()) is True

    def test_rejects_a_plain_object(self) -> None:
        assert CrewAIAdapter().matches(object()) is False

    def test_rejects_a_crewai_object_that_is_neither(self) -> None:
        class Task:
            description = "x"

        Task.__module__ = "crewai.task"
        assert CrewAIAdapter().matches(Task()) is False

    def test_instrument_returns_the_same_object(self) -> None:
        crew = _FakeCrew()
        assert CrewAIAdapter().instrument(crew, default_config()) is crew

    def test_instrument_rejects_a_mismatched_target(self) -> None:
        with pytest.raises(UnsupportedFrameworkError, match="cannot instrument"):
            CrewAIAdapter().instrument(object(), default_config())


class TestClaudeAgentAdapter:
    def test_matches_a_client(self) -> None:
        assert ClaudeAgentAdapter().matches(_FakeClaudeClient()) is True

    def test_matches_the_pre_rename_module(self) -> None:
        # `claude_code_sdk` is the old distribution name and is still installed
        # in the wild; users on it are not helped by "no adapter matches".
        client = _FakeClaudeClient()
        type(client).__module__ = "claude_code_sdk.client"
        try:
            assert ClaudeAgentAdapter().matches(client) is True
        finally:
            type(client).__module__ = "claude_agent_sdk.client"

    def test_rejects_the_options_dataclass(self) -> None:
        # Config, not a client. Accepting it would let `instrument(options)`
        # look like it worked while instrumenting nothing.
        assert ClaudeAgentAdapter().matches(_FakeClaudeOptions()) is False

    def test_rejects_a_plain_object(self) -> None:
        assert ClaudeAgentAdapter().matches(object()) is False

    def test_instrument_returns_the_same_object(self) -> None:
        client = _FakeClaudeClient()
        assert ClaudeAgentAdapter().instrument(client, default_config()) is client

    def test_instrument_rejects_a_mismatched_target(self) -> None:
        with pytest.raises(UnsupportedFrameworkError, match="cannot instrument"):
            ClaudeAgentAdapter().instrument(object(), default_config())


class TestAdaptersDoNotOverlap:
    """Duck-typed resolution returns the first match, so matches must be disjoint."""

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (_FakeGraph(), LangGraphAdapter),
            (_FakeOpenAIAgent(), OpenAIAgentsAdapter),
            (_FakeCrew(), CrewAIAdapter),
            (_FakeCrewAgent(), CrewAIAdapter),
            (_FakeClaudeClient(), ClaudeAgentAdapter),
        ],
    )
    def test_each_target_resolves_to_its_own_adapter(
        self, target: object, expected: type[Adapter]
    ) -> None:
        assert isinstance(resolve_adapter(target), expected)

    @pytest.mark.parametrize(
        "target",
        [_FakeGraph(), _FakeOpenAIAgent(), _FakeCrew(), _FakeClaudeClient()],
    )
    def test_exactly_one_adapter_claims_each_target(self, target: object) -> None:
        # Order-independent version of the above: if two adapters ever both
        # matched, resolution would silently depend on registry ordering.
        claimers = [name for name in available_adapters() if load_adapter(name).matches(target)]
        assert len(claimers) == 1, f"{type(target).__name__} claimed by {claimers}"


class TestSetupWarnsWhenNoProviderIsConfigured:
    """A user with no OTel SDK gets no signals; that must be said at setup."""

    @pytest.mark.parametrize(
        ("adapter", "target"),
        [
            (LangGraphAdapter(), _FakeGraph()),
            (OpenAIAgentsAdapter(), _FakeOpenAIAgent()),
            (CrewAIAdapter(), _FakeCrew()),
            (ClaudeAgentAdapter(), _FakeClaudeClient()),
        ],
    )
    def test_a_missing_provider_warns_but_does_not_raise(
        self,
        adapter: Adapter,
        target: object,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Not an error: the user may wire instrumentation after instrument(),
        # and refusing would break a legitimate ordering (Section 4.2).
        class NoOpProvider:
            def get_tracer(self, *args: object, **kwargs: object) -> object:
                return trace.NoOpTracer()

        with caplog.at_level("WARNING", logger="agentmeter"):
            result = adapter.instrument(target, default_config(), provider=NoOpProvider())  # type: ignore[arg-type]

        assert result is target
        assert "no signals will be emitted" in caplog.text


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
