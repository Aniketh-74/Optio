"""Adapters against the frameworks they actually claim to support (R-TECH-3).

Every other adapter test builds a stand-in: a ``Mock`` with ``__module__``
rewritten to look like the framework. That proves the *matching logic* is right,
and proves nothing about whether the logic matches anything the framework really
produces. The two are different claims, and only the second is what a user
cares about -- "supports LangGraph" is a promise about LangGraph, not about our
opinion of it.

These tests install the real packages and build real objects. They are skipped
when a framework is absent, so a contributor without a 2 GB dependency tree can
still run the suite; CI installs each one and sets
``OPTIO_REQUIRE_FRAMEWORKS=1`` to turn a skip into a failure, because a matrix
that passes by skipping is not a matrix.

**Nothing here calls a model.** The adapters attach to an object; the signals
come from OTel spans the framework emits during a real run. Recognition and
attachment are what these tests cover, and they need no API key, no network,
and no spend.

**Running the full matrix locally.** ``crewai`` declares
``requires_python = <3.14,>=3.10``, so it cannot be installed into the 3.14 dev
environment at all -- pip resolves back to 0.11.2, a 2024 release, and then
fails to build an old ``numpy``. The compiler error that produces is a red
herring: the binding constraint is the interpreter version, and no toolchain
fixes it.

The matrix therefore needs an interpreter of its own::

    py -3.13 -m venv .venv-frameworks
    .venv-frameworks\\Scripts\\python -m pip install -e ".[optimize]" \\
        crewai langgraph claude-agent-sdk openai-agents pytest pytest-timeout
    $env:OPTIO_REQUIRE_FRAMEWORKS = "1"
    .venv-frameworks\\Scripts\\python -m pytest tests/frameworks

All four adapters pass there with the gate on, which is the first time
``crewai`` has been verified anywhere but CI. The same venv is also the only
check this repo has that ``requires-python = ">=3.10"`` is true below 3.14: the
full suite passes on 3.13.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from optio.errors import UnsupportedFrameworkError
from optio.runtime.adapter_registry import resolve_adapter

pytestmark = pytest.mark.frameworks

REQUIRE = os.environ.get("OPTIO_REQUIRE_FRAMEWORKS") == "1"


def _need(module: str) -> None:
    """Skip unless ``module`` is importable -- unless CI says it must be."""
    if importlib.util.find_spec(module) is not None:
        return
    message = f"{module} is not installed"
    if REQUIRE:
        pytest.fail(
            f"{message}, but OPTIO_REQUIRE_FRAMEWORKS=1. This gate exists to "
            f"prove the adapter works against the real package; passing by "
            f"skipping would defeat it."
        )
    pytest.skip(f"{message}; the real-framework matrix runs in CI")


class TestLangGraph:
    def test_a_compiled_graph_is_recognised(self) -> None:
        _need("langgraph")
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class State(TypedDict):
            count: int

        builder = StateGraph(State)
        builder.add_node("bump", lambda s: {"count": s["count"] + 1})
        builder.add_edge(START, "bump")
        builder.add_edge("bump", END)
        graph = builder.compile()

        adapter = resolve_adapter(graph)
        assert type(adapter).__name__ == "LangGraphAdapter"

    def test_an_uncompiled_builder_is_not_recognised(self) -> None:
        # A StateGraph has no invoke(); instrumenting it would attach to
        # something that never runs. Better to fail loudly at setup (Section
        # 4.2) than to silently meter nothing.
        _need("langgraph")
        from typing import TypedDict

        from langgraph.graph import StateGraph

        class State(TypedDict):
            count: int

        with pytest.raises(UnsupportedFrameworkError):
            resolve_adapter(StateGraph(State))

    def test_instrumenting_returns_the_same_graph(self) -> None:
        # The object must come back usable and unwrapped: optio observes via
        # the OTel span processor, it does not proxy the framework (ADR-001).
        _need("langgraph")
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph
        from opentelemetry.sdk.trace import TracerProvider

        from optio import instrument

        class State(TypedDict):
            count: int

        builder = StateGraph(State)
        builder.add_node("bump", lambda s: {"count": s["count"] + 1})
        builder.add_edge(START, "bump")
        builder.add_edge("bump", END)
        graph = builder.compile()

        returned = instrument(graph, provider=TracerProvider())
        assert returned is graph
        assert graph.invoke({"count": 0}) == {"count": 1}


class TestOpenAIAgents:
    def test_an_agent_is_recognised(self) -> None:
        _need("agents")
        from agents import Agent

        agent = Agent(name="assistant", instructions="be helpful")
        adapter = resolve_adapter(agent)
        assert type(adapter).__name__ == "OpenAIAgentsAdapter"

    def test_instrumenting_returns_the_same_agent(self) -> None:
        _need("agents")
        from agents import Agent
        from opentelemetry.sdk.trace import TracerProvider

        from optio import instrument

        agent = Agent(name="assistant", instructions="be helpful")
        assert instrument(agent, provider=TracerProvider()) is agent


class TestClaudeAgentSDK:
    def test_a_client_is_recognised(self) -> None:
        _need("claude_agent_sdk")
        from claude_agent_sdk import ClaudeSDKClient

        adapter = resolve_adapter(ClaudeSDKClient())
        assert type(adapter).__name__ == "ClaudeAgentAdapter"

    def test_the_options_object_is_not_mistaken_for_a_client(self) -> None:
        # Same module, no query()/receive_response(). Claiming it would attach
        # to a config object and meter nothing.
        _need("claude_agent_sdk")
        from claude_agent_sdk import ClaudeAgentOptions

        with pytest.raises(UnsupportedFrameworkError):
            resolve_adapter(ClaudeAgentOptions())


class TestCrewAI:
    def test_a_crew_is_recognised(self) -> None:
        _need("crewai")
        from crewai import Agent, Crew, Task

        researcher = Agent(role="researcher", goal="find things", backstory="curious")
        task = Task(description="look it up", expected_output="an answer", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[task])

        adapter = resolve_adapter(crew)
        assert type(adapter).__name__ == "CrewAIAdapter"

    def test_a_bare_agent_is_recognised(self) -> None:
        _need("crewai")
        from crewai import Agent

        agent = Agent(role="researcher", goal="find things", backstory="curious")
        adapter = resolve_adapter(agent)
        assert type(adapter).__name__ == "CrewAIAdapter"

    def test_a_task_is_not_recognised(self) -> None:
        # A Task is neither a crew nor an agent; it has no run to meter.
        _need("crewai")
        from crewai import Agent, Task

        researcher = Agent(role="researcher", goal="find things", backstory="curious")
        task = Task(description="look it up", expected_output="an answer", agent=researcher)

        with pytest.raises(UnsupportedFrameworkError):
            resolve_adapter(task)


class TestNoFrameworkClaimsAnother:
    """Adapters must not fight over an object -- exactly one may claim it."""

    def test_each_real_object_is_claimed_by_exactly_one_adapter(self) -> None:
        from optio.runtime.adapter_registry import available_adapters, load_adapter

        objects: list[object] = []

        if importlib.util.find_spec("agents") is not None:
            from agents import Agent as OpenAIAgent

            objects.append(OpenAIAgent(name="a", instructions="x"))
        if importlib.util.find_spec("claude_agent_sdk") is not None:
            from claude_agent_sdk import ClaudeSDKClient

            objects.append(ClaudeSDKClient())
        if importlib.util.find_spec("crewai") is not None:
            from crewai import Agent as CrewAgent

            objects.append(CrewAgent(role="r", goal="g", backstory="b"))

        if not objects:
            pytest.skip("no frameworks installed")

        adapters = [load_adapter(name) for name in available_adapters()]
        for target in objects:
            claimers = [type(a).__name__ for a in adapters if a.matches(target)]
            assert len(claimers) == 1, f"{type(target).__name__} claimed by {claimers}"
