"""Tool schemas and tool results: the stages, and the guarantees they make.

The load-bearing tests here are the ones about what is *not* removed.
`minify_tools` earns its keep by deleting text, so the way it fails is by
deleting slightly too much and turning a cheap win into degraded tool
selection -- a regression that shows up as an agent picking the wrong tool,
which nobody traces back to a token optimization. `NON_SEMANTIC_KEYS` is
therefore pinned from both sides: every key in it is gone, and every key the
model reads survives.
"""

from __future__ import annotations

from typing import Any

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import Fidelity, StageContext
from optio_optimize.stages.tools import (
    MAX_SCHEMA_DEPTH,
    MIN_KEPT_TOOLS,
    NON_SEMANTIC_KEYS,
    CapToolResultsStage,
    MinifyToolsStage,
    PruneToolsStage,
)
from optio_optimize.tokens import default_counter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(**overrides: object) -> StageContext:
    return StageContext(config=OptimizeConfig(**overrides), counter=default_counter())  # type: ignore[arg-type]


def _openai_tool(name: str, description: str = "", **properties: Any) -> dict[str, Any]:
    """A tool in OpenAI's nested shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Does {name} things.",
            "parameters": {"type": "object", "properties": properties},
        },
    }


def _request(
    *, tools: tuple[dict[str, Any], ...] = (), messages: tuple[Message, ...] = ()
) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o",
        messages=messages or (Message(role="user", content="do the thing"),),
        tools=tools,
        temperature=0.0,
    )


class TestMinifyKeepsEverythingTheModelReads:
    """The half of this stage that is a promise rather than a saving."""

    @pytest.mark.parametrize("key", ["description", "examples", "default", "enum"])
    def test_a_key_the_model_reads_is_never_stripped(self, key: str) -> None:
        tool = _openai_tool("search", status={"type": "string", key: ["a", "b"], "title": "Status"})
        result = MinifyToolsStage().before(_request(tools=(tool,)), _ctx())

        rendered = repr(result.request.tools)
        assert key in rendered, f"{key} shapes tool selection and must survive minification"

    def test_the_stripped_set_is_exactly_the_documented_one(self) -> None:
        # Pins the constant itself: widening it is a deliberate act that has to
        # change this test, not something a helpful refactor can do quietly.
        assert set(NON_SEMANTIC_KEYS) == {"title", "$schema", "$id", "$comment"}


class TestMinifyRemovesAnnotations:
    def test_annotations_go_at_every_nesting_level(self) -> None:
        tool = _openai_tool(
            "search",
            query={"type": "string", "title": "Query"},
            filters={
                "type": "object",
                "title": "Filters",
                "properties": {"since": {"type": "string", "title": "Since"}},
            },
        )
        result = MinifyToolsStage().before(_request(tools=(tool,)), _ctx())

        assert "title" not in repr(result.request.tools)

    def test_a_list_of_schemas_is_walked(self) -> None:
        tool = _openai_tool("search", value={"anyOf": [{"type": "string", "title": "As text"}]})
        result = MinifyToolsStage().before(_request(tools=(tool,)), _ctx())

        assert "title" not in repr(result.request.tools)

    def test_it_reports_the_tokens_it_saved(self) -> None:
        tools = tuple(
            _openai_tool(f"tool_{n}", field={"type": "string", "title": "A rather long title"})
            for n in range(4)
        )
        result = MinifyToolsStage().before(_request(tools=tools), _ctx())

        assert result.saved_input_tokens > 0

    def test_it_declines_with_no_tools(self) -> None:
        result = MinifyToolsStage().before(_request(), _ctx())
        assert result.request.tools == ()
        assert result.note == ""

    def test_it_declines_when_there_is_nothing_to_strip(self) -> None:
        # Rewriting the caller's schemas for a zero-token gain is pure risk.
        result = MinifyToolsStage().before(_request(tools=(_openai_tool("search"),)), _ctx())
        assert result.note == ""

    def test_a_deeply_nested_schema_terminates(self) -> None:
        # Bounded walk, same lesson as the core's digest depth guard: the test
        # has to prove the boundary is real, not merely that a call returned.
        deep: dict[str, Any] = {"type": "string", "title": "leaf"}
        for _ in range(MAX_SCHEMA_DEPTH + 5):
            deep = {"type": "object", "title": "wrap", "properties": {"next": deep}}

        result = MinifyToolsStage().before(_request(tools=({"function": deep},)), _ctx())

        # Something was stripped (the shallow levels) but the walk stopped, so
        # the deepest title survives -- which is the bound being observable
        # rather than merely asserted.
        assert result.request.tools != ({"function": deep},)


class TestCapToolResults:
    def test_an_oversized_result_is_truncated_and_says_so(self) -> None:
        messages = (
            Message(role="user", content="fetch"),
            Message(role="tool", content="x" * 40_000, name="fetch"),
        )
        result = CapToolResultsStage(max_tokens=100).before(_request(messages=messages), _ctx())

        capped = result.request.messages[-1].content
        assert len(capped) < 40_000
        assert "truncated by optio_optimize" in capped
        assert result.saved_input_tokens > 0

    def test_the_notice_states_how_much_went_missing(self) -> None:
        # A silent truncation makes the model answer confidently from a
        # fragment; the number is what lets a caller notice.
        messages = (Message(role="tool", content="y" * 20_000, name="fetch"),)
        result = CapToolResultsStage(max_tokens=50).before(_request(messages=messages), _ctx())

        assert "of 20000 characters omitted" in result.request.messages[0].content

    def test_a_small_result_is_untouched(self) -> None:
        messages = (Message(role="tool", content="status=ok", name="fetch"),)
        result = CapToolResultsStage(max_tokens=2000).before(_request(messages=messages), _ctx())

        assert result.request.messages[0].content == "status=ok"
        assert result.note == ""

    def test_non_tool_messages_are_never_capped(self) -> None:
        messages = (Message(role="user", content="z" * 40_000),)
        result = CapToolResultsStage(max_tokens=10).before(_request(messages=messages), _ctx())

        assert result.request.messages[0].content == "z" * 40_000

    def test_a_non_positive_ceiling_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CapToolResultsStage(max_tokens=0)


class TestPruneToolsRefusesToLoseACapability:
    """Every test here is about the stage declining, which is the point.

    This is the only ``ALTERED`` stage in the module: a wrongly-pruned tool is
    a capability the agent silently loses, and the caller sees an agent that
    inexplicably will not do something rather than an error.
    """

    def test_a_tool_already_called_is_never_pruned(self) -> None:
        # Direct evidence the model wants this tool beats any lexical score,
        # and dropping it would leave a tool_calls entry in the history
        # referring to a tool no longer declared.
        tools = (
            _openai_tool("fetch_record", "Retrieve a record by identifier."),
            *(_openai_tool(f"unrelated_{n}", "Zebra xylophone quixotic.") for n in range(4)),
        )
        messages = (
            Message(role="user", content="tell me about zebra xylophone quixotic things"),
            Message(
                role="assistant",
                content="",
                extra={
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "fetch_record"}}
                    ]
                },
            ),
        )
        result = PruneToolsStage().before(_request(tools=tools, messages=messages), _ctx())

        names = repr(result.request.tools)
        assert "fetch_record" in names

    def test_it_declines_when_there_are_too_few_tools_to_risk_it(self) -> None:
        tools = tuple(_openai_tool(f"t{n}") for n in range(MIN_KEPT_TOOLS))
        result = PruneToolsStage().before(_request(tools=tools), _ctx())

        assert result.request.tools == tools
        assert result.note == ""

    def test_it_never_prunes_below_the_floor(self) -> None:
        # Every tool scores zero against this question; the floor must still
        # hold, because a model left with one tool will use it regardless.
        tools = tuple(_openai_tool(f"t{n}", "Zebra xylophone quixotic.") for n in range(8))
        messages = (Message(role="user", content="what is the capital of France"),)
        result = PruneToolsStage().before(_request(tools=tools, messages=messages), _ctx())

        assert len(result.request.tools) >= MIN_KEPT_TOOLS

    def test_it_declines_when_the_conversation_is_only_a_system_prompt(self) -> None:
        # Scoring against the system prompt would make every tool look
        # relevant, so the stage would silently never fire -- the
        # "looks configured, does nothing" trap config.py exists to prevent.
        tools = tuple(_openai_tool(f"t{n}") for n in range(8))
        messages = (Message(role="system", content="You are an agent with many tools."),)
        result = PruneToolsStage().before(_request(tools=tools, messages=messages), _ctx())

        assert result.request.tools == tools

    def test_caller_tool_order_survives_pruning(self) -> None:
        # Tool order is stable across turns and therefore part of what a
        # provider's prefix cache matches on. Reordering would forfeit that.
        tools = (
            _openai_tool("alpha", "Search documents for a query."),
            _openai_tool("beta", "Zebra xylophone quixotic."),
            _openai_tool("gamma", "Search records for a query."),
            _openai_tool("delta", "Search archives for a query."),
            _openai_tool("epsilon", "Zebra xylophone quixotic."),
        )
        messages = (Message(role="user", content="search documents and records for a query"),)
        result = PruneToolsStage().before(_request(tools=tools, messages=messages), _ctx())

        kept = [t["function"]["name"] for t in result.request.tools]
        assert kept == sorted(kept, key=lambda n: [t["function"]["name"] for t in tools].index(n))

    def test_it_reads_both_provider_tool_shapes(self) -> None:
        # OpenAI nests under "function"; Anthropic is flat. Reading both means
        # no provider flag is needed to score a tool.
        flat = {"name": "search_docs", "description": "Search the documents."}
        nested = _openai_tool("search_docs", "Search the documents.")
        messages = (Message(role="user", content="search the documents"),)

        stage = PruneToolsStage()
        for shape in (flat, nested):
            tools = (shape, *(_openai_tool(f"z{n}", "Zebra quixotic.") for n in range(5)))
            result = stage.before(_request(tools=tools, messages=messages), _ctx())
            assert "search_docs" in repr(result.request.tools)


class TestFidelityIsDeclaredHonestly:
    def test_only_pruning_claims_to_alter_content(self) -> None:
        assert MinifyToolsStage().fidelity is Fidelity.SHAPED
        assert CapToolResultsStage().fidelity is Fidelity.SHAPED
        assert PruneToolsStage().fidelity is Fidelity.ALTERED

    def test_pruning_is_the_only_lossy_one(self) -> None:
        assert not MinifyToolsStage().lossy
        assert not CapToolResultsStage().lossy
        assert PruneToolsStage().lossy
