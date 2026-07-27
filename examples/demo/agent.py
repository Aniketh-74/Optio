"""A deliberately misbehaving agent (M4-5).

The demo has to show agentmeter catching a real pathology, which means the
agent has to actually misbehave. This one does two things wrong, on purpose:

1. It gets stuck. After a few productive steps the "model" starts calling the
   same tool with the same arguments forever, making no progress.
2. It keeps spending while stuck. Every looping step burns tokens, which is
   exactly the failure that shows up on an invoice rather than in an error log.

**No API keys, no network.** The model is a scripted stand-in. That is a
deliberate constraint, not a shortcut: ADR-006 makes the demo an evaluator-facing
deliverable and it has to run on a fresh machine in one command. A demo that
needs an OpenAI key and costs real money to run is a demo nobody runs.

What is *not* faked is the part being demonstrated. The spans are real OTel
GenAI spans carrying real semconv attributes, the token counts drive the real
pricing table, and the signals come from the real cost and behavior lanes. Swap
``ScriptedModel`` for an SDK call and everything downstream is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from opentelemetry import trace

from agentmeter import semconv

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The model the demo prices against. Real entry in the pricing table, so the
#: dollar figures below are the ones a real gpt-4o run would produce.
MODEL: Final = "gpt-4o"

#: Tokens a step consumes. Input grows with the conversation -- the whole reason
#: agentic workloads cost 5-30x a single-shot chat (Section 1.3) -- so a stuck
#: agent gets progressively more expensive per step, not merely repetitive.
_BASE_INPUT_TOKENS: Final = 800
_INPUT_GROWTH_PER_STEP: Final = 450
_OUTPUT_TOKENS: Final = 120

_tracer: Final = trace.get_tracer("agentmeter.demo")


@dataclass(frozen=True, slots=True)
class Step:
    """One planned agent step.

    Attributes:
        tool: Tool the model decided to call.
        query: Arguments to that tool. Identical values across steps are what
            make a loop detectable.
    """

    tool: str
    query: str


class ScriptedModel:
    """A stand-in for an LLM that gets stuck in a retrieval loop.

    Produces a short run of genuine progress, then repeats one call forever --
    the shape of a real agent that has lost the thread but has no way to know it.
    """

    #: Real work: each step is a different call.
    PRODUCTIVE: Final[tuple[Step, ...]] = (
        Step("search", "quarterly revenue 2026"),
        Step("fetch_document", "q1-earnings.pdf"),
        Step("extract_table", "revenue by segment"),
        Step("search", "segment definitions"),
    )

    #: The trap. The agent decides it needs one more document and asks for it
    #: over and over, with identical arguments, never noticing it already has
    #: the answer.
    STUCK: Final = Step("fetch_document", "q1-earnings-appendix.pdf")

    def __init__(self, *, get_stuck: bool = True) -> None:
        """Create the scripted model.

        Args:
            get_stuck: When ``False``, the agent finishes cleanly after the
                productive steps. This is the "after" side of the demo's
                before/after comparison -- the same agent with the loop fixed.
        """
        self.get_stuck = get_stuck

    def plan(self, max_steps: int) -> Iterator[Step]:
        """Yield the steps this run will take.

        Args:
            max_steps: Hard ceiling, so the demo terminates even while looping.

        Yields:
            One step at a time.
        """
        for index in range(max_steps):
            if index < len(self.PRODUCTIVE):
                yield self.PRODUCTIVE[index]
            elif self.get_stuck:
                yield self.STUCK
            else:
                return


def run_step(step: Step, step_index: int) -> None:
    """Emit one GenAI span for a step.

    The span carries the attributes agentmeter reads (Section 7.2): model,
    operation, tool name, and token usage. Nothing here writes a signal --
    the span tap observes these and the lanes compute from them.

    Args:
        step: The step being executed.
        step_index: Zero-based position in the run, used to grow the context.
    """
    input_tokens = _BASE_INPUT_TOKENS + step_index * _INPUT_GROWTH_PER_STEP

    with _tracer.start_as_current_span(f"{step.tool}") as span:
        span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
        span.set_attribute(semconv.GEN_AI_OPERATION_NAME, "chat")
        span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, MODEL)
        span.set_attribute(semconv.GEN_AI_RESPONSE_MODEL, MODEL)
        span.set_attribute(semconv.GEN_AI_TOOL_NAME, step.tool)
        span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, _OUTPUT_TOKENS)
        # The tool's arguments. Identical values across steps are what the
        # behavior lane hashes into a repeated signature.
        span.set_attribute("gen_ai.tool.query", step.query)

        # A real model call takes time; a little sleep keeps the demo readable
        # as it scrolls rather than dumping thirty steps instantly.
        time.sleep(0.02)


def run_agent(*, get_stuck: bool = True, max_steps: int = 30) -> int:
    """Run the demo agent to completion.

    Args:
        get_stuck: Whether the agent falls into the retrieval loop.
        max_steps: Hard ceiling on steps.

    Returns:
        The number of steps executed.
    """
    model = ScriptedModel(get_stuck=get_stuck)
    executed = 0
    for index, step in enumerate(model.plan(max_steps)):
        run_step(step, index)
        executed += 1
    return executed
