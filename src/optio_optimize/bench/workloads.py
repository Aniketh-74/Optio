"""Benchmark workloads, including ones this library does badly on.

A benchmark suite built only from cases an optimizer handles well produces a
number that is true and useless. These workloads are chosen to span the range,
and two of them exist specifically because the library should show little or no
gain on them:

* ``unique_questions`` — every prompt distinct and short. Nothing to cache,
  nothing to trim. Expected saving: near zero, and a suite that hides this is
  lying by omission.
* ``sampled_creative`` — ``temperature=0.9``. The exact cache must decline
  every request, because caching a sampled call replaces variety the caller
  asked for with one frozen answer.

The remainder are the shapes real agents actually produce. Sizes are chosen to
be representative rather than large: a benchmark that costs $50 to run against a
live API will not be run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from optio_optimize.types import LLMRequest, Message

#: A system prompt of the size real agents carry, *including* the tool schemas
#: that usually sit beside it. The repetition factor is set so this lands above
#: 1024 tokens, which is not arbitrary: Anthropic ignores a prefix-cache marker
#: below that, so a shorter prompt makes the prefix stage correctly decline and
#: the benchmark then reports zero benefit from a feature that would work fine
#: in production. The first version of this file got that wrong and measured a
#: 760-token prompt, which is smaller than any real agent's.
_SYSTEM_PROMPT = (
    "You are a meticulous research assistant operating inside an automated "
    "pipeline. Answer using only the context provided. If the context does not "
    "contain the answer, say exactly: INSUFFICIENT CONTEXT. Never speculate, "
    "never invent citations, and never continue past the answer with offers of "
    "further help. Prefer the shortest complete answer. When asked for a number, "
    "give the number and its unit and nothing else. When asked to compare two "
    "things, give at most three points of difference, each one sentence. "
    "Available tools: search_documents(query: str, top_k: int) -> list[Chunk]; "
    "fetch_record(record_id: str) -> Record; compute_metric(name: str, "
    "period: str) -> float; compare_periods(a: str, b: str) -> Delta. "
) * 9

#: A retrieved document chunk. Deliberately repetitive between chunks: real RAG
#: retrieval returns overlapping passages constantly, which is precisely what
#: deduplication and pruning target.
_CHUNK = (
    "Quarterly revenue for the period increased relative to the prior quarter, "
    "driven primarily by subscription growth in the enterprise segment. "
    "Operating expenses rose more slowly than revenue. "
)


@dataclass(slots=True)
class Workload:
    """A named sequence of requests plus what it is meant to exercise.

    Attributes:
        name: Identifier used in reports.
        description: What shape of agent traffic this imitates.
        build: Produces the request sequence.
        expectation: What the library should do here, in plain words, including
            "very little". Stated so a surprising result is visible as a
            surprise rather than absorbed as a number.
    """

    name: str
    description: str
    build: Callable[[], list[LLMRequest]]
    expectation: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    def requests(self) -> list[LLMRequest]:
        """Return this workload's requests."""
        return self.build()


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)  # type: ignore[arg-type]


def _multi_turn_chat(turns: int = 12, model: str = "gpt-4o") -> list[LLMRequest]:
    """A conversation that grows, resending its whole history each step.

    The most common agent shape and the most expensive: step *n* pays for every
    turn before it, so cost grows quadratically in conversation length.
    """
    requests: list[LLMRequest] = []
    history: list[Message] = [_msg("system", _SYSTEM_PROMPT)]
    for turn in range(turns):
        history.append(_msg("user", f"Question {turn}: summarise section {turn} of the report."))
        requests.append(
            LLMRequest(
                model=model,
                messages=tuple(history),
                temperature=0.0,
            )
        )
        history.append(
            _msg(
                "assistant", f"Section {turn} covers revenue, costs and outlook for period {turn}."
            )
        )
    return requests


def _rag_queries(count: int = 10, chunks: int = 8, model: str = "gpt-4o") -> list[LLMRequest]:
    """Retrieval-augmented queries carrying many overlapping chunks.

    Retrieval commonly returns near-duplicate passages, and every one is billed
    on every call.
    """
    requests: list[LLMRequest] = []
    for query in range(count):
        # Overlapping on purpose: chunk k of query q repeats chunk k of query
        # q-1 about half the time, as a real vector store would.
        context = "\n\n".join(f"[doc {(query + c) % 5}] {_CHUNK}" for c in range(chunks))
        requests.append(
            LLMRequest(
                model=model,
                messages=(
                    _msg("system", _SYSTEM_PROMPT),
                    _msg(
                        "user", f"Context:\n{context}\n\nQuestion: what drove revenue in Q{query}?"
                    ),
                ),
                temperature=0.0,
            )
        )
    return requests


def _tool_loop(steps: int = 20, distinct: int = 4, model: str = "gpt-4o") -> list[LLMRequest]:
    """An agent cycling through a small set of tool calls.

    The behaviour optio's behavior lane calls ``looping``, and the case where
    exact caching pays most: the same call recurs verbatim.
    """
    requests: list[LLMRequest] = []
    for step in range(steps):
        tool = step % distinct
        requests.append(
            LLMRequest(
                model=model,
                messages=(
                    _msg("system", _SYSTEM_PROMPT),
                    _msg("user", f"Call tool {tool} with the standard arguments and report back."),
                ),
                temperature=0.0,
            )
        )
    return requests


def _retry_storm(attempts: int = 15, model: str = "gpt-4o") -> list[LLMRequest]:
    """The same request repeated after transient failures.

    Every retry after the first is pure waste, and it is waste an exact cache
    removes completely.
    """
    request = LLMRequest(
        model=model,
        messages=(
            _msg("system", _SYSTEM_PROMPT),
            _msg("user", "Fetch the current pipeline status and summarise it."),
        ),
        temperature=0.0,
    )
    return [request] * attempts


def _fan_out(branches: int = 12, model: str = "gpt-4o") -> list[LLMRequest]:
    """Parallel sub-tasks, a third of which are identical."""
    requests: list[LLMRequest] = []
    for branch in range(branches):
        requests.append(
            LLMRequest(
                model=model,
                messages=(
                    _msg("system", _SYSTEM_PROMPT),
                    _msg("user", f"Classify item {branch % 4} into one of: revenue, cost, risk."),
                ),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
    return requests


def _unique_questions(count: int = 12, model: str = "gpt-4o") -> list[LLMRequest]:
    """Every prompt different and short. The adversarial case."""
    return [
        LLMRequest(
            model=model,
            messages=(_msg("user", f"In one sentence, what is entity {n} known for?"),),
            temperature=0.0,
        )
        for n in range(count)
    ]


def _sampled_creative(count: int = 8, model: str = "gpt-4o") -> list[LLMRequest]:
    """Sampled requests that the cache must refuse to serve."""
    return [
        LLMRequest(
            model=model,
            messages=(
                _msg("system", "You are a copywriter."),
                _msg("user", "Write a one-line tagline for a coffee brand."),
            ),
            temperature=0.9,
        )
        for _ in range(count)
    ]


#: Every workload, keyed by name.
WORKLOADS: dict[str, Workload] = {
    "multi_turn_chat": Workload(
        name="multi_turn_chat",
        description="12-turn conversation resending full history each step",
        build=_multi_turn_chat,
        expectation="large prefix-cache benefit; history trimming once Phase 2 lands",
        tags=("prefix_cache", "trim_history"),
    ),
    "rag_queries": Workload(
        name="rag_queries",
        description="10 queries carrying 8 overlapping retrieved chunks each",
        build=_rag_queries,
        expectation="deduplication and pruning target this; little gain before Phase 2",
        tags=("deduplicate", "prune_retrieval"),
    ),
    "tool_loop": Workload(
        name="tool_loop",
        description="20 steps cycling 4 distinct tool calls",
        build=_tool_loop,
        expectation="high exact-cache hit rate: 16 of 20 calls avoidable",
        tags=("exact_cache",),
    ),
    "retry_storm": Workload(
        name="retry_storm",
        description="the same request issued 15 times",
        build=_retry_storm,
        expectation="14 of 15 calls avoidable; the strongest case for exact caching",
        tags=("exact_cache",),
    ),
    "fan_out": Workload(
        name="fan_out",
        description="12 parallel classifications over 4 distinct items, JSON output",
        build=_fan_out,
        expectation="two thirds cacheable, plus structured-output preamble suppression",
        tags=("exact_cache", "structured_output"),
    ),
    "unique_questions": Workload(
        name="unique_questions",
        description="12 short, entirely distinct prompts",
        build=_unique_questions,
        expectation="NEAR ZERO saving. Included so the suite reports its own limits.",
        tags=("adversarial",),
    ),
    "sampled_creative": Workload(
        name="sampled_creative",
        description="8 identical prompts at temperature 0.9",
        build=_sampled_creative,
        expectation="ZERO cache hits by design; caching sampled calls would be a bug",
        tags=("adversarial",),
    ),
}
