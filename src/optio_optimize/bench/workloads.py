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

#: A retrieved chunk about a completely unrelated topic -- office parking, not
#: the quarterly report the question is about. A real vector store returns
#: chunks like this constantly: a near-miss on embedding similarity that a
#: cheap lexical check can still catch, because it shares almost no
#: vocabulary with the question. ``_rag_queries`` never includes one of
#: these, which is why ``prune_retrieval`` measured 0 tokens saved on it,
#: both simulated and live (``docs/optimize-benchmarks.md``) -- that is a
#: correct zero on a workload with nothing to prune, not evidence the stage
#: works. ``_rag_queries_noisy`` exists to give it something to prune.
_IRRELEVANT_CHUNK = (
    "The office parking garage will be closed for resurfacing next Tuesday. "
    "Employees should use the visitor lot on Elm Street during the closure. "
    "Badge access will still work at the north entrance."
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


def _timestamped_agent(turns: int = 12, model: str = "gpt-4o") -> list[LLMRequest]:
    """``multi_turn_chat`` with a clock at the top of the system prompt.

    One line different from ``multi_turn_chat``, and that line is the most
    common prompt-caching bug in production: something that varies placed
    *above* the stable instructions, so the provider's prefix cache matches
    nothing and the whole system prompt is billed at full rate on every turn.
    Nothing errors, every response is correct, and the bill is several times
    what it should be.

    Its value here is as a *pair*: run it beside ``multi_turn_chat`` and the
    difference in cached tokens is the price of the bug, measured rather than
    asserted. It is also the workload `detect_unstable_prefix` exists to flag,
    so a run that reports no finding here is a broken detector.
    """
    requests: list[LLMRequest] = []
    history: list[Message] = []
    for turn in range(turns):
        # The timestamp is regenerated per turn, exactly as `datetime.now()` in
        # a prompt template would be. Below the instructions it would be free.
        system = f"Current time: 2026-07-29T09:{turn:02d}:31Z\n{_SYSTEM_PROMPT}"
        history.append(_msg("user", f"Question {turn}: summarise section {turn} of the report."))
        requests.append(
            LLMRequest(
                model=model,
                messages=(_msg("system", system), *history),
                temperature=0.0,
            )
        )
        history.append(
            _msg("assistant", f"Section {turn} covers revenue, costs and outlook for {turn}.")
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


def _multi_turn_chat_long(model: str = "gpt-4o") -> list[LLMRequest]:
    """A 50-turn version of ``multi_turn_chat``.

    The 12-turn default is short enough that `trim_history`'s live result
    (cost -8.4%, docs/optimize-benchmarks.md) could plausibly be a small-scale
    artifact rather than a stable one -- the whole reason a sliding window can
    fight a growing provider-side cache is a *scale* effect (ADR-013's own
    reasoning), and IMPLEMENTATION.md's own problem statement describes
    agentic workloads running 5-30x longer than a single-shot chat, a regime
    12 turns doesn't reach. This exists to check the trend at 50, not to
    replace the 12-turn figure already measured.
    """
    return _multi_turn_chat(turns=50, model=model)


def _rag_queries_noisy(count: int = 10, chunks: int = 6, model: str = "gpt-4o") -> list[LLMRequest]:
    """Retrieval context with one genuinely irrelevant chunk mixed in.

    ``rag_queries`` never exercises what ``prune_retrieval`` actually decides
    to drop: every chunk there is about the same topic, so nothing ever
    scores below the relevance floor and the stage always declines. This
    workload gives it something to prune -- an office-parking-notice chunk,
    landing in the middle of the context (not an edge, so a stage that only
    ever happened to drop the first or last block would not look like it
    works by accident) -- and something it must never touch: every chunk that
    actually shares the question's vocabulary.
    """
    requests: list[LLMRequest] = []
    for query in range(count):
        relevant = [f"[doc {c}] {_CHUNK}" for c in range(chunks)]
        midpoint = chunks // 2
        blocks = [*relevant[:midpoint], f"[doc noise] {_IRRELEVANT_CHUNK}", *relevant[midpoint:]]
        context = "\n\n".join(blocks)
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


def _tool_calling_chat(turns: int = 10, model: str = "gpt-4o") -> list[LLMRequest]:
    """A growing conversation where the agent actually issues tool calls.

    The only workload in this suite that uses ``role="tool"`` messages.
    ``tool_loop`` talks *about* calling tools in plain text; this one produces
    the real assistant-tool_calls / tool-result message pairs a live
    function-calling agent sends -- ``tool_calls`` and ``tool_call_id`` live in
    ``Message.extra`` in OpenAI's own shape, since the request model carries
    provider-specific fields through untouched rather than normalizing them.
    It exists specifically to prove ``trim_history`` never cuts a window that
    separates a tool_calls assistant message from its result -- no other
    workload here could have caught that defect, because none of them have a
    "tool" role message to orphan.
    """
    requests: list[LLMRequest] = []
    history: list[Message] = [_msg("system", _SYSTEM_PROMPT)]
    for turn in range(turns):
        history.append(_msg("user", f"Look up record {turn} and summarise it."))
        requests.append(LLMRequest(model=model, messages=tuple(history), temperature=0.0))

        call_id = f"call_{turn}"
        history.append(
            Message(
                role="assistant",
                content="",
                extra={
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "fetch_record",
                                "arguments": f'{{"record_id": {turn}}}',
                            },
                        }
                    ]
                },
            )
        )
        history.append(
            Message(
                role="tool",
                content=f"record {turn}: status=ok, total={turn * 7}",
                name="fetch_record",
                extra={"tool_call_id": call_id},
            )
        )
        requests.append(LLMRequest(model=model, messages=tuple(history), temperature=0.0))

        history.append(_msg("assistant", f"Record {turn} looks fine, total is {turn * 7}."))
    return requests


#: Tool families an MCP-connected agent typically carries. Each becomes a
#: schema in the generated shape real bridges emit -- a ``title`` on every
#: property, a ``$schema`` header, and a description written for a docs page
#: rather than for a token budget. That shape is the point: a hand-written
#: schema has little for ``minify_tools`` to remove, and measuring only
#: hand-written ones would report that the stage does nothing while real
#: OpenAPI- and MCP-derived tools carry the waste on every turn.
_TOOL_FAMILIES = (
    ("search_documents", "Search the indexed document corpus", "query", "top_k"),
    ("fetch_record", "Retrieve a single record by its identifier", "record_id", "fields"),
    ("compute_metric", "Compute a named metric over a reporting period", "name", "period"),
    ("compare_periods", "Compare two reporting periods and return the delta", "a", "b"),
    ("list_tickets", "List support tickets matching a filter", "status", "assignee"),
    ("send_notification", "Send a notification to a channel", "channel", "body"),
    ("run_query", "Execute a read-only analytical query", "sql", "timeout_seconds"),
    ("summarize_thread", "Summarize a conversation thread", "thread_id", "max_words"),
    ("translate_text", "Translate text between languages", "text", "target_language"),
    ("classify_intent", "Classify a message into a product intent", "message", "taxonomy"),
)


def _mcp_tool(name: str, description: str, *params: str) -> dict[str, object]:
    """One tool in the shape an MCP bridge or OpenAPI generator emits."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                f"{description}. This tool is provided by the platform integration layer "
                f"and should be used whenever the user's request calls for it. Arguments "
                f"are validated before dispatch and errors are returned as structured "
                f"payloads rather than raised."
            ),
            "parameters": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": f"{name}_arguments",
                "type": "object",
                "properties": {
                    param: {
                        "type": "string",
                        "title": param.replace("_", " ").title(),
                        "description": f"The {param.replace('_', ' ')} to use.",
                    }
                    for param in params
                },
                "required": list(params),
            },
        },
    }


def _mcp_agent(steps: int = 10, model: str = "gpt-4o") -> list[LLMRequest]:
    """An agent carrying a full MCP tool library, with oversized tool results.

    The workload that makes tool cost visible at all. Every other workload in
    this suite sends ``tools=()``, so before this existed the three stages in
    ``stages/tools.py`` could not be measured -- which under ADR-016's third
    test means no claim about them could ship.

    Two costs are exercised, and they behave differently:

    * **Schemas**, resent verbatim on all ten steps. ``minify_tools`` and
      ``prune_tools`` target these, and because the prefix is identical every
      turn it is also exactly the region a provider prefix cache covers -- so
      a saving here may be smaller in money than in tokens, the same gap
      ``prefix_cache``'s own live correction turned on.
    * **Results**, which enter the history once and are then billed on every
      later step. One deliberately oversized payload lands at step 3, so the
      run contains the shape ``cap_tool_results`` exists for: a single tool
      response that quietly raises the price of the whole remaining
      conversation.
    """
    tools = tuple(_mcp_tool(*family) for family in _TOOL_FAMILIES)
    requests: list[LLMRequest] = []
    history: list[Message] = [_msg("system", _SYSTEM_PROMPT)]

    for step in range(steps):
        history.append(_msg("user", f"Step {step}: look up record {step} and report the total."))
        requests.append(
            LLMRequest(model=model, messages=tuple(history), tools=tools, temperature=0.0)
        )

        call_id = f"call_{step}"
        history.append(
            Message(
                role="assistant",
                content="",
                extra={
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "fetch_record",
                                "arguments": f'{{"record_id": "{step}"}}',
                            },
                        }
                    ]
                },
            )
        )
        # Step 3 returns a runaway payload: a query that matched far more rows
        # than the agent needed. Every step after this one pays for it.
        payload = (
            "\n".join(
                f'{{"row": {row}, "record": "{step}", "status": "ok", "total": {row * 7}}}'
                for row in range(400)
            )
            if step == 3
            else f'{{"record": "{step}", "status": "ok", "total": {step * 7}}}'
        )
        history.append(
            Message(
                role="tool", content=payload, name="fetch_record", extra={"tool_call_id": call_id}
            )
        )
        history.append(_msg("assistant", f"Record {step} is fine; total {step * 7}."))
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
                    # "json" must appear literally in the messages or OpenAI
                    # rejects response_format=json_object with a 400. Found by
                    # the live benchmark, which failed all 12 calls here; the
                    # simulator had accepted the request happily. Real callers
                    # hit this on their first request, so the workload should
                    # look like one that works.
                    _msg(
                        "user",
                        f"Classify item {branch % 4} as revenue, cost or risk. "
                        'Reply as json: {"label": "..."}',
                    ),
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
    "multi_turn_chat_long": Workload(
        name="multi_turn_chat_long",
        description="50-turn conversation resending full history each step",
        build=_multi_turn_chat_long,
        expectation="checks whether trim_history's live cost win at 12 turns compounds, "
        "plateaus, or reverses at scale",
        tags=("prefix_cache", "trim_history"),
    ),
    "rag_queries": Workload(
        name="rag_queries",
        description="10 queries carrying 8 overlapping retrieved chunks each",
        build=_rag_queries,
        expectation="deduplication and pruning target this; little gain before Phase 2",
        tags=("deduplicate", "prune_retrieval"),
    ),
    "rag_queries_noisy": Workload(
        name="rag_queries_noisy",
        description="10 queries with one genuinely irrelevant chunk mixed into relevant context",
        build=_rag_queries_noisy,
        expectation="prune_retrieval must drop the irrelevant chunk and keep every relevant one",
        tags=("prune_retrieval",),
    ),
    "tool_loop": Workload(
        name="tool_loop",
        description="20 steps cycling 4 distinct tool calls",
        build=_tool_loop,
        expectation="high exact-cache hit rate: 16 of 20 calls avoidable",
        tags=("exact_cache",),
    ),
    "tool_calling_chat": Workload(
        name="tool_calling_chat",
        description="10-turn agent conversation with real tool_calls / tool-result messages",
        build=_tool_calling_chat,
        expectation="every request must stay provider-valid: trim_history must never "
        "orphan a tool result from its assistant tool_calls message",
        tags=("trim_history", "tool_safety"),
    ),
    "timestamped_agent": Workload(
        name="timestamped_agent",
        description="multi_turn_chat with a per-turn timestamp above the system prompt",
        build=_timestamped_agent,
        expectation="ZERO provider prefix-cache hits, by construction. Run beside "
        "multi_turn_chat to price the most common caching bug in production; "
        "detect_unstable_prefix must report a finding here or it is broken.",
        tags=("detect_unstable_prefix", "adversarial"),
    ),
    "mcp_agent": Workload(
        name="mcp_agent",
        description="10 steps carrying 10 MCP-shaped tool schemas, one oversized tool result",
        build=_mcp_agent,
        expectation="the only workload that can measure tool cost at all: minify_tools and "
        "cap_tool_results should both fire, prune_tools only when enabled",
        tags=("minify_tools", "cap_tool_results", "prune_tools"),
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
