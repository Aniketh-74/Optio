"""Translating a normalized request into each provider's own shape.

:class:`~optio_optimize.types.LLMRequest` is provider-neutral by design
(ADR-013), which means *something* has to convert it at the boundary, and that
conversion is where this package's most expensive bug lived. The live
benchmark's OpenAI adapter did not forward ``request.tools``. Nothing failed:
the call succeeded, the report printed, and ``minify_tools`` claimed 3,240
saved tokens while the provider billed byte-identical totals in both arms
because the field the stage had rewritten was dropped on the floor. A whole
live run measured nothing and said so confidently.

ADR-017 adds a second place that has to make the same translation -- batch
submission builds a request body as plain JSON rather than SDK keyword
arguments -- so the translation lives here, once, and both call sites share it.
Two independent copies of this logic is precisely how the same field goes
missing from one of them.

What is deliberately *not* centralized: the final ``create(...)`` call. The
live adapters pass explicit keyword arguments rather than unpacking a dict,
because the SDKs' ``create`` methods are heavily overloaded and
``**dict[str, object]`` type-checks only while the package is absent -- the
unpacked version passed CI, where ``openai`` is not installed, and failed the
moment anyone ran a live benchmark. So this module returns the *parts* that are
easy to forget (messages, tools, system blocks) and each call site assembles
them in the form its transport needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optio_optimize.types import LLMResponse

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest

#: Anthropic's only cache-control type. Named so the three places that emit it
#: -- system blocks, marked turns, and the adapter's raw-param path -- cannot
#: drift apart on a literal. Copy it at each use: a shared mutable dict living
#: on several request bodies is a bug waiting for someone to edit one of them.
EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}

#: Request fields that are deliberately never put on the wire, with the reason.
#: Checked by a test against :class:`~optio_optimize.types.LLMRequest`'s own
#: field list, so adding a field to the request type fails the suite until
#: someone decides which side of this line it belongs on. That is the same
#: guard :func:`~optio_optimize.cache.request_key` applies to the cache key, and
#: for the same reason: a new field that is silently not sent is invisible.
UNSENT_FIELDS: dict[str, str] = {
    "extra": "provider transport details, merged by each adapter rather than named here",
}
# `thinking_budget` was excused here until 2026-07-30, on the grounds that its
# shape is provider-specific. That was true and was not a reason to drop it: a
# caller who set the field had it silently discarded, and reasoning tokens bill
# at *completion* rates, so the excuse hid the most expensive tokens in a
# request. ADR-018 splits the idea into the two shapes the vendors actually
# accept -- `thinking_budget` (Anthropic, a token count) and `reasoning_effort`
# (OpenAI, a category) -- rather than inventing a conversion between them.


def openai_messages(request: LLMRequest) -> list[dict[str, Any]]:
    """Build OpenAI's ``messages`` array.

    ``tool_calls`` and ``tool_call_id`` are lifted out of ``extra`` explicitly
    rather than forwarded wholesale: OpenAI rejects a ``tool`` message with no
    preceding ``tool_calls``, so dropping them silently -- which the first
    version of the live adapter did -- makes every tool-calling workload fail
    with a 400 regardless of what any stage did.

    Args:
        request: The request, after every stage has run.

    Returns:
        Plain dicts. The caller casts to the SDK's ``TypedDict`` if it needs to;
        a batch submission serializes them as JSON instead.
    """
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            entry["name"] = message.name
        if "tool_calls" in message.extra:
            entry["tool_calls"] = message.extra["tool_calls"]
        if "tool_call_id" in message.extra:
            entry["tool_call_id"] = message.extra["tool_call_id"]
        messages.append(entry)
    return messages


def openai_tools(request: LLMRequest) -> list[dict[str, Any]] | None:
    """Return the tool schemas, or ``None`` when there are none.

    ``None`` rather than ``[]``: an empty array is a different request from an
    absent field, and OpenAI rejects the empty one.
    """
    return list(request.tools) or None


def anthropic_system_and_turns(
    request: LLMRequest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into Anthropic's ``system`` blocks and ``messages`` turns.

    Anthropic takes the system prompt as a separate top-level parameter rather
    than a message role, and this is where the library's ``cacheable`` marker
    becomes a real API field -- the entire difference between a ~90% input
    discount and none.

    **The marker is honoured on turns too, not only on system blocks.**
    ``PrefixCacheStage`` marks the *last message of the stable prefix*, which is
    a system message only in the shortest conversations; once there are three or
    more turns the boundary lands on a user or assistant message. An earlier
    version of this function emitted ``cache_control`` for system blocks alone,
    so on any real conversation the marker was computed, placed, reported in the
    savings ledger -- and silently dropped on the way to the wire. Caught by the
    Anthropic adapter's first test run.

    A breakpoint on a turn caches everything above it, the system prompt
    included, so this is strictly better than marking the system block alone.
    Anthropic requires block-shaped content to carry ``cache_control``, so a
    marked turn's string content is promoted to a one-element text block;
    unmarked turns keep the plain string they arrived with.

    Returns:
        ``(system_blocks, turns)``. Either may be empty.
    """
    system_blocks: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            block: dict[str, Any] = {"type": "text", "text": message.content}
            if message.cacheable:
                block["cache_control"] = dict(EPHEMERAL_CACHE_CONTROL)
            system_blocks.append(block)
        elif message.cacheable:
            turns.append(
                {
                    "role": message.role,
                    "content": [
                        {
                            "type": "text",
                            "text": message.content,
                            "cache_control": dict(EPHEMERAL_CACHE_CONTROL),
                        }
                    ],
                }
            )
        else:
            turns.append({"role": message.role, "content": message.content})
    return system_blocks, turns


def anthropic_tools(request: LLMRequest) -> list[dict[str, Any]] | None:
    """Return tool schemas in Anthropic's shape, or ``None``."""
    return [as_anthropic_tool(tool) for tool in request.tools] or None


def as_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one tool schema into Anthropic's shape.

    OpenAI nests ``name``/``description``/``parameters`` under ``function``;
    Anthropic puts ``name``/``description``/``input_schema`` at the top level. A
    schema already in Anthropic's shape passes through, so a caller who writes
    for either provider gets the same behaviour.

    Args:
        tool: A tool schema in either provider's shape.

    Returns:
        The schema as Anthropic expects it.
    """
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
    }


def openai_body(request: LLMRequest, model: str) -> dict[str, Any]:
    """Build a complete OpenAI chat-completions request body as JSON.

    Used by batch submission, which writes request envelopes to a JSONL file and
    therefore has no SDK method to pass keyword arguments to. The live adapter
    does *not* use this -- see the module docstring on why it assembles keyword
    arguments explicitly instead -- so the two must be kept in step by the test
    that compares them field for field.

    ``None`` values are omitted rather than sent: the batch endpoint validates
    each envelope against the same schema as the synchronous one, and an
    explicit ``"temperature": null`` is a validation error rather than a default.

    Args:
        request: The request, after every stage has run.
        model: Model to bill against -- not necessarily ``request.model``, which
            the routing stage may already have changed.

    Returns:
        A JSON-serializable body.
    """
    body: dict[str, Any] = {"model": model, "messages": openai_messages(request)}
    if request.max_tokens is not None:
        body["max_completion_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.response_format is not None:
        body["response_format"] = request.response_format
    # `reasoning_effort`, never a value derived from `thinking_budget`. OpenAI
    # takes a category and Anthropic a token count, and translating between
    # them means inventing thresholds that depend on the model (ADR-018).
    if request.reasoning_effort is not None:
        body["reasoning_effort"] = request.reasoning_effort
    tools = openai_tools(request)
    if tools is not None:
        body["tools"] = tools
    if request.stop:
        body["stop"] = list(request.stop)
    return body


def anthropic_body(
    request: LLMRequest,
    model: str,
    *,
    default_max_tokens: int = 1024,
) -> dict[str, Any]:
    """Build a complete Anthropic messages request body as JSON.

    Args:
        request: The request, after every stage has run.
        model: Model to bill against.
        default_max_tokens: Used when the caller set none. Anthropic requires
            the field, so there is no "omit it" option the way there is on
            OpenAI -- a value has to be invented, and inventing it visibly here
            beats burying it in an adapter.

    Returns:
        A JSON-serializable body.
    """
    system_blocks, turns = anthropic_system_and_turns(request)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": request.max_tokens or default_max_tokens,
        "messages": turns,
    }
    if system_blocks:
        body["system"] = system_blocks
    if request.temperature is not None:
        body["temperature"] = request.temperature
    # `is not None` rather than truthiness: budget 0 means "do not think", which
    # is a real instruction, while absence means the caller said nothing at all.
    # Collapsing the two would silently disable reasoning for everyone who left
    # the field alone -- the absence-is-not-zero rule, on the field where
    # getting it backwards is most expensive.
    if request.thinking_budget is not None:
        body["thinking"] = {"type": "enabled", "budget_tokens": request.thinking_budget}
    tools = anthropic_tools(request)
    if tools is not None:
        body["tools"] = tools
    if request.stop:
        body["stop_sequences"] = list(request.stop)
    return body


def response_from_anthropic_message(message: Any) -> LLMResponse:
    """Normalize an Anthropic message into this package's response model.

    ``input_tokens`` is reported by Anthropic *excluding* both cache reads and
    cache writes, so **both** are added back to make the field mean what it
    means everywhere else here: total prompt tokens, of which some were
    discounted and some carried a premium. Getting that wrong in one place would
    make batched, synchronous and adapter totals silently incomparable.

    Writes were omitted at first, and the omission was not symmetric: it dropped
    the most expensive band of prompt tokens from the total, so a cached call
    reported a fraction of its real cost and every prefix-cache saving derived
    from it came out too high. On the first turn of the run behind this
    package's 53.7% figure it reported 200 tokens against a true 4,805.

    Attribute access rather than isinstance narrowing, because the two callers
    receive structurally identical objects from different SDK code paths --
    a batch result entry's ``.message`` and a live ``messages.create`` return.

    Args:
        message: An ``anthropic.types.Message`` or a batch result's message.

    Returns:
        The normalized response.
    """
    usage = getattr(message, "usage", None)
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
    text = "".join(
        str(getattr(block, "text", ""))
        for block in getattr(message, "content", [])
        if getattr(block, "type", "") == "text"
    )
    return LLMResponse(
        content=text,
        input_tokens=(int(getattr(usage, "input_tokens", 0)) + cached + written) if usage else 0,
        output_tokens=int(getattr(usage, "output_tokens", 0)) if usage else 0,
        cached_input_tokens=cached,
        cache_write_tokens=written,
        model=str(getattr(message, "model", "")),
        finish_reason=getattr(message, "stop_reason", None),
    )
