"""Anthropic adapter: wraps a client's ``messages.create``.

**Why this adapter matters more than its size.**
:class:`~optio_optimize.stages.caching.PrefixCacheStage` is described in its own
source as *"the single largest lossless saving available"*, and on OpenAI it
contributes exactly zero -- OpenAI caches any long prefix automatically, for
both arms of an A/B, which is why a simulated 36.3% saving corrected to -1.8%
against the live API. Anthropic caches nothing without an explicit
``cache_control`` breakpoint, and the ``cacheable`` marker this package's
pipeline places is what puts one there. **This adapter is the only path by
which a user reaches that stage's value**; before it, the translation existed
only inside the benchmark module.

**Why the client, not a framework.** Same reasoning as
:mod:`~optio_optimize.adapters.openai_agents`: ``messages.create`` is the one
place anything built on this SDK actually talks to the provider, so wrapping it
composes with the Claude Agent SDK, LangGraph-on-Anthropic, and direct use
alike, without this package importing any of them.

**Sync and async both.** Unlike OpenAI, where the Agents SDK's ``async def``
``get_response`` forced the choice, ``Anthropic`` and ``AsyncAnthropic`` are
both in ordinary use. :class:`~optio_optimize.pipeline.Pipeline` already has
``execute`` and ``aexecute``, so supporting both costs one branch -- and "async
only" is not plug and play.

**Streaming is optimized, as of ADR-019.** It used not to be, on the grounds
that a pipeline built around one request producing one response can only buffer
a token stream. That holds for the few stages which read a reply and not for the
majority, which only rewrite the request -- so a streaming caller was getting
zero of nineteen stages, including the one this module exists for. A
``stream=True`` call now runs every ``before`` hook, sends the transformed
request, and completes when the stream does; see
:mod:`optio_optimize.adapters.anthropic_streaming`.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Any

from optio_optimize import wire
from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from optio_optimize.pipeline import PreparedRequest

_log = logging.getLogger("optio_optimize")

#: Scratch key under which the caller's original message param rides through
#: untouched -- multimodal content blocks, provider extensions, anything this
#: package does not model -- restored verbatim unless a stage changed the text.
#:
#: Defined in :mod:`~optio_optimize.wire` since ADR-022, because the cache key
#: now reads it too: an image block lives here and nowhere else, so keying only
#: ``Message.content`` made two different images hash identically.
_RAW = wire.RAW_CONTENT_KEY

#: Scratch key on a response for the real SDK object, when one exists. Returned
#: verbatim rather than reconstructed, to preserve every field this package
#: does not model (id, stop_sequence, container, ...).
_NATIVE = "_native"


def wrap_anthropic_client(
    client: Any,
    optimizer: Optimizer | None = None,
    **overrides: Any,
) -> Any:
    """Route ``client.messages.create`` through an ``Optimizer``.

    Mutates and returns the same client, matching the identity contract
    ``optio``'s own adapters use (§6.7) and
    :func:`~optio_optimize.adapters.openai_agents.wrap_openai_client`: callers
    keep the object they built, with one method replaced.

    Works on both ``Anthropic`` and ``AsyncAnthropic``; which one it is gets
    detected rather than declared.

    Args:
        client: The client to wrap. ``create`` calls are optimized from this
            point on, streaming and not (ADR-019).
        optimizer: Optimizer to route through. Built from ``overrides`` when
            omitted, so ``wrap_anthropic_client(client, exact_cache=False)``
            works without constructing one.
        **overrides: Individual ``OptimizeConfig`` fields. Invalid together
            with ``optimizer``, for the same reason ``Optimizer.__init__``
            rejects the combination.

    Returns:
        ``client``, with ``messages.create`` replaced.
    """
    active = optimizer if optimizer is not None else Optimizer(**overrides)
    original_create = client.messages.create

    if _is_async(original_create):
        client.messages.create = _async_create(original_create, active)
    else:
        client.messages.create = _sync_create(original_create, active)
    return client


def _is_async(create: Any) -> bool:
    """Whether ``messages.create`` is a coroutine function.

    ``inspect.iscoroutinefunction`` alone answers **False for both** Anthropic
    clients: the SDK wraps ``create`` in a ``@required_args`` decorator, and the
    coroutine marker does not survive it. Unwrapping first recovers it.

    This is worth the comment because the failure is silent and looks like
    success. With the naive check, an ``AsyncAnthropic`` took the synchronous
    branch, `original(**kwargs)` returned an un-awaited coroutine,
    ``response_from_anthropic_message`` read no attributes off it and produced
    an empty response without raising, and ``_unwrap`` handed the coroutine
    straight back -- which ``await`` then executed as an ordinary unoptimized
    call. **Eight of this module's eleven tests passed that way**, because the
    real request did happen; only the ones needing a cache hit or a stage
    effect failed. An adapter that does nothing, reporting success.
    """
    return inspect.iscoroutinefunction(inspect.unwrap(create))


def _async_create(
    original: Callable[..., Awaitable[Any]],
    optimizer: Optimizer,
) -> Callable[..., Awaitable[Any]]:
    """Build the replacement for an ``AsyncAnthropic``'s ``messages.create``."""

    async def optimized(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return await _astream(original, optimizer, kwargs)
        prepared = _translate(kwargs)
        if prepared is None:
            return await original(**kwargs)

        async def call_provider(sent: LLMRequest) -> LLMResponse:
            reply = await original(**_kwargs_from_request(sent, kwargs))
            return _with_native(wire.response_from_anthropic_message(reply), reply)

        return _unwrap(await optimizer.acall(prepared, call_provider), kwargs)

    return optimized


def _sync_create(original: Callable[..., Any], optimizer: Optimizer) -> Callable[..., Any]:
    """Build the replacement for a sync ``Anthropic``'s ``messages.create``."""

    def optimized(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _stream(original, optimizer, kwargs)
        prepared = _translate(kwargs)
        if prepared is None:
            return original(**kwargs)

        def call_provider(sent: LLMRequest) -> LLMResponse:
            reply = original(**_kwargs_from_request(sent, kwargs))
            return _with_native(wire.response_from_anthropic_message(reply), reply)

        return _unwrap(optimizer.call(prepared, call_provider), kwargs)

    return optimized


def _outgoing_stream_kwargs(
    optimizer: Optimizer,
    kwargs: dict[str, Any],
) -> tuple[PreparedRequest, dict[str, Any]] | LLMResponse | None:
    """Run the request-side half of a streaming call (ADR-019).

    Everything that can fail happens here, *before* the provider is called, so
    the fail-open path cannot retry a call that already billed.

    Args:
        optimizer: The optimizer whose pipeline runs.
        kwargs: The caller's ``create`` arguments.

    Returns:
        ``(prepared, outgoing_kwargs)`` to send; an :class:`LLMResponse` when a
        stage served the answer without a provider; or ``None`` meaning "call the
        provider with the caller's own arguments, untouched".
    """
    request = _translate(kwargs)
    if request is None:
        return None
    try:
        pipeline = optimizer.pipeline
        prepared = pipeline.prepare(request)
        if prepared.bypassed:
            return None
        if prepared.short_circuit is not None:
            return pipeline.complete(prepared, prepared.short_circuit)
        return prepared, _kwargs_from_request(prepared.request, kwargs)
    except Exception as exc:  # noqa: BLE001 - ADR-013 rule 1: never break the caller
        # Type only, never the message (§10).
        _log.warning(
            "optio_optimize: could not optimize a streaming request (%s); "
            "streaming directly from the provider, unoptimized",
            type(exc).__name__,
        )
        return None


def _stream(original: Callable[..., Any], optimizer: Optimizer, kwargs: dict[str, Any]) -> Any:
    """Optimize and return a synchronous stream."""
    from optio_optimize.adapters.anthropic_streaming import (
        ReplayStream,
        StreamProxy,
        replay_events,
    )

    outcome = _outgoing_stream_kwargs(optimizer, kwargs)
    if outcome is None:
        return original(**kwargs)
    if isinstance(outcome, LLMResponse):
        return ReplayStream(replay_events(outcome, kwargs))

    prepared, outgoing = outcome
    stream = original(**outgoing)
    try:
        return StreamProxy(stream, optimizer.pipeline, prepared)
    except Exception as exc:  # noqa: BLE001 - the call already happened
        # Returning the raw stream loses the accounting, never the answer.
        # Re-calling the provider here would bill the caller twice.
        _log.warning(
            "optio_optimize: could not wrap a live stream (%s); returning it unwrapped",
            type(exc).__name__,
        )
        return stream


async def _astream(
    original: Callable[..., Awaitable[Any]],
    optimizer: Optimizer,
    kwargs: dict[str, Any],
) -> Any:
    """Optimize and return an asynchronous stream."""
    from optio_optimize.adapters.anthropic_streaming import (
        AsyncReplayStream,
        AsyncStreamProxy,
        replay_events,
    )

    outcome = _outgoing_stream_kwargs(optimizer, kwargs)
    if outcome is None:
        return await original(**kwargs)
    if isinstance(outcome, LLMResponse):
        return AsyncReplayStream(replay_events(outcome, kwargs))

    prepared, outgoing = outcome
    stream = await original(**outgoing)
    try:
        return AsyncStreamProxy(stream, optimizer.pipeline, prepared)
    except Exception as exc:  # noqa: BLE001 - the call already happened
        _log.warning(
            "optio_optimize: could not wrap a live stream (%s); returning it unwrapped",
            type(exc).__name__,
        )
        return stream


def _translate(kwargs: dict[str, Any]) -> LLMRequest | None:
    """Build an :class:`LLMRequest`, or ``None`` to fall back untouched.

    Returning ``None`` rather than raising is the fail-open path (ADR-013 rule
    1). Translation runs *before* any provider call, so falling back here
    cannot double-bill -- unlike a failure after a real call already happened.
    """
    try:
        return _request_from_kwargs(kwargs)
    except Exception as exc:  # noqa: BLE001 - must never break the caller
        # Type only, never the message: an exception payload can carry prompt
        # content, and §10's rule outlives the package boundary.
        _log.warning(
            "optio_optimize: could not translate the request (%s); "
            "calling the provider directly, unoptimized",
            type(exc).__name__,
        )
        return None


def _request_from_kwargs(kwargs: dict[str, Any]) -> LLMRequest:
    """Translate ``messages.create(**kwargs)`` into this package's model.

    Anthropic's ``system`` is a top-level parameter rather than a message role,
    so it is folded into the message list as leading ``system`` messages --
    which is the shape every stage in this package already reasons about, and
    what lets ``PrefixCacheStage`` mark it at all.
    """
    messages: list[Message] = []
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        messages.append(Message(role="system", content=system))
    elif isinstance(system, (list, tuple)):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                messages.append(
                    Message(role="system", content=str(block.get("text", "")), extra={_RAW: block})
                )

    for param in kwargs.get("messages") or ():
        messages.append(_message_from_param(param))

    max_tokens = kwargs.get("max_tokens")
    temperature = kwargs.get("temperature")
    return LLMRequest(
        model=str(kwargs.get("model") or ""),
        messages=tuple(messages),
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        tools=tuple(kwargs.get("tools") or ()),
        temperature=float(temperature) if isinstance(temperature, (int, float)) else None,
        stop=tuple(kwargs.get("stop_sequences") or ()),
        thinking_budget=_budget_from_thinking(kwargs.get("thinking")),
    )


def _budget_from_thinking(thinking: Any) -> int | None:
    """Read Anthropic's ``thinking`` parameter as a token budget.

    Read as well as written, so a stage can *lower* what the caller asked for.
    Without this the request would carry ``None``, a stage would have nothing to
    reduce, and ADR-018's "only ever lower a budget the caller set" would have no
    budget to work from.

    ``None`` for a disabled or absent config: ``{"type": "disabled"}`` is not a
    budget of zero, it is the absence of one, and turning it into ``0`` would let
    a request round-trip into an *enabled* zero-token config -- a different
    instruction to the provider than the one the caller gave.
    """
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return None
    budget = thinking.get("budget_tokens")
    return budget if isinstance(budget, int) else None


def _message_from_param(param: Any) -> Message:
    """Translate one Anthropic ``MessageParam`` into a :class:`Message`.

    Only ``role`` and text ``content`` are modelled; multimodal content blocks,
    ``tool_use``/``tool_result`` blocks and provider extensions ride through in
    ``extra[_RAW]`` and are restored verbatim unless a stage edits the text.

    The text is *derived* from block content rather than blanked. An earlier
    version set it to ``""`` for any list, which made
    :func:`_param_from_message`'s "did a stage change this?" comparison
    (``raw["content"] == message.content``) compare a list against ``""`` --
    never equal, so every block-shaped message was rebuilt with empty content.
    A ``tool_result`` turn went out as ``{"role": "user", "content": ""}``,
    which is every tool-using Anthropic conversation, the exact composition
    this module's docstring advertises.
    """
    if not isinstance(param, dict):
        return Message(role="user", content=str(param))
    return Message(
        role=param.get("role", "user"),
        content=_text_from_content(param.get("content")),
        extra={_RAW: param},
    )


#: Both shared with the cache key since ADR-022. A block wrongly judged *text*
#: contributes nothing to that key, so a second, subtly different reading of the
#: same wire shape is exactly the divergence :mod:`~optio_optimize.wire` exists
#: to prevent -- and here it would be silent and would serve a wrong answer.
_is_text_block = wire.is_text_block
_block_text = wire.block_text


def _text_from_content(content: Any) -> str:
    """Concatenate the text a caller's ``content`` carries.

    Handles all three shapes the SDK accepts: a plain string, a list of block
    dicts, and a list of pydantic block objects echoed back from a previous
    response. Non-text blocks contribute nothing, which is deliberate -- a
    ``tool_use`` block's ``input`` is structured arguments, not prose, and
    concatenating it into the text a stage may rewrite would invite a stage to
    corrupt it.

    Mirrors :func:`~optio_optimize.wire.response_from_anthropic_message`'s own
    text extraction. Both read the same wire shape, and a second, subtly
    different reading of it is how the two would come to disagree.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(b) for b in content if _is_text_block(b))
    return ""


def _rewritten_content(original: Any, text: str) -> Any:
    """Put a stage's new ``text`` back into ``original``'s shape.

    A string, or a list holding exactly one text block, has one unambiguous
    place for it. Anything else does not: with two text blocks there is no way
    to know which one a stage meant to rewrite, and picking would silently
    move the caller's words between blocks.

    There the original content passes through unchanged -- ADR-013 rule 1's
    fail-open applied to a shape rather than to an exception. That does mean a
    stage's edit is dropped, so it is worth being precise about the cost: the
    stages that rewrite non-system text (``cap_tool_results``, ``deduplicate``,
    ``prune_retrieval``) report a token saving, and a dropped edit makes that
    figure optimistic. Multiple text blocks in one message are rare and the
    alternative is corrupting content, but this is the honest reason to prefer
    the single-block path.
    """
    if not isinstance(original, list):
        return text
    indices = [i for i, block in enumerate(original) if _is_text_block(block)]
    if len(indices) != 1:
        return original
    blocks = [_as_block_dict(b) or b for b in original]
    only = blocks[indices[0]]
    if not isinstance(only, dict):
        return original
    only["text"] = text
    return blocks


def _as_block_dict(block: Any) -> dict[str, Any] | None:
    """Return ``block`` as a mutable dict, or ``None`` if it cannot be one.

    Content blocks arrive as dicts from a caller writing params by hand, and as
    pydantic objects when a previous response's ``content`` is appended to the
    history verbatim -- which is what the SDK's own examples do. Both have to be
    editable here, or a marker lands on the first shape and silently misses the
    second.
    """
    if isinstance(block, dict):
        return dict(block)
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump(exclude_none=True)
        except Exception:  # noqa: BLE001 - a foreign object must not break a call
            return None
        if isinstance(dumped, dict):
            return dumped
    return None


def _param_from_message(message: Message) -> Any:
    """Translate a possibly-transformed :class:`Message` back to a param.

    The original param is returned untouched when no stage changed the text,
    preserving content blocks and every other field this package does not
    model. Only when the text differs is a new param built, and even then only
    ``content`` changes.

    **The ``cacheable`` marker is applied here, not by ``wire``.** This function
    exists to preserve fields ``wire`` cannot see -- the caller's original
    param, with its tool_use blocks and provider extensions -- so turns are
    built from the raw param rather than from ``wire.anthropic_system_and_turns``.
    That means the marker has to be re-applied on this path too. The first
    version applied it only in ``wire``, so the stage marked a turn, the ledger
    recorded the work, and the field never reached the wire: the same silent
    class of defect as ``tools`` going unsent, one function over.
    """
    raw = message.extra.get(_RAW)
    if isinstance(raw, dict):
        original = raw.get("content")
        # Compare against the text *derived* from the original, not against the
        # original itself: for block content those are different types and the
        # comparison could never hold, which is how every tool-using turn came
        # to be rebuilt with empty content.
        if _text_from_content(original) == message.content:
            param = raw
        else:
            param = {**raw, "content": _rewritten_content(original, message.content)}
    else:
        param = {"role": message.role, "content": message.content}

    if message.cacheable:
        param = {
            **param,
            "content": _with_cache_control(
                param.get("content"), message.content, message.cache_ttl
            ),
        }
    return param


def _cache_control(ttl: str | None) -> dict[str, Any]:
    """Build a breakpoint, with a lifetime only if one was chosen.

    ``None`` emits no ``ttl`` key at all rather than an explicit ``"5m"``, and
    the difference is deliberate: absent means this package expressed no
    preference and the provider applies its own default, while a literal ``"5m"``
    is a lifetime this package has taken responsibility for. They bill the same
    today; they are not the same statement, and only one of them stays correct if
    Anthropic ever changes that default (ADR-021).
    """
    control: dict[str, Any] = dict(wire.EPHEMERAL_CACHE_CONTROL)
    if ttl is not None:
        control["ttl"] = ttl
    return control


def _with_cache_control(content: Any, text: str, ttl: str | None = None) -> list[dict[str, Any]]:
    """Return ``content`` as blocks, with a cache breakpoint on the last one.

    Anthropic only accepts ``cache_control`` on a content *block*, so plain
    string content is promoted to a one-element text block. An existing block
    list keeps every block it had and gains the marker on the last, which is
    where a breakpoint belongs: it caches everything above it.

    Blocks are coerced through :func:`_as_block_dict` rather than marked only
    when already dicts. The earlier ``isinstance(last, dict)`` guard fell
    through silently for a pydantic block -- the shape you get by appending a
    response's own ``content`` to the history, as the SDK's examples do -- so
    the stage placed a marker, its note claimed a breakpoint, and nothing
    reached the wire. That is the third appearance of this exact defect in this
    package (``tools`` unsent, then ``cacheable`` unsent from this very path),
    which is why the coercion lives in one named function.
    """
    if isinstance(content, list) and content:
        blocks: list[Any] = [_as_block_dict(b) or b for b in content]
        last = blocks[-1]
        if isinstance(last, dict):
            last["cache_control"] = _cache_control(ttl)
        else:
            # Un-coercible foreign object. Nothing correct is available: marking
            # is impossible and dropping the block would delete content. The
            # marker is lost, the request is intact, and the caller pays the
            # undiscounted rate they would have paid without this package.
            _log.warning(
                "optio_optimize: could not place a cache breakpoint on a %s "
                "content block; the request is unaffected and uncached",
                type(last).__name__,
            )
        return blocks
    return [{"type": "text", "text": text, "cache_control": _cache_control(ttl)}]


def _system_block_from_message(message: Message) -> dict[str, Any]:
    """Rebuild one Anthropic system block, keeping the caller's own fields.

    Built from ``extra[_RAW]`` for the same reason turns are, and the reason is
    sharper here. This path previously used
    :func:`~optio_optimize.wire.anthropic_system_and_turns`, which composes a
    block from text alone -- so a caller who had set their own
    ``cache_control`` with ``ttl: "1h"`` had it replaced by a bare text block.
    They had paid Anthropic's 2x write premium for a one-hour cache and this
    package silently downgraded them to no cache at all: a cost *regression*
    caused by a cost-reduction library, which is the one outcome it must never
    produce.

    An existing ``cache_control`` is therefore never overwritten. The marker
    :class:`~optio_optimize.stages.caching.PrefixCacheStage` places is a
    default, not an override -- the caller's own breakpoint is better
    information than this package's guess, and it is a wider TTL as often as
    not.
    """
    raw = message.extra.get(_RAW)
    block: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    block.setdefault("type", "text")
    if str(block.get("text", "")) != message.content:
        block["text"] = message.content
    if message.cacheable and "cache_control" not in block:
        block["cache_control"] = _cache_control(message.cache_ttl)
    return block


def _kwargs_from_request(sent: LLMRequest, original: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``create(**kwargs)`` from the request the stages left behind.

    Starts from ``original``, preserving every parameter this package never
    models (``top_p``, ``metadata``, ``extra_headers``, ...), and overrides only
    what a stage could plausibly have changed.
    """
    system_blocks = [_system_block_from_message(m) for m in sent.messages if m.role == "system"]
    kwargs = dict(original)
    kwargs["model"] = sent.model
    kwargs["messages"] = [_param_from_message(m) for m in sent.messages if m.role != "system"]
    if system_blocks:
        kwargs["system"] = system_blocks
    elif "system" in kwargs:
        del kwargs["system"]
    if sent.max_tokens is not None:
        kwargs["max_tokens"] = sent.max_tokens
    tools = wire.anthropic_tools(sent)
    if tools is not None:
        kwargs["tools"] = tools
    if sent.stop:
        kwargs["stop_sequences"] = list(sent.stop)
    # Overrides the caller's own `thinking`, unlike `cache_control` on a system
    # block, which is left alone. The two look similar and are not: a breakpoint
    # is the caller's better information about their own reuse pattern, while a
    # budget on the request is a value a stage has already decided to lower --
    # and ADR-018 lets it only ever lower. Preserving the original here would
    # make the stage report a saving the provider never granted, which is the
    # defect class this adapter has now hit three times.
    #
    # Note this reads `sent`, not `original`: a caller-set budget arrives here
    # via `_request_from_kwargs` and comes back out unchanged when no stage
    # touched it, so the round trip is lossless without a second branch.
    if sent.thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": sent.thinking_budget}
    return kwargs


def _with_native(response: LLMResponse, native: Any) -> LLMResponse:
    """Attach the SDK's own object so it can be handed back verbatim."""
    from dataclasses import replace

    return replace(response, extra={**response.extra, _NATIVE: native})


def _unwrap(response: LLMResponse, kwargs: dict[str, Any]) -> Any:
    """Return the provider's own object, or a fresh one for a cache hit.

    ``served_from`` is the gate rather than "is a native object available".
    ``ExactCacheStage`` stores the prior response and ``dataclasses.replace``
    preserves ``extra``, so a cache hit *does* carry a native object -- the
    original call's, usage numbers included. Returning it would re-bill a call
    that cost nothing, on every hit. The OpenAI adapter documents the same
    defect, found by its own test suite.
    """
    native = response.extra.get(_NATIVE)
    if response.served_from is None and native is not None:
        return native
    return _message_from_response(response, kwargs)


def _message_from_response(response: LLMResponse, kwargs: dict[str, Any]) -> Any:
    """Build a valid SDK ``Message`` for a response with no real call behind it."""
    from anthropic.types import Message as AnthropicMessage

    return AnthropicMessage.model_validate(
        {
            "id": f"optio-optimize-{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": response.model or kwargs.get("model") or "",
            "content": [{"type": "text", "text": response.content}],
            "stop_reason": response.finish_reason or "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        }
    )
