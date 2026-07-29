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

**Streaming is not optimized.** A stage pipeline built around one request
producing one response can only buffer a token stream, which defeats the reason
to ask for one. A ``stream=True`` call bypasses this wrapper entirely.
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

_log = logging.getLogger("optio_optimize")

#: Scratch key under which the caller's original message param rides through
#: untouched -- multimodal content blocks, provider extensions, anything this
#: package does not model -- restored verbatim unless a stage changed the text.
_RAW = "_raw"

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
        client: The client to wrap. Non-streaming ``create`` calls are
            optimized from this point on; streaming calls pass through.
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
            return await original(**kwargs)
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
            return original(**kwargs)
        prepared = _translate(kwargs)
        if prepared is None:
            return original(**kwargs)

        def call_provider(sent: LLMRequest) -> LLMResponse:
            reply = original(**_kwargs_from_request(sent, kwargs))
            return _with_native(wire.response_from_anthropic_message(reply), reply)

        return _unwrap(optimizer.call(prepared, call_provider), kwargs)

    return optimized


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
    )


def _message_from_param(param: Any) -> Message:
    """Translate one Anthropic ``MessageParam`` into a :class:`Message`.

    Only ``role`` and text ``content`` are modelled; multimodal content blocks,
    ``tool_use``/``tool_result`` blocks and provider extensions ride through in
    ``extra[_RAW]`` and are restored verbatim unless a stage edits the text.
    """
    if not isinstance(param, dict):
        return Message(role="user", content=str(param))
    content = param.get("content")
    text = content if isinstance(content, str) else ""
    return Message(role=param.get("role", "user"), content=text, extra={_RAW: param})


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
        unchanged = raw.get("content") == message.content
        param = raw if unchanged else {**raw, "content": message.content}
    else:
        param = {"role": message.role, "content": message.content}

    if message.cacheable:
        param = {**param, "content": _with_cache_control(param.get("content"), message.content)}
    return param


def _with_cache_control(content: Any, text: str) -> list[dict[str, Any]]:
    """Return ``content`` as blocks, with a cache breakpoint on the last one.

    Anthropic only accepts ``cache_control`` on a content *block*, so plain
    string content is promoted to a one-element text block. An existing block
    list keeps every block it had and gains the marker on the last, which is
    where a breakpoint belongs: it caches everything above it.
    """
    if isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        last = blocks[-1]
        if isinstance(last, dict):
            last["cache_control"] = dict(wire.EPHEMERAL_CACHE_CONTROL)
        return blocks
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": dict(wire.EPHEMERAL_CACHE_CONTROL),
        }
    ]


def _kwargs_from_request(sent: LLMRequest, original: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``create(**kwargs)`` from the request the stages left behind.

    Starts from ``original``, preserving every parameter this package never
    models (``top_p``, ``metadata``, ``extra_headers``, ...), and overrides only
    what a stage could plausibly have changed.
    """
    system_blocks, _ = wire.anthropic_system_and_turns(sent)
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
