"""OpenAI Agents SDK adapter: wraps an ``AsyncOpenAI`` client's chat calls.

**Why the client, not the SDK's ``Model`` protocol.** ``agents.Model.get_response``
takes and returns OpenAI *Responses*-API shapes -- a complex, evolving union
of typed items (function calls, reasoning blocks, computer-use actions) that
this package's ``LLMRequest``/``LLMResponse`` were never designed to
represent. Reimplementing that surface to satisfy the ``Model`` ABC would be
a second translation layer with its own defect surface, for API shapes this
package has no other use for.

The SDK's own ``OpenAIChatCompletionsModel`` already does the harder half of
that translation, and at the one place it actually talks to a provider it
calls exactly ``openai_client.chat.completions.create(**kwargs)`` -- Chat
Completions shape, the same one ``bench/providers.py``'s ``OpenAIProvider``
already speaks. So this adapter wraps the client instead: it intercepts
``chat.completions.create``, which is the SDK's own, real extension point
for a Chat-Completions-backed model, not a shape this package invented. That
also means it composes with anything else built on an ``AsyncOpenAI``
client, the Agents SDK included but not required to import.

**Why async.** ``Model.get_response`` is ``async def`` and abstract, with no
synchronous alternative -- the reason ``Optimizer.acall``/``Pipeline.aexecute``
exist at all (see ``pipeline.py``).

**Streaming is not optimized.** A stage pipeline built around one request
producing one response has nothing meaningful to do with a token stream, and
trying would mean buffering it -- defeating the reason to stream in the
first place. A ``stream=True`` call bypasses this wrapper entirely and goes
straight to the real client, unmodified.

**Double-counting, if ``emit_spans`` is on.** If you already have GenAI OTel
instrumentation capturing spans for the OpenAI SDK (an
``opentelemetry-instrumentation-openai``-style package, or the Agents SDK's
own OTel bridge), do not also enable ``emit_spans`` on the ``Optimizer`` this
wraps -- both would emit `gen_ai.usage.*` for calls this wrapper handles, and
optio's cost lane sums whatever it sees per run. Recorded as unsolved in
ADR-014's Consequences rather than solved implicitly here: detecting another
instrumentor at runtime is fragile, so the tradeoff is made legible instead
(ADR-013's own stated preference), not chosen automatically.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion

_log = logging.getLogger("optio_optimize")

#: Scratch key under which the original message param -- everything this
#: package does not model (multimodal content, ``cache_control``, provider
#: extensions) -- rides through untouched unless a stage actually changed
#: the text.
_RAW = "_raw"

#: Scratch key on a response for the real ChatCompletion, when one exists.
#: Returned verbatim rather than reconstructed, to preserve every field this
#: package does not model (id, system_fingerprint, service_tier, ...).
_NATIVE = "_native"


def wrap_openai_client(
    client: AsyncOpenAI,
    optimizer: Optimizer | None = None,
    **overrides: Any,
) -> AsyncOpenAI:
    """Route ``client.chat.completions.create`` through an ``Optimizer``.

    Mutates and returns the same client (matching the identity contract
    ``optio``'s own adapters use, §6.7): callers keep the object they built,
    with one method replaced.

    Args:
        client: The client to wrap. Non-streaming ``create`` calls on it are
            optimized from this point on; streaming calls pass through
            untouched.
        optimizer: Optimizer to route through. Built from ``overrides`` when
            omitted, so ``wrap_openai_client(client, exact_cache=False)``
            works without constructing one by hand.
        **overrides: Individual ``OptimizeConfig`` fields, e.g.
            ``trim_history=False``. Invalid together with ``optimizer``, for
            the same reason ``Optimizer.__init__`` rejects the combination.

    Returns:
        ``client``, with ``chat.completions.create`` replaced.
    """
    active = optimizer if optimizer is not None else Optimizer(**overrides)
    original_create = cast(
        "Callable[..., Awaitable[ChatCompletion]]", client.chat.completions.create
    )

    async def optimized_create(**kwargs: Any) -> ChatCompletion:
        if kwargs.get("stream"):
            # Nothing here can act on a token stream without buffering it,
            # which defeats the point of asking for one.
            return await original_create(**kwargs)

        try:
            request = _request_from_kwargs(kwargs)
        except Exception as exc:  # noqa: BLE001 - must never break the caller
            # Translation runs before any provider call, so falling back
            # here costs nothing extra -- unlike a failure after a real call
            # already happened, this can't double-bill anyone. Pipeline's own
            # fail-open guarantee covers every stage; it does not cover this
            # wrapper's own kwargs-to-LLMRequest step, which runs outside it.
            _log.warning(
                "optio_optimize: could not translate the request (%s); "
                "calling the provider directly, unoptimized",
                type(exc).__name__,
            )
            return await original_create(**kwargs)

        async def call_provider(sent: LLMRequest) -> LLMResponse:
            completion = await original_create(**_kwargs_from_request(sent, kwargs))
            return _response_from_completion(completion)

        response = await active.acall(request, call_provider)
        native = response.extra.get(_NATIVE)
        if response.served_from is None and native is not None:
            # A real call happened on *this* request (possibly transformed);
            # hand back exactly what the provider returned rather than a
            # reconstruction that would drop fields this package never
            # modeled (system_fingerprint, service_tier, moderation, ...).
            return cast("ChatCompletion", native)
        # served_from is set: some stage served this without calling the
        # provider *this time* -- exact_cache stores the prior real
        # ChatCompletion in extra[_NATIVE] via dataclasses.replace, which
        # preserves every field it doesn't explicitly zero, `extra` included.
        # Returning that object here would hand back its original non-zero
        # `usage` for a call that cost nothing, silently double-billing every
        # cache hit in anyone's eyes but LLMResponse's own zeroed fields.
        # Reconstructed fresh instead, from the numbers optio_optimize
        # actually reports.
        return _completion_from_response(response, kwargs)

    # The real method is a complex @overload (sync/async, stream/non-stream);
    # replacing it with one concrete async signature is exactly what this
    # wrapper needs to do, so both codes mypy raises for it are expected.
    client.chat.completions.create = optimized_create  # type: ignore[method-assign,assignment]
    return client


def _numeric_or_none(value: object) -> float | None:
    """Coerce an OpenAI-SDK-style optional numeric field to a real number.

    The Agents SDK (and the ``openai`` SDK itself) represents "the caller
    never set this" as a falsy sentinel object (``openai.Omit`` /
    ``NotGiven``), not ``None`` -- ``OpenAIChatCompletionsModel`` builds every
    optional field with exactly this pattern. A bare ``kwargs.get(...)``
    therefore hands this package a live sentinel instead of ``None``, and
    every ``is None`` check downstream silently treats "unset" as "set to a
    mystery object" -- ``LLMRequest.is_deterministic`` reads it as non-zero
    (harmless, since ``None`` would too), but ``AdaptiveMaxTokensStage``'s
    "the caller already capped this" check reads it as *present*, and
    declines a stage that should have run. Found by driving this adapter
    through the real ``agents.OpenAIChatCompletionsModel``, whose default
    ``ModelSettings()`` never sets these explicitly -- no hand-built test
    kwargs would have hit this.

    ``bool`` is excluded even though it is a real ``int`` subclass, for the
    same reason ``optio.lanes.cost.lane._int_attribute`` excludes it: never a
    meaningful temperature or token count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _dict_or_none(value: object) -> dict[str, Any] | None:
    """Same normalization as :func:`_numeric_or_none`, for ``response_format``."""
    return value if isinstance(value, dict) else None


def _request_from_kwargs(kwargs: dict[str, Any]) -> LLMRequest:
    """Translate ``create(**kwargs)`` into this package's request model."""
    messages = tuple(_message_from_param(m) for m in kwargs.get("messages") or ())
    max_tokens = _numeric_or_none(kwargs.get("max_completion_tokens"))
    if max_tokens is None:
        max_tokens = _numeric_or_none(kwargs.get("max_tokens"))
    return LLMRequest(
        model=kwargs.get("model") or "",
        messages=messages,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        tools=tuple(kwargs.get("tools") or ()),
        temperature=_numeric_or_none(kwargs.get("temperature")),
        response_format=_dict_or_none(kwargs.get("response_format")),
    )


def _message_from_param(param: dict[str, Any]) -> Message:
    """Translate one ``ChatCompletionMessageParam`` into a :class:`Message`.

    Only ``role``/``content``/``name`` are modeled; everything else --
    ``tool_calls``, ``tool_call_id``, multimodal content parts, provider
    extensions -- rides through in ``extra[_RAW]`` and is restored verbatim
    by :func:`_param_from_message` unless a stage actually edited the text.
    """
    content = param.get("content")
    text = content if isinstance(content, str) else ""
    return Message(
        role=param.get("role", "user"),
        content=text,
        name=param.get("name"),
        extra={_RAW: param},
    )


def _param_from_message(message: Message) -> dict[str, Any]:
    """Translate a (possibly stage-transformed) :class:`Message` back.

    The raw original param is returned untouched when no stage changed the
    text -- preserving tool_calls, multimodal content, and every other field
    this package does not model. Only when the text differs is a new param
    built, and even then only ``content`` changes; everything else from the
    original survives.
    """
    raw = message.extra.get(_RAW)
    if isinstance(raw, dict):
        if raw.get("content") == message.content:
            return raw
        return {**raw, "content": message.content}
    # A message this package's own stages created (StructuredOutputStage
    # inserting a fresh system message, when the caller had none) has no
    # raw param to start from.
    param: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        param["name"] = message.name
    return param


def _kwargs_from_request(sent: LLMRequest, original: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``create(**kwargs)`` from the request stages left behind.

    Starts from ``original`` -- preserving every parameter this package
    never models (``extra_headers``, ``seed``, ``top_p``, ...) -- and
    overrides only what a stage could plausibly have changed.
    """
    kwargs = dict(original)
    kwargs["model"] = sent.model
    kwargs["messages"] = [_param_from_message(m) for m in sent.messages]
    if sent.max_tokens is not None:
        if "max_completion_tokens" in original:
            kwargs["max_completion_tokens"] = sent.max_tokens
        else:
            kwargs["max_tokens"] = sent.max_tokens
    if sent.response_format is not None:
        kwargs["response_format"] = sent.response_format
    return kwargs


def _response_from_completion(completion: ChatCompletion) -> LLMResponse:
    """Translate a real ``ChatCompletion`` into this package's response model.

    Deliberately unable to raise on a structurally unusual but real response
    (e.g. zero choices): this runs *after* a real call already succeeded, so
    the only fail-open option left that doesn't risk a redundant billed call
    is to degrade gracefully here rather than needing a recovery path above.
    """
    choice = completion.choices[0] if completion.choices else None
    usage = completion.usage
    cached = 0
    if usage is not None and usage.prompt_tokens_details is not None:
        cached = usage.prompt_tokens_details.cached_tokens or 0

    return LLMResponse(
        content=(choice.message.content or "") if choice is not None else "",
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        cached_input_tokens=cached,
        model=completion.model,
        finish_reason=choice.finish_reason if choice is not None else None,
        extra={_NATIVE: completion},
    )


def _completion_from_response(response: LLMResponse, kwargs: dict[str, Any]) -> ChatCompletion:
    """Build a valid ``ChatCompletion`` for a response with no real call behind it.

    Only reached for a cache hit: :func:`wrap_openai_client` returns the real
    object directly whenever one exists.
    """
    from openai.types.chat import ChatCompletion as _ChatCompletion
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    return _ChatCompletion(
        id=f"optio-optimize-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=response.model or kwargs.get("model") or "",
        choices=[
            Choice(
                index=0,
                finish_reason=response.finish_reason or "stop",  # type: ignore[arg-type]
                message=ChatCompletionMessage(role="assistant", content=response.content),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            total_tokens=response.input_tokens + response.output_tokens,
        ),
    )
