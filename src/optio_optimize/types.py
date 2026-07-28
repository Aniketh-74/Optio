"""The provider-agnostic request and response model.

Every optimization in this package is a transformation on these types. Keeping
them small and provider-neutral is what lets a stage like history trimming be
written once rather than four times, and what keeps the stages testable without
a network or an API key.

The model is deliberately *lossy about provider specifics*: it carries the
fields every optimization needs to reason about (messages, model, token
ceiling, tools) and passes everything else through untouched in ``extra``. A
stage that does not understand a provider's field cannot corrupt it, because it
never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

#: Message roles we distinguish. ``system`` is separated because it is the
#: highest-value target for prefix caching (stable across every call in a run)
#: and the one part of a prompt that must never be trimmed.
Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation.

    Attributes:
        role: Who produced it.
        content: The text. Kept as a plain string: multimodal parts are held in
            ``extra`` so that text-oriented stages cannot silently drop an image.
        name: Optional participant or tool name.
        cacheable: Whether a provider prefix-cache marker may be placed after
            this message. Set by the prefix stage, read by provider adapters.
        extra: Provider-specific fields, passed through untouched.
    """

    role: Role
    content: str
    name: str | None = None
    cacheable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def with_content(self, content: str) -> Message:
        """Return a copy carrying different text.

        Args:
            content: Replacement text.

        Returns:
            A new message; the original is unchanged.
        """
        return replace(self, content=content)


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A normalized model call, before it reaches the provider.

    Attributes:
        model: Requested model identifier.
        messages: The conversation, oldest first.
        max_tokens: Output ceiling, if the caller set one.
        tools: Tool/function schemas, in the provider's own shape.
        temperature: Sampling temperature; ``0`` makes exact caching sound.
        response_format: Structured-output schema, if any.
        extra: Everything else, passed through untouched.
    """

    model: str
    messages: tuple[Message, ...]
    max_tokens: int | None = None
    tools: tuple[dict[str, Any], ...] = ()
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def with_messages(self, messages: tuple[Message, ...]) -> LLMRequest:
        """Return a copy carrying a different conversation."""
        return replace(self, messages=messages)

    @property
    def is_deterministic(self) -> bool:
        """Whether this request should return the same output every time.

        Only ``temperature == 0`` qualifies. Exact caching a sampled request
        would replace variety the caller explicitly asked for with a frozen
        answer -- cheaper, and wrong in a way that is very hard to notice.
        ``None`` does not qualify: the provider default is not zero.
        """
        return self.temperature == 0.0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A model reply, normalized.

    Attributes:
        content: The text produced.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        cached_input_tokens: Prompt tokens served from the *provider's* prefix
            cache. Billed at a discount rather than free, which is why they are
            counted separately from tokens we avoided sending at all.
        model: The model that actually served the request, which is not
            necessarily the one requested -- the routing stage may have changed
            it, and reporting the requested model would make routing savings
            invisible.
        finish_reason: Why generation stopped.
        served_from: Which stage produced this without calling the provider,
            or ``None`` for a real call.
        extra: Provider-specific fields.
    """

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    model: str = ""
    finish_reason: str | None = None
    served_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def billable_input_tokens(self) -> int:
        """Prompt tokens charged at full rate."""
        return max(0, self.input_tokens - self.cached_input_tokens)
