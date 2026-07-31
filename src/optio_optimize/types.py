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
        cache_ttl: Lifetime to request for that marker -- ``"1h"``, or ``None``
            to say nothing and take the provider's five-minute default. Only
            meaningful when ``cacheable`` is set.

            A separate field rather than widening ``cacheable`` to a union,
            which would break every existing reader of it for no gain. ``None``
            rather than a ``"5m"`` default because the two are not the same
            instruction: absent means "this package expressed no preference",
            and a request that says ``5m`` explicitly is one this package has
            taken responsibility for (ADR-021).
        extra: Provider-specific fields, passed through untouched.
    """

    role: Role
    content: str
    name: str | None = None
    cacheable: bool = False
    cache_ttl: str | None = None
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
        stop: Sequences that halt generation. Output tokens are billed as they
            are produced, so a model that keeps writing past the answer is
            billed for every word of it -- a stop sequence is the only
            mechanism that stops the meter mid-completion rather than
            capping it in advance the way ``max_tokens`` does.
        thinking_budget: Ceiling on reasoning tokens, as a **token count**, for
            models that expose one. Separate from ``max_tokens`` because
            reasoning tokens are billed at the output rate while never
            appearing in the output: a model can spend twenty thousand of them
            on a question whose visible answer is two hundred, which no
            ``max_tokens`` value constrains. Reaches Anthropic as
            ``thinking={"type": "enabled", "budget_tokens": N}``.
        reasoning_effort: The same lever as ``thinking_budget``, expressed the
            way OpenAI accepts it: a **category** (``"none"`` through
            ``"max"``) rather than a number.

            Two fields for one idea, because the vendors do not take the same
            kind of thing and no honest conversion exists between them
            (ADR-018 amendment). Whether a 2,000-token budget is ``"low"`` or
            ``"medium"`` depends on the model, so a threshold table would be
            invented -- and this package has twice published an invented
            number that a live run then corrected. Each adapter sends the field
            its provider understands and ignores the other; a caller targeting
            both vendors sets both.
        extra: Everything else, passed through untouched.
    """

    model: str
    messages: tuple[Message, ...]
    max_tokens: int | None = None
    tools: tuple[dict[str, Any], ...] = ()
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    stop: tuple[str, ...] = ()
    thinking_budget: int | None = None
    reasoning_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def with_messages(self, messages: tuple[Message, ...]) -> LLMRequest:
        """Return a copy carrying a different conversation."""
        return replace(self, messages=messages)

    def with_tools(self, tools: tuple[dict[str, Any], ...]) -> LLMRequest:
        """Return a copy carrying a different tool set.

        Args:
            tools: Replacement schemas, in the provider's own shape.

        Returns:
            A new request; the original is unchanged.
        """
        return replace(self, tools=tools)

    @property
    def is_deterministic(self) -> bool:
        """Whether this request should return the same output every time.

        Only ``temperature == 0`` qualifies. Exact caching a sampled request
        would replace variety the caller explicitly asked for with a frozen
        answer -- cheaper, and wrong in a way that is very hard to notice.
        ``None`` does not qualify: the provider default is not zero.
        """
        return self.temperature == 0.0


#: Every spelling a provider uses for "the model ran out of room".
#:
#: OpenAI says ``length``; **Anthropic says ``max_tokens``**; Gemini says
#: ``max_output_tokens``. This package compared against ``"length"`` alone until
#: 2026-07-31, which made its truncation guards dead code on Anthropic -- see
#: :attr:`LLMResponse.was_truncated` and ADR-033 for what that cost.
#:
#: A third vendor is one entry here, in the one place that holds this knowledge,
#: and adding it fixes every caller at once. Carrying an unused member costs
#: nothing; omitting one cost a correctness bug in a lossless default-on stage.
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})


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
        cache_write_tokens: Prompt tokens written *into* the provider's cache on
            this call, included in ``input_tokens``. Anthropic charges these at a
            **premium**, not a discount -- 1.25x the base input rate for the
            5-minute TTL and 2x for the one-hour -- so a report that folds them
            into ordinary input tokens understates what a cached call cost and
            therefore overstates what caching saved. Always ``0`` for providers
            that cache automatically without charging to populate it (OpenAI).
        cache_write_1h_tokens: The **subset** of ``cache_write_tokens`` that
            populated a one-hour entry rather than a five-minute one, billed at
            2x base input instead of 1.25x. A subset rather than a sibling
            because that is how the provider reports it -- the total in
            ``cache_creation_input_tokens``, the breakdown in ``cache_creation``
            -- and adding the two would double-count the most expensive tokens
            in the request (ADR-021).
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
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    model: str = ""
    finish_reason: str | None = None
    served_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def was_truncated(self) -> bool:
        """Whether the model ran out of room before finishing.

        Asks the question rather than comparing a string, because the vendors
        spell the answer differently and comparing to one spelling is how this
        check came to be dead code on Anthropic for every caller.

        The live cascade run that found it carried ``max_tokens=16`` as the
        *guaranteed* escalation -- ``default_verifier`` rejects a truncated
        answer -- and the chopped-off answer was accepted and returned as final,
        because Anthropic reports ``max_tokens`` where OpenAI reports
        ``length``. The same literal guarded ``ExactCacheStage``, which refuses
        to serve a truncated entry precisely because ``request_key`` omits
        ``max_tokens`` so calls with different ceilings share one. Inert, that
        let a ``max_tokens=16`` call poison the entry for every later caller --
        in a stage that is ``Fidelity.IDENTICAL`` and on by default (ADR-033).

        An unrecognised reason reads as *not* truncated. That fails the safe
        way: the alternative would make the exact cache refuse to serve anything
        from a provider whose vocabulary is not listed here, turning a missing
        entry into a silently disabled cache.
        """
        return self.finish_reason in TRUNCATION_REASONS

    @property
    def billable_input_tokens(self) -> int:
        """Prompt tokens not served from the provider's cache.

        Not the same as "charged at the base rate", which is what this said
        before ``cache_write_tokens`` existed: a subset of these are cache
        writes and carry a premium. Use :func:`~optio_optimize.savings._cost`
        for money; this property is a token count.
        """
        return max(0, self.input_tokens - self.cached_input_tokens)
