"""Providers for benchmarking: real APIs, and a simulator for everything else.

Real calls are the authoritative measurement — only a live model can tell you
what an optimization did to output length, end-to-end latency, or answer
quality. They also cost money and need credentials, so this module is built
around two rules:

**We never handle the key.** Adapters read the provider SDK's own environment
variable and pass nothing. This library does not accept, store, log, or forward
a credential, matching §10's position in the core.

**Nothing spends money without being asked twice.** A live run requires an
explicit flag *and* passes a spend estimate to a guard that aborts above a cap.
A benchmark that quietly bills someone is a far worse bug than a slow one.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from optio_optimize import wire
from optio_optimize.config import pricing_for
from optio_optimize.tokens import TokenCounter, count_request, default_counter
from optio_optimize.types import LLMResponse

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest

#: Prefix length below which automatic caching does not engage. Confirmed
#: twice live against OpenAI, most recently 2026-07-29 (a 1422-token prompt
#: reported 0 cached on the first call, 1408 on the first repeat) -- the
#: floor is real. The exact reported value both times was a multiple of
#: :data:`_AUTO_CACHE_QUANTUM_TOKENS`, not an arbitrary number past the floor.
_AUTO_CACHE_FLOOR_TOKENS = 1024

#: Granularity of OpenAI's automatic cache. Confirmed live 2026-07-29 across
#: an 8-call growing conversation: reported ``cached_tokens`` moved in exact
#: multiples of 128 (0 -> 1408 -> plateau -> 1536 -> plateau), never landing
#: between them, even though the prompt itself grew by an uneven number of
#: tokens every call. An earlier trace (2026-07-28, docs/optimize-benchmarks.md)
#: recorded a 256-token jump between two calls -- consistent with crossing two
#: 128-token boundaries in one step, not a different quantum; not re-verified
#: closely enough at the time to tell the difference, which is why this
#: comment gives the date and the method rather than just the number.
_AUTO_CACHE_QUANTUM_TOKENS = 128

#: Default ceiling for a live run. Deliberately small: the suite should be
#: cheap enough to run often, and a benchmark nobody runs proves nothing.
DEFAULT_SPEND_CAP_USD = 1.00


class BenchProvider(Protocol):
    """Something that answers a request and reports what it cost."""

    @property
    def is_live(self) -> bool:
        """Whether this reaches a real model."""
        ...

    @property
    def models_latency(self) -> bool:
        """Whether a call takes a realistic amount of time.

        Declared rather than measured. Our own tokenizer runs inside the timed
        region of a benchmark, and on a large prompt it costs enough to look
        like network latency -- so the clock cannot distinguish "a provider
        answered slowly" from "we did expensive work". Only the provider knows.
        """
        ...

    @property
    def model(self) -> str:
        """The model actually served. Costs must be priced against this.

        Not the model the caller asked for: a benchmark that calls
        ``gpt-4o-mini`` and prices it as ``gpt-4o`` overstates spend by about
        17x, which is the difference between "this optimization is worth it"
        and "this optimization is essential".
        """
        ...

    @property
    def label(self) -> str:
        """Identifier for reports."""
        ...

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Answer one request."""
        ...

    def reset(self) -> None:
        """Clear any per-run state before an A/B arm begins.

        Both arms share one provider, so cache state left by the first leaks
        into the second and credits this library with hits the baseline
        actually earned. A live provider cannot honour this -- its cache is
        server-side -- which is a real limit on live A/B accuracy and is
        stated in the report rather than papered over.
        """
        ...


class SpendGuard:
    """Aborts a live run before it exceeds a cap.

    Checks *before* each call using an estimate, and tracks actual spend after.
    Estimating first is the point: discovering the overspend afterwards is not
    a guard, it is a receipt.
    """

    def __init__(self, cap_usd: float = DEFAULT_SPEND_CAP_USD) -> None:
        """Build a guard.

        Args:
            cap_usd: Maximum to spend across the run.

        Raises:
            ValueError: If the cap is not positive.
        """
        if cap_usd <= 0:
            raise ValueError(f"cap_usd must be positive, got {cap_usd}")
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0

    def check(self, estimated_usd: float) -> None:
        """Raise if this call would breach the cap.

        Raises:
            RuntimeError: When the projected total exceeds the cap.
        """
        if self.spent_usd + estimated_usd > self.cap_usd:
            raise RuntimeError(
                f"spend cap reached: ${self.spent_usd:.4f} spent, this call is "
                f"~${estimated_usd:.4f}, cap is ${self.cap_usd:.2f}. "
                "Raise cap_usd or shrink the workload."
            )

    def record(self, actual_usd: float) -> None:
        """Add a completed call's real cost."""
        self.spent_usd += actual_usd
        self.calls += 1


@dataclass(slots=True)
class SimulatedProvider:
    """A deterministic stand-in that bills honestly for what it is sent.

    Its one job that matters: **count the request it actually receives.** That
    is what makes token-reduction measurable without a live model -- a stage
    that shrinks the prompt shows up here as a smaller bill, exactly as it would
    at a real provider.

    What it cannot do is tell you anything real about completion length,
    latency, or answer quality. It is honest about that by construction:
    :attr:`is_live` is ``False``, and the metrics layer refuses to report those
    figures for a non-live arm rather than printing a fabricated one.

    Attributes:
        output_tokens: Completion length to report. Fixed, so both arms are
            comparable, and meaningless as a measurement.
        latency_ms: Artificial delay per call. Zero by default to keep the
            suite fast; set it to model a real provider's round trip when
            comparing throughput.
        model: Model name used for pricing.
    """

    output_tokens: int = 120
    latency_ms: float = 0.0
    model: str = "gpt-4o"
    prefix_cache_style: str = "automatic"
    _seen_prefixes: set[str] = field(default_factory=set)

    @property
    def is_live(self) -> bool:
        """Never live."""
        return False

    @property
    def models_latency(self) -> bool:
        """True only when a deliberate delay was configured."""
        return self.latency_ms > 0

    @property
    def label(self) -> str:
        """Identifier for reports."""
        return f"simulated({self.model})"

    def reset(self) -> None:
        """Forget every cached prefix, so each arm starts cold."""
        self._seen_prefixes.clear()

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Answer a request, billing for its real size."""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        counter = default_counter()
        input_tokens = count_request(request, counter)
        cached = self._prefix_cache_hit(request, counter)

        # Deterministic content derived from the request, so the quality
        # comparison is meaningful: identical requests produce identical
        # answers, and a stage that changed the prompt shows up as a change.
        digest = hashlib.blake2b(
            "|".join(m.content for m in request.messages).encode("utf-8", "replace"),
            digest_size=8,
        ).hexdigest()
        return LLMResponse(
            content=f"simulated-answer-{digest}",
            input_tokens=input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=cached,
            model=request.model,
            finish_reason="stop",
        )

    def _prefix_cache_hit(self, request: LLMRequest, counter: TokenCounter) -> int:
        """Model provider-side prefix caching.

        Vendors do this two incompatible ways, and conflating them produced the
        worst measurement error in this project's history:

        ``automatic`` (OpenAI, and the default here)
            The provider caches any prompt prefix over ~1024 tokens with no
            cooperation from the caller. Measured live: 1280 of 1401 tokens
            served from cache on the *second* identical call, with no marker
            sent. **Both A/B arms get this**, so the incremental value of our
            prefix stage on such a provider is zero.

        ``explicit`` (Anthropic)
            Nothing is cached unless the request carries a ``cache_control``
            breakpoint. Our marker is what puts it there, so the stage is the
            difference between a 90% discount and none.

        The first version modelled only the explicit style, and therefore
        credited our library with a discount OpenAI grants unconditionally --
        reporting a 36.3% saving on ``multi_turn_chat`` that the live run showed
        to be **-1.8%**. Attributing a provider feature to your own library is
        the single easiest way to publish a benchmark that collapses on contact
        with reality, and it took a real API call to catch.
        """
        if self.prefix_cache_style == "automatic":
            return self._automatic_prefix_hit(request, counter)

        boundary = 0
        for index, message in enumerate(request.messages):
            if message.cacheable:
                boundary = index + 1
                break
        if boundary == 0:
            return 0

        prefix_text = "|".join(m.content for m in request.messages[:boundary])

        # Longest *common* prefix, not exact match on the marked boundary.
        # Vendors cache by token prefix, so a conversation that grows still hits
        # on everything it shares with the previous call. Exact-matching the
        # marked text made every turn of a growing chat a miss and reported
        # zero benefit from a feature that works -- a simulator bug that would
        # have been read as a stage bug.
        best = max(
            (seen for seen in self._seen_prefixes if prefix_text.startswith(seen)),
            key=len,
            default="",
        )
        self._seen_prefixes.add(prefix_text)
        if not best:
            return 0  # First call writes the cache; it does not read it.
        return counter.count_text(best, request.model)

    def _automatic_prefix_hit(self, request: LLMRequest, counter: TokenCounter) -> int:
        """Model OpenAI-style caching, which needs no marker at all.

        Applies to every request regardless of what our stages did, which is the
        whole point: it lands on the baseline arm too, so the A/B difference
        reflects only what this library contributed.
        """
        # Longest previously-seen prefix of the full message text. No marker is
        # consulted -- the provider does not need one.
        full = "|".join(m.content for m in request.messages)
        best = max(
            (seen for seen in self._seen_prefixes if full.startswith(seen)),
            key=len,
            default="",
        )
        # Vendors cache at a coarse granularity; the observed floor is 1024
        # tokens. Below the floor, nothing.
        for boundary in range(len(request.messages), 0, -1):
            candidate = "|".join(m.content for m in request.messages[:boundary])
            if counter.count_text(candidate, request.model) >= _AUTO_CACHE_FLOOR_TOKENS:
                self._seen_prefixes.add(candidate)
                break

        if not best:
            return 0
        cached = counter.count_text(best, request.model)
        if cached < _AUTO_CACHE_FLOOR_TOKENS:
            return 0
        # Round down to the cache quantum: the message boundary that crosses
        # the floor lands at an arbitrary token count, but the provider only
        # ever reports a multiple of _AUTO_CACHE_QUANTUM_TOKENS (confirmed
        # live, see the constant's docstring) -- reporting the raw boundary
        # count would overstate the discount by up to one quantum on every
        # call, small per-call but compounding across a whole benchmark.
        return (cached // _AUTO_CACHE_QUANTUM_TOKENS) * _AUTO_CACHE_QUANTUM_TOKENS


class OpenAIProvider:
    """Real OpenAI chat completions.

    Reads ``OPENAI_API_KEY`` from the environment via the SDK's own mechanism.
    This class never sees the value.
    """

    def __init__(self, model: str = "gpt-4o-mini", guard: SpendGuard | None = None) -> None:
        """Build the adapter.

        Args:
            model: Model to call. Defaults to the cheap one -- a benchmark
                should not default to the expensive option.
            guard: Spend guard. One with the default cap when omitted.

        Raises:
            RuntimeError: If the SDK is missing or no key is configured, with
                the specific remedy rather than a generic import error.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the openai package is required for live benchmarking: pip install openai"
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Live benchmarking needs it; the "
                "simulator needs nothing."
            )
        self._client = OpenAI()
        self.model = model
        self.guard = guard if guard is not None else SpendGuard()

    @property
    def is_live(self) -> bool:
        """Always live."""
        return True

    @property
    def models_latency(self) -> bool:
        """A real network round trip always does."""
        return True

    def reset(self) -> None:
        """No-op: the provider's cache lives on their servers, not ours.

        This is why a live A/B slightly favours whichever arm runs second, and
        why the baseline is run first -- so the bias works against the result
        this library wants, not for it.
        """

    @property
    def label(self) -> str:
        """Identifier for reports."""
        return f"openai({self.model})"

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Send the request and normalize the reply."""
        estimated = _estimate_cost(request, self.model)
        self.guard.check(estimated)

        # Built as explicit keyword arguments rather than an unpacked dict. The
        # SDK's `create` is heavily overloaded, and `**dict[str, object]` type-
        # checks only while the package is absent -- so the dict version passed
        # in CI, where openai is not installed, and failed the moment anyone ran
        # a live benchmark locally. NOT_GIVEN is the SDK's own "omit this field"
        # sentinel, which is not the same as passing None.
        from openai.types.chat import ChatCompletionMessageParam

        # Our Message is provider-neutral by design (ADR-013), so it has to be
        # cast to each SDK's own TypedDict at the boundary. The cast is checked
        # by the round trip: a wrong role or key fails the live call loudly.
        #
        # The translation itself lives in optio_optimize.wire, shared with
        # batch submission (ADR-017), so the two paths cannot drift into
        # sending different requests. What stays here is the `create` call
        # itself -- see below on why it is not an unpacked dict.
        messages = wire.openai_messages(request)
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            # `None` rather than the SDK's omit sentinel: the sentinel's name
            # and type have changed across releases (NotGiven, then Omit), and
            # every version documents None as "unset" for these fields. Pinning
            # to the sentinel would make this adapter break on SDK upgrades for
            # no benefit.
            max_completion_tokens=request.max_tokens,
            temperature=request.temperature,
            response_format=cast("Any", request.response_format),
            # Tool schemas are ordinary billed input, and forgetting them here
            # does not fail -- it silently measures the wrong thing. The first
            # version of this method omitted them, so the entire live run of
            # `mcp_agent` sent no tools at all: `minify_tools` reported saving
            # 3,240 tokens while the provider billed byte-identical totals in
            # both arms (76,439 either way), because the field the stage had
            # rewritten was dropped on the floor. That incident is why the
            # translation is shared with the batch path rather than written
            # twice, and why `wire.UNSENT_FIELDS` makes the omissions explicit.
            tools=cast("Any", wire.openai_tools(request)),
            stop=cast("Any", list(request.stop) or None),
        )
        usage = completion.usage
        cached = 0
        if usage is not None and getattr(usage, "prompt_tokens_details", None) is not None:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        message = completion.choices[0].message
        # Surface any proposed tool call so cascade routing can vet it before
        # the agent executes it (ADR-023 step 3). Same ``extra["tool_calls"]``
        # convention the request side uses (see optio_optimize.wire); OpenAI's
        # ``arguments`` is already a JSON string, so it passes through as-is.
        extra: dict[str, Any] = {}
        # Only "function" tool calls carry a ``.function``; the SDK's union also
        # includes a custom-tool variant that does not, so skip anything without
        # one rather than assume the shape.
        function_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": fn.name, "arguments": fn.arguments},
            }
            for tc in (message.tool_calls or [])
            if (fn := getattr(tc, "function", None)) is not None
        ]
        if function_calls:
            extra["tool_calls"] = function_calls

        response = LLMResponse(
            content=message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=cached,
            model=completion.model,
            finish_reason=completion.choices[0].finish_reason,
            extra=extra,
        )
        self.guard.record(_actual_cost(response, self.model))
        return response


class AnthropicProvider:
    """Real Anthropic messages, with prefix-cache markers applied.

    The one provider where this library's ``cacheable`` marker becomes a real
    API field, so it is where the prefix stage can be measured rather than
    assumed.
    """

    #: Default model. **Verified against `models.list` on 2026-07-31**, which is
    #: not pedantry: this was `claude-haiku-4`, and that name returns `404
    #: not_found_error`. So did `claude-sonnet-4` and `claude-opus-4`. Those three
    #: are family keys in `PRICING`, not model IDs the API will serve, and using
    #: one as the default meant no live Anthropic benchmark had ever completed a
    #: single call.
    DEFAULT_MODEL = "claude-haiku-4-5"

    def __init__(self, model: str = DEFAULT_MODEL, guard: SpendGuard | None = None) -> None:
        """Build the adapter.

        Raises:
            RuntimeError: If the SDK is missing or no key is configured.
        """
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the anthropic package is required for live benchmarking: pip install anthropic"
            ) from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Live benchmarking needs it; the "
                "simulator needs nothing."
            )
        self._client = Anthropic()
        self.model = model
        self.guard = guard if guard is not None else SpendGuard()

    @property
    def is_live(self) -> bool:
        """Always live."""
        return True

    @property
    def models_latency(self) -> bool:
        """A real network round trip always does."""
        return True

    def reset(self) -> None:
        """No-op: the provider's cache lives on their servers, not ours.

        This is why a live A/B slightly favours whichever arm runs second, and
        why the baseline is run first -- so the bias works against the result
        this library wants, not for it.
        """

    @property
    def label(self) -> str:
        """Identifier for reports."""
        return f"anthropic({self.model})"

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Send the request, translating cacheable markers to cache_control."""
        estimated = _estimate_cost(request, self.model)
        self.guard.check(estimated)

        # Our Message is provider-neutral by design (ADR-013), so it has to be
        # cast to the SDK's own TypedDicts at the boundary, the same pattern
        # as OpenAIProvider. Built as plain dicts first because a Message's
        # role is a superset (it includes "tool", which Anthropic has no
        # MessageParam role for) of what either TypedDict accepts. The split
        # itself -- including turning a `cacheable` marker into the provider's
        # own cache_control directive, which is the whole point of the prefix
        # stage -- lives in optio_optimize.wire, shared with batch submission.
        system_blocks, turns = wire.anthropic_system_and_turns(request)

        from anthropic.types import MessageParam

        # `tools` and `stop_sequences` are **omitted when empty, never passed as
        # None**, and the difference is not cosmetic: Anthropic rejects a null
        # with `400 tools: Input should be a valid array`. The first live
        # benchmark run against Anthropic failed all twelve calls on exactly
        # this. `system` really does accept None -- the comment that used to sit
        # here reasoned that out correctly and then applied it one argument too
        # far. `adapters/anthropic.py` had it right all along, which is why every
        # live measurement script worked while this provider had never once
        # succeeded.
        optional: dict[str, Any] = {}
        # `system` too, and this one was asserted otherwise in a comment: "every
        # anthropic version accepts None here as 'no system prompt'". Current
        # Anthropic answers `400 system: Input should be a valid array`. It only
        # surfaced after the `tools` fix, because each rejection hides the next.
        if system_blocks:
            optional["system"] = cast("Any", system_blocks)
        # Anthropic's schema shape differs from OpenAI's -- name and
        # input_schema at the top level rather than nested under "function" --
        # so a request built for one is translated rather than forwarded. Same
        # omission as OpenAIProvider had, and it would fail the same silent
        # way: tools rewritten by a stage, then never sent.
        tools = wire.anthropic_tools(request)
        if tools:
            optional["tools"] = tools
        if request.stop:
            optional["stop_sequences"] = list(request.stop)

        reply = self._client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens or 1024,
            messages=cast("list[MessageParam]", turns),
            temperature=request.temperature if request.temperature is not None else 1.0,
            **optional,
        )
        # Usage comes from the shared extractor, not from a private reading of
        # `reply.usage`. This used to take `cache_read_input_tokens` and stop,
        # so `cache_creation_input_tokens` was never read: written tokens were
        # dropped from `input_tokens` outright, and the cache-write premium in
        # `ABResult.cost_usd` was inert because nothing populated the field it
        # prices. A live Sonnet 4.5 run reported `reads 18,300 writes 0`, which
        # cannot happen -- nothing is read from a cache that was never written.
        #
        # `wire.response_from_anthropic_message` has been correct since ADR-021
        # and its docstring records this exact defect being fixed there; the
        # streaming adapter uses it. The benchmark, whose entire output is
        # savings figures, had kept its own copy.
        response = wire.response_from_anthropic_message(reply)
        # Normalise Anthropic's tool_use blocks to the same ``tool_calls``
        # convention OpenAI uses, so cascade's verifier reads one shape (ADR-023
        # step 3). Anthropic's ``input`` is a dict, not a JSON string, so it is
        # serialised to match ``function.arguments``. Detected by the block's
        # ``type`` discriminator rather than an ``isinstance`` against a class
        # name, so this survives SDK versions that spell or place ToolUseBlock
        # differently (ADR-023 improvement #5); getattr keeps it type-clean.
        extra: dict[str, Any] = {}
        tool_uses = [b for b in reply.content if getattr(b, "type", None) == "tool_use"]
        if tool_uses:
            extra["tool_calls"] = [
                {
                    "id": getattr(b, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(b, "name", None),
                        "arguments": json.dumps(getattr(b, "input", {})),
                    },
                }
                for b in tool_uses
            ]

        if extra:
            response = replace(response, extra=extra)
        self.guard.record(_actual_cost(response, self.model))
        return response


def _estimate_cost(request: LLMRequest, model: str) -> float:
    """Rough pre-call cost, for the spend guard.

    Assumes a generous completion length. Over-estimating is correct here: the
    guard should stop early rather than let the last call be the one that
    breaks the cap.
    """
    pricing = pricing_for(model)
    if pricing is None:
        return 0.0
    input_tokens = count_request(request, default_counter())
    assumed_output = 1024
    return (
        input_tokens * pricing.input_usd_per_m + assumed_output * pricing.output_usd_per_m
    ) / 1_000_000


def _actual_cost(response: LLMResponse, model: str) -> float:
    """Real cost of a completed call."""
    from optio_optimize.savings import _cost

    pricing = pricing_for(model)
    if pricing is None:
        return 0.0
    return _cost(
        pricing, response.input_tokens, response.output_tokens, response.cached_input_tokens
    )


def available_live_provider(
    guard: SpendGuard | None = None,
) -> BenchProvider | None:
    """Return a live provider if one is configured, else ``None``.

    Checks Anthropic first: it is the only provider whose prefix caching this
    library controls explicitly, so it measures strictly more than the others.
    """
    for factory in (AnthropicProvider, OpenAIProvider):
        try:
            return factory(guard=guard)
        except RuntimeError:
            continue
    return None
