"""Cascade routing: call cheap, verify, escalate on failure.

The one cost technique that needs **two** provider calls, and the reason it is
not a :class:`~optio_optimize.stages.base.Stage` (ADR-023). A stage's contract is
that it transforms a request and the pipeline makes one provider call around it;
cascade has to make the call itself, look at what came back, and possibly call
again. That is the same wall batch dispatch hit (ADR-017), and the same answer:
the technique lives outside the stage list, wrapped around the provider callable
the caller already passes in.

**What it does.** Static ``route_models`` guesses cheap-vs-expensive *before* the
call from prompt length, and its own docstring records the guess failing --
"What is 17 times 24, minus 89?" gets routed and the cheap model answers 329
against 319. Cascade removes the guess: it sends the request to ``cheap_model``,
runs a verifier over the answer, returns it if the verifier accepts, and
otherwise re-sends the *original* request to its original model. Worst case is
slow-and-right (both calls paid, correct answer) rather than wrong-and-cheap.

**Why the cheap attempt does not touch the cache.** The cheap call is made here,
directly against the wrapped provider, *inside* the wrapper -- it never passes
through the stage pipeline. So a rejected cheap answer is never handed to any
stage's ``after`` hook and therefore never written to ``exact_cache`` under the
request's key, which is the cache conflict the spec flagged (ADR-023 open
decision 2, resolved by not caching rejected attempts at all). Only the accepted
final answer -- cheap-passed or escalated -- is what the pipeline sees and caches.

**Fidelity: ``ALTERED``.** A cheap answer that passes the verifier may still
differ from what the expensive model would have said; the verifier bounds that
risk but does not erase it. Off by default and gated behind ADR-015 live
evidence, like every altered technique.

**Streaming is out of scope.** Cascade wraps a unary provider call: it needs the
whole cheap answer in hand to verify it before deciding whether to escalate, and
a streamed response is produced incrementally with tokens already delivered to
the caller. A streaming cascade would have to buffer the entire cheap stream
(discarding its latency benefit) and could not un-send a rejected partial, so
streamed calls bypass cascade entirely rather than pretend to support it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from optio_optimize.stages.routing import MAX_ROUTABLE_TOKENS, is_routable
from optio_optimize.tokens import default_counter

if TYPE_CHECKING:
    from optio_optimize.config import OptimizeConfig
    from optio_optimize.pipeline import AsyncProviderCall, ProviderCall
    from optio_optimize.tokens import TokenCounter
    from optio_optimize.types import LLMRequest, LLMResponse

_log = logging.getLogger("optio_optimize")

#: A caller-supplied check: given the request and the cheap model's answer,
#: return ``True`` to accept it and ``False`` to escalate to the expensive
#: model. Must not raise for cause -- a raising verifier is treated as
#: "escalate", so an exception here costs the saving, never the answer.
Verifier = Callable[["LLMRequest", "LLMResponse"], bool]


def break_even_escalation_rate(expensive_model: str, cheap_model: str) -> float | None:
    """The escalation rate above which cascade costs more than doing nothing.

    Straight off the rate card. An accepted-cheap request pays ``C`` instead of
    ``E`` and saves ``E - C``; an escalated one pays ``C + E`` instead of ``E``
    and loses ``C``. Setting those equal gives ``r = 1 - C/E``: **66.7%** for
    Haiku 4.5 -> Sonnet 4.5, 80% for Haiku 4.5 -> Opus 5, 60% for Sonnet 5 ->
    Opus 5. The wider the price gap, the more escalation the technique tolerates.

    Computed rather than documented because a threshold a user has to derive by
    hand is one they will not derive -- and cascade's first live run reported a
    50% escalation rate while losing 25%, so the comparison needs to be easy to
    make (ADR-034).

    Uses the input rate. Output bills several times higher and at a ratio that
    differs per model, but the ratio ``C/E`` is what matters here and it is
    close enough between the bands to make a second figure noise rather than
    precision -- and this is a threshold to read a measured rate against, not
    itself a measurement.

    Args:
        expensive_model: The model the request originally targeted.
        cheap_model: The model cascade attempts first.

    Returns:
        The break-even rate as a fraction, or ``None`` when either model is
        unpriced or the "cheap" model is not actually cheaper -- the latter
        being a configuration error rather than a rate.
    """
    from optio_optimize.config import pricing_for

    exp = pricing_for(expensive_model)
    cheap = pricing_for(cheap_model)
    if exp is None or cheap is None or exp.input_usd_per_m <= 0:
        return None
    if cheap.input_usd_per_m >= exp.input_usd_per_m:
        return None
    return 1.0 - cheap.input_usd_per_m / exp.input_usd_per_m


def _demands_json(response_format: dict[str, Any] | None) -> bool:
    """Whether a ``response_format`` asks for JSON this verifier can check."""
    return isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    )


def _required_keys(response_format: dict[str, Any]) -> list[str]:
    """Top-level keys a ``json_schema`` marks required, if any.

    A deliberately shallow read of the schema's ``required`` list -- full JSON
    Schema validation would need a dependency this package does not carry, and
    the top-level required keys catch the common "the model returned JSON but
    dropped a field" failure without pretending to validate the whole shape.
    """
    if response_format.get("type") != "json_schema":
        return []
    schema = response_format.get("json_schema", {})
    inner = schema.get("schema", {}) if isinstance(schema, dict) else {}
    required = inner.get("required") if isinstance(inner, dict) else None
    return [k for k in required if isinstance(k, str)] if isinstance(required, list) else []


def _tool_specs(tools: tuple[dict[str, Any], ...]) -> dict[str, list[str]]:
    """Map each tool's name to the parameters its schema marks required.

    Handles both the OpenAI shape (``{"type": "function", "function":
    {"name": ..., "parameters": {"required": [...]}}}``) and the flat Anthropic
    shape (``{"name": ..., "input_schema": {"required": [...]}}``). A tool with
    no required list maps to ``[]`` -- present, nothing enforced.
    """
    specs: dict[str, list[str]] = {}
    for tool in tools:
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        schema = fn.get("parameters")
        if not isinstance(schema, dict):
            schema = fn.get("input_schema") if isinstance(fn.get("input_schema"), dict) else {}
        required = schema.get("required") if isinstance(schema, dict) else None
        specs[name] = (
            [k for k in required if isinstance(k, str)] if isinstance(required, list) else []
        )
    return specs


def _proposed_calls(response: LLMResponse) -> list[dict[str, Any]]:
    """Tool calls the model proposed, read from the ``tool_calls`` convention.

    The request side carries tool calls in ``message.extra["tool_calls"]``
    (see :mod:`optio_optimize.wire`); a response surfaces a *proposed* call the
    same way. Empty when the provider adapter does not populate it, in which
    case a tool request has no call to vet and falls through to the text checks.
    """
    calls = response.extra.get("tool_calls") if isinstance(response.extra, dict) else None
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def default_verifier(request: LLMRequest, response: LLMResponse) -> bool:
    """The built-in structural verifier: accept unless the answer is visibly broken.

    Deliberately conservative and deterministic. It escalates on the failures a
    cheap deterministic check can be *sure* of and accepts everything else:

    * a truncated answer (:attr:`~optio_optimize.types.LLMResponse.was_truncated`
      -- the model ran out of room and the answer is cut off. Asked as a
      question because the vendors spell it differently, and comparing against
      ``"length"`` alone made this check dead code on Anthropic, ADR-033);
    * for a **tool** request that proposed a call, one that names a tool not in
      the request, whose ``arguments`` are not valid JSON, or that omits a
      parameter the tool's schema marks ``required`` (ADR-023 steps 3 and #4) --
      the cheap model's proposal is vetted before the agent runs it;
    * an empty text answer (only when no tool call was proposed -- a genuine
      tool call legitimately carries empty content); and
    * for a JSON request (``response_format`` of type ``json_object`` or
      ``json_schema``), an answer that does not parse as JSON, or parses but
      drops a top-level key the schema marked ``required`` (ADR-023 step 1).

    **It does not judge whether the answer is correct.** The motivating failure,
    a cheap model confidently answering "17 times 24, minus 89" as 329, is
    fluent, complete, and untruncated, so this verifier accepts it. Catching that
    class needs a model or a domain check, which is what a caller-supplied
    :data:`Verifier` is for (ADR-023 decision 1). This default exists so the
    technique is useful out of the box on the failures structure alone can catch,
    not so it can be left as the whole safety story.

    Args:
        request: The request that produced ``response``; read for its
            ``response_format`` and ``tools`` so conformance can be checked.
        response: The cheap model's answer.

    Returns:
        ``True`` to accept the cheap answer, ``False`` to escalate.
    """
    # `was_truncated` rather than a comparison against "length": Anthropic
    # reports `max_tokens`, so this check -- the verifier's first and most basic
    # -- was dead code against Anthropic and a truncated cheap answer was
    # accepted as final (ADR-033).
    if response.was_truncated:
        return False

    # A tool request that proposed a call: vet the proposal, not the text --
    # a genuine tool call carries empty content, so the empty-text check below
    # must not fire on it.
    if request.tools:
        calls = _proposed_calls(response)
        if calls:
            specs = _tool_specs(request.tools)
            for call in calls:
                fn = call.get("function", call)
                fn = fn if isinstance(fn, dict) else {}
                name = fn.get("name")
                if name not in specs:
                    return False
                args = fn.get("arguments")
                parsed: Any = args if isinstance(args, dict) else None
                if isinstance(args, str) and args.strip():
                    try:
                        parsed = json.loads(args)
                    except (ValueError, TypeError):
                        return False
                required = specs[name]
                if required and (
                    not isinstance(parsed, dict) or any(r not in parsed for r in required)
                ):
                    return False
            return True

    content = response.content
    if not content or not content.strip():
        return False
    fmt = request.response_format
    if _demands_json(fmt):
        assert fmt is not None  # _demands_json is False for None; narrows for mypy
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return False
        required = _required_keys(fmt)
        if required and (not isinstance(parsed, dict) or any(k not in parsed for k in required)):
            return False
    return True


@dataclass
class ModelJudge:
    """A verifier that asks a model whether the cheap answer is adequate.

    Addresses ADR-023 improvement #2: a model-based verifier spends real money,
    and cascade would otherwise credit a saving without subtracting what the
    judgment cost. This judge records the tokens each grading call billed and
    exposes them through :meth:`pop_cost`, which :class:`CascadeRouter` reads and
    folds into ``CascadeStats.verifier_usd`` -- so "cascade saved X" is net of the
    verifier, not gross.

    Attributes:
        provider: The call used to run the judge. Usually a *cheaper* model than
            the one being escalated to -- a judge as expensive as the escalation
            it guards can erase the saving, which is exactly the risk this makes
            visible.
        model: The judge model, used both to make the call and to price it.
        instruction: The grading prompt; the default asks for a PASS/FAIL verdict.
    """

    provider: ProviderCall
    model: str
    instruction: str = (
        "You are a strict grader. Reply with exactly one word: PASS if the candidate answer is "
        "complete, correct, and adequate for the task; otherwise FAIL."
    )
    _last: tuple[int, int] = (0, 0)

    def __call__(self, request: LLMRequest, response: LLMResponse) -> bool:
        """Grade ``response`` against ``request``; ``True`` accepts, ``False`` escalates."""
        from optio_optimize.types import LLMRequest as _Req
        from optio_optimize.types import Message

        task = request.messages[-1].content if request.messages else ""
        candidate = response.content or "(empty)"
        judged = self.provider(
            _Req(
                model=self.model,
                messages=(
                    Message(role="system", content=self.instruction),
                    Message(
                        role="user",
                        content=f"TASK:\n{task}\n\nCANDIDATE:\n{candidate}\n\nVerdict:",
                    ),
                ),
                temperature=0.0,
                max_tokens=4,
            )
        )
        self._last = (judged.input_tokens, judged.output_tokens)
        return (judged.content or "").strip().upper().startswith("PASS")

    def pop_cost(self) -> tuple[int, int, float]:
        """Return and reset (input_tokens, output_tokens, usd) of the last judgment."""
        from optio_optimize.config import pricing_for
        from optio_optimize.savings import _cost

        vin, vout = self._last
        self._last = (0, 0)
        pricing = pricing_for(self.model)
        usd = _cost(pricing, vin, vout, cached=0) if pricing is not None else 0.0
        return vin, vout, usd


@dataclass(frozen=True, slots=True)
class CascadeCost:
    """Cascade's economics, priced. Measured where possible, estimated where not.

    The honest part is the split. ``cheap_spend`` and ``escalation_spend`` are
    *measured* from the tokens the provider actually billed. ``escalation_waste``
    is also measured -- it is the cost of the cheap attempts that were rejected,
    pure overhead cascade added. Only ``all_expensive_baseline`` contains an
    estimate: for a request the cheap model answered, the expensive model was
    never called, so what it *would* have cost is projected from the cheap call's
    token counts. That one figure is a projection; everything else is a receipt.

    Attributes:
        cheap_spend_usd: Every cheap attempt (accepted and rejected). Measured.
        escalation_spend_usd: Every expensive call made on escalation. Measured.
        verifier_spend_usd: What a model-based verifier cost (0 for a free one).
        total_spend_usd: What cascade actually caused to be spent. Measured.
        all_expensive_baseline_usd: What the same requests would have cost sent
            straight to the expensive model. Measured for escalated requests
            (the expensive call happened); *estimated* for accepted-cheap ones.
        net_saving_usd: ``all_expensive_baseline - total_spend``. Positive means
            cascade paid off; negative means it cost more than doing nothing.
        escalation_waste_usd: Cost of the rejected cheap attempts. Measured, and
            the number that was previously invisible (ADR-023 improvement #1).
    """

    @property
    def cost_weighted_escalation_rate(self) -> float | None:
        """Escalated spend over the all-expensive baseline.

        **The rate to read against**
        :func:`break_even_escalation_rate`, and not the same as
        :attr:`CascadeStats.escalation_rate`, which counts requests.

        The first live run escalated 4 of 8 -- 50%, comfortably under a 66.7%
        break-even -- and lost 25.3%, because those four requests were 92% of
        the baseline spend. The correlation is structural rather than an
        artefact of that mix: a request is likelier to fail a verifier when it
        is long, carries tools, or demands a schema, and each of those also
        makes it expensive. So a count-weighted rate flatters cascade on any
        realistic workload (ADR-034).

        ``None`` when nothing has been attempted -- a ratio over no baseline is
        a division, not a measurement.
        """
        if self.all_expensive_baseline_usd <= 0:
            return None
        return self.escalation_spend_usd / self.all_expensive_baseline_usd

    cheap_spend_usd: float
    escalation_spend_usd: float
    verifier_spend_usd: float
    total_spend_usd: float
    all_expensive_baseline_usd: float
    net_saving_usd: float
    escalation_waste_usd: float


@dataclass(slots=True)
class CascadeStats:
    """A running tally of what the cascade did, for the ADR-015 gate.

    The gate cares about more than cost: a cascade that escalates every request
    saves nothing (two calls where one would do), and one that escalates nothing
    is just static routing with an extra call's latency. The escalation rate is
    the number that tells those apart, so it is measured rather than assumed.

    Token and timing fields exist so cascade's cost is never invisible: an
    escalated request really pays for the cheap attempt *plus* the expensive
    one, and that waste is spent at the provider whether or not anything counts
    it. Cascade counts it here (ADR-023 improvement #1), separately from the
    stage :class:`~optio_optimize.savings.SavingsReport`, which correctly
    accounts only for the single final response the pipeline was handed.

    Attributes:
        skipped: Requests not eligible for a cheap attempt (per
            :func:`~optio_optimize.stages.routing.is_routable`); sent straight to
            the requested model, exactly as if the cascade were not installed.
        attempted: Requests a cheap attempt was made for.
        cheap_passed: Attempts whose cheap answer the verifier accepted.
        escalated: Attempts that fell back to the expensive model -- verifier
            rejected the cheap answer, or the cheap call itself failed.
        cheap_input_tokens: Prompt tokens across *all* cheap attempts.
        cheap_output_tokens: Completion tokens across all cheap attempts.
        accepted_cheap_input_tokens: The subset from attempts that were accepted.
        accepted_cheap_output_tokens: Likewise for completions.
        escalated_input_tokens: Prompt tokens billed by the expensive calls made
            on escalation.
        escalated_output_tokens: Likewise for completions.
        verifier_input_tokens: Prompt tokens a model-based verifier billed (#2).
        verifier_output_tokens: Likewise for completions.
        verifier_usd: What the verifier cost, if it reported it (#2).
        cheap_ms: Wall time spent in cheap attempts (#3).
        verifier_ms: Wall time spent verifying (#3).
        escalation_ms: Wall time spent in escalation calls (#3).
    """

    skipped: int = 0
    attempted: int = 0
    cheap_passed: int = 0
    escalated: int = 0
    cheap_input_tokens: int = 0
    cheap_output_tokens: int = 0
    accepted_cheap_input_tokens: int = 0
    accepted_cheap_output_tokens: int = 0
    escalated_input_tokens: int = 0
    escalated_output_tokens: int = 0
    verifier_input_tokens: int = 0
    verifier_output_tokens: int = 0
    verifier_usd: float = 0.0
    cheap_ms: float = 0.0
    verifier_ms: float = 0.0
    escalation_ms: float = 0.0

    @property
    def escalation_rate(self) -> float | None:
        """Fraction of attempts that fell back to the expensive model.

        ``None`` when nothing has been attempted yet -- a rate over zero
        attempts is a division, not a measurement.
        """
        if not self.attempted:
            return None
        return self.escalated / self.attempted

    @property
    def total_latency_ms(self) -> float:
        """Wall time cascade added across cheap, verify, and escalation phases."""
        return self.cheap_ms + self.verifier_ms + self.escalation_ms

    def cost_summary(self, expensive_model: str, cheap_model: str) -> CascadeCost | None:
        """Price cascade's activity, or ``None`` if either model is unpriced.

        Follows the package's rule of never fabricating a rate for an unknown
        model (returns ``None`` rather than assuming one), and labels the single
        estimated figure as such -- see :class:`CascadeCost`.
        """
        from optio_optimize.config import pricing_for
        from optio_optimize.savings import _cost

        exp = pricing_for(expensive_model)
        cheap = pricing_for(cheap_model)
        if exp is None or cheap is None:
            return None

        cheap_spend = _cost(cheap, self.cheap_input_tokens, self.cheap_output_tokens, cached=0)
        escalation_spend = _cost(
            exp, self.escalated_input_tokens, self.escalated_output_tokens, cached=0
        )
        total = cheap_spend + escalation_spend + self.verifier_usd
        # All-expensive baseline: for escalated requests the expensive cost is
        # measured; for accepted-cheap ones it is projected from the cheap call's
        # token counts (the expensive model was never called).
        baseline = _cost(
            exp,
            self.accepted_cheap_input_tokens + self.escalated_input_tokens,
            self.accepted_cheap_output_tokens + self.escalated_output_tokens,
            cached=0,
        )
        waste = _cost(
            cheap,
            self.cheap_input_tokens - self.accepted_cheap_input_tokens,
            self.cheap_output_tokens - self.accepted_cheap_output_tokens,
            cached=0,
        )
        return CascadeCost(
            cheap_spend_usd=cheap_spend,
            escalation_spend_usd=escalation_spend,
            verifier_spend_usd=self.verifier_usd,
            total_spend_usd=total,
            all_expensive_baseline_usd=baseline,
            net_saving_usd=baseline - total,
            escalation_waste_usd=waste,
        )


@dataclass(slots=True)
class CascadeRouter:
    """Wraps a provider call so it tries cheap, verifies, and escalates.

    Constructed by :class:`~optio_optimize.optimizer.Optimizer` when
    ``cascade_routing`` is on; not usually built by hand. The wrapper it produces
    has the same signature as the provider call it wraps, so it drops into the
    pipeline where the raw call would have gone and every stage runs around it
    unchanged.

    Attributes:
        config: Active configuration; read for ``cheap_model``.
        verify: The acceptance check. Defaults to :func:`default_verifier`.
        stats: Running tally, exposed via ``Optimizer.cascade_stats``.
        counter: Token counter for the eligibility length check. Shares the
            pipeline's default so the two agree on what "short" means.
    """

    config: OptimizeConfig
    verify: Verifier = default_verifier
    stats: CascadeStats = field(default_factory=CascadeStats)
    counter: TokenCounter = field(default_factory=default_counter)

    def _eligible(self, request: LLMRequest) -> bool:
        """Whether cascade should attempt a cheap call for this request."""
        return is_routable(
            request,
            self.config.cheap_model,
            self.counter,
            allow_response_format=self.config.cascade_structured_output,
            allow_tools=self.config.cascade_tools,
            max_tokens=self.config.cascade_max_tokens or MAX_ROUTABLE_TOKENS,
        )

    def _accepts(self, request: LLMRequest, response: LLMResponse) -> bool:
        """Run the verifier, time it (#3), fold any reported cost (#2), fail open.

        A verifier may optionally expose ``pop_cost() -> (input_tokens,
        output_tokens, usd)`` returning and resetting the cost of its most recent
        judgment; :class:`ModelJudge` does. Cascade reads it so a model-based
        verifier's spend is subtracted from the saving rather than hidden. A
        plain function has no such method and simply reports no verifier cost.
        """
        started = time.perf_counter()
        try:
            ok = self.verify(request, response)
        except Exception as exc:  # noqa: BLE001 - a broken verifier must not break the call
            _log.warning(
                "optio_optimize: cascade verifier raised (%s); escalating",
                type(exc).__name__,
            )
            ok = False
        self.stats.verifier_ms += (time.perf_counter() - started) * 1000.0
        pop = getattr(self.verify, "pop_cost", None)
        if callable(pop):
            try:
                vin, vout, vusd = pop()
                self.stats.verifier_input_tokens += int(vin)
                self.stats.verifier_output_tokens += int(vout)
                self.stats.verifier_usd += float(vusd)
            except Exception:  # noqa: BLE001 - a broken cost hook must not break the call
                pass
        return ok

    def _add_cheap(self, response: LLMResponse) -> None:
        self.stats.cheap_input_tokens += response.input_tokens
        self.stats.cheap_output_tokens += response.output_tokens

    def _accept_cheap(self, response: LLMResponse) -> None:
        self.stats.cheap_passed += 1
        self.stats.accepted_cheap_input_tokens += response.input_tokens
        self.stats.accepted_cheap_output_tokens += response.output_tokens

    def _record_escalation(self, response: LLMResponse) -> None:
        self.stats.escalated += 1
        self.stats.escalated_input_tokens += response.input_tokens
        self.stats.escalated_output_tokens += response.output_tokens

    def wrap(self, call: ProviderCall) -> ProviderCall:
        """Return a provider call that cascades cheap -> verify -> escalate.

        Args:
            call: The real provider call to wrap.

        Returns:
            A call with the same signature. On an eligible request it sends the
            cheap model first and escalates on rejection; on everything else it
            is a passthrough.
        """
        cheap_model = self.config.cheap_model

        def escalate(request: LLMRequest) -> LLMResponse:
            started = time.perf_counter()
            resp = call(request)
            self.stats.escalation_ms += (time.perf_counter() - started) * 1000.0
            self._record_escalation(resp)
            return resp

        def cascade_call(request: LLMRequest) -> LLMResponse:
            if not self._eligible(request):
                self.stats.skipped += 1
                return call(request)
            assert cheap_model is not None  # eligibility guarantees this; narrows for mypy
            self.stats.attempted += 1
            started = time.perf_counter()
            try:
                cheap_response = call(replace(request, model=cheap_model))
            except Exception as exc:  # noqa: BLE001 - fall back to the model that would have run
                self.stats.cheap_ms += (time.perf_counter() - started) * 1000.0
                _log.warning(
                    "optio_optimize: cascade cheap attempt failed (%s); escalating",
                    type(exc).__name__,
                )
                return escalate(request)
            self.stats.cheap_ms += (time.perf_counter() - started) * 1000.0
            self._add_cheap(cheap_response)
            if self._accepts(request, cheap_response):
                self._accept_cheap(cheap_response)
                return cheap_response
            return escalate(request)

        return cascade_call

    def awrap(self, call: AsyncProviderCall) -> AsyncProviderCall:
        """The async twin of :meth:`wrap`.

        Identical control flow; only the provider calls are awaited. Exists for
        the same reason :meth:`~optio_optimize.pipeline.Pipeline.aexecute` does:
        the realistic adapter targets are ``async def``.

        Args:
            call: The real async provider call to wrap.

        Returns:
            An async call with the same signature.
        """
        cheap_model = self.config.cheap_model

        async def escalate(request: LLMRequest) -> LLMResponse:
            started = time.perf_counter()
            resp = await call(request)
            self.stats.escalation_ms += (time.perf_counter() - started) * 1000.0
            self._record_escalation(resp)
            return resp

        async def cascade_call(request: LLMRequest) -> LLMResponse:
            if not self._eligible(request):
                self.stats.skipped += 1
                return await call(request)
            assert cheap_model is not None  # eligibility guarantees this; narrows for mypy
            self.stats.attempted += 1
            started = time.perf_counter()
            try:
                cheap_response = await call(replace(request, model=cheap_model))
            except Exception as exc:  # noqa: BLE001 - fall back to the model that would have run
                self.stats.cheap_ms += (time.perf_counter() - started) * 1000.0
                _log.warning(
                    "optio_optimize: cascade cheap attempt failed (%s); escalating",
                    type(exc).__name__,
                )
                return await escalate(request)
            self.stats.cheap_ms += (time.perf_counter() - started) * 1000.0
            self._add_cheap(cheap_response)
            if self._accepts(request, cheap_response):
                self._accept_cheap(cheap_response)
                return cheap_response
            return await escalate(request)

        return cascade_call
