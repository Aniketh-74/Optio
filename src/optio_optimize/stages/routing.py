"""Send short, simple requests to a cheaper model.

The one ``ALTERED``-tier stage that needs no auxiliary call and no new
dependency: it never reads or rewrites the prompt, only ``request.model``.
That is also its whole risk surface -- the answer itself may simply be worse,
which is what makes it ``Fidelity.ALTERED`` rather than ``SHAPED``, and why it
stays off by default (ADR-013) even though the mechanism is cheap.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from optio_optimize.stages.base import Fidelity, Stage, StageResult
from optio_optimize.tokens import count_request

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.tokens import TokenCounter
    from optio_optimize.types import LLMRequest

#: Estimated prompt tokens at or under which a request is a routing
#: candidate. There is no way to judge a step's real difficulty without
#: calling a model to grade it, which would spend money to decide whether to
#: spend less -- so length is the only signal used, on the theory that a
#: short, standalone prompt is more often a lookup than a multi-step
#: reasoning task. A theory, not a guarantee; see the class docstring.
MAX_ROUTABLE_TOKENS = 500


def is_routable(
    request: LLMRequest,
    cheap_model: str | None,
    counter: TokenCounter,
    *,
    allow_response_format: bool = False,
    allow_tools: bool = False,
    max_tokens: int = MAX_ROUTABLE_TOKENS,
) -> bool:
    """Whether a request is eligible to be sent to ``cheap_model``.

    The single source of truth for "this request looks safe to downgrade",
    shared by :class:`RouteModelsStage` (which acts on it immediately) and by
    cascade routing (ADR-023, which uses it to decide whether to *attempt* a
    cheap call before verifying). Extracted so the two can never drift: a
    request the static router would downgrade and the cascade would not, or
    vice versa, would be a difference nobody chose.

    Eligible means all of: a ``cheap_model`` exists and differs from the
    request's current model; no tools attached (tool selection is where a
    weaker model degrades first); no ``response_format`` unless
    ``allow_response_format`` is set; and an estimated prompt at or under
    :data:`MAX_ROUTABLE_TOKENS` (length is weak evidence of ease, but the only
    evidence available without a model call to produce better).

    Args:
        request: The request under consideration.
        cheap_model: The model routing would downgrade to, or ``None``.
        counter: Token counter for the length check.
        allow_response_format: Permit requests carrying a ``response_format``.
            Left ``False`` for static ``route_models``, which has no way to
            check the cheap model honoured the schema. Set ``True`` by cascade
            when ``cascade_structured_output`` is on: the schema is itself a
            verifier, so the cheap attempt can be checked for conformance and
            escalated if it does not fit (ADR-023 step 1).
        allow_tools: Permit requests carrying ``tools``. Left ``False`` for
            static ``route_models`` (tool selection is where a weak model
            degrades first, with no recovery). Set ``True`` by cascade when
            ``cascade_tools`` is on: the cheap model's *proposed* call can be
            vetted before the agent executes it, and escalated if it names an
            unknown tool or malformed arguments (ADR-023 step 3).
        max_tokens: Prompt-token ceiling. Defaults to
            :data:`MAX_ROUTABLE_TOKENS`. Cascade may raise it via
            ``cascade_max_tokens`` (ADR-023 step 2): its escalation net means a
            long-but-hard prompt the cheap model fumbles is recovered rather
            than served wrong, so the ceiling can be a cost knob instead of a
            safety one -- a raised ceiling only stops paying when *enough* long
            prompts escalate that the wasted cheap attempts outweigh the wins.

    Returns:
        ``True`` if the request clears every guardrail above.
    """
    if not cheap_model or request.model == cheap_model:
        return False
    if request.tools and not allow_tools:
        return False
    if request.response_format is not None and not allow_response_format:
        return False
    return count_request(request, counter) <= max_tokens


class RouteModelsStage(Stage):
    """Downgrade the model on requests short enough to look easy.

    Declines whenever there is a reason to think the step needs the
    requested model's full capability: tools attached (tool selection is
    exactly where a weaker model degrades first), a ``response_format``
    schema (precision matters more for structured extraction), the prompt
    already targets ``cheap_model``, or the prompt is longer than
    :data:`MAX_ROUTABLE_TOKENS` (length is weak evidence of complexity, but
    it is the only evidence available without a model call to produce
    better evidence).

    **What this cannot know.** Whether the cheap model actually answers as
    well as the requested one is a model-*capability* question, and nothing
    here can check it -- "short and simple" is a proxy for "easy", not proof
    of it. The ADR-013 eval gate (``eval.cases.DecisionBoundaryCase``) can
    only confirm this stage routes exactly what its own rule above says it
    should; confirming the *answers* stay acceptable needs a live comparison
    of the two models, and turning this stage on at all is the caller's
    judgment call to make, not this docstring's to settle.

    **What one live measurement found (2026-07-29, ADR-015).** Twelve short
    prompts with known answers, asked of ``gpt-4o`` and ``gpt-4o-mini``
    (16.7x cheaper), graded against ground truth rather than a judge: the
    cheap model matched on every lookup and on 7 of 8 short-but-hard probes.
    The one it missed is the whole risk in one line -- *"What is 17 times 24,
    minus 89?"*, eight words, well inside :data:`MAX_ROUTABLE_TOKENS`, no
    tools, no schema, so this stage routes it. ``gpt-4o`` answers 319;
    ``gpt-4o-mini`` answers 329. Reproduced identically three times.

    That is a *floor* on the risk, not a measurement of it: twelve probes,
    one model pair, all single-turn and all answer-checkable by construction,
    which excludes exactly the open-ended requests where a weaker model
    degrades in ways no string comparison would catch. Note also that four
    famous reasoning traps (bat-and-ball, the strawberry letter count) were
    tried first and the cheap model passed all of them -- a probe set can be
    "hard" and still prove nothing if the hard parts are memorized. See
    ``docs/optimize-benchmarks.md``.
    """

    fidelity = Fidelity.ALTERED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "route_models"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Retarget the request to the cheap model, if it looks safe to."""
        cheap_model = ctx.config.cheap_model
        if not is_routable(request, cheap_model, ctx.counter):
            return self.declines(request)
        assert cheap_model is not None  # is_routable guarantees this; narrows for mypy

        estimated = count_request(request, ctx.counter)
        return StageResult(
            request=replace(request, model=cheap_model),
            note=f"routed {request.model} -> {cheap_model} (~{estimated} prompt tokens)",
        )
