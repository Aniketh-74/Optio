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
    from optio_optimize.types import LLMRequest

#: Estimated prompt tokens at or under which a request is a routing
#: candidate. There is no way to judge a step's real difficulty without
#: calling a model to grade it, which would spend money to decide whether to
#: spend less -- so length is the only signal used, on the theory that a
#: short, standalone prompt is more often a lookup than a multi-step
#: reasoning task. A theory, not a guarantee; see the class docstring.
MAX_ROUTABLE_TOKENS = 500


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
    should; confirming the *answers* stay acceptable needs a live A/B run
    with a judge (``bench/harness.py``), and turning this stage on at all is
    the caller's judgment call to make, not this docstring's to settle.
    """

    fidelity = Fidelity.ALTERED

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "route_models"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Retarget the request to the cheap model, if it looks safe to."""
        cheap_model = ctx.config.cheap_model
        if not cheap_model or request.model == cheap_model:
            return self.declines(request)
        if request.tools or request.response_format is not None:
            return self.declines(request)

        estimated = count_request(request, ctx.counter)
        if estimated > MAX_ROUTABLE_TOKENS:
            return self.declines(request)

        return StageResult(
            request=replace(request, model=cheap_model),
            note=f"routed {request.model} -> {cheap_model} (~{estimated} prompt tokens)",
        )
