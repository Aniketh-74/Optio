"""The Stage contract — one optimization, one object.

Every technique in this package is a stage. The uniform shape is what keeps the
package from becoming forty special cases: stages compose in a list, each one is
independently testable against a plain request, and adding one requires no
change to the pipeline.

A stage does exactly one of three things:

* **Transform** the request — trim history, compress a prompt, swap the model.
* **Short-circuit** it — return a response without calling the provider, which
  is what a cache hit is.
* **Decline** — return the request unchanged, which must always be safe.

Declining is the default and the fallback. A stage that cannot help, does not
understand the request, or hits its latency budget returns the input untouched,
and the pipeline continues as though it were not installed. That property is
what makes ADR-013's fail-open rule implementable: skipping a broken stage is
always a valid execution.

**Stages are permitted to raise.** The pipeline wraps every call. Stage authors
should write direct code rather than scattering defensive try/except, exactly as
lane authors do in the core (``optio.lanes.base``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optio_optimize.config import OptimizeConfig
    from optio_optimize.tokens import TokenCounter
    from optio_optimize.types import LLMRequest, LLMResponse


class Fidelity(Enum):
    """How much a stage may change the model's answer.

    A two-way lossy/lossless split was the first design and it was wrong, in a
    way the benchmark caught immediately: ``structured_output`` was marked
    lossless, and the A/B suite reported **every** response in the ``fan_out``
    workload as divergent. Both facts were correct. The stage appends an
    instruction to the system prompt, so the model answers differently -- that
    is the entire point of it -- and no amount of it being *desirable* makes the
    output identical.

    Three levels, because the middle case is real and common:

    ``IDENTICAL``
        The response is byte-for-byte what an unoptimized call would return.
        Achievable only by not calling the model differently at all: serving a
        cached answer for the same request, or marking a prefix the provider
        bills at a discount. Divergence here is a bug, and the benchmark fails
        the build over it.

    ``SHAPED``
        The information is preserved; the presentation changes. Suppressing a
        prose wrapper around JSON, or capping a reply that would have rambled.
        A user gets what they asked for in a more compact form. Enabled by
        default, because that is a trade almost everyone wants -- but it is
        stated rather than hidden behind a "lossless" claim.

    ``ALTERED``
        The content itself may differ: a near-match served from a semantic
        cache, a compressed prompt, a cheaper model, a summarized history.
        Off by default, and gated by the eval suite (ADR-013 rule 3).
    """

    IDENTICAL = "identical"
    SHAPED = "shaped"
    ALTERED = "altered"

    @property
    def guarantees_identical_output(self) -> bool:
        """Whether a response must match the unoptimized one exactly."""
        return self is Fidelity.IDENTICAL


@dataclass(slots=True)
class StageContext:
    """What a stage is allowed to see.

    Deliberately narrow. A stage gets configuration, a token counter, and a
    scratch dict scoped to one request -- not the pipeline, not other stages,
    and no way to reach across to another request. Stages that could see each
    other would acquire ordering dependencies, and ordering dependencies in a
    list of forty transformations is how this kind of package becomes
    unmaintainable.

    Attributes:
        config: Active configuration.
        counter: Token counter for measurement decisions.
        scratch: Per-request storage. A stage may leave a note for its own
            ``after`` hook here -- the cache stages use it to carry the key
            computed during ``before`` -- but must not read another stage's.
        run_id: The optio run this request belongs to, when known. Lets savings
            be attributed to the same run the cost lane is metering.
    """

    config: OptimizeConfig
    counter: TokenCounter
    scratch: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    """What a stage did.

    Attributes:
        request: The request to pass on. The input, unchanged, when declining.
        response: Set only to short-circuit -- the pipeline then skips every
            remaining stage *and* the provider call.
        saved_input_tokens: Prompt tokens this stage avoided sending. Measured
            by the stage, not inferred by the pipeline, because only the stage
            knows what the request would have looked like otherwise.
        saved_output_tokens: Completion tokens avoided.
        note: Short human-readable reason, surfaced in reports and logs. Must
            never contain prompt content.
    """

    request: LLMRequest
    response: LLMResponse | None = None
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0
    note: str = ""

    @property
    def short_circuited(self) -> bool:
        """Whether this result ends the pipeline."""
        return self.response is not None


class Stage(ABC):
    """One optimization.

    Attributes:
        name: Stable identifier, used in savings reports, configuration and
            logs. Must not change once released -- it is the key users write
            in ``disabled_stages``.
    """

    #: How far this stage may move the answer. See :class:`Fidelity`.
    #: ``ALTERED`` stages are off unless explicitly enabled, and the eval suite
    #: (ADR-013 rule 3) is required to cover every one of them.
    fidelity: Fidelity = Fidelity.IDENTICAL

    @property
    def lossy(self) -> bool:
        """Whether this stage may change the answer's content.

        ``SHAPED`` is deliberately *not* lossy: the information survives, only
        its presentation changes. Kept as a property so the warning on
        construction and the eval gate both key off content risk rather than
        firing on every stage that reformats a reply.
        """
        return self.fidelity is Fidelity.ALTERED

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable stage identifier."""

    @abstractmethod
    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Transform or short-circuit a request on its way to the provider.

        Args:
            request: The request as the previous stages left it.
            ctx: Per-request context.

        Returns:
            A result carrying the request to pass on, or a response to return
            immediately.
        """

    def after(  # noqa: B027 - optional hook; most stages have nothing to observe
        self,
        request: LLMRequest,
        response: LLMResponse,
        ctx: StageContext,
    ) -> None:
        """Observe the completed exchange.

        The write half of read-through caching, and the only place a stage can
        learn what the provider actually returned. Default is to do nothing.

        Errors here are swallowed by the pipeline like any other stage failure:
        the response has already been produced, so a failed cache write must not
        turn a successful call into an exception.

        Args:
            request: The request as sent to the provider, after all transforms.
            response: What came back.
            ctx: The same context object ``before`` received.
        """

    def declines(self, request: LLMRequest) -> StageResult:
        """Return a no-op result. The safe answer whenever unsure."""
        return StageResult(request=request)
