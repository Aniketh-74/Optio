"""Output-side optimizations: token ceilings and structured replies.

Input tokens get most of the attention because prompts are visibly large, but
output tokens are billed at three to five times the input rate on every major
provider. A 200-token reduction in output is worth more than a 600-token
reduction in input on GPT-4o. These three stages are where that leverage is.

Two of them work by *adding* input tokens to remove output ones. That is a good
trade at a 4-6x price ratio and a bad one if the added instruction grows, which
is why both instructions are a single sentence and why the cost of each is
netted off the saving it claims rather than reported gross.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio_optimize.stages.base import Fidelity, Stage, StageResult
from optio_optimize.types import Message

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest, LLMResponse

#: Scratch key for the observed-length history shared across requests.
_OBSERVED = "adaptive_observed_lengths"

#: Multiple of the observed p95 output length used as the ceiling. Generous on
#: purpose: the cost of guessing low is a truncated answer -- a correctness
#: failure the caller sees -- while guessing high merely forgoes some saving.
#: The asymmetry justifies erring high.
CEILING_MULTIPLIER = 2.0

#: Observations required before a ceiling is imposed. Below this the sample
#: cannot support a p95 and the ceiling would be an invented number.
MIN_OBSERVATIONS = 20

#: Never cap below this. A ceiling in the low hundreds truncates ordinary
#: answers, and a truncated answer that looks complete is the worst outcome
#: this stage can produce.
FLOOR_TOKENS = 256


class AdaptiveMaxTokensStage(Stage):
    """Cap output length from what this workload actually produces.

    Most callers never set ``max_tokens``, so the model generates until it stops
    naturally -- which for a chatty model on an open-ended prompt means
    paragraphs of preamble nobody reads. This stage watches real completion
    lengths and sets a ceiling from them.

    Not lossy in the usual sense: it does not change a response that fits. It
    *can* truncate one that would have run longer than anything previously
    observed, which is why the multiplier is generous, the floor is high, and
    the ceiling only applies once there is a real sample.
    """

    # SHAPED, not IDENTICAL: a ceiling that binds truncates the reply. Rare by
    # design, but "rare" is not "never", and a stage claiming identical output
    # must be able to guarantee it.
    fidelity = Fidelity.SHAPED

    def __init__(self) -> None:
        """Build the stage with an empty observation history."""
        self._lengths: list[int] = []

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "adaptive_max_tokens"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Impose a ceiling derived from observed output lengths."""
        if request.max_tokens is not None:
            # The caller stated a ceiling. Overriding it -- in either direction
            # -- substitutes our guess for their explicit instruction.
            return self.declines(request)
        if len(self._lengths) < MIN_OBSERVATIONS:
            return self.declines(request)

        ceiling = max(FLOOR_TOKENS, int(_percentile(self._lengths, 0.95) * CEILING_MULTIPLIER))
        typical = int(_percentile(self._lengths, 0.5))
        # The saving is the tail we expect not to generate, not the difference
        # from an imaginary unbounded reply. Reporting `ceiling` as saved would
        # be inventing a baseline nobody would have hit.
        expected_saving = max(0, int(_percentile(self._lengths, 0.99)) - ceiling)

        from dataclasses import replace

        return StageResult(
            request=replace(request, max_tokens=ceiling),
            saved_output_tokens=expected_saving,
            note=f"ceiling {ceiling} (p50 observed {typical})",
        )

    def after(self, request: LLMRequest, response: LLMResponse, ctx: StageContext) -> None:
        """Record the real completion length."""
        if response.served_from is not None:
            return  # A cached reply is not a fresh observation.
        if response.output_tokens > 0:
            self._lengths.append(response.output_tokens)
            # Bounded: this lives for the process lifetime, and §11's memory
            # rule applies to this package too.
            if len(self._lengths) > 1000:
                del self._lengths[:500]


class StructuredOutputStage(Stage):
    """Steer the model toward compact structured replies.

    When the caller supplied a ``response_format`` schema, adds a short
    instruction reinforcing it and suppressing the prose wrapper models like to
    add around JSON ("Sure! Here's the JSON you asked for: ... Let me know if
    you'd like me to adjust anything."). That wrapper is pure billed output that
    the caller then has to strip.

    Only acts when a schema is already present. Inventing one would change the
    contract between the caller and their model, which is not this library's
    call to make.
    """

    # SHAPED. This appends an instruction to the prompt, so the model answers
    # differently -- more tersely, which is the point. The A/B suite reported
    # every fan_out response as divergent while this claimed to be lossless,
    # and the suite was right.
    fidelity = Fidelity.SHAPED

    #: Appended to the system prompt. Deliberately terse -- it costs input
    #: tokens on every call, so a long instruction can outweigh what it saves.
    INSTRUCTION = "Respond only with the requested structure. No preamble or explanation."

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "structured_output"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Reinforce an existing structured-output request."""
        if request.response_format is None and not request.tools:
            return self.declines(request)
        if any(self.INSTRUCTION in m.content for m in request.messages):
            return self.declines(request)  # Already applied on a previous pass.

        messages = list(request.messages)
        instruction_cost = ctx.counter.count_text(self.INSTRUCTION, request.model)

        if messages and messages[0].role == "system":
            messages[0] = messages[0].with_content(f"{messages[0].content}\n{self.INSTRUCTION}")
        else:
            messages.insert(0, Message(role="system", content=self.INSTRUCTION))

        # Typical preamble suppressed, minus what the instruction costs. A
        # modest, defensible figure rather than the flattering one: measured
        # preambles run 30-60 output tokens on chat-tuned models.
        net = max(0, 40 - instruction_cost)
        return StageResult(
            request=request.with_messages(tuple(messages)),
            saved_output_tokens=net,
            note="preamble suppressed",
        )


class ConcisionStage(Stage):
    """Suppress the conversational scaffolding around a chat reply.

    Chat-tuned models wrap an answer in three habits the caller is billed for
    and rarely wants: restating the question, summarizing the reply they just
    wrote, and offering follow-up help. The field literature puts this at
    **30-50% of output tokens in chat products**, and output is billed at four
    to six times the input rate on most frontier models -- which is what makes
    a fifteen-token instruction a good trade against it.

    That trade is the entire design constraint here, and it is why
    :attr:`INSTRUCTION` is one sentence. The instruction is paid on every
    request, in input tokens, whether or not the model was going to pad the
    reply. A paragraph explaining *how* to be concise would be self-defeating
    in a way that is genuinely easy to miss, because the cost lands in a
    different column from the saving.

    Declines when a schema or tools are present:
    :class:`StructuredOutputStage` already covers that case with a stricter
    instruction, and stacking both would pay twice for one effect.

    ``SHAPED``. The answer survives; the packaging around it does not. Callers
    who *want* the follow-up offer -- consumer chat products often do -- should
    leave this off, which is why it is a named stage rather than folded into
    another.
    """

    fidelity = Fidelity.SHAPED

    #: Appended to the system prompt. Names the three specific habits rather
    #: than saying "be concise", which models reliably interpret as license to
    #: shorten the *answer* -- the one part that should not shrink.
    INSTRUCTION = "Do not restate the question, summarize your own reply, or offer further help."

    #: Conservative estimate of scaffolding suppressed per reply, in output
    #: tokens, before the instruction's own input cost is netted off. Lower
    #: than :class:`StructuredOutputStage`'s equivalent on purpose: that stage
    #: suppresses a known wrapper around a known structure, while this one
    #: suppresses a habit whose size varies with the model and the question.
    #: The live A/B suite is the authority on the real figure; this is what the
    #: report says until it runs.
    ESTIMATED_SCAFFOLDING_TOKENS = 25

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "concision"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Add the anti-scaffolding instruction to a free-form chat request."""
        if request.response_format is not None or request.tools:
            return self.declines(request)
        if any(self.INSTRUCTION in m.content for m in request.messages):
            return self.declines(request)

        messages = list(request.messages)
        instruction_cost = ctx.counter.count_text(self.INSTRUCTION, request.model)
        if messages and messages[0].role == "system":
            messages[0] = messages[0].with_content(f"{messages[0].content}\n{self.INSTRUCTION}")
        else:
            messages.insert(0, Message(role="system", content=self.INSTRUCTION))

        return StageResult(
            request=request.with_messages(tuple(messages)),
            saved_output_tokens=max(0, self.ESTIMATED_SCAFFOLDING_TOKENS - instruction_cost),
            note="chat scaffolding suppressed",
        )


class ChainOfDraftStage(Stage):
    """Constrain reasoning to shorthand instead of prose.

    Chain-of-Thought produces verbose intermediate reasoning, and verbosity is
    billed as output at the highest per-token rate on the request. Chain of
    Draft (Xu et al., arXiv:2502.18600) asks for the same reasoning in
    telegraphic notes of roughly five words a step. The published result is
    striking: **as little as 7.6% of CoT's tokens** at equal accuracy, and on
    one task average output fell from 189.4 to 14.3 tokens *while* accuracy
    rose from 93.2% to 97.3%.

    ``ALTERED``, unambiguously. This does not reshape how an answer is
    presented -- it changes how the model is asked to think, and a model that
    reasons in five-word steps can reach a different conclusion than one
    reasoning in sentences. Off by default and eval-gated like every other
    ``ALTERED`` stage (ADR-013 rule 3).

    **Where the published number does not transfer.** A follow-up on
    software-engineering tasks (arXiv:2506.10987) found it degrades on code,
    where a reasoning step genuinely needs more than five words, and smaller
    models handle the constraint worse than large ones. It also interacts
    unpredictably with models that have their own native thinking format,
    since it is competing with a mechanism it cannot see. So the honest
    summary is: excellent on arithmetic, extraction and classification;
    untested-to-poor on everything this package's own workloads look like.
    Nothing here has measured it, and until something has, the numbers above
    belong to the paper and not to this library.

    Declines when the caller supplied a ``response_format``. A schema already
    constrains the reply, and stacking a reasoning-style instruction on top is
    how you get a model that emits its shorthand notes *inside* the JSON.
    """

    fidelity = Fidelity.ALTERED

    #: Appended to the system prompt. The word "step" matters more than the
    #: word count: the paper's own operationalisation asks for minimal drafts
    #: per step, not for a short answer, and models shorten the *answer* when
    #: told only to be brief.
    INSTRUCTION = (
        "Think step by step, but write at most five words per step. "
        "Return the final answer after a '####' separator."
    )

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "chain_of_draft"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Ask for shorthand reasoning rather than prose."""
        if request.response_format is not None:
            return self.declines(request)
        if any(self.INSTRUCTION in m.content for m in request.messages):
            return self.declines(request)

        messages = list(request.messages)
        instruction_cost = ctx.counter.count_text(self.INSTRUCTION, request.model)
        if messages and messages[0].role == "system":
            messages[0] = messages[0].with_content(f"{messages[0].content}\n{self.INSTRUCTION}")
        else:
            messages.insert(0, Message(role="system", content=self.INSTRUCTION))

        # **Claims nothing.** Unlike `concision`, which suppresses a wrapper of
        # roughly knowable size, the saving here is some fraction of however
        # much reasoning the model would have done -- a quantity this stage
        # cannot see and must not guess. A negative saving is not the answer
        # either: the pipeline derives `baseline = actual + saved`, so a
        # negative would put baseline below actual and break the one invariant
        # `Pipeline._finish` exists to hold. So the report gets a zero and a
        # note carrying the cost, and the live A/B supplies the other side.
        return StageResult(
            request=request.with_messages(tuple(messages)),
            note=f"reasoning constrained to shorthand (costs {instruction_cost} input tokens)",
        )


def _percentile(values: list[int], fraction: float) -> float:
    """Return a percentile of ``values`` by nearest rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return float(ordered[index])
