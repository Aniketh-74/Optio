"""Output-side optimizations: token ceilings and structured replies.

Input tokens get most of the attention because prompts are visibly large, but
output tokens are billed at three to five times the input rate on every major
provider. A 200-token reduction in output is worth more than a 600-token
reduction in input on GPT-4o. These two stages are where that leverage is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio_optimize.stages.base import Fidelity, Stage, StageResult

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
            from optio_optimize.types import Message

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


def _percentile(values: list[int], fraction: float) -> float:
    """Return a percentile of ``values`` by nearest rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return float(ordered[index])
