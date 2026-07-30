"""Does a lowered reasoning budget still get the right answer? Measure both.

ADR-018 ships ``reasoning_budget`` off and states the condition for ever
recommending it:

> A cost number without an accuracy number is not evidence for this stage; it is
> evidence for the half of the trade that flatters it.

So this script grades every reply as well as pricing it, on two task sets chosen
to separate the two ways the stage can be judged. The **easy** set is where a
smaller budget should be free. The **hard** set is where it should hurt, and
hard is the whole reason anyone turns extended thinking on -- a stage that looks
good on easy questions alone has been measured on the half that cannot fail.

Every hard answer was computed in Python before this script was written, not
recalled. The failure mode of a graded eval is a wrong answer key, which scores
a correct model as incorrect and makes any reduction look free.

**Isolation** (ADR-015 rule 2). One variable: the reasoning budget. Every arm
sends the same prompts, the same system instruction and the same ``max_tokens``
-- the ceiling is deliberately *not* reduced alongside the budget, or a
truncated reply could not be attributed to either. Every other stage is off.

**Three arms, control-treated-control.** Extended thinking runs at temperature 1
because the API requires it, so trace length is stochastic and a single pair of
arms cannot tell "the budget shortened the trace" from "the second arm drew
shorter traces". The two controls bracket the treated arm and their spread is
printed as the noise floor any claim has to clear. That is not caution for its
own sake: the first two-arm version of this script reported an 18.5% saving that
the diagnostics then showed the stage could not have produced.

**How the treated arm gets its ceiling.** The stage needs ``MIN_OBSERVATIONS``
completed reasoning calls before it will act, so it is warmed from the first
control arm's real per-call output lengths -- the same numbers it would have
collected itself in a long-running process, replayed rather than re-billed.

What it measured, 2026-07-30, ``claude-haiku-4-5``, budget 16,000 -> 4,438:

* Output tokens 12,892 and 14,537 in the two controls, **10,644** treated. Cost
  **-21.9%** against their mean, against a control-to-control noise floor of
  11.7%. Below both controls, so ordering does not explain it.
* Accuracy **100% on both sets in all three arms**, and no reply truncated.
* **Zero of forty control calls exceeded the ceiling** -- longest was 2,480
  tokens against a 4,438 ceiling. So the reduction is *not* the ceiling
  truncating anything, which is the mechanism the stage's own docstring
  described. ``budget_tokens`` shapes how long the model thinks even when it had
  room to spare: a target, not just a cap.
* That reading cuts against the stage rather than for it, and it is why the flag
  stays off. The saving is real and the safety argument that justified it is not:
  "the ceiling cannot bind on the observed distribution" is beside the point if
  the model thinks less merely for having been told it had less room. **And the
  accuracy number is close to vacuous** -- a hard set every arm scores 100% on
  cannot detect degradation. Ten tasks Haiku 4.5 finds easy is not evidence a
  reduced budget is safe; it is evidence the task set needs to be harder.

Usage::

    python scripts/measure_reasoning_budget.py

Spends real money -- roughly $0.20 for the three arms.
"""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

MODEL = "claude-haiku-4-5"

#: What an over-provisioned caller sets, and the number this measurement is
#: about. Nobody tunes this per request: a default goes in once and every call
#: inherits it, which is exactly the situation a stage is for.
CALLER_BUDGET = 16_000

#: Held identical across both arms. Must exceed the budget -- Anthropic rejects
#: a ``max_tokens`` at or below ``thinking.budget_tokens`` -- and reducing it in
#: the treated arm would confound the measurement with a second variable.
MAX_TOKENS = CALLER_BUDGET + 1_024

#: Identical in both arms. Grading needs a parseable final answer; asking for one
#: is not an optimization and costs the same handful of tokens either way.
SYSTEM_PROMPT = (
    "Solve the problem. End your reply with a line of exactly the form "
    "'ANSWER: <value>' and nothing after it."
)


@dataclass(frozen=True, slots=True)
class Task:
    """One graded question.

    Attributes:
        task_id: Short label used in the report.
        prompt: The question, sent verbatim in both arms.
        answer: The correct value, computed independently of any model.
    """

    task_id: str
    prompt: str
    answer: str


EASY: tuple[Task, ...] = (
    Task("e1", "What is 17 times 23?", "391"),
    Task("e2", "What is the capital city of Australia?", "canberra"),
    Task("e3", "What is 144 divided by 12?", "12"),
    Task("e4", "How many days are in February in a leap year?", "29"),
    Task("e5", "What is 2 to the power of 10?", "1024"),
    Task("e6", "What is the chemical symbol for gold?", "au"),
    Task("e7", "What is 15 percent of 200?", "30"),
    Task("e8", "How many sides does a hexagon have?", "6"),
    Task("e9", "What is 100 minus 37?", "63"),
    Task("e10", "What is the square root of 169?", "13"),
)

#: Every answer below was produced by running the calculation, then written
#: here. See the module docstring on why that ordering matters.
HARD: tuple[Task, ...] = (
    Task(
        "h1",
        "What is the sum of all three-digit integers divisible by 7 but not by 5?",
        "56231",
    ),
    Task("h2", "What are the last two digits of 7 raised to the power 222?", "49"),
    Task(
        "h3",
        "How many integers from 1 to 1000 inclusive are divisible by 3 or by 5 but not by both?",
        "401",
    ),
    Task(
        "h4",
        "How many times does the digit 7 appear when writing out every integer "
        "from 1 to 1000 inclusive?",
        "300",
    ),
    Task("h5", "How many positive divisors does 5040 have?", "60"),
    Task(
        "h6",
        "What are the last two digits of the 30th Fibonacci number, where the "
        "first two Fibonacci numbers are 1 and 1?",
        "40",
    ),
    Task("h7", "What is the sum of all prime numbers below 100?", "1060"),
    Task(
        "h8",
        "24 workers finish a job in 18 days working 8 hours a day. How many days "
        "would 16 workers need working 6 hours a day?",
        "36",
    ),
    Task(
        "h9",
        "How many squares of any size are there on a standard 8 by 8 chessboard?",
        "204",
    ),
    Task(
        "h10",
        "How many arrangements of the five distinct letters A, B, C, D, E have A "
        "not in the first position and E not in the last position?",
        "78",
    ),
)


def _load_key() -> None:
    """Read the key in-process only; never onto a command line."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {"ANTHROPIC_API_KEY", "ANTHROPIC_KEY"}:
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("\"'")
            return


_load_key()

from anthropic import Anthropic  # noqa: E402
from anthropic.types import TextBlock  # noqa: E402

from optio_optimize import Optimizer  # noqa: E402
from optio_optimize.adapters.anthropic import wrap_anthropic_client  # noqa: E402
from optio_optimize.config import PRICING, OptimizeConfig  # noqa: E402
from optio_optimize.savings import _cost  # noqa: E402
from optio_optimize.stages.base import StageContext  # noqa: E402
from optio_optimize.stages.output import MIN_OBSERVATIONS, ReasoningBudgetStage  # noqa: E402
from optio_optimize.tokens import HeuristicCounter  # noqa: E402
from optio_optimize.types import LLMRequest, LLMResponse, Message  # noqa: E402


def _config(*, reasoning_budget: bool) -> OptimizeConfig:
    """One flag differs between the arms; every other stage is off in both.

    Spelled out rather than splatted from a dict so that adding a
    default-on stage later is a type error here rather than a silent second
    variable in the measurement.
    """
    return OptimizeConfig(
        reasoning_budget=reasoning_budget,
        exact_cache=False,
        prefix_cache=False,
        trim_history=False,
        cap_tool_results=False,
        minify_tools=False,
        structured_output=False,
        adaptive_max_tokens=False,
        deduplicate=False,
        prune_retrieval=False,
        detect_unstable_prefix=False,
    )


@dataclass(slots=True)
class ArmResult:
    """What one arm cost and how often it was right.

    Attributes:
        input_tokens: Prompt tokens billed, cache reads and writes included.
        output_tokens: Completion tokens billed. **Thinking is in here** -- no
            provider in ``PRICING`` reports the trace apart from the answer,
            which is why the stage claims no saving and why this script exists.
        lengths: Per-call output lengths, used to warm the treated arm's stage.
        correct: Task ids answered correctly.
        wrong: ``(task_id, expected, got)`` for each miss.
        truncated: Task ids whose generation hit the ceiling.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    lengths: list[int] = field(default_factory=list)
    correct: list[str] = field(default_factory=list)
    wrong: list[tuple[str, str, str]] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)

    @property
    def usd(self) -> float:
        """Billed cost.

        Cache reads and writes are zero here: both arms run with
        ``prefix_cache`` off, so there is no discounted or premium band.
        """
        cost: float = _cost(PRICING[MODEL], self.input_tokens, self.output_tokens, 0, 0)
        return cost

    def accuracy(self, tasks: tuple[Task, ...]) -> float | None:
        """Fraction of ``tasks`` answered correctly, or ``None`` on no data.

        ``None`` rather than ``0.0`` when nothing in ``tasks`` was graded:
        absence is not zero, the rule this package applies to every rate it
        reports.
        """
        graded = {task.task_id for task in tasks} & (
            set(self.correct) | {entry[0] for entry in self.wrong}
        )
        if not graded:
            return None
        return len(graded & set(self.correct)) / len(graded)


def _final_answer(text: str) -> str:
    """Pull the graded value out of a reply.

    Only the ``ANSWER:`` line is read, and only that line is ever printed --
    the reasoning trace is model output and §10's content rule applies to a
    script as much as to the library.
    """
    marker = text.upper().rfind("ANSWER:")
    if marker < 0:
        return ""
    tail = text[marker + len("ANSWER:") :].strip().splitlines()
    value = tail[0] if tail else ""
    return value.strip().strip(".*` ").replace(",", "").lower()


def _graded(task: Task, text: str) -> tuple[bool, str]:
    """Whether the reply answered ``task``, and the value it gave."""
    got = _final_answer(text)
    return got == task.answer.lower(), got


def _ctx() -> StageContext:
    """The context the stage sees, with only its own flag on."""
    return StageContext(config=OptimizeConfig(reasoning_budget=True), counter=HeuristicCounter())


def _warm(stage: ReasoningBudgetStage, lengths: list[int]) -> None:
    """Replay the control arm's observed output lengths into the stage.

    The stage records the length of any completed call that carried a budget, so
    replaying real lengths puts it in exactly the state twenty live calls would
    have -- without paying for them twice.
    """
    sent = LLMRequest(
        model=MODEL,
        messages=(Message(role="user", content="warm"),),
        thinking_budget=CALLER_BUDGET,
    )
    ctx = _ctx()
    for length in lengths:
        stage.after(sent, LLMResponse(content="", output_tokens=length, model=MODEL), ctx)


def _chosen_ceiling(stage: ReasoningBudgetStage) -> int | None:
    """The budget the warmed stage will send, or ``None`` if it declines.

    A probe, not a side effect: ``before`` returns a new request and mutates
    nothing. Worth printing, because a treated arm that quietly declined every
    call is a broken measurement and looks identical to a stage that helped
    nothing.
    """
    probe = LLMRequest(
        model=MODEL,
        messages=(Message(role="user", content="probe"),),
        thinking_budget=CALLER_BUDGET,
    )
    result = stage.before(probe, _ctx())
    return result.request.thinking_budget if result.note else None


def run(*, stage: ReasoningBudgetStage | None) -> ArmResult:
    """Answer every task once.

    Args:
        stage: The reasoning-budget stage to run, already warmed, or ``None``
            for the control arm.

    Returns:
        Cost, accuracy and per-task detail for this arm.
    """
    client = Anthropic()
    if stage is None:
        optimizer = Optimizer(_config(reasoning_budget=False))
    else:
        # The warmed instance, passed explicitly -- the registry would build a
        # fresh one with no observation history, which declines everything.
        optimizer = Optimizer(_config(reasoning_budget=True), stages=[stage])
    wrap_anthropic_client(client, optimizer=optimizer)

    result = ArmResult()
    for task in EASY + HARD:
        reply = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "enabled", "budget_tokens": CALLER_BUDGET},
            messages=[{"role": "user", "content": task.prompt}],
        )
        # isinstance, not `.type == "text"`: only the former narrows the block
        # union for mypy, and here it also drops the thinking blocks.
        text = "".join(b.text for b in reply.content if isinstance(b, TextBlock))
        ok, got = _graded(task, text)
        if ok:
            result.correct.append(task.task_id)
        else:
            result.wrong.append((task.task_id, task.answer, got or "<no ANSWER line>"))
        if reply.stop_reason == "max_tokens":
            result.truncated.append(task.task_id)
        result.input_tokens += reply.usage.input_tokens
        result.output_tokens += reply.usage.output_tokens
        result.lengths.append(reply.usage.output_tokens)
    return result


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    print(f"control arm: budget {CALLER_BUDGET:,}, {len(EASY + HARD)} tasks")
    control = run(stage=None)

    if len(control.lengths) < MIN_OBSERVATIONS:
        raise SystemExit(
            f"only {len(control.lengths)} observations; the stage needs "
            f"{MIN_OBSERVATIONS} and would decline every request, which is a "
            f"broken measurement rather than a negative result"
        )
    stage = ReasoningBudgetStage()
    _warm(stage, control.lengths)
    ceiling = _chosen_ceiling(stage)
    if ceiling is None:
        raise SystemExit(
            f"the warmed stage declines a {CALLER_BUDGET:,}-token budget, so both "
            f"arms would be identical. Observed output p95 is already above "
            f"half the budget; a bigger over-provision or a shorter workload is "
            f"needed for this to measure anything."
        )
    print(f"treated arm: budget {CALLER_BUDGET:,} -> {ceiling:,} as the stage decides")
    treated = run(stage=stage)

    # A second control arm, run *after* the treated one, and it is not
    # redundant. Extended thinking runs at temperature 1, so a single pair of
    # arms cannot distinguish "the budget shortened the trace" from "the second
    # arm drew shorter traces". Two controls bracketing the treated arm can:
    # if both land together and above it, order is ruled out.
    print("control arm again, after the treated one, to bracket it")
    control_2 = run(stage=None)

    print(
        f"\n{'arm':<10} {'input':>8} {'output':>9} {'longest':>8} {'cost':>10} "
        f"{'easy':>6} {'hard':>6}"
    )
    print("-" * 64)
    for label, arm in (("control", control), ("treated", treated), ("control-2", control_2)):
        print(
            f"{label:<10} {arm.input_tokens:>8,} {arm.output_tokens:>9,} "
            f"{max(arm.lengths):>8,} ${arm.usd:>9.5f} "
            f"{_pct(arm.accuracy(EASY)):>6} {_pct(arm.accuracy(HARD)):>6}"
        )

    # Against the mean of the two controls, not the first one: the first is a
    # single sample of a stochastic quantity, and picking whichever control
    # flatters the stage is how a measurement becomes a marketing number.
    baseline = (control.usd + control_2.usd) / 2
    if baseline:
        change = (baseline - treated.usd) / baseline
        print(f"\ncost change against the mean of both controls: {change:+.1%}")
        print(
            f"spread between the two controls alone: "
            f"{abs(control.usd - control_2.usd) / baseline:.1%} -- the noise floor "
            f"any claim above has to clear"
        )
    print(
        f"output tokens: {control.output_tokens:,} / {control_2.output_tokens:,} "
        f"control, {treated.output_tokens:,} treated"
    )
    print(
        f"truncated: control {len(control.truncated)}, treated {len(treated.truncated)}, "
        f"control-2 {len(control_2.truncated)}"
    )

    # **The number that decides whether the cost line above means anything.**
    # Extended thinking runs at temperature 1 -- the API requires it -- so the
    # trace length is stochastic and two runs of twenty tasks differ by tens of
    # percent on their own. The budget can only have bound on a call whose
    # unconstrained output exceeded the ceiling. If none did, the cost delta is
    # sampling noise wearing a stage's name, which is the exact mistake that
    # published 36.3% before a live run said -1.8%.
    unconstrained = control.lengths + control_2.lengths
    bindable = [n for n in unconstrained if n > ceiling]
    print(
        f"\nunconstrained output lengths across both controls: max "
        f"{max(unconstrained):,}, {len(bindable)}/{len(unconstrained)} above the "
        f"{ceiling:,} ceiling"
    )
    if not bindable:
        print(
            "  -> no call could have been cut off by the ceiling. Any difference "
            "above is therefore NOT the ceiling binding. Either it is variance, "
            "or the provider treats budget_tokens as a target that shapes the "
            "trace rather than a cap that truncates it -- and those have very "
            "different safety implications. The bracketing controls are what "
            "tells them apart."
        )
    for label, arm in (("control", control), ("treated", treated), ("control-2", control_2)):
        if arm.wrong:
            detail = ", ".join(f"{tid} expected {want} got {got}" for tid, want, got in arm.wrong)
            print(f"{label} missed: {detail}")

    print(
        "\nThe token counts and the accuracy are the measurement. A cost "
        "reduction here is only evidence for this stage if the hard-set accuracy "
        "held; ADR-018 requires both numbers or neither."
    )
