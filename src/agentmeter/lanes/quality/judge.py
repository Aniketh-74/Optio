"""The sampled LLM-judge (M5-3).

The deep quality signal, and the only component in agentmeter that ever reads
trace content. Everything about its design follows from two constraints.

## It must never block the agent

The judge is a model call: hundreds of milliseconds at best, and it can hang.
The SC-5 budget is 5 ms per step. So the judge never runs on the hot path -- it
is dispatched at run end, to a worker thread, and the run does not wait for it.

If the judge has not answered by the time the caller wants signals, **no quality
signal is emitted**. Absent, not zero, not "pending" (docs/signals.md). A late
score is worth less than a fast run, and the cost lane's numbers are already
correct without it.

## It must never be a credential or content risk

Section 10 is explicit: we store no keys, and the judge runs on *the user's own*
SDK and credentials. This module therefore defines only a **callable type**, not
an integration -- agentmeter never constructs a model client, never reads an API
key from the environment, and never has a default provider that silently starts
calling a paid API. A user who enables the quality lane without supplying a
judge gets the heuristic tier and a warning, not a surprise invoice.

The content the judge sees is the user's own trace, passed to the user's own
model. agentmeter neither logs it nor retains it: :class:`JudgeRequest` is
handed to the callable and dropped, and only the numeric scores come back.

## Scores are bounded and validated

A judge is itself a model, and models return malformed output. Scores outside
``[0, 1]`` are rejected rather than clamped -- a judge returning ``7`` has
misunderstood the scale, and clamping to ``1.0`` would publish a confident wrong
number where absence is the honest answer (R-TECH-5, "who evals the evaluator").
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

_log: Final = logging.getLogger("agentmeter")

#: Worker threads for judge calls. Small on purpose: the judge is off the hot
#: path, so throughput does not matter, and an unbounded pool would let a slow
#: judge accumulate threads in a long-lived agent process.
_MAX_WORKERS: Final = 2

#: Seconds to wait for a judge result at collection time. Not the judge's own
#: timeout -- the user's client owns that. This is how long the *run* is willing
#: to wait, which is approximately not at all.
DEFAULT_COLLECT_TIMEOUT: Final = 0.0

#: Valid range for every score. Inclusive at both ends.
SCORE_MIN: Final = 0.0
SCORE_MAX: Final = 1.0


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """What the judge is asked to evaluate.

    Attributes:
        run_id: The run's identifier, for the user's own correlation.
        step_count: How many steps the run took.
        content: Trace content for the judge to read. Supplied by the caller
            and passed straight through -- agentmeter does not read, log, or
            retain it (Section 10). Empty when the user has redaction on.
    """

    run_id: str
    step_count: int
    content: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class JudgeScores:
    """What a judge returned.

    Attributes:
        groundedness: Whether the answer is supported by retrieved context,
            in ``[0, 1]``. ``None`` when the judge did not score it.
        task_success: Whether the run accomplished the task, in ``[0, 1]``.
            ``None`` when the judge did not score it.
    """

    groundedness: float | None = None
    task_success: float | None = None

    def validated(self) -> JudgeScores:
        """Return a copy with out-of-range or non-numeric scores dropped.

        Returns:
            Scores with every invalid entry replaced by ``None``. Dropping
            rather than clamping: a score of ``7`` means the judge misread the
            scale, and clamping it to ``1.0`` would publish a confident wrong
            number where absence is honest.
        """
        return JudgeScores(
            groundedness=_valid_score(self.groundedness),
            task_success=_valid_score(self.task_success),
        )

    @property
    def is_empty(self) -> bool:
        """Whether both scores are absent."""
        return self.groundedness is None and self.task_success is None


#: A user-supplied outcome evaluator.
#:
#: Implemented by the user against their own SDK and credentials. agentmeter
#: never constructs one and ships no default, because a default would mean
#: silently spending the user's money the moment they enabled the lane.
#:
#: The judge may block and may raise; both are handled by the runner, which
#: calls it on a worker thread and never on the agent's path. Returning an empty
#: :class:`JudgeScores` declines to score.
#:
#: A plain ``Callable`` alias rather than a ``Protocol`` with ``__call__``:
#: mypy will not accept a bare function against such a protocol, so the obvious
#: way to write a judge --
#:
#:     def my_judge(request: JudgeRequest) -> JudgeScores: ...
#:
#: -- would fail to type-check for every user. An API whose documented usage its
#: own type checker rejects is a broken API.
Judge = Callable[[JudgeRequest], "JudgeScores"]


class JudgeRunner:
    """Dispatches judge calls off the hot path and collects what is ready.

    One instance per lane. Thread-safe.
    """

    __slots__ = ("_judge", "_lock", "_pending", "_pool")

    def __init__(self, judge: Judge | None) -> None:
        """Create a runner.

        Args:
            judge: The user's judge, or ``None`` to disable judging entirely.
        """
        self._judge = judge
        self._lock = threading.Lock()
        self._pending: dict[str, Future[JudgeScores]] = {}
        # Created lazily: a user with the quality lane on but no judge supplied
        # should not pay for a thread pool that will never be used.
        self._pool: ThreadPoolExecutor | None = None

    @property
    def enabled(self) -> bool:
        """Whether a judge is available to call."""
        return self._judge is not None

    def submit(self, request: JudgeRequest) -> bool:
        """Dispatch a judge call without waiting for it.

        Args:
            request: The run to evaluate.

        Returns:
            Whether the call was dispatched.
        """
        if self._judge is None:
            return False

        with self._lock:
            if request.run_id in self._pending:
                # Run end can fire more than once (M1-2). A second dispatch
                # would pay for the same judgement twice, on the user's money.
                return False
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS, thread_name_prefix="agentmeter-judge"
                )
            judge = self._judge
            self._pending[request.run_id] = self._pool.submit(judge, request)
        return True

    def collect(self, run_id: str, timeout: float = DEFAULT_COLLECT_TIMEOUT) -> JudgeScores | None:
        """Take a completed judgement, if one is ready.

        Args:
            run_id: The run to collect for.
            timeout: Seconds to wait. Defaults to not waiting at all -- the run
                is over and the caller is on its way out.

        Returns:
            Validated scores, or ``None`` when the judge is unfinished, absent,
            or failed. A failure is a missing signal, never an error the agent
            sees (ADR-004).
        """
        with self._lock:
            future = self._pending.pop(run_id, None)
        if future is None:
            return None

        try:
            # Typed as `object`, not `JudgeScores`: the annotation on a
            # user-supplied callable is a promise, not a guarantee, and this is
            # the boundary where an untyped or wrong judge arrives. Trusting the
            # declared type here would make the isinstance check below dead code
            # to the type checker while remaining live at runtime.
            scores: object = future.result(timeout=timeout)
        except Exception as error:  # noqa: BLE001 - a judge is user code (ADR-004)
            # Type only, never the message: an exception raised by a model
            # client routinely carries the prompt in its payload (Section 10).
            _log.warning(
                "agentmeter: quality judge did not produce a score (%s); "
                "no quality signal for this run",
                type(error).__name__,
            )
            # Not cancelled on timeout: the call is already in flight and
            # cancelling would not stop the spend, only discard what it bought.
            return None

        if not isinstance(scores, JudgeScores):
            _log.warning(
                "agentmeter: quality judge returned %s, expected JudgeScores; "
                "no quality signal for this run",
                type(scores).__name__,
            )
            return None

        validated = scores.validated()
        return None if validated.is_empty else validated

    def discard(self, run_id: str) -> None:
        """Forget a pending judgement.

        Args:
            run_id: The run to drop.
        """
        with self._lock:
            self._pending.pop(run_id, None)

    def pending_count(self) -> int:
        """Return how many judgements are outstanding.

        Exposed so a test can observe that state is released rather than
        accumulating in a long-lived process.
        """
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        """Release the worker pool.

        Safe to call more than once.
        """
        with self._lock:
            pool, self._pool = self._pool, None
            self._pending.clear()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def _valid_score(value: float | None) -> float | None:
    """Return a score if it is a real number in range, else ``None``.

    Args:
        value: The candidate score.

    Returns:
        The score, or ``None`` when absent, non-numeric, NaN, or out of range.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # NaN fails every comparison, so the range check below rejects it -- stated
    # here because "why does this catch NaN" is otherwise a puzzle.
    if not SCORE_MIN <= value <= SCORE_MAX:
        return None
    return float(value)
