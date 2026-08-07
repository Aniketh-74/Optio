"""The sampled LLM-judge (M5-3).

The deep quality signal, and the only component in optio that ever reads
trace content. Everything about its design follows from two constraints.

## It must never block the agent

The judge is a model call: hundreds of milliseconds at best, and it can hang.
The SC-5 budget is 5 ms per step. So the judge never runs on the hot path -- it
is dispatched at run end, to a worker thread, and the run does not wait for it.

**The answer therefore arrives after the run span has closed, and an ended span
cannot be modified.** The score is pushed to a callback when it lands
(:meth:`JudgeRunner.submit`), and the lane emits it on a span of its own linked
back to the run's -- see :mod:`optio.lanes.quality.deferred`.

This replaces dispatching and immediately polling with a zero-second timeout,
which is what this module did through 0.3.0. That could only ever succeed if the
judge finished between two adjacent statements on the same thread: measured
against a warm worker pool it collected 2 scores in 200 runs with an *instant*
in-process judge, and none at all with a 200 ms one. The tests passed because
each built a fresh lane, whose cold ``ThreadPoolExecutor`` blocks in
``Thread.start`` until the worker is running -- handing an instant judge exactly
the head start a real deployment never gives it. So the tier that spends the
user's money delivered nothing, and every test agreed it worked.

A judge that fails, hangs, or declines still emits **no quality signal at all**.
Absent, not zero, not "pending" (docs/signals.md).

## It must never be a credential or content risk

Section 10 is explicit: we store no keys, and the judge runs on *the user's own*
SDK and credentials. This module therefore defines only a **callable type**, not
an integration -- optio never constructs a model client, never reads an API
key from the environment, and never has a default provider that silently starts
calling a paid API. A user who enables the quality lane without supplying a
judge gets the heuristic tier and a warning, not a surprise invoice.

The content the judge sees is the user's own trace, passed to the user's own
model. optio neither logs it nor retains it: :class:`JudgeRequest` is
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
from functools import partial
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

_log: Final = logging.getLogger("optio")

#: Worker threads for judge calls. Small on purpose: the judge is off the hot
#: path, so throughput does not matter, and an unbounded pool would let a slow
#: judge accumulate threads in a long-lived agent process.
_MAX_WORKERS: Final = 2

#: Seconds to wait for a judge result at collection time. Not the judge's own
#: timeout -- the user's client owns that. This is how long the *run* is willing
#: to wait, which is approximately not at all.
DEFAULT_COLLECT_TIMEOUT: Final = 0.0

#: Most judgements allowed in flight at once. Beyond this, new ones are refused.
#:
#: The pool has :data:`_MAX_WORKERS` threads, so sustained throughput is bounded
#: by the judge's own latency -- roughly ten runs a second for a 200 ms judge.
#: An agent that ends sampled runs faster than that would otherwise queue them
#: without limit, and every queued entry pins a request until it is delivered.
#: Refusing is the correct outcome: the judge is a sampled, best-effort signal,
#: and dropping one is a gap in a graph (ADR-004), whereas an unbounded queue is
#: a memory leak that ends the process. Sized so an ordinary burst rides through
#: and a genuine mismatch of rates is caught early.
MAX_PENDING: Final = 64

#: Seconds :meth:`JudgeRunner.drain` waits for outstanding judgements. Generous
#: next to :data:`DEFAULT_COLLECT_TIMEOUT` because it is paid only when the
#: caller has explicitly asked to shut down, never on a run's path -- and the
#: thing being waited for is a model call the user has already paid for.
DEFAULT_DRAIN_TIMEOUT: Final = 5.0

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
            and passed straight through -- optio does not read, log, or
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
#: Implemented by the user against their own SDK and credentials. optio
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

    __slots__ = ("_judge", "_lock", "_pending", "_pool", "_saturated")

    def __init__(self, judge: Judge | None) -> None:
        """Create a runner.

        Args:
            judge: The user's judge, or ``None`` to disable judging entirely.
        """
        self._judge = judge
        # A `Condition`, not a `Lock`, so `drain` can wait on delivery finishing
        # rather than poll for it. Used as a plain lock everywhere else.
        self._lock = threading.Condition()
        # Holds a run until its scores have been *delivered*, not merely
        # computed. `Future.set_result` wakes waiters before it runs the done
        # callbacks, so a drain that waited on the futures could return while a
        # score was still on its way to being emitted.
        self._pending: dict[str, Future[JudgeScores]] = {}
        # Created lazily: a user with the quality lane on but no judge supplied
        # should not pay for a thread pool that will never be used.
        self._pool: ThreadPoolExecutor | None = None
        self._saturated = False

    @property
    def enabled(self) -> bool:
        """Whether a judge is available to call."""
        return self._judge is not None

    def submit(
        self,
        request: JudgeRequest,
        on_scores: Callable[[JudgeScores], None] | None = None,
    ) -> bool:
        """Dispatch a judge call without waiting for it.

        Two delivery modes, and the caller picks one by whether it passes
        ``on_scores``:

        * **Push** -- ``on_scores`` is called on a worker thread as soon as the
          judge answers. The only mode that can actually deliver a model call's
          result, because the alternative below has to be *polled*, and the run
          is over by the time there is anything to poll for.
        * **Pull** -- omit it and call :meth:`collect`. Suits a caller that can
          afford to wait.

        Args:
            request: The run to evaluate.
            on_scores: Called with the validated scores when they arrive.
                Never called when the judge failed, timed out, or declined --
                absence is not a score (ADR-044). Runs on a worker thread, so
                it must not raise; one that does is logged and swallowed.

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
            if len(self._pending) >= MAX_PENDING:
                # Runs are ending faster than the judge can answer. Refusing
                # sheds load in the only direction that is safe: this run goes
                # unscored, rather than the queue growing until the process
                # dies. Also stops a hung judge from accumulating requests
                # forever behind it.
                self._note_saturation()
                return False
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS, thread_name_prefix="optio-judge"
                )
            judge = self._judge
            future = self._pool.submit(judge, request)
            self._pending[request.run_id] = future

        if on_scores is not None:
            # Registered outside the lock: a future that finished while we held
            # it runs this callback synchronously on *this* thread, and
            # `_deliver` takes the same lock.
            future.add_done_callback(partial(self._deliver, request.run_id, on_scores))
        return True

    def _deliver(
        self,
        run_id: str,
        on_scores: Callable[[JudgeScores], None],
        future: Future[JudgeScores],
    ) -> None:
        """Hand a finished judgement to its callback, on the worker's thread.

        Total by construction. This runs on a pool thread with nothing above it
        to catch anything -- the fail-open guard wraps the *agent's* call stack,
        and this is no longer on it. An escaping exception here would be
        swallowed by ``concurrent.futures`` and logged as an unhandled callback
        error, which is a confusing way for a user to learn their judge is
        broken.

        Args:
            run_id: The run being judged.
            on_scores: The caller's delivery callback.
            future: The completed future.
        """
        try:
            # Zero timeout is not a wait here: the future is already done, which
            # is the only reason this callback is running.
            scores = self._read(future, timeout=0.0)
            if scores is not None:
                on_scores(scores)
        except Exception as error:  # noqa: BLE001 - see docstring: must be total
            _log.warning(
                "optio: could not deliver a quality judgement (%s); no quality signal for this run",
                type(error).__name__,
            )
        finally:
            # Released last, and in a `finally`: a run that stayed pending after
            # a failed delivery would make `drain` wait out its whole timeout at
            # every shutdown.
            self._release(run_id)

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
        return self._read(future, timeout=timeout)

    def _read(self, future: Future[JudgeScores], timeout: float) -> JudgeScores | None:
        """Turn a future into validated scores, or into nothing.

        Shared by both delivery modes so that a judge which returns garbage is
        rejected identically whether it was collected or pushed -- a validation
        rule that exists on one path only is a rule that stops running the day
        the other path becomes the common one.

        Args:
            future: The judge's future.
            timeout: Seconds to wait for it.

        Returns:
            Validated scores, or ``None`` when the judge is unfinished, failed,
            returned the wrong type, or declined to score.
        """
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
                "optio: quality judge did not produce a score (%s); no quality signal for this run",
                type(error).__name__,
            )
            # Not cancelled on timeout: the call is already in flight and
            # cancelling would not stop the spend, only discard what it bought.
            return None

        if not isinstance(scores, JudgeScores):
            _log.warning(
                "optio: quality judge returned %s, expected JudgeScores; "
                "no quality signal for this run",
                type(scores).__name__,
            )
            return None

        validated = scores.validated()
        return None if validated.is_empty else validated

    def _note_saturation(self) -> None:
        """Log the first refusal, then stay quiet. Caller holds the lock.

        Once, because a saturated judge saturates on every run, and a line per
        run would flood the logs at exactly the moment the user's system is
        under load -- turning our dropped signal into their second incident.
        """
        if self._saturated:
            return
        self._saturated = True
        _log.warning(
            "optio: %d quality judgements are already in flight; skipping this run's. "
            "The judge is answering more slowly than runs are ending, so some runs "
            "will go unscored. Further skips are not logged. See docs/quality.md.",
            MAX_PENDING,
        )

    def _release(self, run_id: str) -> None:
        """Drop a run from the pending set and wake anything draining.

        Args:
            run_id: The run to release.
        """
        with self._lock:
            self._pending.pop(run_id, None)
            self._lock.notify_all()

    def discard(self, run_id: str) -> None:
        """Forget a pending judgement.

        Args:
            run_id: The run to drop.
        """
        self._release(run_id)

    def drain(self, timeout: float = DEFAULT_DRAIN_TIMEOUT) -> int:
        """Wait for in-flight judgements to finish being delivered.

        The judge is deliberately asynchronous, which means a process that
        stops immediately after its last run loses the scores it just paid for.
        This is how a caller says "I am finished, let the answers land" --
        bounded, because a hung judge must not become a hung shutdown.

        Args:
            timeout: Seconds to wait.

        Returns:
            How many judgements were still undelivered when the wait ended.
            Non-zero means scores were abandoned, which is a dropped signal and
            never an error (ADR-004).
        """
        with self._lock:
            self._lock.wait_for(lambda: not self._pending, timeout=timeout)
            return len(self._pending)

    def pending_count(self) -> int:
        """Return how many judgements are outstanding.

        Exposed so a test can observe that state is released rather than
        accumulating in a long-lived process.
        """
        with self._lock:
            return len(self._pending)

    def shutdown(self, drain_timeout: float = DEFAULT_DRAIN_TIMEOUT) -> None:
        """Let outstanding judgements land, then release the worker pool.

        Safe to call more than once.

        Args:
            drain_timeout: Seconds to wait for in-flight judgements before
                dropping them. Pass ``0`` to discard immediately. Draining
                first because the alternative is throwing away a model call the
                user has already been billed for, purely because the process
                asked to stop a few milliseconds too early.
        """
        if drain_timeout > 0:
            self.drain(drain_timeout)
        with self._lock:
            pool, self._pool = self._pool, None
            self._pending.clear()
            self._lock.notify_all()
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
