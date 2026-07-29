"""Asynchronous batch dispatch — the second surface (ADR-017).

Every major provider sells asynchronous processing at roughly **50% off**, with
a turnaround measured in hours. It is the largest unconditional discount in the
field and the only one that costs no quality at all: the same model returns the
same answer, later.

It cannot be a :class:`~optio_optimize.stages.base.Stage`. Every stage answers
*what should this request look like*, and the pipeline's contract is that a
request goes in and a response comes back on the same stack frame. Batch answers
*when should this be sent, and by whom*, and its answer is "in a few hours, by
someone who is no longer here". There is no response to return and no error to
fail open into, because nothing has gone wrong.

So it is a separate class, and it is honest about being asynchronous:

* **The caller declares latency tolerance; this library never infers it.**
  There is no heuristic for "this looks like it can wait". Submitting a
  user-facing request to a 24-hour queue to save half a cent is a product
  failure this package must be structurally incapable of causing, which is why
  it is a different method rather than a flag on a shared path.
* **The optimization pipeline runs first, unchanged.** A batched request is
  still worth trimming and minifying, and the discounts compose -- the batch
  rate applies to whatever tokens survive the stages. :class:`BatchOptimizer`
  *owns* an :class:`~optio_optimize.Optimizer` rather than reimplementing it.
* **The exact cache is checked before submission and populated on retrieval.**
  A request already answered should not enter a queue at all.
* **Failure is explicit, not fail-open.** ADR-013's rule 1 works synchronously
  because there is somewhere to fall back to. A failed batch submission has not
  degraded to a slower path; it has not happened. Quietly converting it into
  10,000 synchronous calls would be a fail-open that costs twice the money it
  was asked to save, so :meth:`BatchOptimizer.submit` raises.
* **Polling is the caller's, with a helper.** No background thread: a library
  that spawns one inside somebody's web worker behaves differently in every
  deployment.

**On the savings number.** Batch savings reported here are *arithmetic* -- the
provider's published discount applied to measured token counts -- not measured
end-to-end the way every claim in ``docs/optimize-benchmarks.md`` is. The
benchmark harness is built around a synchronous ``compare()`` and cannot express
a result that arrives tomorrow. That is a weaker class of evidence than this
project otherwise ships and the reason is the clock rather than a shortcut, so
:class:`BatchReport` says so in its own output rather than leaving a reader to
assume parity.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from optio_optimize.config import PRICING
from optio_optimize.errors import OptimizeError
from optio_optimize.optimizer import Optimizer

if TYPE_CHECKING:
    from optio_optimize.config import OptimizeConfig
    from optio_optimize.pipeline import PreparedRequest
    from optio_optimize.types import LLMRequest, LLMResponse

_log = logging.getLogger("optio_optimize")

#: Fraction of the synchronous price a batched call is billed at. Both OpenAI
#: and Anthropic publish 50% on input and output alike. Held as a constant
#: rather than a per-model column in :data:`~optio_optimize.config.PRICING`
#: because it is a uniform multiplier on whatever that table says -- but it is
#: a *published* figure this package has not measured, which is exactly why
#: :meth:`BatchReport.summary_lines` labels it.
BATCH_DISCOUNT = 0.5

#: In-flight batches whose stage state is retained. The state is what lets a
#: retrieved result populate the exact cache and land in the savings report,
#: and it is per-process by construction: nothing here is persisted, because
#: ADR-017's "no queue, no persistence" excludes infrastructure the caller
#: would have to operate. A ceiling because this runs in a long-lived agent
#: process and an unbounded dict of held requests is the leak §11 exists to
#: catch. Eviction loses accounting, never correctness -- results for an
#: evicted handle still fetch, they just arrive unattributed.
MAX_TRACKED_BATCHES = 64


class BatchState(Enum):
    """Where a submitted batch is.

    Deliberately coarser than either provider's own status vocabulary. OpenAI
    distinguishes ``validating``/``in_progress``/``finalizing`` and Anthropic
    reports ``in_progress``/``canceling``/``ended``; none of that difference is
    actionable to a caller, who can only wait or stop waiting. Mapping both
    onto four states is what keeps the calling code from having to know which
    provider it is talking to.
    """

    #: Queued, running, or finalizing. Results are not available yet.
    PENDING = "pending"
    #: Finished. Results are available, though individual items may still have
    #: errored -- a completed batch is not a successful one.
    COMPLETED = "completed"
    #: The batch itself failed. No results.
    FAILED = "failed"
    #: The provider's completion window elapsed. Partial results may exist.
    EXPIRED = "expired"
    #: Cancelled, by this library or elsewhere.
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether waiting any longer cannot change the answer."""
        return self is not BatchState.PENDING


class BatchError(OptimizeError):
    """Base class for batch dispatch failures.

    This package otherwise has no runtime exception type at all -- ADR-013's
    rule 1 turns every runtime failure into a skipped stage, so a runtime error
    class would advertise a control flow the pipeline does not offer. Batch is
    the exception, and it is one on purpose (ADR-017 decision 4): there is no
    degraded path to fall back to, so silence here would mean a caller believing
    work is queued that is not.
    """


class BatchSubmissionError(BatchError):
    """A batch was not accepted, and here is exactly what happened to each item.

    Attributes:
        accepted: ``custom_id`` values the provider took. Empty in the usual
            case -- both providers validate a submission as a unit -- but not
            assumed empty, because a future provider that partially accepts
            must not be representable as "everything failed".
        rejected: ``custom_id`` values that were not submitted.
    """

    def __init__(self, message: str, *, accepted: Sequence[str], rejected: Sequence[str]) -> None:
        super().__init__(message)
        self.accepted = tuple(accepted)
        self.rejected = tuple(rejected)


class BatchTimeoutError(BatchError):
    """:meth:`BatchOptimizer.await_results` gave up before the batch finished.

    Not a failure of the batch. The work is still queued and the handle is still
    good -- which is why this carries it.

    Attributes:
        handle: The batch still in flight. Fetch it later with
            :meth:`BatchOptimizer.results`.
        waited_seconds: How long the caller actually blocked.
    """

    def __init__(self, message: str, *, handle: BatchHandle, waited_seconds: float) -> None:
        super().__init__(message)
        self.handle = handle
        self.waited_seconds = waited_seconds


@dataclass(frozen=True, slots=True)
class BatchItem:
    """One request in a batch, with the id its answer will come back under.

    Attributes:
        custom_id: The caller's key for this request. Unique within a batch;
            both providers enforce that and this library checks it before
            submission so the error names the duplicate rather than arriving
            as a provider validation failure with no context.
        request: The call to make, before optimization.
    """

    custom_id: str
    request: LLMRequest


@dataclass(frozen=True, slots=True)
class BatchHandle:
    """What a caller must store to collect results later.

    Deliberately small and JSON-serializable: ADR-017 rules out persistence
    inside this library, so the caller keeps this in whatever they already use
    for job state. Everything needed to *fetch* is here. What is not here is the
    stage state -- savings accounting and cache population need the same
    :class:`BatchOptimizer` instance that submitted, and if the process has
    restarted those are lost while the results themselves are not.

    Attributes:
        id: The provider's batch identifier.
        backend: Which backend issued it, so a caller holding several knows
            where to send it back.
        model: Model the batch was submitted against.
        custom_ids: Every id in the batch, in submission order.
        submitted_at: Unix timestamp of submission.
    """

    id: str
    backend: str
    model: str
    custom_ids: tuple[str, ...]
    submitted_at: float


@dataclass(frozen=True, slots=True)
class BatchSubmission:
    """The outcome of :meth:`BatchOptimizer.submit`.

    Two channels, because a submission genuinely has two outcomes. Requests the
    exact cache already had an answer for are returned immediately and never
    enter the queue; the rest are in flight behind ``handle``. Collapsing those
    into one would mean either waiting a day for an answer already in memory or
    losing track of which is which.

    Attributes:
        handle: The in-flight batch, or ``None`` when every request was served
            from cache and nothing was submitted.
        served: ``custom_id`` to response, for requests answered without the
            queue.
        submitted: ``custom_id`` values that went to the provider.
    """

    handle: BatchHandle | None
    served: Mapping[str, LLMResponse]
    submitted: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """Whether everything was answered without submitting anything."""
        return self.handle is None


@dataclass(frozen=True, slots=True)
class BatchResults:
    """What came back.

    Attributes:
        state: Where the batch is. ``PENDING`` means the other two fields are
            empty and nothing has gone wrong.
        responses: ``custom_id`` to response, for items that succeeded.
        errors: ``custom_id`` to a provider error message, for items that did
            not. A completed batch with errors is normal and is not a failure
            of the batch -- one malformed request does not sink the other 9,999.
    """

    state: BatchState
    responses: Mapping[str, LLMResponse]
    errors: Mapping[str, str]

    @property
    def is_pending(self) -> bool:
        """Whether the provider is still working."""
        return self.state is BatchState.PENDING


class BatchBackend(Protocol):
    """A provider's batch endpoint.

    Narrow on purpose. Everything above this line -- optimization, cache
    checking, accounting, waiting -- is provider-independent and lives in
    :class:`BatchOptimizer`; a backend only has to know one vendor's transport.
    """

    @property
    def name(self) -> str:
        """Stable identifier, recorded on the handle."""
        ...

    @property
    def model(self) -> str:
        """Model this backend submits against."""
        ...

    def submit(self, items: Sequence[BatchItem]) -> str:
        """Send a batch and return the provider's id.

        Args:
            items: Requests, already optimized, with unique ``custom_id``s.

        Returns:
            The provider-issued batch id.

        Raises:
            Exception: Any transport or validation failure.
                :meth:`BatchOptimizer.submit` wraps it in a
                :class:`BatchSubmissionError` naming the affected items.
        """
        ...

    def poll(self, batch_id: str) -> BatchState:
        """Return where the batch is, without downloading results."""
        ...

    def fetch(self, batch_id: str) -> tuple[Mapping[str, LLMResponse], Mapping[str, str]]:
        """Download results.

        Returns:
            ``(responses, errors)``, both keyed by ``custom_id``.
        """
        ...

    def cancel(self, batch_id: str) -> None:
        """Ask the provider to stop."""
        ...


@dataclass(slots=True)
class BatchReport:
    """What batching saved, arithmetically.

    Kept separate from :class:`~optio_optimize.savings.SavingsReport` rather
    than folded into it, because they measure different things and adding them
    together would misrepresent both. A stage saving is *tokens not sent*, and
    it is measured. A batch discount is *the same tokens billed at half rate*,
    and it is the provider's published multiplier -- the same distinction the
    prefix-cache stage draws when it reports zero tokens saved. Reporting the
    discount as avoided tokens would inflate every reduction ratio in the
    package by an amount nobody had measured.

    Attributes:
        submitted: Requests that entered a queue.
        served_from_cache: Requests answered before submission, which cost
            nothing at all rather than half.
        failed: Items the provider returned an error for.
        input_tokens: Prompt tokens billed at the batch rate.
        output_tokens: Completion tokens billed at the batch rate.
        batches: Submissions made.
    """

    submitted: int = 0
    served_from_cache: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    batches: int = 0

    def estimated_discount_usd(self, model: str) -> float | None:
        """Money not spent by batching rather than calling synchronously.

        Args:
            model: Model to price against.

        Returns:
            USD, or ``None`` for an unpriced model. ``None`` rather than a
            zero, which would read as "batching saved nothing" instead of "we
            do not know this model's price" -- the absence-is-not-zero rule the
            rest of the package follows.
        """
        pricing = PRICING.get(model)
        if pricing is None:
            return None
        full_rate = (
            self.input_tokens * pricing.input_usd_per_m
            + self.output_tokens * pricing.output_usd_per_m
        ) / 1_000_000
        return full_rate * (1.0 - BATCH_DISCOUNT)

    def summary_lines(self, model: str = "") -> list[str]:
        """Render a human-readable report.

        Args:
            model: Model to price against; omit for token counts only.

        Returns:
            Lines suitable for printing or logging. The last one is a caveat,
            and it is not optional: this figure is arithmetic where every other
            number this package prints is measured, and a reader comparing them
            side by side has no way to tell unless it says so.
        """
        lines = [
            f"batches:  {self.batches}  ({self.submitted} requests submitted, "
            f"{self.served_from_cache} served from cache, {self.failed} failed)",
            f"tokens:   {self.input_tokens:,} in / {self.output_tokens:,} out, "
            f"billed at {BATCH_DISCOUNT:.0%} of the synchronous rate",
        ]
        if model:
            discount = self.estimated_discount_usd(model)
            lines.append(
                f"discount: ${discount:.4f} not spent on {model}"
                if discount is not None
                else f"discount: unpriced model {model!r}; token counts only"
            )
        lines.append(
            "note:     computed from the provider's published discount, not measured "
            "end-to-end -- the A/B harness cannot express a result that arrives tomorrow"
        )
        return lines


@dataclass(slots=True)
class _TrackedBatch:
    """Stage state held between submission and retrieval."""

    prepared: dict[str, PreparedRequest]
    model: str
    run_id: str | None


class BatchOptimizer:
    """Optimize a set of requests and dispatch them at the provider's batch rate.

    Typical use::

        from optio_optimize import BatchOptimizer
        from optio_optimize.batch_backends import OpenAIBatchBackend

        batcher = BatchOptimizer(OpenAIBatchBackend(model="gpt-4o-mini"))
        submission = batcher.submit([BatchItem("row-1", request), ...])
        # ... hours later, in this process or after storing submission.handle:
        results = batcher.results(submission.handle)

    Calling this is a statement that the work tolerates hours of latency. There
    is no check on that and there cannot be one -- nothing in a request
    distinguishes a nightly enrichment job from a user waiting -- which is why
    it is a separate class rather than a flag.

    Attributes:
        backend: The provider endpoint submissions go to.
        optimizer: The synchronous optimizer whose stages, exact cache and
            savings report this shares.
        report: Batch-specific accounting. Token savings from the stages land
            in ``optimizer.report``; the discount lands here.
    """

    __slots__ = ("_tracked", "backend", "optimizer", "report")

    def __init__(
        self,
        backend: BatchBackend,
        config: OptimizeConfig | None = None,
        *,
        optimizer: Optimizer | None = None,
    ) -> None:
        """Build a batch optimizer.

        Args:
            backend: Provider batch endpoint.
            config: Configuration for the optimizer built when ``optimizer`` is
                omitted. Ignored -- and rejected -- when one is passed, since
                that optimizer already has a config and silently having two
                would mean stages differing between the two surfaces for
                reasons invisible at the call site.
            optimizer: An existing optimizer to share. **Pass the one your
                synchronous path uses.** ADR-017 decision 3 makes the exact
                cache shared deliberately: a request already answered should
                not enter a queue, and an answer that arrives hours later is as
                cacheable as one that arrives immediately. Omit it and this
                builds its own, which is correct but shares nothing.

        Raises:
            OptimizeConfigError: If both ``config`` and ``optimizer`` are given.
        """
        if config is not None and optimizer is not None:
            from optio_optimize.errors import OptimizeConfigError

            raise OptimizeConfigError(
                "pass either a config or an existing optimizer, not both; the "
                "optimizer already carries one, and two would mean the batch and "
                "synchronous surfaces silently running different stages"
            )
        self.backend = backend
        self.optimizer = optimizer if optimizer is not None else Optimizer(config)
        self.report = BatchReport()
        self._tracked: OrderedDict[str, _TrackedBatch] = OrderedDict()

    def submit(
        self,
        items: Sequence[BatchItem],
        *,
        run_id: str | None = None,
    ) -> BatchSubmission:
        """Optimize every request, serve what is cached, queue the rest.

        Args:
            items: Requests to batch, each with a ``custom_id`` unique within
                the call.
            run_id: optio run these belong to, for attributing savings.

        Returns:
            The submission: a handle for whatever was queued, plus any
            responses the exact cache answered outright.

        Raises:
            ValueError: If ``items`` is empty or two share a ``custom_id``.
                Checked here so the message names the duplicate, rather than
                surfacing as a provider validation error with no context.
            BatchSubmissionError: If the provider rejected the submission. The
                error names which items were and were not accepted. Nothing is
                retried and nothing falls back to synchronous calls -- that
                would cost twice what batching was asked to save.
        """
        if not items:
            raise ValueError("submit() needs at least one item; got an empty sequence")
        self._reject_duplicate_ids(items)

        pipeline = self.optimizer.pipeline
        served: dict[str, LLMResponse] = {}
        pending: dict[str, PreparedRequest] = {}
        to_send: list[BatchItem] = []

        for item in items:
            prepared = pipeline.prepare(item.request, run_id=run_id)
            if prepared.short_circuit is not None:
                # Already answered. Completing it here runs the after-hooks and
                # books the saving exactly as a synchronous cache hit would --
                # and, more to the point, keeps it out of a queue it would have
                # spent a day in to re-learn something already in memory.
                served[item.custom_id] = pipeline.complete(
                    prepared, prepared.short_circuit, run_id=run_id
                )
                continue
            pending[item.custom_id] = prepared
            to_send.append(BatchItem(custom_id=item.custom_id, request=prepared.request))

        self.report.served_from_cache += len(served)
        if not to_send:
            return BatchSubmission(handle=None, served=served, submitted=())

        batch_id = self._send(to_send)
        handle = BatchHandle(
            id=batch_id,
            backend=self.backend.name,
            model=self.backend.model,
            custom_ids=tuple(item.custom_id for item in to_send),
            submitted_at=time.time(),
        )
        self._track(
            batch_id,
            _TrackedBatch(prepared=pending, model=self.backend.model, run_id=run_id),
        )
        self.report.submitted += len(to_send)
        self.report.batches += 1
        return BatchSubmission(
            handle=handle,
            served=served,
            submitted=tuple(item.custom_id for item in to_send),
        )

    def poll(self, handle: BatchHandle) -> BatchState:
        """Return where a batch is, without downloading results.

        Args:
            handle: What :meth:`submit` returned.

        Returns:
            The batch's state.
        """
        return self.backend.poll(handle.id)

    def results(self, handle: BatchHandle) -> BatchResults:
        """Fetch results, completing the pipeline for each one.

        Every response that comes back runs its stages' ``after`` hooks, which
        is what populates the exact cache and books the request in the savings
        report -- so an answer produced hours later is reusable by the
        synchronous path immediately.

        That accounting needs the stage state held since submission, which is
        per-process (ADR-017 rules out persistence). After a restart the results
        still fetch and are still correct; they simply arrive unattributed, and
        a debug line says so rather than leaving a silently empty report.

        Args:
            handle: What :meth:`submit` returned.

        Returns:
            The results. When still pending, an empty result carrying
            ``BatchState.PENDING`` -- not an error, and not an exception, since
            "not finished yet" is the expected answer for most of a batch's
            life.
        """
        state = self.backend.poll(handle.id)
        if state is BatchState.PENDING:
            return BatchResults(state=state, responses={}, errors={})
        if state is BatchState.FAILED:
            self._tracked.pop(handle.id, None)
            return BatchResults(state=state, responses={}, errors={})

        raw_responses, errors = self.backend.fetch(handle.id)
        tracked = self._tracked.pop(handle.id, None)
        if tracked is None:
            _log.debug(
                "optio_optimize: batch %r is not tracked by this instance; returning "
                "results without savings accounting or cache population",
                handle.id,
            )

        responses = self._complete_all(raw_responses, tracked)
        self.report.failed += len(errors)
        return BatchResults(state=state, responses=responses, errors=errors)

    def await_results(
        self,
        handle: BatchHandle,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 60.0,
    ) -> BatchResults:
        """Block until the batch finishes or the timeout elapses.

        A convenience, not a scheduler. It blocks the calling thread and spawns
        nothing -- a library that starts a background thread inside somebody's
        web worker is a dependency that behaves differently in every deployment.
        In an async application, run this in an executor.

        ``timeout_seconds`` has no default on purpose. Provider completion
        windows run to 24 hours, so any default this library picked would either
        be too short to ever succeed or long enough to hang a process until
        someone killed it.

        Args:
            handle: What :meth:`submit` returned.
            timeout_seconds: How long to wait in total.
            poll_interval_seconds: Gap between checks. A minute by default:
                turnaround is measured in hours, so polling faster buys nothing
                and rate limits are real.

        Returns:
            The completed results.

        Raises:
            ValueError: If either interval is not positive.
            BatchTimeoutError: If the batch is still pending at the deadline.
                The batch is unaffected and the handle still works -- the error
                carries it back for exactly that reason.
        """
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if poll_interval_seconds <= 0:
            raise ValueError(f"poll_interval_seconds must be positive, got {poll_interval_seconds}")

        started = time.monotonic()
        deadline = started + timeout_seconds
        while True:
            results = self.results(handle)
            if not results.is_pending:
                return results
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BatchTimeoutError(
                    f"batch {handle.id!r} still pending after {timeout_seconds:.0f}s. "
                    "It has not failed -- fetch it later with results(handle).",
                    handle=handle,
                    waited_seconds=time.monotonic() - started,
                )
            time.sleep(min(poll_interval_seconds, remaining))

    def cancel(self, handle: BatchHandle) -> None:
        """Ask the provider to stop, and drop the held stage state.

        Args:
            handle: The batch to cancel.
        """
        self.backend.cancel(handle.id)
        self._tracked.pop(handle.id, None)

    def _send(self, items: Sequence[BatchItem]) -> str:
        """Submit to the backend, converting any failure into an explicit error.

        Raises:
            BatchSubmissionError: Always, on any backend failure. The exception
                *type* is included and the message is not: an exception payload
                can carry prompt content, and §10's rule outlives the package
                boundary just as it does in the pipeline's fail-open guard.
        """
        ids = [item.custom_id for item in items]
        try:
            return self.backend.submit(items)
        except Exception as exc:
            raise BatchSubmissionError(
                f"{self.backend.name} rejected the batch ({type(exc).__name__}); "
                f"none of the {len(ids)} requests were queued. Nothing was retried "
                "and nothing fell back to synchronous calls.",
                accepted=(),
                rejected=ids,
            ) from exc

    def _complete_all(
        self,
        raw: Mapping[str, LLMResponse],
        tracked: _TrackedBatch | None,
    ) -> dict[str, LLMResponse]:
        """Run each response through its pipeline half and account for it."""
        pipeline = self.optimizer.pipeline
        completed: dict[str, LLMResponse] = {}
        for custom_id, response in raw.items():
            self.report.input_tokens += response.input_tokens
            self.report.output_tokens += response.output_tokens
            prepared = tracked.prepared.get(custom_id) if tracked is not None else None
            if prepared is None:
                completed[custom_id] = response
                continue
            completed[custom_id] = pipeline.complete(
                prepared, response, run_id=tracked.run_id if tracked else None
            )
        return completed

    def _track(self, batch_id: str, tracked: _TrackedBatch) -> None:
        """Hold stage state, evicting the oldest batch past the ceiling."""
        self._tracked[batch_id] = tracked
        self._tracked.move_to_end(batch_id)
        while len(self._tracked) > MAX_TRACKED_BATCHES:
            evicted, _ = self._tracked.popitem(last=False)
            # Worth a warning rather than a debug line: the results for that
            # batch will still fetch and still be correct, but they will not
            # reach the exact cache or the savings report, and a report quietly
            # missing a few thousand requests is the kind of wrong number
            # someone acts on.
            _log.warning(
                "optio_optimize: tracking more than %d in-flight batches; dropped "
                "state for %r, whose results will fetch without savings accounting",
                MAX_TRACKED_BATCHES,
                evicted,
            )

    @staticmethod
    def _reject_duplicate_ids(items: Sequence[BatchItem]) -> None:
        """Raise if two items share a ``custom_id``.

        Raises:
            ValueError: Naming the first duplicate. Both providers key results
                by this id, so a collision does not fail -- it silently returns
                one answer where two were expected, which is far harder to
                notice than a rejected submission.
        """
        seen: set[str] = set()
        for item in items:
            if item.custom_id in seen:
                raise ValueError(
                    f"duplicate custom_id {item.custom_id!r}. Results are keyed by it, "
                    "so a collision loses one of the two answers silently."
                )
            seen.add(item.custom_id)


def items_from(requests: Sequence[LLMRequest], *, prefix: str = "req") -> list[BatchItem]:
    """Wrap plain requests in :class:`BatchItem`s with generated ids.

    For the common case where the caller has a list and no natural key of their
    own. Anyone who *does* have one -- a row id, a document id -- should use it
    instead, since that is what makes a result reconnectable to its input after
    a process restart.

    Args:
        requests: The calls to batch.
        prefix: Leading text for generated ids, to keep them legible in a
            provider's dashboard.

    Returns:
        Items with unique ids, in the order given.
    """
    token = uuid.uuid4().hex[:8]
    return [
        BatchItem(custom_id=f"{prefix}-{token}-{index}", request=request)
        for index, request in enumerate(requests)
    ]
