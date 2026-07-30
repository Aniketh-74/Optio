"""Batch dispatch: the second surface (ADR-017).

The properties worth pinning are the ones that make batching *safe* rather than
the ones that make it cheap. Cheap is arithmetic. Safe is:

* a request the exact cache already answered never enters a 24-hour queue;
* a failed submission raises and says what happened, because there is no
  degraded path to fall back to and silence would mean a caller believing work
  is queued that is not;
* an answer that arrives tomorrow still runs its ``after`` hooks, so it lands in
  the savings report and the cache exactly as a synchronous one would;
* nothing anywhere logs a prompt or an exception message.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from optio_optimize import (
    BatchHandle,
    BatchItem,
    BatchOptimizer,
    BatchResults,
    BatchState,
    BatchSubmission,
    BatchSubmissionError,
    BatchTimeoutError,
    LLMRequest,
    LLMResponse,
    Message,
    OptimizeConfig,
    OptimizeConfigError,
    Optimizer,
    items_from,
)
from optio_optimize.batch import BATCH_DISCOUNT, MAX_TRACKED_BATCHES, BatchReport

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

pytestmark = pytest.mark.optimize

#: Turns the size of real ones. A two-token turn is not a conversation worth
#: trimming, and since ADR-026 the stage correctly declines to trim one.
_TURN_PADDING = " ".join(f"context{n}" for n in range(120))


def _request(text: str = "summarize row 1", *, temperature: float | None = 0.0) -> LLMRequest:
    return LLMRequest(
        model="gpt-4o-mini",
        messages=(
            Message(role="system", content="You are terse."),
            Message(role="user", content=text),
        ),
        temperature=temperature,
    )


def _handle(submission: BatchSubmission) -> BatchHandle:
    """Unwrap the handle, so a typo that submits nothing fails here and loudly."""
    assert submission.handle is not None, "expected a submission; everything was cached"
    return submission.handle


def _response(content: str = "done", *, input_tokens: int = 400) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=25,
        model="gpt-4o-mini",
        finish_reason="stop",
    )


class FakeBackend:
    """A backend that records what it was asked to send.

    Deliberately not a mock of a provider SDK: what these tests need to know is
    what reached the *backend boundary*, which is where the optimization
    pipeline's output becomes somebody else's input.
    """

    def __init__(
        self,
        *,
        state: BatchState = BatchState.COMPLETED,
        fail_with: Exception | None = None,
    ) -> None:
        self.submitted: list[Sequence[BatchItem]] = []
        self.state = state
        self.fail_with = fail_with
        self.cancelled: list[str] = []
        self.polls = 0
        self.responses: dict[str, LLMResponse] = {}
        self.errors: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "gpt-4o-mini"

    def submit(self, items: Sequence[BatchItem]) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self.submitted.append(list(items))
        if not self.responses:
            self.responses = {item.custom_id: _response() for item in items}
        return f"batch_{len(self.submitted)}"

    def poll(self, batch_id: str) -> BatchState:
        self.polls += 1
        return self.state

    def fetch(self, batch_id: str) -> tuple[Mapping[str, LLMResponse], Mapping[str, str]]:
        return self.responses, self.errors

    def cancel(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)


def _batcher(backend: FakeBackend | None = None, **overrides: Any) -> BatchOptimizer:
    backend = backend if backend is not None else FakeBackend()
    return BatchOptimizer(backend, OptimizeConfig(**overrides))


# --------------------------------------------------------------------------
# ADR-017 decision 2: the pipeline runs first, unchanged.
# --------------------------------------------------------------------------


def test_stages_run_before_submission():
    backend = FakeBackend()
    # A long history so trimming has something to do.
    request = LLMRequest(
        model="gpt-4o-mini",
        messages=(
            Message(role="system", content="You are terse."),
            *[Message(role="user", content=f"turn {i} {_TURN_PADDING}") for i in range(30)],
        ),
        temperature=0.0,
    )
    batcher = _batcher(backend, recent_turns=4)
    batcher.submit([BatchItem("row-1", request)])

    sent = backend.submitted[0][0].request
    assert len(sent.messages) < len(request.messages), (
        "the batch path submitted the caller's original request; the discounts "
        "are supposed to compose, not substitute"
    )


def test_savings_from_stages_land_in_the_shared_report():
    backend = FakeBackend()
    optimizer = Optimizer(OptimizeConfig(recent_turns=2))
    batcher = BatchOptimizer(backend, optimizer=optimizer)
    request = LLMRequest(
        model="gpt-4o-mini",
        messages=(
            Message(role="system", content="You are terse."),
            *[Message(role="user", content=f"turn {i} " + "padding " * 40) for i in range(20)],
        ),
        temperature=0.0,
    )
    submission = batcher.submit([BatchItem("row-1", request)])
    batcher.results(_handle(submission))

    assert optimizer.report.requests == 1
    assert optimizer.report.total_saved_tokens > 0


def test_batch_and_sync_surfaces_share_one_optimizer():
    optimizer = Optimizer()
    batcher = BatchOptimizer(FakeBackend(), optimizer=optimizer)
    assert batcher.optimizer is optimizer


def test_config_and_optimizer_together_is_a_setup_error():
    with pytest.raises(OptimizeConfigError, match="not both"):
        BatchOptimizer(FakeBackend(), OptimizeConfig(), optimizer=Optimizer())


# --------------------------------------------------------------------------
# ADR-017 decision 3: the exact cache is checked before submission.
# --------------------------------------------------------------------------


def test_a_cached_request_never_enters_the_queue():
    backend = FakeBackend()
    optimizer = Optimizer()
    batcher = BatchOptimizer(backend, optimizer=optimizer)
    request = _request()

    # Answer it synchronously first, populating the shared exact cache.
    optimizer.call(request, lambda r: _response("the real answer"))

    submission = batcher.submit([BatchItem("row-1", request)])
    assert submission.is_complete
    assert submission.handle is None
    assert submission.served["row-1"].content == "the real answer"
    assert backend.submitted == [], "an already-answered request was queued for a day"


def test_a_batch_answer_populates_the_cache_for_the_synchronous_path():
    backend = FakeBackend()
    optimizer = Optimizer()
    batcher = BatchOptimizer(backend, optimizer=optimizer)
    request = _request()
    backend.responses = {"row-1": _response("from the batch")}

    submission = batcher.submit([BatchItem("row-1", request)])
    batcher.results(_handle(submission))

    calls: list[LLMRequest] = []

    def provider(r: LLMRequest) -> LLMResponse:
        calls.append(r)
        return _response("a second call")

    answer = optimizer.call(request, provider)
    assert answer.content == "from the batch"
    assert calls == [], "the batch answer did not reach the cache it should have filled"


def test_mixed_batch_splits_cached_from_submitted():
    backend = FakeBackend()
    optimizer = Optimizer()
    batcher = BatchOptimizer(backend, optimizer=optimizer)
    cached, fresh = _request("row one"), _request("row two")
    optimizer.call(cached, lambda r: _response("already known"))

    submission = batcher.submit([BatchItem("a", cached), BatchItem("b", fresh)])
    assert set(submission.served) == {"a"}
    assert submission.submitted == ("b",)
    assert [i.custom_id for i in backend.submitted[0]] == ["b"]


def test_sampled_requests_are_not_cache_checked_and_do_get_submitted():
    # temperature != 0 is never exact-cached, in batch as synchronously: a
    # frozen answer would replace variety the caller asked for.
    backend = FakeBackend()
    batcher = _batcher(backend)
    request = _request(temperature=0.9)
    batcher.submit([BatchItem("a", request), BatchItem("b", request)])
    assert len(backend.submitted[0]) == 2


# --------------------------------------------------------------------------
# ADR-017 decision 4: failure is explicit, not fail-open.
# --------------------------------------------------------------------------


def test_submission_failure_raises_and_names_the_rejected_items():
    backend = FakeBackend(fail_with=RuntimeError("upstream 503"))
    batcher = _batcher(backend)
    with pytest.raises(BatchSubmissionError) as caught:
        batcher.submit([BatchItem("a", _request("a")), BatchItem("b", _request("b"))])
    assert caught.value.rejected == ("a", "b")
    assert caught.value.accepted == ()


def test_submission_failure_does_not_fall_back_to_synchronous_calls():
    # The fail-open rule that governs stages deliberately does not apply here:
    # turning a failed batch into N synchronous calls costs twice what batching
    # was asked to save, and does it without asking.
    backend = FakeBackend(fail_with=RuntimeError("nope"))
    batcher = _batcher(backend)
    with pytest.raises(BatchSubmissionError):
        batcher.submit([BatchItem("a", _request())])
    assert backend.submitted == []


def test_submission_error_carries_the_exception_type_not_its_message():
    # An exception payload can quote the request back at you. Section 10's
    # never-log-prompt-content rule does not stop at the package boundary just
    # because the string arrived from outside it.
    secret = "confidential customer prompt text"
    backend = FakeBackend(fail_with=RuntimeError(secret))
    batcher = _batcher(backend)
    with pytest.raises(BatchSubmissionError) as caught:
        batcher.submit([BatchItem("a", _request())])
    assert secret not in str(caught.value)
    assert "RuntimeError" in str(caught.value)


def test_original_exception_is_chained_for_debugging():
    backend = FakeBackend(fail_with=RuntimeError("upstream detail"))
    batcher = _batcher(backend)
    with pytest.raises(BatchSubmissionError) as caught:
        batcher.submit([BatchItem("a", _request())])
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_duplicate_custom_ids_are_rejected_before_submission():
    backend = FakeBackend()
    batcher = _batcher(backend)
    with pytest.raises(ValueError, match="duplicate custom_id 'a'"):
        batcher.submit([BatchItem("a", _request("one")), BatchItem("a", _request("two"))])
    assert backend.submitted == [], "a colliding batch reached the provider"


def test_empty_submission_is_rejected():
    with pytest.raises(ValueError, match="at least one item"):
        _batcher().submit([])


# --------------------------------------------------------------------------
# ADR-017 decision 5: polling is the caller's, with a helper.
# --------------------------------------------------------------------------


def test_pending_is_a_result_not_an_error():
    backend = FakeBackend(state=BatchState.PENDING)
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    results = batcher.results(_handle(submission))
    assert results.is_pending
    assert results.responses == {}
    assert results.errors == {}


def test_await_results_returns_once_complete():
    backend = FakeBackend()
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    results = batcher.await_results(
        _handle(submission), timeout_seconds=5.0, poll_interval_seconds=0.01
    )
    assert results.state is BatchState.COMPLETED
    assert set(results.responses) == {"a"}


def test_await_results_times_out_with_the_handle_intact():
    backend = FakeBackend(state=BatchState.PENDING)
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    with pytest.raises(BatchTimeoutError) as caught:
        batcher.await_results(_handle(submission), timeout_seconds=0.05, poll_interval_seconds=0.01)
    # The batch has not failed; the caller merely stopped waiting. Losing the
    # handle here would strand work that is still going to complete.
    assert caught.value.handle == submission.handle
    assert caught.value.waited_seconds > 0
    assert "has not failed" in str(caught.value)


def test_await_results_requires_a_positive_timeout():
    batcher = _batcher()
    submission = batcher.submit([BatchItem("a", _request())])
    with pytest.raises(ValueError, match="timeout_seconds"):
        batcher.await_results(_handle(submission), timeout_seconds=0)
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        batcher.await_results(_handle(submission), timeout_seconds=1.0, poll_interval_seconds=0)


def test_poll_does_not_download_results():
    backend = FakeBackend(state=BatchState.PENDING)
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    assert batcher.poll(_handle(submission)) is BatchState.PENDING


def test_cancel_reaches_the_backend_and_drops_held_state():
    backend = FakeBackend()
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    batcher.cancel(_handle(submission))
    assert backend.cancelled == [_handle(submission).id]


# --------------------------------------------------------------------------
# Results and their accounting.
# --------------------------------------------------------------------------


def test_failed_batch_returns_no_results_and_does_not_hang():
    backend = FakeBackend(state=BatchState.FAILED)
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    results = batcher.results(_handle(submission))
    assert results.state is BatchState.FAILED
    assert not results.is_pending


def test_per_item_errors_do_not_sink_the_batch():
    backend = FakeBackend()
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request("a")), BatchItem("b", _request("b"))])
    backend.responses = {"a": _response("fine")}
    backend.errors = {"b": "invalid_request"}

    results = batcher.results(_handle(submission))
    assert results.state is BatchState.COMPLETED
    assert set(results.responses) == {"a"}
    assert results.errors == {"b": "invalid_request"}
    assert batcher.report.failed == 1


def test_results_for_an_untracked_handle_still_return():
    # After a process restart the stage state is gone -- ADR-017 rules out
    # persistence. The results are still correct; they simply arrive
    # unattributed, which must not be an exception.
    backend = FakeBackend()
    first = _batcher(backend)
    submission = first.submit([BatchItem("a", _request())])

    second = BatchOptimizer(backend)
    results = second.results(_handle(submission))
    assert results.responses["a"].content == "done"
    assert second.optimizer.report.requests == 0


def test_results_are_completed_only_once():
    backend = FakeBackend()
    optimizer = Optimizer()
    batcher = BatchOptimizer(backend, optimizer=optimizer)
    submission = batcher.submit([BatchItem("a", _request())])
    batcher.results(_handle(submission))
    batcher.results(_handle(submission))
    # A second fetch must not re-book the request; a report that double-counts
    # every retried poll is worse than one that reports nothing.
    assert optimizer.report.requests == 1


def test_tracking_is_bounded_and_says_so_when_it_evicts(caplog):
    backend = FakeBackend()
    batcher = _batcher(backend)
    with caplog.at_level(logging.WARNING, logger="optio_optimize"):
        for index in range(MAX_TRACKED_BATCHES + 1):
            backend.responses = {}
            batcher.submit([BatchItem(f"row-{index}", _request(f"text {index}"))])
    assert any("in-flight batches" in record.message for record in caplog.records)


def test_nothing_logged_contains_prompt_content(caplog):
    backend = FakeBackend()
    batcher = _batcher(backend)
    secret = "the customer's confidential question"
    with caplog.at_level(logging.DEBUG, logger="optio_optimize"):
        submission = batcher.submit([BatchItem("a", _request(secret))])
        batcher.results(_handle(submission))
    assert all(secret not in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------
# The report, and its caveat.
# --------------------------------------------------------------------------


def test_discount_is_half_the_synchronous_price():
    report = BatchReport(input_tokens=1_000_000, output_tokens=1_000_000, submitted=1)
    # gpt-4o-mini: $0.15/M in, $0.60/M out -> $0.75 synchronous, half saved.
    assert report.estimated_discount_usd("gpt-4o-mini") == pytest.approx(0.375)
    assert BATCH_DISCOUNT == 0.5


def test_unpriced_model_reports_none_not_zero():
    report = BatchReport(input_tokens=1000, output_tokens=100)
    assert report.estimated_discount_usd("some-unreleased-model") is None


def test_summary_always_states_that_the_figure_is_not_measured():
    lines = BatchReport(input_tokens=1000, output_tokens=100).summary_lines("gpt-4o-mini")
    joined = " ".join(lines)
    assert "not measured end-to-end" in joined, (
        "this is the one number in the package derived from a published figure "
        "rather than an A/B run; a reader comparing it to the measured ones has "
        "no way to tell unless it says so"
    )


def test_report_counts_cache_hits_separately_from_submissions():
    backend = FakeBackend()
    optimizer = Optimizer()
    batcher = BatchOptimizer(backend, optimizer=optimizer)
    cached, fresh = _request("one"), _request("two")
    optimizer.call(cached, lambda r: _response("known"))

    batcher.submit([BatchItem("a", cached), BatchItem("b", fresh)])
    assert batcher.report.served_from_cache == 1
    assert batcher.report.submitted == 1
    assert batcher.report.batches == 1


def test_token_totals_come_from_the_provider_not_our_estimate():
    backend = FakeBackend()
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    backend.responses = {"a": _response(input_tokens=1234)}
    batcher.results(_handle(submission))
    assert batcher.report.input_tokens == 1234


# --------------------------------------------------------------------------
# Handles and ids.
# --------------------------------------------------------------------------


def test_handle_records_where_to_send_it_back():
    backend = FakeBackend()
    batcher = _batcher(backend)
    submission = batcher.submit([BatchItem("a", _request())])
    handle = _handle(submission)
    assert handle.backend == "fake"
    assert handle.model == "gpt-4o-mini"
    assert handle.custom_ids == ("a",)
    assert handle.submitted_at > 0


def test_items_from_generates_unique_ids():
    items = items_from([_request(f"row {i}") for i in range(50)])
    assert len({item.custom_id for item in items}) == 50


def test_items_from_preserves_order():
    requests = [_request(f"row {i}") for i in range(5)]
    items = items_from(requests, prefix="doc")
    assert [item.request for item in items] == requests
    assert all(item.custom_id.startswith("doc-") for item in items)


def test_batch_results_pending_flag():
    assert BatchResults(BatchState.PENDING, {}, {}).is_pending
    assert not BatchResults(BatchState.COMPLETED, {}, {}).is_pending


@pytest.mark.parametrize(
    ("state", "terminal"),
    [
        (BatchState.PENDING, False),
        (BatchState.COMPLETED, True),
        (BatchState.FAILED, True),
        (BatchState.EXPIRED, True),
        (BatchState.CANCELLED, True),
    ],
)
def test_terminal_states(state, terminal):
    assert state.is_terminal is terminal


# --------------------------------------------------------------------------
# The disabled path still works.
# --------------------------------------------------------------------------


def test_disabled_optimizer_still_batches():
    # enabled=False is the A/B control arm. It must not mean "batching stops
    # working", only "no stage rewrites the request".
    backend = FakeBackend()
    batcher = _batcher(backend, enabled=False)
    request = _request()
    submission = batcher.submit([BatchItem("a", request)])
    assert backend.submitted[0][0].request == request
    results = batcher.results(_handle(submission))
    assert results.responses["a"].content == "done"
