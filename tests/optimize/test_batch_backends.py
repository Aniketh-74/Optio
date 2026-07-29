"""The two provider batch backends, against fake clients.

Nothing here reaches a network. What is being checked is the translation layer
on both ends -- what we put in the envelope, and what we read out of the result
-- because that is where a batch surface can be confidently wrong: a submission
that omits a field still succeeds, and a result parsed slightly wrong still
returns a plausible string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from optio_optimize import BatchItem, BatchState, LLMRequest, Message
from optio_optimize.batch_backends import (
    COMPLETION_WINDOW,
    AnthropicBatchBackend,
    OpenAIBatchBackend,
    _openai_state,
)

pytestmark = pytest.mark.optimize

_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


def _request(text: str = "hello") -> LLMRequest:
    return LLMRequest(
        model="gpt-4o-mini",
        messages=(
            Message(role="system", content="You are terse.", cacheable=True),
            Message(role="user", content=text),
        ),
        max_tokens=128,
        tools=(_TOOL,),
        temperature=0.0,
        stop=("<END>",),
    )


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


class _FakeFiles:
    def __init__(self, contents: dict[str, str]) -> None:
        self.uploaded: list[bytes] = []
        self.contents = contents

    def create(self, *, file: Any, purpose: str) -> Any:
        assert purpose == "batch"
        self.uploaded.append(file.read())
        return type("File", (), {"id": "file_1"})()

    def content(self, file_id: str) -> Any:
        return type("Content", (), {"text": self.contents.get(file_id, "")})()


class _FakeBatches:
    def __init__(self, batch: Any) -> None:
        self.batch = batch
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return self.batch

    def retrieve(self, batch_id: str) -> Any:
        return self.batch

    def cancel(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)


@dataclass
class _FakeBatch:
    id: str = "batch_abc"
    status: str = "completed"
    output_file_id: str | None = "out_1"
    error_file_id: str | None = None


class _FakeOpenAI:
    def __init__(self, *, batch: _FakeBatch | None = None, contents: dict[str, str] | None = None):
        self.files = _FakeFiles(contents or {})
        self.batches = _FakeBatches(batch or _FakeBatch())


def _completion_line(custom_id: str = "a", *, status: int = 200) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": status,
                "body": {
                    "model": "gpt-4o-mini-2024",
                    "choices": [{"message": {"content": "the answer"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 410,
                        "completion_tokens": 30,
                        "prompt_tokens_details": {"cached_tokens": 256},
                    },
                },
            },
        }
    )


def test_openai_submission_writes_one_jsonl_line_per_item():
    client = _FakeOpenAI()
    backend = OpenAIBatchBackend(model="gpt-4o-mini", client=client)
    backend.submit([BatchItem("a", _request("one")), BatchItem("b", _request("two"))])

    lines = client.files.uploaded[0].decode().splitlines()
    assert len(lines) == 2
    envelope = json.loads(lines[0])
    assert envelope["custom_id"] == "a"
    assert envelope["method"] == "POST"
    assert envelope["url"] == "/v1/chat/completions"


def test_openai_envelope_carries_the_optimized_request_in_full():
    client = _FakeOpenAI()
    backend = OpenAIBatchBackend(model="gpt-4o-mini", client=client)
    backend.submit([BatchItem("a", _request())])

    body = json.loads(client.files.uploaded[0].decode().splitlines()[0])["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["max_completion_tokens"] == 128
    assert body["temperature"] == 0.0
    assert body["stop"] == ["<END>"]
    # The field whose omission from the synchronous adapter made a whole live
    # benchmark measure nothing.
    assert body["tools"][0]["function"]["name"] == "lookup"


def test_openai_requests_the_discounted_completion_window():
    client = _FakeOpenAI()
    OpenAIBatchBackend(model="gpt-4o-mini", client=client).submit([BatchItem("a", _request())])
    created = client.batches.created[0]
    assert created["completion_window"] == COMPLETION_WINDOW
    assert created["input_file_id"] == "file_1"


def test_openai_results_are_parsed_with_provider_token_counts():
    client = _FakeOpenAI(contents={"out_1": _completion_line("a")})
    backend = OpenAIBatchBackend(client=client)
    responses, errors = backend.fetch("batch_abc")
    assert errors == {}
    answer = responses["a"]
    assert answer.content == "the answer"
    assert answer.input_tokens == 410
    assert answer.output_tokens == 30
    assert answer.cached_input_tokens == 256
    assert answer.model == "gpt-4o-mini-2024"


def test_openai_non_200_items_become_errors_not_empty_answers():
    line = json.dumps({"custom_id": "a", "response": {"status_code": 429, "body": {}}})
    backend = OpenAIBatchBackend(client=_FakeOpenAI(contents={"out_1": line}))
    responses, errors = backend.fetch("batch_abc")
    assert responses == {}
    assert errors == {"a": "HTTP 429"}


def test_openai_error_file_reports_the_code_not_the_message():
    # An error body can quote the offending request back at you.
    error_line = json.dumps(
        {
            "custom_id": "b",
            "error": {"code": "invalid_request", "message": "bad prompt: SECRET TEXT"},
        }
    )
    batch = _FakeBatch(output_file_id=None, error_file_id="err_1")
    client = _FakeOpenAI(batch=batch, contents={"err_1": error_line})
    _, errors = OpenAIBatchBackend(client=client).fetch("batch_abc")
    assert errors == {"b": "invalid_request"}


def test_openai_expired_batch_still_yields_whatever_finished():
    batch = _FakeBatch(status="expired", output_file_id="out_1")
    client = _FakeOpenAI(batch=batch, contents={"out_1": _completion_line("a")})
    backend = OpenAIBatchBackend(client=client)
    assert backend.poll("batch_abc") is BatchState.EXPIRED
    responses, _ = backend.fetch("batch_abc")
    assert set(responses) == {"a"}


def test_openai_cancel_reaches_the_client():
    client = _FakeOpenAI()
    OpenAIBatchBackend(client=client).cancel("batch_abc")
    assert client.batches.cancelled == ["batch_abc"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("validating", BatchState.PENDING),
        ("in_progress", BatchState.PENDING),
        ("finalizing", BatchState.PENDING),
        ("cancelling", BatchState.PENDING),
        ("completed", BatchState.COMPLETED),
        ("failed", BatchState.FAILED),
        ("expired", BatchState.EXPIRED),
        ("cancelled", BatchState.CANCELLED),
    ],
)
def test_openai_status_mapping(status, expected):
    assert _openai_state(status) is expected


def test_unknown_status_keeps_the_caller_waiting():
    # The conservative direction is "poll again". Reading an unrecognized new
    # provider status as FAILED would make a caller abandon work that was about
    # to succeed; reading it as PENDING costs one more HTTP request.
    assert _openai_state("some_future_status") is BatchState.PENDING


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


@dataclass
class _Usage:
    input_tokens: int = 300
    output_tokens: int = 40
    cache_read_input_tokens: int = 100


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Message:
    content: list[_TextBlock]
    usage: _Usage
    model: str = "claude-haiku-4"
    stop_reason: str = "end_turn"


@dataclass
class _Result:
    type: str
    message: _Message | None = None


@dataclass
class _Entry:
    custom_id: str
    result: _Result


class _FakeAnthropicBatches:
    def __init__(self, *, status: str = "ended", entries: list[_Entry] | None = None) -> None:
        self.status = status
        self.entries = entries or []
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

    def create(self, *, requests: list[dict[str, Any]]) -> Any:
        self.created.append({"requests": requests})
        return type("Batch", (), {"id": "msgbatch_1"})()

    def retrieve(self, batch_id: str) -> Any:
        return type("Batch", (), {"processing_status": self.status})()

    def results(self, batch_id: str) -> list[_Entry]:
        return self.entries

    def cancel(self, batch_id: str) -> None:
        self.cancelled.append(batch_id)


class _FakeAnthropic:
    def __init__(self, batches: _FakeAnthropicBatches) -> None:
        self.messages = type("Messages", (), {"batches": batches})()


def test_anthropic_submits_a_typed_array_with_translated_schemas():
    batches = _FakeAnthropicBatches()
    backend = AnthropicBatchBackend(model="claude-haiku-4", client=_FakeAnthropic(batches))
    backend.submit([BatchItem("a", _request())])

    entry = batches.created[0]["requests"][0]
    assert entry["custom_id"] == "a"
    params = entry["params"]
    assert params["model"] == "claude-haiku-4"
    assert params["max_tokens"] == 128
    assert params["stop_sequences"] == ["<END>"]
    # OpenAI's nested shape must be translated, not forwarded.
    assert params["tools"][0]["name"] == "lookup"
    assert "function" not in params["tools"][0]


def test_anthropic_batch_carries_the_prefix_cache_marker():
    # The one provider where our marker becomes a real API field. Both
    # discounts then compose on the same request.
    batches = _FakeAnthropicBatches()
    AnthropicBatchBackend(client=_FakeAnthropic(batches)).submit([BatchItem("a", _request())])
    params = batches.created[0]["requests"][0]["params"]
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_in_progress_is_pending():
    batches = _FakeAnthropicBatches(status="in_progress")
    assert AnthropicBatchBackend(client=_FakeAnthropic(batches)).poll("x") is BatchState.PENDING


def test_anthropic_ended_is_completed_even_though_items_may_have_failed():
    # "ended" covers success, cancellation and expiry alike. Which it was is a
    # per-item fact in the results, so mapping it to CANCELLED here would throw
    # away results that did finish.
    batches = _FakeAnthropicBatches(status="ended")
    assert AnthropicBatchBackend(client=_FakeAnthropic(batches)).poll("x") is BatchState.COMPLETED


def test_anthropic_results_add_cache_reads_back_into_input_tokens():
    # Anthropic reports input_tokens excluding cache reads. Everywhere else in
    # this package input_tokens means "total prompt tokens, some discounted",
    # so getting this wrong in one place makes batched and synchronous totals
    # silently incomparable.
    entry = _Entry("a", _Result("succeeded", _Message([_TextBlock("hi")], _Usage())))
    backend = AnthropicBatchBackend(client=_FakeAnthropic(_FakeAnthropicBatches(entries=[entry])))
    responses, errors = backend.fetch("x")
    assert errors == {}
    assert responses["a"].input_tokens == 400
    assert responses["a"].cached_input_tokens == 100
    assert responses["a"].billable_input_tokens == 300
    assert responses["a"].content == "hi"


@pytest.mark.parametrize("kind", ["errored", "canceled", "expired"])
def test_anthropic_non_success_results_become_errors(kind):
    entry = _Entry("a", _Result(kind))
    backend = AnthropicBatchBackend(client=_FakeAnthropic(_FakeAnthropicBatches(entries=[entry])))
    responses, errors = backend.fetch("x")
    assert responses == {}
    assert errors == {"a": kind}


def test_anthropic_cancel_reaches_the_client():
    batches = _FakeAnthropicBatches()
    AnthropicBatchBackend(client=_FakeAnthropic(batches)).cancel("msgbatch_1")
    assert batches.cancelled == ["msgbatch_1"]


# --------------------------------------------------------------------------
# Credentials: this library never handles one.
# --------------------------------------------------------------------------


def test_backends_never_accept_a_key_argument():
    import inspect

    for backend in (OpenAIBatchBackend, AnthropicBatchBackend):
        params = set(inspect.signature(backend.__init__).parameters)
        assert not params & {"api_key", "key", "token", "credential"}, (
            f"{backend.__name__} grew a credential parameter; this library reads "
            "the SDK's own environment variable and never sees the value"
        )


def test_missing_credentials_explain_the_remedy(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIBatchBackend()
