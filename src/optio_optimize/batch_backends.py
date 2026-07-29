"""Provider batch endpoints.

Two vendors, two file formats, one :class:`~optio_optimize.batch.BatchBackend`
protocol. OpenAI takes a JSONL upload of request envelopes and hands back a file
of result envelopes; Anthropic takes a typed array in the request body and
streams results back through the SDK. Everything above them --  optimization,
cache checking, accounting, waiting -- is in :mod:`optio_optimize.batch` and
knows about neither.

Both follow the rule the live benchmark's adapters follow: **we never handle the
key.** Each reads the provider SDK's own environment variable through the SDK
itself and passes nothing. This library does not accept, store, log, or forward
a credential.

The request bodies are built by :mod:`optio_optimize.wire`, shared with the
synchronous adapters. That sharing is not tidiness -- the live OpenAI adapter
once omitted ``tools`` from its call, which did not fail, and instead made a
whole benchmark run report savings against a request that never carried the
field the stage had rewritten. A second, independent translation here would be
a second chance at exactly that.
"""

from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING, Any

from optio_optimize.batch import BatchState
from optio_optimize.types import LLMResponse
from optio_optimize.wire import anthropic_body, openai_body

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from optio_optimize.batch import BatchItem

#: OpenAI batch statuses that mean "still working". Listed positively rather
#: than as "not completed", so a status this library has never seen maps to
#: PENDING and the caller keeps waiting, instead of being told a live batch
#: failed.
_OPENAI_PENDING = frozenset({"validating", "in_progress", "finalizing"})

#: How the terminal OpenAI statuses map. ``cancelling`` is deliberately absent:
#: it belongs with the pending set above, because a cancellation in progress is
#: not yet a cancellation.
_OPENAI_TERMINAL = {
    "completed": BatchState.COMPLETED,
    "failed": BatchState.FAILED,
    "expired": BatchState.EXPIRED,
    "cancelled": BatchState.CANCELLED,
}

#: Provider completion window requested at submission. Both vendors offer 24
#: hours and price the discount against it; there is no shorter paid tier to
#: choose, so this is a constant rather than a parameter.
COMPLETION_WINDOW = "24h"


class OpenAIBatchBackend:
    """OpenAI's ``/v1/batches`` endpoint.

    Reads ``OPENAI_API_KEY`` from the environment via the SDK's own mechanism.
    This class never sees the value.

    Attributes:
        model: Model every request in a batch is submitted against.
    """

    def __init__(self, model: str = "gpt-4o-mini", *, client: Any = None) -> None:
        """Build the backend.

        Args:
            model: Model to submit against. Defaults to the cheap one -- a
                batch of ten thousand should not default to the expensive
                option.
            client: An ``openai.OpenAI`` instance to use instead of building
                one. For tests, and for callers who already configure a client
                with their own base URL or retry policy.

        Raises:
            RuntimeError: If the SDK is missing or no key is configured, with
                the specific remedy rather than a generic import error.
        """
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "the openai package is required for batch dispatch: pip install openai"
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Batch dispatch needs it, the same "
                    "way a synchronous call does."
                )
            client = OpenAI()
        self._client = client
        self.model = model

    @property
    def name(self) -> str:
        """Stable identifier, recorded on the handle."""
        return "openai"

    def submit(self, items: Sequence[BatchItem]) -> str:
        """Upload a JSONL of request envelopes and create the batch.

        Args:
            items: Optimized requests with unique ids.

        Returns:
            The provider-issued batch id.
        """
        payload = "\n".join(
            json.dumps(
                {
                    "custom_id": item.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": openai_body(item.request, self.model),
                },
                separators=(",", ":"),
            )
            for item in items
        )
        # Built in memory rather than through a temporary file. A batch is
        # bounded by the provider's own size limit, so this cannot grow without
        # bound -- and writing prompt content to disk is a side effect this
        # package has no business having, even briefly.
        upload = io.BytesIO(payload.encode("utf-8"))
        upload.name = "batch.jsonl"
        stored = self._client.files.create(file=upload, purpose="batch")
        batch = self._client.batches.create(
            input_file_id=stored.id,
            endpoint="/v1/chat/completions",
            completion_window=COMPLETION_WINDOW,
        )
        return str(batch.id)

    def poll(self, batch_id: str) -> BatchState:
        """Return the batch's state without downloading results."""
        batch = self._client.batches.retrieve(batch_id)
        return _openai_state(str(batch.status))

    def fetch(self, batch_id: str) -> tuple[Mapping[str, LLMResponse], Mapping[str, str]]:
        """Download and parse the result file.

        Returns:
            ``(responses, errors)`` keyed by ``custom_id``. An expired batch
            has an output file covering whatever finished, so this is not
            conditional on the batch having completed cleanly.
        """
        batch = self._client.batches.retrieve(batch_id)
        responses: dict[str, LLMResponse] = {}
        errors: dict[str, str] = {}

        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            for line in _jsonl(self._client.files.content(output_file_id)):
                custom_id = str(line.get("custom_id", ""))
                envelope = line.get("response") or {}
                status = envelope.get("status_code")
                body = envelope.get("body") or {}
                if status == 200 and body:
                    responses[custom_id] = _response_from_openai_body(body)
                else:
                    errors[custom_id] = f"HTTP {status}"

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            for line in _jsonl(self._client.files.content(error_file_id)):
                custom_id = str(line.get("custom_id", ""))
                detail = line.get("error") or {}
                # The provider's own code, never its message. An error payload
                # can quote the offending request back at you, and §10's
                # never-log-prompt-content rule does not stop at this package's
                # boundary just because the string came from outside it.
                errors[custom_id] = str(detail.get("code", "error"))

        return responses, errors

    def cancel(self, batch_id: str) -> None:
        """Ask OpenAI to stop processing."""
        self._client.batches.cancel(batch_id)


class AnthropicBatchBackend:
    """Anthropic's Message Batches endpoint.

    Reads ``ANTHROPIC_API_KEY`` from the environment via the SDK's own
    mechanism. This class never sees the value.

    The provider where a prefix-cache marker is a real API field, so a batched
    request carries the ``cache_control`` block this library's prefix stage
    placed -- the two discounts compose, and neither is this package inventing
    a number.

    Attributes:
        model: Model every request in a batch is submitted against.
    """

    def __init__(self, model: str = "claude-haiku-4", *, client: Any = None) -> None:
        """Build the backend.

        Args:
            model: Model to submit against.
            client: An ``anthropic.Anthropic`` instance to use instead of
                building one.

        Raises:
            RuntimeError: If the SDK is missing or no key is configured.
        """
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "the anthropic package is required for batch dispatch: pip install anthropic"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Batch dispatch needs it, the same "
                    "way a synchronous call does."
                )
            client = Anthropic()
        self._client = client
        self.model = model

    @property
    def name(self) -> str:
        """Stable identifier, recorded on the handle."""
        return "anthropic"

    def submit(self, items: Sequence[BatchItem]) -> str:
        """Send a typed array of requests.

        No file upload: Anthropic takes the whole batch in the request body,
        which is why this is shorter than its OpenAI counterpart rather than
        differently careful.
        """
        batch = self._client.messages.batches.create(
            requests=[
                {
                    "custom_id": item.custom_id,
                    "params": anthropic_body(item.request, self.model),
                }
                for item in items
            ]
        )
        return str(batch.id)

    def poll(self, batch_id: str) -> BatchState:
        """Return the batch's state without downloading results."""
        batch = self._client.messages.batches.retrieve(batch_id)
        # Anthropic reports only `in_progress` / `canceling` / `ended`. "Ended"
        # covers success, cancellation and expiry alike -- which of those it was
        # is a per-item fact in the results, so this maps to COMPLETED and lets
        # `fetch` sort the items out. Reporting CANCELLED here on the strength
        # of a `canceling` status would discard results that did finish.
        status = str(getattr(batch, "processing_status", ""))
        return BatchState.COMPLETED if status == "ended" else BatchState.PENDING

    def fetch(self, batch_id: str) -> tuple[Mapping[str, LLMResponse], Mapping[str, str]]:
        """Stream the results and split successes from failures."""
        responses: dict[str, LLMResponse] = {}
        errors: dict[str, str] = {}
        for entry in self._client.messages.batches.results(batch_id):
            custom_id = str(getattr(entry, "custom_id", ""))
            result = getattr(entry, "result", None)
            kind = str(getattr(result, "type", "errored"))
            message = getattr(result, "message", None)
            if kind == "succeeded" and message is not None:
                responses[custom_id] = _response_from_anthropic_message(message)
            else:
                # The result *type* only -- `succeeded`, `errored`, `canceled`,
                # `expired`. Same reasoning as the OpenAI backend: an error body
                # can quote the prompt back.
                errors[custom_id] = kind
        return responses, errors

    def cancel(self, batch_id: str) -> None:
        """Ask Anthropic to stop processing."""
        self._client.messages.batches.cancel(batch_id)


def _openai_state(status: str) -> BatchState:
    """Map an OpenAI batch status onto :class:`BatchState`.

    An unrecognized status maps to ``PENDING`` rather than ``FAILED``. The
    conservative direction is "keep waiting": a new provider status that this
    library reads as a failure would make a caller abandon work that was about
    to succeed, whereas reading it as pending costs one more poll.
    """
    if status in _OPENAI_PENDING:
        return BatchState.PENDING
    return _OPENAI_TERMINAL.get(status, BatchState.PENDING)


def _jsonl(content: Any) -> list[dict[str, Any]]:
    """Parse a provider file response into JSON objects, one per line.

    The SDK returns a response wrapper rather than a string, and its accessor
    has moved across versions, so the text is pulled out by trying the
    attributes that have existed rather than pinning to one.
    """
    text = getattr(content, "text", None)
    if text is None:
        raw = getattr(content, "content", content)
        text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _response_from_openai_body(body: dict[str, Any]) -> LLMResponse:
    """Normalize an OpenAI chat completion body from a batch result file.

    Parsed from raw JSON rather than an SDK model: the batch output file is
    downloaded as text, so there is no typed object to read. The fields are the
    same ones the synchronous adapter reads off ``completion``.
    """
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return LLMResponse(
        content=message.get("content") or "",
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        cached_input_tokens=int(details.get("cached_tokens", 0) or 0),
        model=str(body.get("model", "")),
        finish_reason=choices[0].get("finish_reason"),
    )


def _response_from_anthropic_message(message: Any) -> LLMResponse:
    """Normalize an Anthropic message from a batch result.

    ``input_tokens`` is reported *excluding* cache reads, so the cached count is
    added back to make the field mean the same thing it means everywhere else in
    this package: total prompt tokens, of which some were discounted. The
    synchronous adapter does the same, and getting it wrong in one place would
    make batched and synchronous totals silently incomparable.
    """
    usage = getattr(message, "usage", None)
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    text = "".join(
        str(getattr(block, "text", ""))
        for block in getattr(message, "content", [])
        if getattr(block, "type", "") == "text"
    )
    return LLMResponse(
        content=text,
        input_tokens=(int(getattr(usage, "input_tokens", 0)) + cached) if usage else 0,
        output_tokens=int(getattr(usage, "output_tokens", 0)) if usage else 0,
        cached_input_tokens=cached,
        model=str(getattr(message, "model", "")),
        finish_reason=getattr(message, "stop_reason", None),
    )
