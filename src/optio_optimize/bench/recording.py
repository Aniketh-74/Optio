"""Keeping the exchanges this project pays for (ADR-039).

Every live figure here was printed to a terminal, read once, copied into an ADR
by hand, and the response discarded. The consequence is not that the numbers are
wrong -- it is that **re-checking one costs money**, so nobody does. That is the
shape behind the longest-lived defect in the changelog: ``prefix_cache``
reported a saving for months on a prompt below the provider's cacheable minimum,
hitting nothing, because no cheap test could have noticed.

A recorded exchange is the same evidence with the receipt kept. The call is paid
for once; every later run replays it for nothing.

**What a replay proves, exactly:** the library still builds the request the
provider was measured on. It does not prove the provider still answers that way
-- only a fresh call does that. So the recording carries the date it was made,
and staleness is a fact a reader can see rather than one they must assume.

Two decisions here are load-bearing:

*Exchanges are flushed as they happen.* Buffering to the end would be faster and
would have lost the entire ADR-037 probe when its account ran dry mid-scan --
money spent, evidence gone.

*A miss raises.* A replay that answered anyway would report savings for a path
no provider ever saw: a fabricated number with a receipt attached, which is
worse than no recording at all.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from optio_optimize.types import LLMRequest, LLMResponse

if TYPE_CHECKING:
    from optio_optimize.bench.providers import BenchProvider

#: Bumped when the on-disk shape changes in a way older readers cannot handle.
#: Recordings are evidence and outlive releases, so a reader that met an
#: unknown version and guessed would be inventing measurements.
RECORDING_VERSION = 1


def exchange_key(request: LLMRequest) -> str:
    """Digest every part of ``request`` that reaches the provider.

    Deliberately **not** :func:`~optio_optimize.cache.request_key`, which omits
    ``max_tokens`` and the ``cacheable`` markers. Both omissions are right there
    and wrong here:

    * ``max_tokens`` truncates a reply rather than changing it, so an exact
      cache may share one entry across ceilings. A recording is evidence about
      one exchange, and a truncated reply is a different exchange with different
      billed output tokens.
    * ``cacheable`` markers do not change the *answer*, which is why the cache
      ignores them -- but they are the entire mechanism ``prefix_cache`` is paid
      for. A key blind to them would replay the old cache numbers after the
      stage stopped emitting them, and report the saving intact.

    A miss is the regression signal, so this must move whenever the wire does.

    Args:
        request: The request as it would be sent.

    Returns:
        A hex digest, stable across processes and releases.
    """
    payload = json.dumps(
        {
            "model": request.model,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "name": m.name,
                    "cacheable": m.cacheable,
                    "cache_ttl": m.cache_ttl,
                }
                for m in request.messages
            ],
            "max_tokens": request.max_tokens,
            "tools": list(request.tools),
            "temperature": request.temperature,
            "response_format": request.response_format,
            "stop": list(request.stop),
            "thinking_budget": request.thinking_budget,
            "reasoning_effort": request.reasoning_effort,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    # blake2b rather than hash(): PYTHONHASHSEED randomizes str hashing per
    # process, and a recording read by a later run must key identically.
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class RecordingProvider:
    """Wrap a live provider and keep every exchange it pays for.

    Observes only: the response reaches the caller untouched, so a run that is
    being recorded measures exactly what it would have measured otherwise.

    Attributes:
        inner: The provider actually being called.
        path: File to append to. Created with a provenance header.
    """

    inner: BenchProvider
    path: Path

    def __post_init__(self) -> None:
        """Write the provenance header, replacing any earlier recording."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "version": RECORDING_VERSION,
            "provider": self.inner.label,
            "model": self.inner.model,
            # The reader's only defence against a stale recording. A replay
            # cannot tell that a provider changed its behaviour; a date lets a
            # human decide whether to re-measure.
            # `timezone.utc` rather than `datetime.UTC`: the latter is 3.11+,
            # and pyproject declares 3.10.
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    @property
    def is_live(self) -> bool:
        """Whatever the wrapped provider is -- recording changes nothing."""
        return self.inner.is_live

    @property
    def models_latency(self) -> bool:
        """Whatever the wrapped provider is."""
        return self.inner.models_latency

    @property
    def model(self) -> str:
        """The model the wrapped provider serves."""
        return self.inner.model

    @property
    def label(self) -> str:
        """Identifier for reports, marked so a reader knows a file was written."""
        return f"recording({self.inner.label})"

    def reset(self) -> None:
        """Delegate. Arm boundaries are the inner provider's business."""
        self.inner.reset()

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Call the provider, append the exchange, return the reply unchanged."""
        response = self.inner(request)
        line = json.dumps(
            {"key": exchange_key(request), "response": asdict(response)},
            default=str,
        )
        # Appended per call rather than buffered: these exchanges cost real
        # money, and a run that dies at request 40 must keep the first 39.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return response


@dataclass
class ReplayProvider:
    """Serve exchanges recorded from a live run. Calls nothing, costs nothing.

    Repeats are served **in the order they were recorded**, because that order
    carries the measurement: two identical requests are how prefix caching is
    observed at all, and the responses differ -- the first writes the cache, the
    second reads it. Serving either one twice would erase the finding.

    Attributes:
        path: A file written by :class:`RecordingProvider`.
    """

    path: Path
    _exchanges: dict[str, list[LLMResponse]] = field(default_factory=dict, init=False)
    _header: dict[str, Any] = field(default_factory=dict, init=False)
    _served: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)

    def __post_init__(self) -> None:
        """Load the recording.

        Raises:
            ValueError: If the file was written by a newer, unknown version.
        """
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self._header = json.loads(lines[0]) if lines else {}
        version = self._header.get("version")
        if version != RECORDING_VERSION:
            raise ValueError(
                f"{self.path} declares recording version {version!r}, and this build "
                f"reads {RECORDING_VERSION}. Refusing to guess at a shape it does not "
                f"know: a misread recording is a fabricated measurement."
            )
        by_key: dict[str, list[LLMResponse]] = defaultdict(list)
        for line in lines[1:]:
            if not line.strip():
                continue
            entry = json.loads(line)
            by_key[entry["key"]].append(LLMResponse(**entry["response"]))
        self._exchanges = dict(by_key)

    @property
    def recorded_at(self) -> str:
        """When the live run happened, so staleness is visible rather than assumed."""
        return str(self._header.get("recorded_at", ""))

    @property
    def is_live(self) -> bool:
        """False. The token and cache counts are real; the call is not.

        This is what stops the metrics layer printing a wall-clock comparison
        that would be measuring a file read, and what keeps a replayed arm from
        being reported as fresh evidence.
        """
        return False

    @property
    def models_latency(self) -> bool:
        """False, for the same reason."""
        return False

    @property
    def model(self) -> str:
        """The model the recording was made against."""
        return str(self._header.get("model", ""))

    @property
    def label(self) -> str:
        """Identifier carrying the date, so reports cannot hide staleness."""
        return f"replay({self._header.get('provider', self.path.name)}@{self.recorded_at})"

    def reset(self) -> None:
        """No-op, deliberately.

        The cursor is not rewound between A/B arms: the recording is one linear
        run and the arms appear in it in order. Rewinding would serve the
        baseline's answers to the optimized arm.
        """

    def __call__(self, request: LLMRequest) -> LLMResponse:
        """Serve the next recorded response for this request.

        Raises:
            KeyError: If this exchange was never recorded, or was recorded fewer
                times than it has now been asked for. Never fabricates: a
                replay that answered anyway would credit this library with a
                saving no provider ever measured.
        """
        key = exchange_key(request)
        recorded = self._exchanges.get(key, [])
        index = self._served[key]
        if index >= len(recorded):
            seen = f"recorded {len(recorded)}x, asked {index + 1}x" if recorded else "never seen"
            raise KeyError(
                f"this request is not in the recording ({seen}). {self.path.name} was "
                f"made on {self.recorded_at or 'an unknown date'}; either the library "
                f"now builds a different request -- which is the regression this "
                f"replay exists to catch -- or the recording needs remaking."
            )
        self._served[key] = index + 1
        return recorded[index]


__all__ = [
    "RECORDING_VERSION",
    "RecordingProvider",
    "ReplayProvider",
    "exchange_key",
]
