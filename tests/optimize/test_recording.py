"""Live measurements, kept (ADR-039).

Every figure this project has paid for was printed to a terminal, read once,
and written into an ADR by hand. The response itself was discarded. So a
measurement costs money *every time it is checked*, and in practice that means
it is checked once and trusted forever -- the shape that let ``prefix_cache``
claim a saving for months on a prompt below the cacheable minimum, hitting
nothing.

A recorded exchange changes that. The call is paid for once; every later run
replays it for nothing. What replay proves is narrow and worth stating: **the
library still builds the request the provider was measured on**. It cannot
prove the provider still answers that way -- only a fresh call does that, which
is why the recording carries the date it was made.

The failure mode to design against is a replay that quietly invents an answer.
A recording that returns *something* for an unrecorded request would report
savings for code paths that were never measured, which is worse than no
recording at all: it is the fabricated-number failure with a receipt attached.
So a miss raises, and the key covers everything that reaches the wire --
including the ``cacheable`` markers, since a library that stopped emitting them
would otherwise replay the old cache numbers and show no regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optio_optimize.bench.recording import (
    RECORDING_VERSION,
    RecordingProvider,
    ReplayProvider,
    exchange_key,
)
from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


def _request(text: str = "hello", *, cacheable: bool = False, **kw: object) -> LLMRequest:
    return LLMRequest(
        model="claude-haiku-4-5",
        messages=(Message(role="user", content=text, cacheable=cacheable),),
        temperature=0.0,
        **kw,  # type: ignore[arg-type]
    )


class _FakeLive:
    """A stand-in for a provider that costs money."""

    is_live = True
    models_latency = True
    model = "claude-haiku-4-5"
    label = "fake(claude-haiku-4-5)"

    def __init__(self, *responses: LLMResponse) -> None:
        self._queue = list(responses)
        self.calls = 0

    def __call__(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return self._queue.pop(0) if self._queue else LLMResponse(content="ok", input_tokens=10)

    def reset(self) -> None:
        pass


class TestRecording:
    def test_the_response_reaches_the_caller_unchanged(self, tmp_path: Path) -> None:
        """Recording observes. A wrapper that altered the reply would corrupt
        the very run it exists to preserve."""
        answer = LLMResponse(content="42", input_tokens=99, output_tokens=3)
        provider = RecordingProvider(_FakeLive(answer), tmp_path / "r.jsonl")

        assert provider(_request()) is answer

    def test_each_exchange_is_on_disk_before_the_next_call(self, tmp_path: Path) -> None:
        """These calls cost money; a crash must not take the run with it.

        Buffering until close would be faster and would have thrown away the
        $7.60 probe that produced ADR-037 when its account ran dry mid-scan.
        """
        path = tmp_path / "r.jsonl"
        provider = RecordingProvider(_FakeLive(), path)

        provider(_request("first"))
        after_one = len(path.read_text(encoding="utf-8").splitlines())
        provider(_request("second"))

        assert after_one == 2  # header + one exchange
        assert len(path.read_text(encoding="utf-8").splitlines()) == 3

    def test_the_header_says_what_was_measured_and_when(self, tmp_path: Path) -> None:
        """A number with no provenance is the thing this package keeps deleting."""
        path = tmp_path / "r.jsonl"
        RecordingProvider(_FakeLive(), path)(_request())

        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["provider"] == "fake(claude-haiku-4-5)"
        assert header["recorded_at"]


class TestReplay:
    def test_it_serves_the_recorded_answer(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        RecordingProvider(_FakeLive(LLMResponse(content="42")), path)(_request())

        assert ReplayProvider(path)(_request()).content == "42"

    def test_the_cache_numbers_survive_the_round_trip(self, tmp_path: Path) -> None:
        """The whole reason to keep the response rather than a summary.

        ``cache_write_1h_tokens`` is a *subset* of ``cache_write_tokens``
        (ADR-021); a serializer that treated them as siblings would double-count
        the most expensive tokens in the request on every replay.
        """
        path = tmp_path / "r.jsonl"
        live = LLMResponse(
            content="ok",
            input_tokens=6_317,
            output_tokens=120,
            cached_input_tokens=5_000,
            cache_write_tokens=1_317,
            cache_write_1h_tokens=1_000,
            model="claude-haiku-4-5",
            finish_reason="end_turn",
        )
        RecordingProvider(_FakeLive(live), path)(_request())

        assert ReplayProvider(path)(_request()) == live

    def test_it_calls_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "r.jsonl"
        inner = _FakeLive()
        RecordingProvider(inner, path)(_request())
        before = inner.calls

        ReplayProvider(path)(_request())

        assert inner.calls == before

    def test_an_unrecorded_request_raises(self, tmp_path: Path) -> None:
        """Never fabricate. A replay that answered anyway would report savings
        for a path no provider ever saw."""
        path = tmp_path / "r.jsonl"
        RecordingProvider(_FakeLive(), path)(_request("recorded"))

        with pytest.raises(KeyError, match="not in the recording"):
            ReplayProvider(path)(_request("never recorded"))

    def test_repeats_replay_in_the_order_they_happened(self, tmp_path: Path) -> None:
        """Two identical requests are how caching is measured at all: the first
        writes the cache and the second reads it, and the *responses differ*.
        Serving either one twice would erase the measurement."""
        path = tmp_path / "r.jsonl"
        write = LLMResponse(content="ok", input_tokens=1_000, cache_write_tokens=1_000)
        read = LLMResponse(content="ok", input_tokens=1_000, cached_input_tokens=1_000)
        recorder = RecordingProvider(_FakeLive(write, read), path)
        recorder(_request())
        recorder(_request())

        replay = ReplayProvider(path)
        assert replay(_request()).cache_write_tokens == 1_000
        assert replay(_request()).cached_input_tokens == 1_000

    def test_exhausting_the_repeats_raises_rather_than_reusing_one(self, tmp_path: Path) -> None:
        """A third call was never measured, however many times the first two were."""
        path = tmp_path / "r.jsonl"
        RecordingProvider(_FakeLive(), path)(_request())

        replay = ReplayProvider(path)
        replay(_request())
        with pytest.raises(KeyError, match="not in the recording"):
            replay(_request())

    def test_reset_does_not_rewind_between_arms(self, tmp_path: Path) -> None:
        """The harness calls ``reset()`` at every A/B arm boundary.

        A recording is one linear run with both arms inside it in order, so
        rewinding there would replay the baseline's answers to the optimized arm
        -- and since the baseline is deliberately run first, the optimized arm
        would inherit its uncached numbers and the measured saving would collapse
        to zero. Silently.
        """
        path = tmp_path / "r.jsonl"
        first = LLMResponse(content="baseline", input_tokens=1_000)
        second = LLMResponse(content="optimized", input_tokens=1_000, cached_input_tokens=900)
        recorder = RecordingProvider(_FakeLive(first, second), path)
        recorder(_request())
        recorder(_request())

        replay = ReplayProvider(path)
        replay(_request())
        replay.reset()

        assert replay(_request()).content == "optimized"

    def test_a_recording_from_an_unknown_version_is_refused(self, tmp_path: Path) -> None:
        """Reading a shape this build does not know would invent measurements.

        The failure has to be loud at load, not at the first odd number: a
        recording is evidence, and evidence read through the wrong schema is
        indistinguishable from evidence that was never gathered.
        """
        path = tmp_path / "r.jsonl"
        RecordingProvider(_FakeLive(), path)(_request())
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["version"] = RECORDING_VERSION + 1
        path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="recording version"):
            ReplayProvider(path)

    def test_it_does_not_claim_to_be_live(self, tmp_path: Path) -> None:
        """Tokens and cache counts are real; latency and answer quality are not.

        ``is_live`` is what stops the metrics layer printing a wall-clock
        comparison that would be measuring a file read.
        """
        path = tmp_path / "r.jsonl"
        RecordingProvider(_FakeLive(), path)(_request())
        replay = ReplayProvider(path)

        assert replay.is_live is False
        assert replay.models_latency is False


class TestItDrivesTheRealHarness:
    """A replay nobody can run a benchmark through saves nothing.

    The unit tests above call the providers directly. This runs the actual
    ``compare`` both ways -- once recording, once replaying -- because the value
    on offer is *the second run costing nothing*, and that only holds if the
    optimizer builds byte-identical requests across processes. Any hidden
    per-run variation (a timestamp, a set iterated in hash order) shows up here
    as a replay miss and nowhere else.
    """

    def test_a_recorded_benchmark_replays_to_the_same_numbers(self, tmp_path: Path) -> None:
        from optio_optimize.bench.harness import compare
        from optio_optimize.bench.providers import SimulatedProvider
        from optio_optimize.bench.workloads import WORKLOADS

        workload = WORKLOADS["multi_turn_chat"]
        path = tmp_path / "bench.jsonl"

        recorded = compare(workload, RecordingProvider(SimulatedProvider(), path))
        replayed = compare(workload, ReplayProvider(path))

        assert replayed.optimized.input_tokens == recorded.optimized.input_tokens
        assert replayed.baseline.input_tokens == recorded.baseline.input_tokens
        assert replayed.total_token_reduction == recorded.total_token_reduction

    def test_the_replayed_arm_is_not_reported_as_live(self, tmp_path: Path) -> None:
        """Its tokens are the recorded run's; its clock is a file read."""
        from optio_optimize.bench.harness import compare
        from optio_optimize.bench.providers import SimulatedProvider
        from optio_optimize.bench.workloads import WORKLOADS

        workload = WORKLOADS["multi_turn_chat"]
        path = tmp_path / "bench.jsonl"
        compare(workload, RecordingProvider(SimulatedProvider(), path))

        assert compare(workload, ReplayProvider(path)).optimized.live is False


class TestTheKeyCoversWhatReachesTheWire:
    """A miss is the regression signal, so the key must move when the wire does."""

    def test_dropping_a_cache_marker_is_a_miss(self) -> None:
        """The regression this exists to catch.

        ``prefix_cache`` earns its saving by marking a prefix ``cacheable``. If
        it stopped, a key blind to the marker would replay the old cache numbers
        and the suite would report the saving intact -- exactly the months-long
        silent failure recorded in the changelog.
        """
        assert exchange_key(_request(cacheable=True)) != exchange_key(_request(cacheable=False))

    def test_a_different_ceiling_is_a_miss(self) -> None:
        """Unlike ``cache.request_key``, which omits ``max_tokens`` on purpose.

        That omission is right for serving a cached answer -- a ceiling
        truncates a reply rather than changing it -- and wrong here: a recording
        is evidence about one exchange, and a truncated reply is a different
        exchange with different billed output tokens.
        """
        assert exchange_key(_request(max_tokens=16)) != exchange_key(_request(max_tokens=4_000))

    def test_the_same_request_keys_the_same_in_another_process(self) -> None:
        """Recordings outlive the process that made them.

        Run in a subprocess with a different ``PYTHONHASHSEED``, because that is
        the only way this fails: ``hash()`` on a string is salted per process,
        so a digest built from it agrees with itself all through one test
        session and disagrees with yesterday's recording. Every exchange would
        miss, and the failure would look like the regression this replay exists
        to report.
        """
        import os
        import subprocess
        import sys

        script = (
            "from optio_optimize.bench.recording import exchange_key\n"
            "from optio_optimize.types import LLMRequest, Message\n"
            "print(exchange_key(LLMRequest(model='claude-haiku-4-5',"
            " messages=(Message(role='user', content='stable'),), temperature=0.0)))\n"
        )
        env = {**os.environ, "PYTHONHASHSEED": "1"}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
        )

        assert out.stdout.strip() == exchange_key(_request("stable"))

    def test_tools_are_in_the_key(self) -> None:
        tools = ({"name": "search", "parameters": {}},)
        assert exchange_key(_request(tools=tools)) != exchange_key(_request())
