"""Self-observability instruments (Section 12).

Two things are being proved here. The first is that the instruments actually
record -- Section 12 named four of them and, until this module existed, all four
were constants nothing ever wrote to.

The second matters more: **self-observability must not be able to break the
agent.** It runs on the hot path, and the failure modes are real -- a user with
no metrics SDK, an exporter that raises, an OTel version whose metrics API moved.
Monitoring that takes down what it monitors is worse than no monitoring
(ADR-004), so every entry point is expected to swallow whatever it hits and turn
itself off rather than propagate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from optio import semconv
from optio.runtime import selfobs

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    selfobs.reset_for_test()
    yield
    selfobs.reset_for_test()


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> InMemoryMetricReader:
    """A real SDK meter provider whose output can be read back."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    # Patched rather than set globally: OTel's global meter provider is
    # write-once per process, so setting it would leak into every other test.
    monkeypatch.setattr("opentelemetry.metrics.get_meter", provider.get_meter)
    return reader


def _points(reader: InMemoryMetricReader, name: str) -> list[object]:
    """Pull the data points recorded for one instrument."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    found: list[object] = []
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name == name:
                    found.extend(metric.data.data_points)
    return found


class TestTheInstrumentsRecord:
    def test_signals_emitted_counts_per_lane(self, reader: InMemoryMetricReader) -> None:
        selfobs.record_signals_emitted(3, "cost")
        selfobs.record_signals_emitted(1, "behavior")

        points = _points(reader, semconv.INTERNAL_SIGNALS_EMITTED)
        by_lane = {p.attributes["lane"]: p.value for p in points}  # type: ignore[attr-defined]
        assert by_lane == {"cost": 3, "behavior": 1}

    def test_lane_errors_counts_per_component(self, reader: InMemoryMetricReader) -> None:
        selfobs.record_lane_error("cost")
        selfobs.record_lane_error("cost")
        selfobs.record_lane_error("quality")

        points = _points(reader, semconv.INTERNAL_LANE_ERRORS)
        by_component = {p.attributes["component"]: p.value for p in points}  # type: ignore[attr-defined]
        assert by_component == {"cost": 2, "quality": 1}

    def test_overhead_records_a_distribution(self, reader: InMemoryMetricReader) -> None:
        for seconds in (0.001, 0.002, 0.003):
            selfobs.record_overhead(seconds)

        points = _points(reader, semconv.INTERNAL_OVERHEAD_SECONDS)
        assert len(points) == 1
        assert points[0].count == 3  # type: ignore[attr-defined]
        assert points[0].sum == pytest.approx(0.006)  # type: ignore[attr-defined]

    def test_sampling_rate_publishes_the_configured_value(
        self, reader: InMemoryMetricReader
    ) -> None:
        selfobs.record_sampling_rate(0.25)

        points = _points(reader, semconv.INTERNAL_SAMPLING_RATE)
        assert [p.value for p in points] == [0.25]  # type: ignore[attr-defined]

    def test_zero_and_negative_counts_are_not_recorded(self, reader: InMemoryMetricReader) -> None:
        # A counter must never go backwards, and recording nothing is not an
        # event worth a data point.
        selfobs.record_signals_emitted(0, "cost")
        selfobs.record_signals_emitted(-1, "cost")

        assert _points(reader, semconv.INTERNAL_SIGNALS_EMITTED) == []

    def test_metrics_land_outside_the_gen_ai_namespace(self, reader: InMemoryMetricReader) -> None:
        # The load-bearing separation: a consumer policy gates on gen_ai.*, and
        # a rule reading our health metrics would turn our bug into their
        # outage. Nothing here may be mistakable for an agent signal.
        selfobs.record_signals_emitted(1, "cost")
        selfobs.record_lane_error("cost")
        selfobs.record_overhead(0.001)

        data = reader.get_metrics_data()
        names = [
            metric.name
            for resource in data.resource_metrics  # type: ignore[union-attr]
            for scope in resource.scope_metrics
            for metric in scope.metrics
        ]
        assert names, "expected instruments to have recorded"
        for name in names:
            assert name.startswith(semconv.INTERNAL_NAMESPACE)
            assert not name.startswith(semconv.GENAI_NAMESPACE)
            assert name not in semconv.EMITTED_SIGNALS


class TestNothingHereCanBreakTheAgent:
    """ADR-004 applied to the code that reports our own health."""

    def test_no_metrics_sdk_configured_is_survivable(self) -> None:
        # The default state for most users: the API's no-op meter. Must be
        # silent, not an error.
        selfobs.record_signals_emitted(1, "cost")
        selfobs.record_lane_error("cost")
        selfobs.record_overhead(0.001)
        selfobs.record_sampling_rate(0.1)

    def test_an_exploding_meter_disables_itself_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(_name: str) -> Mock:
            raise RuntimeError("metrics backend is down")

        monkeypatch.setattr("opentelemetry.metrics.get_meter", explode)

        selfobs.record_signals_emitted(1, "cost")

        assert not selfobs.is_enabled(), "a failing meter must disable, not retry forever"

    def test_an_exploding_instrument_disables_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meter = Mock()
        counter = Mock()
        counter.add.side_effect = RuntimeError("exporter died mid-flight")
        meter.create_counter.return_value = counter
        meter.create_histogram.return_value = Mock()
        monkeypatch.setattr("opentelemetry.metrics.get_meter", lambda _name: meter)

        selfobs.record_signals_emitted(1, "cost")

        assert not selfobs.is_enabled()

    def test_once_disabled_later_calls_are_cheap_no_ops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def explode(name: str) -> Mock:
            calls.append(name)
            raise RuntimeError("down")

        monkeypatch.setattr("opentelemetry.metrics.get_meter", explode)

        for _ in range(100):
            selfobs.record_signals_emitted(1, "cost")

        # One attempt, not a hundred: retrying a broken exporter on every span
        # would put the failure back on the hot path it was removed from.
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("boom"), ValueError("bad"), MemoryError(), TypeError("wrong")],
        ids=lambda e: type(e).__name__,
    )
    def test_any_exception_type_is_absorbed(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def explode(_name: str) -> Mock:
            raise error

        monkeypatch.setattr("opentelemetry.metrics.get_meter", explode)
        selfobs.record_lane_error("cost")
        selfobs.record_overhead(0.5)
        selfobs.record_sampling_rate(0.1)

    def test_every_entry_point_short_circuits_once_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each recorder has its own early return. Covering only one of them
        # would leave the others free to call into a broken meter forever.
        calls: list[str] = []

        def explode(name: str) -> Mock:
            calls.append(name)
            raise RuntimeError("down")

        monkeypatch.setattr("opentelemetry.metrics.get_meter", explode)
        selfobs.record_signals_emitted(1, "cost")  # disables
        assert not selfobs.is_enabled()

        selfobs.record_signals_emitted(1, "cost")
        selfobs.record_lane_error("cost")
        selfobs.record_overhead(0.001)
        selfobs.record_sampling_rate(0.1)

        assert len(calls) == 1, "no entry point may retry after the first failure"

    def test_a_failing_histogram_disables_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        meter = Mock()
        histogram = Mock()
        histogram.record.side_effect = RuntimeError("histogram exploded")
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = histogram
        monkeypatch.setattr("opentelemetry.metrics.get_meter", lambda _name: meter)

        selfobs.record_overhead(0.001)

        assert not selfobs.is_enabled()

    def test_a_failing_lane_error_counter_disables_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        meter = Mock()
        counter = Mock()
        counter.add.side_effect = RuntimeError("counter exploded")
        meter.create_counter.return_value = counter
        meter.create_histogram.return_value = Mock()
        monkeypatch.setattr("opentelemetry.metrics.get_meter", lambda _name: meter)

        selfobs.record_lane_error("cost")

        assert not selfobs.is_enabled()

    def test_a_failing_gauge_registration_disables_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        meter = Mock()
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = Mock()
        meter.create_observable_gauge.side_effect = RuntimeError("gauge exploded")
        monkeypatch.setattr("opentelemetry.metrics.get_meter", lambda _name: meter)

        selfobs.record_sampling_rate(0.1)

        assert not selfobs.is_enabled()

    def test_the_gauge_callback_produces_a_real_observation(
        self, reader: InMemoryMetricReader
    ) -> None:
        # The callback is invoked by the SDK on collection, not at registration,
        # so a broken _observation() would only surface at read time.
        selfobs.record_sampling_rate(0.42)
        points = _points(reader, semconv.INTERNAL_SAMPLING_RATE)
        assert [p.value for p in points] == [0.42]  # type: ignore[attr-defined]

    def test_overhead_and_gauge_bail_out_when_instruments_are_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `_enabled` is still True but instrument construction yields nothing:
        # both recorders must return quietly rather than index into None.
        monkeypatch.setattr(selfobs, "_instruments", lambda: None)
        selfobs.record_overhead(0.001)
        selfobs.record_sampling_rate(0.1)
        assert selfobs.is_enabled(), "bailing out is not itself a failure"

    def test_instruments_returns_none_when_already_disabled(self) -> None:
        # Line-level cover for the fast path: once off, _instruments() must not
        # even reach the lock.
        selfobs.record_signals_emitted(1, "cost")  # no SDK; harmless
        selfobs._enabled = False
        assert selfobs._instruments() is None

    def test_instruments_is_reused_rather_than_rebuilt(self, reader: InMemoryMetricReader) -> None:
        # The cached path, and the reason the lock is not taken per span.
        first = selfobs._instruments()
        second = selfobs._instruments()
        assert first is not None
        assert first is second or tuple(first) == tuple(second)  # type: ignore[arg-type]

    def test_a_thread_that_disabled_us_mid_construction_is_respected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The in-lock re-check. Simulates: this thread passed the outer check,
        # another thread failed and disabled, then this thread took the lock.
        real_lock = selfobs._lock

        class DisablingLock:
            def __enter__(self) -> None:
                real_lock.acquire()
                selfobs._enabled = False

            def __exit__(self, *_args: object) -> None:
                real_lock.release()

        monkeypatch.setattr(selfobs, "_lock", DisablingLock())
        assert selfobs._instruments() is None

    def test_failure_logs_the_exception_type_but_never_its_message(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # An exporter's exception carries endpoint URLs and headers, and
        # headers carry credentials (Section 10).
        secret = "otlp-token-abc123"

        def explode(_name: str) -> Mock:
            raise RuntimeError(f"auth failed for header {secret}")

        monkeypatch.setattr("opentelemetry.metrics.get_meter", explode)
        with caplog.at_level("DEBUG", logger="optio"):
            selfobs.record_signals_emitted(1, "cost")

        assert secret not in caplog.text
        assert "RuntimeError" in caplog.text
