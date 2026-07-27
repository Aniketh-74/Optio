"""Self-observability instruments (Section 12).

optio is an observability *producer*; this is where it observes itself. Four
instruments, all under ``optio.internal.*``:

* ``signals_emitted`` -- counter, how many signals were written
* ``lane_errors`` -- counter, how many times the fail-open guard absorbed a
  failure. A rising value is the single best indicator of a bug in this library
  (R-TECH-4), and it is invisible to the agent by design, so it has to surface
  somewhere.
* ``overhead`` -- histogram of seconds spent inside the tap, the measurement
  SC-5 budgets at 5 ms p99
* ``sampling_rate`` -- gauge, the configured judge sample rate, so a quality
  score's provenance can be reconstructed later

## Why metrics and not span attributes

The namespace is deliberately not ``gen_ai.*``. These describe optio's health,
not the agent's economics, and a consumer's policy must never be able to gate on
them -- a rule that says "deny if lane_errors > 0" would convert our bug into
the user's outage, which is exactly the inversion ADR-004 exists to prevent.
Keeping them on a separate namespace *and* a separate signal type (metrics, not
span attributes) makes that mistake hard to make by accident.

## Why every call is guarded

This runs on the hot path. A user with no metrics SDK configured, a misbehaving
exporter, or an OTel version whose metrics API differs must not be able to break
an agent through the code that reports our own health -- self-monitoring that
takes down the process it monitors is worse than none (ADR-004). Every
instrument is resolved lazily and every recording is wrapped; failures disable
the instrument for the process rather than raising or logging on each step.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Final

from optio import semconv

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter, Observation

_log: Final = logging.getLogger("optio")

_lock: Final = threading.Lock()

#: Resolved once. ``False`` means metrics are unavailable or broken in this
#: process and every entry point becomes a no-op -- checked before the lock so
#: the disabled path costs one attribute read.
_enabled = True

_signals_emitted: Counter | None = None
_lane_errors: Counter | None = None
_overhead: Histogram | None = None
_meter: Meter | None = None


def _disable(reason: str, error: BaseException) -> None:
    """Turn self-observability off for the process.

    Args:
        reason: What was being attempted.
        error: The failure.
    """
    global _enabled
    _enabled = False
    # Type only, never the message (Section 10): an exporter's exception can
    # carry endpoint URLs and headers, and headers carry credentials.
    _log.debug(
        "optio: self-observability disabled after %s failed (%s); signals are unaffected",
        reason,
        type(error).__name__,
    )


def _instruments() -> tuple[Counter, Counter, Histogram] | None:
    """Return the instrument triple, creating it on first use.

    Returns:
        The instruments, or ``None`` when metrics are unavailable.
    """
    global _signals_emitted, _lane_errors, _overhead, _meter

    if not _enabled:
        return None
    if _signals_emitted is not None and _lane_errors is not None and _overhead is not None:
        return _signals_emitted, _lane_errors, _overhead

    with _lock:
        # Re-check inside the lock: another thread may have built them, or
        # hit a failure and disabled us, between the checks above and here.
        if _signals_emitted is not None and _lane_errors is not None and _overhead is not None:
            return _signals_emitted, _lane_errors, _overhead
        # `globals()` rather than the bare name: mypy narrows `_enabled` to
        # True from the check above and calls this branch unreachable, which is
        # only true single-threaded. The read has to be opaque to preserve it.
        if not globals()["_enabled"]:
            return None
        try:
            from opentelemetry import metrics

            _meter = metrics.get_meter("optio")
            _signals_emitted = _meter.create_counter(
                semconv.INTERNAL_SIGNALS_EMITTED,
                unit="{signal}",
                description="Signals written to a run span.",
            )
            _lane_errors = _meter.create_counter(
                semconv.INTERNAL_LANE_ERRORS,
                unit="{error}",
                description="Failures absorbed by the fail-open guard.",
            )
            _overhead = _meter.create_histogram(
                semconv.INTERNAL_OVERHEAD_SECONDS,
                unit="s",
                description="Seconds spent inside optio per span dispatch.",
            )
        except Exception as error:  # noqa: BLE001 - never break the agent
            _disable("instrument creation", error)
            return None
        return _signals_emitted, _lane_errors, _overhead


def record_signals_emitted(count: int, lane: str) -> None:
    """Count signals written.

    Args:
        count: How many were written.
        lane: The lane that produced them.
    """
    if not _enabled or count <= 0:
        return
    instruments = _instruments()
    if instruments is None:
        return
    try:
        instruments[0].add(count, {"lane": lane})
    except Exception as error:  # noqa: BLE001 - never break the agent
        _disable("signals_emitted.add", error)


def record_lane_error(component: str) -> None:
    """Count one fail-open activation.

    Args:
        component: The lane or subsystem that failed.
    """
    if not _enabled:
        return
    instruments = _instruments()
    if instruments is None:
        return
    try:
        instruments[1].add(1, {"component": component})
    except Exception as error:  # noqa: BLE001 - never break the agent
        _disable("lane_errors.add", error)


def record_overhead(seconds: float) -> None:
    """Record time spent inside optio for one dispatch.

    Args:
        seconds: Elapsed seconds.
    """
    if not _enabled:
        return
    instruments = _instruments()
    if instruments is None:
        return
    try:
        instruments[2].record(seconds)
    except Exception as error:  # noqa: BLE001 - never break the agent
        _disable("overhead.record", error)


def record_sampling_rate(rate: float) -> None:
    """Publish the configured judge sampling rate.

    An observable gauge rather than a counter: the rate is a configured value,
    not an event stream, and a consumer reading a quality score later needs to
    know what fraction of runs it came from.

    Called once at setup, so unlike the others this is off the hot path.

    Args:
        rate: The configured rate, in ``[0, 1]``.
    """
    if not _enabled:
        return
    if _instruments() is None:
        return
    try:
        from opentelemetry import metrics

        meter = _meter if _meter is not None else metrics.get_meter("optio")
        meter.create_observable_gauge(
            semconv.INTERNAL_SAMPLING_RATE,
            callbacks=[lambda _options: [_observation(rate)]],
            unit="1",
            description="Configured fraction of runs routed to the quality judge.",
        )
    except Exception as error:  # noqa: BLE001 - never break the agent
        _disable("sampling_rate gauge", error)


def _observation(value: float) -> Observation:
    """Build a metrics Observation.

    Imported at call time so this module adds nothing to core import cost
    (Section 11) for users who never configure a metrics SDK.

    Args:
        value: The observed value.

    Returns:
        An ``Observation``.
    """
    from opentelemetry.metrics import Observation as _Observation

    return _Observation(value)


def reset_for_test() -> None:
    """Re-enable and drop cached instruments.

    Exposed for tests, which need a clean slate per case and must be able to
    exercise the disabled path without leaking it into the next test.
    """
    global _enabled, _signals_emitted, _lane_errors, _overhead, _meter
    with _lock:
        _enabled = True
        _signals_emitted = None
        _lane_errors = None
        _overhead = None
        _meter = None


def is_enabled() -> bool:
    """Whether self-observability is still active in this process."""
    return _enabled
