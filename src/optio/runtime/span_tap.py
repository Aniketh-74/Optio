"""Framework-agnostic span ingestion (M1-3).

The tap is an OTel :class:`~opentelemetry.sdk.trace.SpanProcessor` installed on
the user's existing tracer provider. That placement is what makes ``optio``
portable: any framework emitting GenAI spans is observable without a
per-framework integration, and adapters (M1-4, M4) shrink to "make sure spans
are emitted and the tap is installed".

**Read here, write elsewhere.** A span processor cannot annotate the span it
observes. By the time any hook fires the SDK has already ended the span:
``_on_ending`` sees ``is_recording() == False`` and rejects writes with
"Setting attribute on ended span", while ``on_end`` receives a genuine
``ReadableSpan`` that has no ``set_attribute`` at all. So step spans are an
input only. Signals are written to the *run* span, which is still open while its
child steps come and go -- and which is the correct home for them anyway, since
every signal optio emits is run-scoped (``docs/signals.md``).

Three further rules govern this module:

* **Only GenAI spans are processed.** A span with no ``gen_ai.*`` attribute is
  the user's own application traffic and none of our business. Rejecting it
  cheaply keeps hot-path cost proportional to *our* spans rather than to all of
  the user's (SC-5).
* **Every dispatch is guarded.** Lanes may raise; the tap is where that is
  contained. ``on_end`` is called from inside the agent's call stack, so an
  escaping exception would be an agent-visible error (ADR-004).
* **A span outside any run is dropped.** Signals are run-scoped, so there is
  nowhere to attribute it.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Final

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

from optio import semconv
from optio.lanes.registry import enabled_lanes
from optio.runtime import selfobs
from optio.runtime.failopen import guard, guard_signals
from optio.runtime.run_context import current_run
from optio.runtime.signal_writer import write_signals

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace import Span as SdkSpan
    from opentelemetry.trace import Span, Tracer

    from optio.config import Config
    from optio.lanes.base import Lane, RunLike

_log: Final = logging.getLogger("optio")

#: Prefix identifying spans this library cares about.
_GENAI_PREFIX: Final = f"{semconv.GENAI_NAMESPACE}."


def is_genai_span(span: ReadableSpan) -> bool:
    """Whether a span is GenAI *input* this library should process.

    Two things are excluded, and the second is the subtle one:

    * Spans with no ``gen_ai.*`` attribute at all -- the user's own traffic.
    * Spans carrying **only** attributes optio itself emitted. Writing
      signals to the run span makes that span match ``gen_ai.*``, so when it
      ends it comes back through the tap. Feeding it to the lanes would let a
      run's own cost signal be re-ingested as if it were a fresh step, double
      counting it -- silently, in the direction that makes the product's core
      number wrong (R-TECH-1).

    Args:
        span: The span to classify.

    Returns:
        ``True`` if the span is GenAI input worth dispatching.
    """
    attributes = span.attributes
    if not attributes:
        return False

    has_genai = False
    for key in attributes:
        if not key.startswith(_GENAI_PREFIX):
            continue
        if key in semconv.EMITTED_SIGNALS:
            # Our own output; does not make the span an input.
            continue
        has_genai = True
        break
    return has_genai


class OptioSpanTap(SpanProcessor):
    """Dispatches finished GenAI spans to the enabled lanes.

    Attributes:
        config: The active configuration.
        lanes: The lanes this tap dispatches to.
    """

    def __init__(
        self,
        config: Config,
        lanes: list[Lane] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Build a tap.

        Args:
            config: Active configuration.
            lanes: Explicit lane list, primarily for tests. Resolved from
                ``config`` when omitted.
            tracer: Tracer for lanes that emit spans of their own rather than
                attributes on the run span -- currently the quality lane's
                deferred judge score. Comes from the provider this tap is
                installed on, so those spans are recorded where the user is
                actually listening.
        """
        self.config = config
        self.lanes: list[Lane] = enabled_lanes(config, tracer) if lanes is None else lanes

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        """Handle span start.

        Nothing to do: a GenAI span has no usage or response attributes yet, so
        no lane could compute anything from it. Present because the
        ``SpanProcessor`` protocol requires it.

        Args:
            span: The span being started.
            parent_context: The parent context, unused.
        """

    def on_end(self, span: ReadableSpan) -> None:
        """Dispatch a finished span to every enabled lane.

        The whole body is guarded -- this runs inside the agent's call stack.

        Args:
            span: The span that just ended.
        """
        # perf_counter around the whole guarded dispatch: the number SC-5
        # budgets is what the agent actually pays, which includes the guard
        # itself, not just the lane bodies.
        started = time.perf_counter()
        guard(self._dispatch, None, span, component="span_tap")
        selfobs.record_overhead(time.perf_counter() - started)

    def _dispatch(self, span: ReadableSpan) -> None:
        """Route one span to the lanes and write whatever they produce.

        Args:
            span: The span to process.
        """
        if not self.lanes or not is_genai_span(span):
            return

        run = current_run()
        if run is None:
            # A GenAI span outside any run: work we were never told to meter.
            return

        target = self._signal_target()
        for lane in self.lanes:
            signals = guard_signals(lane.process_span, span, run, component=lane.name)
            if signals:
                write_signals(target, signals)
                selfobs.record_signals_emitted(len(signals), lane.name)

    def _signal_target(self) -> Span | None:
        """Return the span signals should be written to.

        The currently-active span, which during a step is the run span that
        encloses it. Returns ``None`` when nothing is recording, in which case
        the signals are dropped -- correct behaviour, since there is nowhere to
        put them.

        Returns:
            The active recording span, or ``None``.
        """
        span = trace.get_current_span()
        if span is trace.INVALID_SPAN or not span.is_recording():
            return None
        return span

    def on_run_end(self, run: RunLike, span: Span | None = None) -> None:
        """Collect end-of-run signals from every lane.

        Called by :class:`~optio.runtime.run_context.RunContext` at run end,
        not by the OTel SDK. This is where the cost lane's reconcile sweep (M2)
        and quality scoring (M5) will attach.

        Args:
            run: The run that just ended.
            span: Span to write to. Defaults to the active span.
        """
        target = span if span is not None else self._signal_target()
        for lane in self.lanes:
            signals = guard_signals(lane.on_run_end, run, component=lane.name)
            if signals:
                write_signals(target, signals)
                selfobs.record_signals_emitted(len(signals), lane.name)

    def shutdown(self) -> None:
        """Release resources. Nothing to release; lanes hold no handles."""

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Flush pending work.

        The tap is synchronous, so nothing is ever pending.

        Args:
            timeout_millis: Ignored.

        Returns:
            Always ``True``.
        """
        return True
