"""Every documented way to scope a run must actually produce signals.

The three entry points the README offers -- ``@meter`` on a sync function,
``@meter`` on an async one, and a bare ``with RunContext(...)`` -- are tested
here against the same yardstick: run an agent, then look for numbers.

Written because two of the three produced none, and said nothing about it.

* ``@meter`` had one wrapper, a synchronous one. Applied to an ``async def`` it
  measured the *construction of the coroutine*: ``fn(...)`` returns a coroutine
  object without executing a line of the body, so the ``with`` block closed the
  run before the agent had started. Most agent code is async.
* ``RunContext`` used alone opened no span. Signals are span attributes, so
  every one of them was computed and then dropped for want of anywhere to put
  it -- the documented path for a raw SDK loop or an unsupported framework.

Both failed in the same direction: the integration looks applied, the agent
works perfectly, and the telemetry is empty. Neither had a test, which is the
whole reason they survived. So this file asserts the property that actually
matters to a user -- *did I get numbers* -- rather than any mechanism, and it
asserts it once per entry point so a new one cannot quietly join without.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio import RunContext, meter, semconv
from optio.config import Config
from optio.runtime import failopen, installer
from optio.runtime.run_context import current_run

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    failopen.reset_activations()
    installer.reset_installations()
    yield
    failopen.reset_activations()
    installer.reset_installations()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def step(provider: TracerProvider) -> None:
    """One priceable GenAI span, as a framework would emit it."""
    with provider.get_tracer("app").start_as_current_span("chat gpt-4o") as span:
        span.set_attribute(semconv.GEN_AI_SYSTEM, "openai")
        span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
        span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 1_000)
        span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 200)


def emitted(exporter: InMemorySpanExporter) -> dict[str, object]:
    """Every optio signal on every exported span, wherever it landed.

    Deliberately indifferent to *which* span carries what. The question this
    file asks is whether the user got numbers at all, and pinning them to a
    particular span here would just restate what the per-lane suites already
    cover.
    """
    found: dict[str, object] = {}
    for span in exporter.get_finished_spans():
        for name, value in (span.attributes or {}).items():
            if name in semconv.EMITTED_SIGNALS:
                found[name] = value
    return found


#: What any working entry point must produce from the run above: a priced run
#: and a behavioural verdict. Quality is off by default (ADR-003) and absent.
_EXPECTED = (semconv.RUN_ACTUAL_COST, semconv.RUN_LOOP_STATE, semconv.RUN_REPEAT_COUNT)


class TestTheDecoratorOnASyncFunction:
    def test_it_emits_signals(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        @meter(config=Config(), provider=provider)
        def agent() -> str:
            step(provider)
            return "done"

        assert agent() == "done"

        signals = emitted(exporter)
        for name in _EXPECTED:
            assert name in signals, f"{name} is missing from a synchronous metered run"


class TestTheDecoratorOnAnAsyncFunction:
    """The failure mode here is silence, so these test for presence."""

    def test_it_emits_signals(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        @meter(config=Config(), provider=provider)
        async def agent() -> str:
            await asyncio.sleep(0)
            step(provider)
            return "done"

        assert asyncio.run(agent()) == "done"

        signals = emitted(exporter)
        for name in _EXPECTED:
            assert name in signals, (
                f"{name} is missing from an async metered run; the decorator is "
                "measuring coroutine construction rather than the agent"
            )

    def test_the_run_is_current_inside_the_body(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """The mechanism, stated once.

        Every missing signal above follows from this one fact: with a sync
        wrapper, the run had already ended by the time the coroutine ran, so
        spans emitted inside it belonged to no run and were dropped by the tap.
        """
        seen: list[object] = []

        @meter(config=Config(), provider=provider)
        async def agent() -> None:
            seen.append(current_run())

        asyncio.run(agent())

        assert seen[0] is not None, "no run was current inside the agent body"

    def test_it_returns_a_coroutine_not_a_finished_result(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Wrapping must not change the caller's contract.

        An async function still has to return an awaitable, or every caller
        that does `await agent()` breaks.
        """

        @meter(config=Config(), provider=provider)
        async def agent() -> str:
            return "done"

        coro = agent()
        assert asyncio.iscoroutine(coro)
        assert asyncio.run(coro) == "done"

    def test_an_exception_still_propagates(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """optio observes runs; it never swallows the user's errors."""

        @meter(config=Config(), provider=provider)
        async def agent() -> None:
            step(provider)
            raise RuntimeError("agent exploded")

        with pytest.raises(RuntimeError, match="agent exploded"):
            asyncio.run(agent())

        # And the run still ended, so the signals for the work it did survive.
        assert semconv.RUN_ACTUAL_COST in emitted(exporter)

    def test_concurrent_runs_do_not_share_a_context(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """The reason a `ContextVar` is safe inside a coroutine.

        Each asyncio task runs with its own copy of the context, so runs that
        overlap in time do not overwrite each other's current run -- which is
        the failure the sync wrapper would have hidden, since its run never
        overlapped anything.
        """
        ids: list[str] = []

        @meter(config=Config(), provider=provider)
        async def agent() -> None:
            run = current_run()
            assert run is not None
            await asyncio.sleep(0.01)
            after = current_run()
            assert after is not None
            # Still the same run after suspending and resuming.
            assert after.run_id == run.run_id
            ids.append(run.run_id)

        async def main() -> None:
            await asyncio.gather(*(agent() for _ in range(8)))

        asyncio.run(main())

        assert len(set(ids)) == 8, "concurrent async runs shared a run id"


class TestABareRunContext:
    """The README's "or scope it yourself", and the answer for "No framework?"."""

    def test_it_emits_signals_without_an_enclosing_span(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        installer.install_tap(Config(), provider)

        with RunContext(budget="$0.50"):
            step(provider)

        signals = emitted(exporter)
        for name in _EXPECTED:
            assert name in signals, (
                f"{name} is missing from a bare RunContext; there was no span to write it to"
            )

    def test_the_budget_signal_arrives_too(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """A budget nobody can see is not a budget.

        This is the signal a downstream policy gates on, and the entry point
        whose whole documented purpose is setting one.
        """
        installer.install_tap(Config(), provider)

        with RunContext(budget="$0.50"):
            step(provider)

        assert emitted(exporter)[semconv.RUN_BUDGET_REMAINING] == pytest.approx(0.5 - 0.0045)

    def test_an_enclosing_span_is_not_duplicated(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """A run gets one span, not two.

        ``@meter`` already opens one, and so do the adapters' own run spans, so
        opening unconditionally would double every metered run in the trace.
        """
        installer.install_tap(Config(), provider)
        tracer = provider.get_tracer("app")

        with tracer.start_as_current_span("caller.run"), RunContext():
            step(provider)

        names = [s.name for s in exporter.get_finished_spans()]
        assert names.count("caller.run") == 1
        assert not any(name.startswith("optio.run.") for name in names)

    def test_the_span_closes_with_the_run(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """An unclosed span holds its trace open until the process exits."""
        installer.install_tap(Config(), provider)

        with RunContext():
            step(provider)

        run_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("optio.run.")]
        assert len(run_spans) == 1
        assert run_spans[0].end_time is not None

    def test_the_span_lands_on_the_configured_provider(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Not the global one.

        A user who passes their own ``TracerProvider`` has exporters attached to
        *it*. Opening the run span on the global provider would put every signal
        somewhere they are not listening -- the same defect as the run span not
        existing at all, one step further along.
        """
        installer.install_tap(Config(), provider)

        with RunContext():
            step(provider)

        assert [s.name for s in exporter.get_finished_spans() if s.name.startswith("optio.run.")]

    def test_an_exception_still_propagates(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        installer.install_tap(Config(), provider)

        with pytest.raises(RuntimeError, match="agent exploded"), RunContext():
            step(provider)
            raise RuntimeError("agent exploded")

        assert semconv.RUN_ACTUAL_COST in emitted(exporter)
