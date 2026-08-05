"""Quality signals from a real instrumented run (M5-4 gate).

The path a user actually takes: enable the lane, supply a judge, run an agent,
read the attributes off the exported spans.

**Two spans, deliberately.** The heuristic answers synchronously, so its verdict
is an attribute on the run span. The judge is a model call dispatched at run
end, so it answers after that span has closed -- and an ended span cannot be
modified. Its scores go on a linked ``optio.quality`` span instead.

Until 0.4.0 they were asserted on the run span and the tests passed, because
each built a fresh lane whose cold thread pool let an instant in-process judge
win a race that a real one always loses. That is why the helpers here drain the
judge explicitly rather than assuming it has finished, and why one test measures
that the run itself did not wait.

This is also where ``gen_ai.run.cost_per_successful_task`` is proven end to end.
It is the only cross-lane signal in the system -- the cost lane owns the
numerator, the judge owns the denominator, and neither imports the other -- so
it only works if the registry orders cost ahead of quality, so that the cost is
on the run object before the judge is dispatched. Nothing in the unit tests
would catch that ordering breaking.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio import RunContext, semconv
from optio.config import Config
from optio.lanes.quality.judge import Judge, JudgeRequest, JudgeScores
from optio.runtime import failopen, installer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

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


def judge_scoring(groundedness: float = 0.92, task_success: float = 0.85) -> Judge:
    """A judge returning fixed scores."""

    def judge(_request: JudgeRequest) -> JudgeScores:
        return JudgeScores(groundedness=groundedness, task_success=task_success)

    return judge


def run_agent(
    provider: TracerProvider,
    config: Config,
    *,
    steps: int = 3,
    output_tokens: int = 40,
) -> None:
    """Drive a short instrumented run, and let any judge finish."""
    tap = installer.install_tap(config, provider)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("optio.run.test"), RunContext(config=config):
        for _ in range(steps):
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
                span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 200)
                span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

    # The judge is asynchronous by design, so the assertions below have to wait
    # for it rather than race it. Waiting *here*, after the run has closed, is
    # also the proof that the run itself did not: a test that could only pass by
    # blocking the run would fail at the `elapsed` assertion in
    # `TestTheRunDoesNotWaitForTheJudge`.
    if tap is not None:
        drain(tap)


def drain(tap: object, timeout: float = 5.0) -> None:
    """Wait for every enabled lane's outstanding async work."""
    for lane in tap.lanes:  # type: ignore[attr-defined]
        drain_lane = getattr(lane, "drain", None)
        if drain_lane is not None:
            assert drain_lane(timeout) == 0, "a judge did not finish within the drain timeout"


def signals(exporter: InMemorySpanExporter) -> dict[str, object]:
    """Return the optio signals on the run span."""
    span: ReadableSpan = next(
        s for s in exporter.get_finished_spans() if s.name.startswith("optio.run")
    )
    return {k: v for k, v in (span.attributes or {}).items() if k in semconv.EMITTED_SIGNALS}


def link_span_ids(span: ReadableSpan) -> list[int]:
    """Return the span ids this span links to."""
    ids = []
    for link in span.links:
        context = link.context
        if context is not None:
            ids.append(context.span_id)
    return ids


def quality_signals(exporter: InMemorySpanExporter) -> dict[str, object]:
    """Return the optio signals on the deferred quality span.

    A separate span because the judge answers after the run span has closed and
    an ended span cannot be modified -- so this is where every judge-derived
    signal lives. Returns an empty mapping when no quality span was emitted,
    which is what "the judge did not score this run" looks like to a consumer.
    """
    span = next(
        (s for s in exporter.get_finished_spans() if s.name == semconv.QUALITY_SPAN_NAME),
        None,
    )
    if span is None:
        return {}
    return {k: v for k, v in (span.attributes or {}).items() if k in semconv.EMITTED_SIGNALS}


class TestTheLaneIsOffByDefault:
    def test_no_quality_signals_without_opting_in(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # ADR-003. The default configuration must emit nothing from this lane.
        run_agent(provider, Config())
        emitted = signals(exporter)

        for name in (
            semconv.RUN_SUCCESS,
            semconv.RUN_QUALITY_GROUNDEDNESS,
            semconv.RUN_QUALITY_TASK_SUCCESS,
            semconv.RUN_COST_PER_SUCCESSFUL_TASK,
        ):
            assert name not in emitted

    def test_cost_and_behavior_still_work(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The lanes are independent: quality being off changes nothing else.
        run_agent(provider, Config())
        emitted = signals(exporter)

        assert semconv.RUN_ACTUAL_COST in emitted
        assert semconv.RUN_LOOP_STATE in emitted


class TestWithAJudge:
    def test_all_quality_signals_are_emitted(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        config = Config(quality_lane=True, quality_sample_rate=1.0, judge=judge_scoring())
        run_agent(provider, config)
        emitted = quality_signals(exporter)

        assert emitted[semconv.RUN_QUALITY_GROUNDEDNESS] == 0.92
        assert emitted[semconv.RUN_QUALITY_TASK_SUCCESS] == 0.85
        assert emitted[semconv.RUN_SUCCESS] is True

    def test_the_quality_span_links_back_to_the_run(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """Scores on an unjoinable span are scores nobody can use.

        Linked rather than parented: the run span was exported before this one
        existed, and a child of a finished span is rendered by backends as a
        gap or dropped outright.
        """
        config = Config(quality_lane=True, quality_sample_rate=1.0, judge=judge_scoring())
        run_agent(provider, config)

        spans = exporter.get_finished_spans()
        run = next(s for s in spans if s.name == "optio.run.test")
        quality = next(s for s in spans if s.name == semconv.QUALITY_SPAN_NAME)

        run_context = run.get_span_context()
        assert run_context is not None
        assert link_span_ids(quality) == [run_context.span_id]
        assert (quality.attributes or {})[semconv.RUN_ID] is not None

    def test_the_run_does_not_wait_for_the_judge(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        """The constraint the whole deferred design exists to satisfy.

        Through 0.4.0 this was satisfied trivially, by discarding every score.
        Now that scores actually arrive, something has to keep the wait from
        creeping back onto the agent's path -- a judge slow enough to be
        unmissable in the timing is the cheapest way to say so.
        """
        judged = threading.Event()

        def slow_judge(_request: JudgeRequest) -> JudgeScores:
            judged.wait(timeout=5.0)
            return JudgeScores(task_success=0.9)

        config = Config(quality_lane=True, quality_sample_rate=1.0, judge=slow_judge)
        tap = installer.install_tap(config, provider)
        tracer = provider.get_tracer("test")

        started = time.perf_counter()
        with (
            tracer.start_as_current_span("optio.run.test"),
            RunContext(config=config),
            tracer.start_as_current_span("chat") as span,
        ):
            span.set_attribute(semconv.GEN_AI_REQUEST_MODEL, "gpt-4o")
            span.set_attribute(semconv.GEN_AI_USAGE_INPUT_TOKENS, 200)
            span.set_attribute(semconv.GEN_AI_USAGE_OUTPUT_TOKENS, 40)
        elapsed = time.perf_counter() - started

        assert elapsed < 0.5, (
            f"the run took {elapsed:.3f}s with a judge that had not answered; "
            "run end is waiting for the judge"
        )
        assert semconv.RUN_QUALITY_TASK_SUCCESS not in quality_signals(exporter)

        # Only now let it answer, and the score still arrives.
        judged.set()
        assert tap is not None
        drain(tap)
        assert quality_signals(exporter)[semconv.RUN_QUALITY_TASK_SUCCESS] == 0.9

    def test_cost_per_successful_task_is_emitted(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The cross-lane signal. Absent through M2-M4 because its denominator
        # did not exist; this is the first time it can be computed.
        config = Config(quality_lane=True, quality_sample_rate=1.0, judge=judge_scoring())
        run_agent(provider, config)

        # On the quality span, because the numerator is final at run end and the
        # denominator is not: the judge is the only thing that can say a run
        # succeeded, and it answers afterwards. So this is the one place both
        # halves are known, and the cost has to travel there on the run object.
        emitted = quality_signals(exporter)
        assert (
            emitted[semconv.RUN_COST_PER_SUCCESSFUL_TASK]
            == signals(exporter)[semconv.RUN_ACTUAL_COST]
        )

    def test_a_failed_run_reports_no_successful_tasks(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # Zero successes means cost-per-success is undefined, not infinite.
        config = Config(
            quality_lane=True,
            quality_sample_rate=1.0,
            judge=judge_scoring(task_success=0.1),
        )
        run_agent(provider, config)
        emitted = quality_signals(exporter)

        assert emitted[semconv.RUN_SUCCESS] is False
        assert semconv.RUN_COST_PER_SUCCESSFUL_TASK not in emitted


class TestWithoutAJudge:
    def test_a_normal_run_emits_no_success_signal(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        # The heuristic never claims success, so a healthy-looking run scores
        # nothing rather than True.
        run_agent(provider, Config(quality_lane=True))
        emitted = signals(exporter)

        assert semconv.RUN_SUCCESS not in emitted
        assert semconv.RUN_COST_PER_SUCCESSFUL_TASK not in emitted

    def test_a_run_producing_no_output_is_reported_as_failed(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        run_agent(provider, Config(quality_lane=True), output_tokens=0)
        emitted = signals(exporter)

        assert emitted[semconv.RUN_SUCCESS] is False


class TestSamplingGovernsSpend:
    def test_a_zero_sample_rate_never_calls_the_judge(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        called: list[str] = []

        def judge(request: JudgeRequest) -> JudgeScores:
            called.append(request.run_id)
            return JudgeScores(task_success=1.0)

        config = Config(quality_lane=True, quality_sample_rate=0.0, judge=judge)
        run_agent(provider, config)

        assert called == []
        assert quality_signals(exporter) == {}


class TestFailOpen:
    def test_a_broken_judge_does_not_break_the_run(
        self, provider: TracerProvider, exporter: InMemorySpanExporter
    ) -> None:
        def raises(_request: JudgeRequest) -> JudgeScores:
            raise RuntimeError("judge exploded")

        config = Config(quality_lane=True, quality_sample_rate=1.0, judge=raises)
        run_agent(provider, config)
        emitted = signals(exporter)

        # The run completed and the other lanes are unaffected.
        assert semconv.RUN_ACTUAL_COST in emitted
        assert semconv.RUN_LOOP_STATE in emitted
        # And no quality span at all: a judge that raised produced no score, and
        # an empty span asserting nothing is worse than no span (ADR-044).
        assert quality_signals(exporter) == {}
