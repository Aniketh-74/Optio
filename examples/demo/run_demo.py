"""The optio demo (M4-5, ADR-006).

Runs the misbehaving agent twice and shows what changes when a policy engine
can see the signals:

* **Ungoverned** -- the agent loops until its step ceiling. Nothing stops it.
* **Governed** -- a policy reads the signals optio emits and stops the run
  the moment the agent is provably stuck.

The saving between those two numbers is the whole product argument, so the demo
computes it from real signals rather than asserting it in prose.

**optio does not stop anything.** It emits signals; the policy in
``policy.py`` decides (ADR-001). That separation is the point of the product and
the demo is built to make it visible: the same signals, evaluated by rules you
can edit, are what end the run.

Runs offline with no API keys -- see ``agent.py`` for why that is a design
constraint rather than a shortcut.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from optio import RunContext, semconv
from optio.config import Config, default_config
from optio.runtime.installer import install_tap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import MODEL, ScriptedModel, run_step

from policy import Decision, evaluate

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import Span

#: Per-run budget. Deliberately generous: the point is that the loop is caught
#: by the *behavior* signal well before the budget runs out, which is the case
#: a cost-only tool misses.
BUDGET: Final = "$2.00"

#: Ceiling for the ungoverned run. A real stuck agent runs until something else
#: kills it; this stands in for that something else.
MAX_STEPS: Final = 60

#: Behavior window for the demo, overriding the default 50.
#:
#: Not cosmetic, and worth understanding before changing. `looping` requires the
#: window to contain almost nothing but the repeated call, so the agent's four
#: productive opening steps must age out before the verdict can fire. With the
#: default 50-step window that takes 53 steps -- by which point this agent has
#: spent $1.72, and a *cost* rule would have stopped it first. The demo would
#: then be showing off the wrong signal: cost gating is the thing several other
#: tools already do.
#:
#: A 20-step window catches the loop at step 23 for $0.36, which is the case
#: that actually distinguishes optio -- behavioral evidence firing while
#: the run still looks affordable. Shorter windows detect faster and tolerate
#: less legitimate repetition; 20 is a reasonable production value, not a number
#: invented to make the demo work.
WINDOW_SIZE: Final = 20

_RESET: Final = "\033[0m"
_BOLD: Final = "\033[1m"
_DIM: Final = "\033[2m"
_RED: Final = "\033[31m"
_GREEN: Final = "\033[32m"
_YELLOW: Final = "\033[33m"


def _supports_colour() -> bool:
    """Whether to emit ANSI colour.

    Returns:
        ``False`` when piped, when ``NO_COLOR`` is set, or on a dumb terminal.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


_COLOUR: Final = _supports_colour()


def _c(text: str, colour: str) -> str:
    """Wrap text in a colour when the terminal supports it."""
    return f"{colour}{text}{_RESET}" if _COLOUR else text


def _config() -> Config:
    """Return the demo's configuration.

    Returns:
        Default config with the shortened behavior window (see WINDOW_SIZE).
    """
    return default_config().merged_with(behavior_window_size=WINDOW_SIZE)


def _add_otlp_export(provider: TracerProvider) -> bool:
    """Export to a collector when one is configured.

    Under ``docker compose`` this is what carries the signals into a real OTel
    pipeline, which is the point of the compose stack: the attributes travel the
    same path they would in production instead of being printed by the process
    that computed them.

    Absent (running the script directly), the demo still works -- it just has no
    collector to talk to. That degradation is deliberate: `python run_demo.py`
    with no infrastructure is the fastest way to see the result, and requiring
    Docker for it would undercut SC-1.

    Args:
        provider: The tracer provider to add the exporter to.

    Returns:
        Whether an OTLP exporter was attached.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        # Endpoint set but the exporter package is missing. Worth saying out
        # loud -- the user asked for export and is not getting it -- but not
        # worth failing the demo over, since the summary is still correct.
        print(
            _c(
                "  note: OTEL_EXPORTER_OTLP_ENDPOINT is set but "
                "opentelemetry-exporter-otlp-proto-grpc is not installed; "
                "running without export",
                _YELLOW,
            )
        )
        return False

    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return True


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one run of the agent produced.

    Attributes:
        steps: Steps actually executed.
        cost: Reconciled spend in USD.
        loop_state: Final behavior verdict.
        repeat_count: Highest repeated-signature count.
        stopped_by: The policy rule that ended the run, if any.
    """

    steps: int
    cost: float
    loop_state: str
    repeat_count: int
    stopped_by: str | None = None


def _signals(spans: list[ReadableSpan]) -> dict[str, object]:
    """Collect the signals optio wrote onto the run span.

    Args:
        spans: Finished spans from the exporter.

    Returns:
        The run span's optio attributes, or ``{}`` if no run span exists.
    """
    for span in spans:
        if span.name.startswith("optio.demo.run") and span.attributes:
            return {
                key: value
                for key, value in span.attributes.items()
                if key in semconv.EMITTED_SIGNALS
            }
    return {}


def _live_signals(span: Span) -> dict[str, object]:
    """Read optio's signals off a span that is still open.

    This is what a policy integration does mid-run: the lanes write onto the
    active run span after each step (ADR-009), and the orchestrator reads them
    back to decide whether to continue. In a production setup the same
    attributes reach OPA or Cedar via the collector; here they are read in
    process so the demo needs no engine installed.

    Args:
        span: The open run span.

    Returns:
        The optio signals currently on the span.
    """
    attributes = getattr(span, "attributes", None) or {}
    return {k: v for k, v in attributes.items() if k in semconv.EMITTED_SIGNALS}


def run_ungoverned(provider: TracerProvider, exporter: InMemorySpanExporter) -> Outcome:
    """Run the agent with no policy watching. It loops to the ceiling.

    Args:
        provider: Tracer provider carrying the span tap.
        exporter: Exporter to read signals back from.

    Returns:
        What the run cost and how it behaved.
    """
    exporter.clear()
    tracer = provider.get_tracer("optio.demo")
    model = ScriptedModel(get_stuck=True)

    with (
        tracer.start_as_current_span("optio.demo.run.ungoverned"),
        RunContext(budget=BUDGET, config=_config()),
    ):
        steps = 0
        for index, step in enumerate(model.plan(MAX_STEPS)):
            run_step(step, index)
            steps += 1

    signals = _signals(exporter.get_finished_spans())
    return Outcome(
        steps=steps,
        cost=float(signals.get(semconv.RUN_ACTUAL_COST, 0.0)),  # type: ignore[arg-type]
        loop_state=str(signals.get(semconv.RUN_LOOP_STATE, "unknown")),
        repeat_count=int(signals.get(semconv.RUN_REPEAT_COUNT, 0)),  # type: ignore[arg-type]
    )


def run_governed(provider: TracerProvider, exporter: InMemorySpanExporter) -> Outcome:
    """Run the agent with a policy reading the signals after every step.

    This is the loop a real integration has: the agent takes a step, optio
    emits signals onto the span, the policy engine evaluates them, and the
    orchestrator acts on the decision. optio is not in the deciding.

    Args:
        provider: Tracer provider carrying the span tap.
        exporter: Exporter to read signals back from.

    Returns:
        What the run cost and which rule stopped it.
    """
    exporter.clear()
    tracer = provider.get_tracer("optio.demo")
    model = ScriptedModel(get_stuck=True)

    steps = 0
    decision = Decision.allow()

    with (
        tracer.start_as_current_span("optio.demo.run.governed") as run_span,
        RunContext(budget=BUDGET, config=_config()),
    ):
        for index, step in enumerate(model.plan(MAX_STEPS)):
            run_step(step, index)
            steps += 1

            # Read the signals off the *run* span, not the step span that just
            # closed. Step spans are input only: by the time a span processor
            # sees one it has already ended, and the SDK silently discards
            # writes to it (ADR-009). The lanes therefore write to the enclosing
            # run span, which is still open -- and being run-scoped is also
            # where these signals belong, since `actual_cost` is a property of
            # a run rather than of one LLM call.
            decision = evaluate(_live_signals(run_span))
            if decision.denied:
                break

    signals = _signals(exporter.get_finished_spans())
    return Outcome(
        steps=steps,
        cost=float(signals.get(semconv.RUN_ACTUAL_COST, 0.0)),  # type: ignore[arg-type]
        loop_state=str(signals.get(semconv.RUN_LOOP_STATE, "unknown")),
        repeat_count=int(signals.get(semconv.RUN_REPEAT_COUNT, 0)),  # type: ignore[arg-type]
        stopped_by=decision.reason if decision.denied else None,
    )


def _print_header() -> None:
    print()
    print(_c("  optio demo", _BOLD))
    print(_c("  ---------------", _DIM))
    print()
    print("  An agent gets stuck in a retrieval loop and keeps paying for it.")
    print("  optio emits the signals; the policy in policy.py decides.")
    print()
    print(_c(f"  model {MODEL}   budget {BUDGET}   step ceiling {MAX_STEPS}", _DIM))
    print(_c("  no API keys, no network -- the model is scripted", _DIM))
    print()


def _print_outcome(title: str, outcome: Outcome, colour: str) -> None:
    print(_c(f"  {title}", _BOLD))
    print(f"    steps        {outcome.steps}")
    print(f"    cost         ${outcome.cost:.4f}")
    state = outcome.loop_state
    state_text = _c(state, _RED if state in {"looping", "retry_storm"} else _GREEN)
    print(f"    loop_state   {state_text}   (repeat_count {outcome.repeat_count})")
    if outcome.stopped_by:
        print(f"    stopped by   {_c(outcome.stopped_by, colour)}")
    print()


def main() -> int:
    """Run both scenarios and print the comparison.

    Returns:
        Process exit code. Non-zero if the demo did not demonstrate what it
        claims -- so the CI smoke test is the demo itself, not a copy of it.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # The in-memory exporter is how the demo reads its own results back. It is
    # not how a user consumes signals -- that is the OTLP export below.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    exporting = _add_otlp_export(provider)
    trace.set_tracer_provider(provider)
    install_tap(_config(), provider)

    _print_header()

    ungoverned = run_ungoverned(provider, exporter)
    _print_outcome("without optio signals", ungoverned, _RED)

    governed = run_governed(provider, exporter)
    _print_outcome("with optio signals + policy", governed, _YELLOW)

    saved = ungoverned.cost - governed.cost
    percent = (saved / ungoverned.cost * 100) if ungoverned.cost else 0.0

    print(_c("  result", _BOLD))
    print(f"    caught the loop after {governed.steps} steps instead of {ungoverned.steps}")
    print(
        f"    saved {_c(f'${saved:.4f}', _GREEN)} of ${ungoverned.cost:.4f} "
        f"({percent:.0f}% of this run)"
    )
    print()
    print(_c("    One run. Multiply by every stuck agent in a fleet.", _DIM))
    print()

    if exporting:
        # Batched spans are still in memory at this point. Without the flush the
        # process can exit before they are sent and the collector shows nothing
        # -- the signals would look broken when only the shutdown was.
        provider.force_flush()
        print(_c("  signals exported to the collector (see its log above)", _DIM))
        print()

    return 0 if _verify(ungoverned, governed) else 1


def _verify(ungoverned: Outcome, governed: Outcome) -> bool:
    """Check the demo actually demonstrated its claim.

    A demo that silently stops working is worse than no demo, and this runs in
    CI as the M4-5 smoke test. Each condition is the literal claim made above.

    Args:
        ungoverned: The unwatched run.
        governed: The policy-watched run.

    Returns:
        ``True`` when every claim held.
    """
    problems: list[str] = []

    if ungoverned.loop_state != semconv.LOOP_STATE_LOOPING:
        problems.append(f"ungoverned run should end `looping`, got {ungoverned.loop_state!r}")
    if ungoverned.cost <= 0:
        problems.append("ungoverned run recorded no cost; is the pricing table current?")
    if governed.stopped_by is None:
        problems.append("policy did not stop the governed run")
    if governed.steps >= ungoverned.steps:
        problems.append(
            f"governed run took {governed.steps} steps, no better than {ungoverned.steps}"
        )
    if governed.cost >= ungoverned.cost:
        problems.append("governed run cost no less than the ungoverned one")

    for problem in problems:
        print(_c(f"  FAIL: {problem}", _RED), file=sys.stderr)
    return not problems


if __name__ == "__main__":
    raise SystemExit(main())
