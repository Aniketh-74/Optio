"""The cost lane (M2-4) -- turns GenAI spans into economic signals.

Wires together the three pieces: :mod:`pricing` says what a step cost,
:mod:`ledger` holds the reserve/reconcile invariant, and :mod:`project` derives
the forward-looking numbers.

**Reserve and reconcile both happen at span end**, which is later than the
ledger's own contract would suggest, and the reason deserves stating. A span
processor only observes *finished* spans (ADR-009) -- there is no pre-step hook
to reserve from. So a step is reserved and immediately reconciled against its
real token counts. The reserve/reconcile machinery still earns its place:

* It keeps the ledger's invariant intact for callers that *can* reserve early --
  an adapter with a pre-step callback, or the projection path.
* It makes a step whose cost cannot be computed visible as a leak rather than
  silently absent, since the reservation stays open.

A step with no usable price is reserved at zero and left unreconciled, so it
surfaces as a leak at run end. Reserving a guessed cost would put a fabricated
number into the total; skipping the step entirely would make it invisible.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from optio import semconv
from optio.lanes.base import Lane, Signal
from optio.lanes.cost.ledger import CostLedger
from optio.lanes.cost.pricing import DEFAULT_PROVIDER, cost_of
from optio.lanes.cost.project import (
    budget_remaining,
    cost_per_successful_task,
    project_cost,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from optio.config import Config
    from optio.lanes.base import RunLike
    from optio.lanes.cost.ledger import LedgerSnapshot
    from optio.lanes.cost.pricing import PricingProvider

_log: Final = logging.getLogger("optio")


class CostLane(Lane):
    """Computes cost signals from GenAI spans.

    Attributes:
        ledger: The reserve/reconcile ledger backing this lane.
        pricing: The pricing provider used to value steps.
    """

    name = "cost"

    def __init__(
        self,
        config: Config,
        ledger: CostLedger | None = None,
        pricing: PricingProvider | None = None,
    ) -> None:
        """Build the lane.

        Args:
            config: Active configuration.
            ledger: Ledger to use. A fresh one is created when omitted.
            pricing: Pricing source. Defaults to the built-in table.
        """
        super().__init__(config)
        self.ledger = ledger if ledger is not None else CostLedger()
        self.pricing = pricing if pricing is not None else DEFAULT_PROVIDER

    def process_span(self, span: ReadableSpan, run: RunLike) -> list[Signal]:
        """Price one step and emit the cost signals available so far.

        Args:
            span: The finished GenAI span.
            run: The run the span belongs to.

        Returns:
            Signals to write. Empty when the span is not a priceable LLM call.
        """
        step_id = _step_id(span)
        cost = self._price(span)

        if cost is None:
            # Reserved at zero and deliberately left open: the step becomes a
            # leak at run end rather than vanishing. An unpriceable step is a
            # gap in the evidence, and the run's cost should say so.
            self.ledger.reserve(run.run_id, step_id, 0.0)
            return self._forward_signals(run)

        self.ledger.reserve(run.run_id, step_id, cost)
        self.ledger.reconcile(run.run_id, step_id, cost)

        snapshot = self.ledger.snapshot(run.run_id)
        signals = [Signal(semconv.RUN_ACTUAL_COST, snapshot.actual)]
        signals.extend(self._forward_signals(run, snapshot))
        return signals

    def on_run_end(self, run: RunLike) -> list[Signal]:
        """Close the run, emit its final cost signals, and release its state.

        Args:
            run: The run that just ended.

        Returns:
            Final signals for the run.
        """
        # Run end can fire more than once (M1-2), and the state is released on
        # the first call. A second call therefore sees an all-zero snapshot,
        # from which `budget_remaining` would compute the *full* budget and
        # overwrite the correct value on the run span -- telling a policy the
        # run spent nothing. Emitting nothing is the only honest answer: the
        # evidence is gone, and re-deriving signals from its absence invents
        # them.
        # Run end is broadcast to every registered observer, so a lane belonging
        # to a *different* tracer provider is asked about this run too. Its
        # ledger has never heard of it, and an untouched ledger's zeros read as
        # "nothing attempted yet" -- the one state where a full budget really is
        # available. It would then emit `budget_remaining = <the whole limit>`
        # for a run that has been spending money throughout, which is the exact
        # value that guarantees `deny if budget_remaining < 0.50` never fires
        # (ADR-044).
        if not self.ledger.knows(run.run_id):
            return []
        if self.ledger.is_finalised(run.run_id):
            return []

        snapshot = self.ledger.close_run(run.run_id)
        # Read the snapshot first, then release. Agents are long-lived
        # processes: without this the ledger retains every run it has ever seen
        # (~368 bytes each), which is an unbounded leak rather than a slow one.
        self.ledger.evict(run.run_id)

        signals: list[Signal] = []

        # A run where nothing could be priced reports *no* cost, not zero.
        # Emitting 0.0 would tell a budget policy the run was free, when the
        # truth is that we do not know what it cost -- the exact confusion
        # `docs/signals.md` forbids.
        if snapshot.reconciled_steps > 0:
            signals.append(Signal(semconv.RUN_ACTUAL_COST, snapshot.actual))
            _record_actual_cost(run, snapshot.actual)

        remaining = budget_remaining(snapshot, run.budget)
        if remaining is not None:
            signals.append(Signal(semconv.RUN_BUDGET_REMAINING, remaining))

        # cost_per_successful_task needs a success count, which the quality lane
        # owns (M5). Until then the denominator is unknown, so the signal is
        # omitted rather than assumed -- treating an unscored run as one success
        # would publish a headline number derived from a guess. It also needs a
        # real numerator: dividing an unpriced run's zero would report perfect
        # efficiency for a run we could not price at all.
        successes = _success_count(run)
        if successes is not None and snapshot.reconciled_steps > 0:
            per_task = cost_per_successful_task(snapshot, successes)
            if per_task is not None:
                signals.append(Signal(semconv.RUN_COST_PER_SUCCESSFUL_TASK, per_task))

        return signals

    def _price(self, span: ReadableSpan) -> float | None:
        """Return the cost of one span, or ``None`` when it cannot be priced.

        Args:
            span: The span to price.

        Returns:
            Cost in USD, or ``None``.
        """
        attributes = span.attributes or {}

        model = attributes.get(semconv.GEN_AI_RESPONSE_MODEL) or attributes.get(
            semconv.GEN_AI_REQUEST_MODEL
        )
        if not isinstance(model, str):
            return None

        input_tokens = _int_attribute(attributes.get(semconv.GEN_AI_USAGE_INPUT_TOKENS))
        output_tokens = _int_attribute(attributes.get(semconv.GEN_AI_USAGE_OUTPUT_TOKENS))
        if input_tokens is None and output_tokens is None:
            # A tool-call span with no token usage. Not an error: not every
            # GenAI span is a billable model call.
            return None

        return cost_of(model, input_tokens or 0, output_tokens or 0, self.pricing)

    def _forward_signals(
        self, run: RunLike, snapshot: LedgerSnapshot | None = None
    ) -> list[Signal]:
        """Return the forward-looking signals for a run.

        Args:
            run: The run being metered.
            snapshot: Ledger state to derive from. Taken here when omitted.
                Callers that already hold one should pass it: a snapshot is a
                locked scan of the run's open reservations, and taking a second
                one costs that again for no new information.

                It also keeps the signals *consistent*. Two snapshots taken
                separately can straddle a concurrent step, so `actual_cost`
                would describe one ledger state and `budget_remaining` another
                -- two numbers that were never true at the same instant, which
                is worse for a policy than either alone.

        Returns:
            Projection and budget signals, omitting any that cannot be computed.
        """
        if snapshot is None:
            snapshot = self.ledger.snapshot(run.run_id)
        signals: list[Signal] = []

        projected = project_cost(snapshot, run.budget)
        if projected is not None:
            signals.append(Signal(semconv.RUN_PROJECTED_COST, projected))

        remaining = budget_remaining(snapshot, run.budget)
        if remaining is not None:
            signals.append(Signal(semconv.RUN_BUDGET_REMAINING, remaining))

        return signals


def _step_id(span: ReadableSpan) -> str:
    """Return a stable identifier for the step a span represents.

    Uses the span id, which is unique per span and stable for its lifetime. A
    framework retry produces a *new* span, so retries are separate ledger
    entries rather than replacements -- correct here, because each attempt
    burned its own tokens.

    Args:
        span: The span to identify.

    Returns:
        A step identifier.
    """
    context = span.get_span_context()
    if context is not None and context.span_id:
        return format(context.span_id, "016x")
    return f"anonymous-{id(span):x}"


def _int_attribute(value: object) -> int | None:
    """Coerce a span attribute to a non-negative int, or ``None``.

    Args:
        value: The raw attribute value.

    Returns:
        The integer value, or ``None`` when absent or unusable.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(value)


def _success_count(run: RunLike) -> int | None:
    """Return the run's successful-task count, if anything has recorded one.

    The quality lane (M5) owns this. Until it lands there is no success signal,
    so the denominator is unknown.

    Args:
        run: The run being metered.

    Returns:
        Number of successes, or ``None`` when unknown.
    """
    successes = getattr(run, "successes", None)
    if isinstance(successes, bool) or not isinstance(successes, int):
        return None
    return successes


def _record_actual_cost(run: RunLike, cost: float) -> None:
    """Publish the run's final cost on the run object.

    The mirror image of ``successes``, which the quality lane writes and this
    lane reads. Both lanes need one number the other owns, and neither may
    import the other (Section 3.1) -- so the exchange happens through the run
    object they both already hold.

    The quality lane needs it because its judge answers after the run span has
    closed, which makes the deferred quality span the only place where a final
    cost and a judged outcome are both known. Written only when the run could
    actually be priced, so an unpriced run leaves the attribute absent rather
    than reading as free (ADR-044).

    Args:
        run: The run being metered.
        cost: Final reconciled cost in USD.
    """
    try:
        setattr(run, "actual_cost", cost)  # noqa: B010
    except AttributeError:
        # A minimal run stub, or an older RunContext without the slot. The
        # deferred signal is then omitted rather than wrong.
        _log.debug("run object does not accept a cost; skipping")
