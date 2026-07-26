"""The single write path for every signal agentmeter emits (M1-5).

Centralising emission here buys three things that matter more than the small
amount of indirection:

1. **Names come from** :mod:`agentmeter.semconv` **only.** Downstream OPA, Cedar
   and AGT policies match on these exact strings, so a typo is a broken
   deployment in someone else's production. One write path means one place where
   names can be validated (Section 16 rule 5).
2. **Absence is preserved.** A value that cannot be computed is *omitted*, never
   coerced to zero. A policy reading ``budget_remaining == 0`` must be able to
   trust that the budget is actually exhausted rather than that a lane failed
   (ADR-004, ``docs/signals.md``).
3. **Nothing raises.** Writing runs on the agent's critical path, so every entry
   point is guarded. A closed or non-recording span is normal, not exceptional.

The writer never touches prompt or completion content. It handles numbers,
booleans and enum strings -- nothing else reaches it (Section 7.2, Section 10).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Final

from agentmeter import semconv
from agentmeter.errors import SignalWriteError
from agentmeter.runtime.failopen import guarded

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opentelemetry.trace import Span

    from agentmeter.lanes.base import Signal

_log: Final = logging.getLogger("agentmeter")

#: Values that are structurally unrepresentable as OTel attributes. NaN and the
#: infinities are the realistic case: a cost divided by zero successful tasks
#: produces one, and exporting it would poison a downstream average silently.
#: Omitting is the honest answer -- the value is unknown, not infinite.
_NON_FINITE = "non-finite"


def _is_emittable(name: str, value: object) -> bool:
    """Decide whether a signal is safe and meaningful to write.

    Args:
        name: The attribute name, expected to be a ``semconv`` constant.
        value: The candidate value.

    Returns:
        ``True`` when the signal should be written.
    """
    if value is None:
        # Absence is a valid signal state; write nothing rather than a null.
        return False

    if name not in semconv.EMITTED_SIGNALS:
        # A name outside the frozen contract means a lane bypassed semconv.
        # Refusing to write it keeps an unreviewed name from reaching consumers
        # and hardening into a contract we never agreed to.
        raise SignalWriteError(f"{name!r} is not a declared signal")

    if isinstance(value, bool):
        # Checked before the numeric branch: bool is a subclass of int, and
        # `math.isfinite(True)` is a meaningless question.
        return True

    if isinstance(value, (int, float)) and not math.isfinite(value):
        _log.debug("dropping %s: %s value", name, _NON_FINITE)
        return False

    if name == semconv.RUN_LOOP_STATE and value not in semconv.LOOP_STATES:
        # An undeclared enum value would be matched by nobody's policy and would
        # look like a healthy run to a consumer checking `!= "looping"`.
        raise SignalWriteError(f"{value!r} is not a valid loop_state")

    return True


@guarded(fallback=False, component="signal_writer")
def write_signal(span: Span | None, signal: Signal) -> bool:
    """Write one signal onto a span.

    Args:
        span: Target span. ``None`` or a non-recording span is a no-op -- outside
            a trace there is nowhere to put the value, which is not an error.
        signal: The signal to write.

    Returns:
        ``True`` if the attribute was written, ``False`` if it was skipped or the
        write failed. Never raises.
    """
    if span is None or not span.is_recording():
        return False

    if not _is_emittable(signal.name, signal.value):
        return False

    span.set_attribute(signal.name, signal.value)
    return True


@guarded(fallback=0, component="signal_writer")
def write_signals(span: Span | None, signals: Iterable[Signal]) -> int:
    """Write a batch of signals, skipping any that cannot be written.

    One bad signal does not discard its neighbours: a lane emitting three
    correct values and one non-finite one should land three.

    Args:
        span: Target span.
        signals: Signals to write.

    Returns:
        How many signals were actually written.
    """
    return sum(1 for signal in signals if write_signal(span, signal))
