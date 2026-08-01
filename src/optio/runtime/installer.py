"""Installs the span tap on the user's tracer provider (M1-4).

Kept separate from the adapters because it is the part that is *not*
framework-specific: every adapter ends up doing the same thing here, and the
idempotence rule below has to hold globally rather than per adapter.

**Install once per provider.** ``instrument()`` is easy to call twice -- two
agents in one process, a module-level call plus one in a test fixture, a
framework that re-wraps. Two taps on one provider means every span dispatched
twice, which in M2 is a doubled cost signal: silently wrong, in the direction
that makes the product's core number wrong (R-TECH-1). So installs are tracked
and repeat calls return the existing tap.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import TYPE_CHECKING, Final

from opentelemetry import trace

from optio.runtime import selfobs
from optio.runtime.run_context import (
    register_run_end_observer,
    unregister_run_end_observer,
)
from optio.runtime.span_tap import OptioSpanTap

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    # `trace.get_tracer_provider()` returns the API base class, not the SDK
    # subclass, so that is what a weak reference to `target` is a reference to.
    from opentelemetry.trace import TracerProvider as AnyTracerProvider

    from optio.config import Config

_log: Final = logging.getLogger("optio")

_lock: Final = threading.Lock()

#: Taps by the ``id()`` of the provider they are installed on, each stored
#: beside a **weak reference to that provider**. Keyed by identity because
#: tracer providers are not reliably hashable across SDK versions.
#:
#: The weak reference is what makes the key safe. ``id()`` is a memory address
#: and CPython reuses addresses the moment the object at one is collected --
#: measured at **189 of 200** short-lived ``TracerProvider`` instances landing
#: on an address a previous one had used. Without the reference to compare
#: against, a brand-new provider inherits a dead provider's entry, this function
#: returns early, and **no processor is ever added to it**: nothing is tapped,
#: nothing is priced, and the run span carries no cost (ADR-043).
#:
#: Weak rather than strong so the dict cannot keep providers alive; a stale
#: entry is detected on lookup and replaced.
_installed: Final[dict[int, tuple[weakref.ref[AnyTracerProvider], OptioSpanTap]]] = {}


def _retire_dead_taps() -> None:
    """Drop entries whose provider has been collected, and their observers.

    Swept here rather than from a ``weakref`` callback on purpose: a callback
    fires at whatever moment the garbage collector chooses, including while this
    module's non-reentrant lock is already held by the same thread, which would
    deadlock. Sweeping at install time runs the same cleanup at a point where
    the lock is known to be safe.

    Without it, ``register_run_end_observer`` grows once per provider a process
    ever creates. Each stale observer keeps firing with its own empty ledger,
    reporting ``reconciled_steps == 0`` -- a run that "cost nothing" rather than
    a run nobody priced.

    Caller must hold :data:`_lock`.
    """
    dead = [key for key, (ref, _) in _installed.items() if ref() is None]
    for key in dead:
        _, tap = _installed.pop(key)
        unregister_run_end_observer(tap.on_run_end)


def install_tap(config: Config, provider: TracerProvider | None = None) -> OptioSpanTap | None:
    """Install the span tap on a tracer provider, once.

    Args:
        config: Active configuration.
        provider: The provider to install on. Defaults to the global one.
            Injectable because OTel's global provider is write-once per process,
            so tests cannot swap it and would otherwise all share one.

    Returns:
        The installed tap, or the one already present. ``None`` when the
        provider cannot accept processors -- the case when the user has not
        configured an OTel SDK. Not an error: it means no spans are being
        recorded, so there is nothing to tap, and saying so is more useful than
        failing a setup that is otherwise valid.
    """
    target = provider if provider is not None else trace.get_tracer_provider()
    add_processor = getattr(target, "add_span_processor", None)

    if add_processor is None:
        _log.debug(
            "no OTel SDK tracer provider configured; optio will emit no signals until one is set up"
        )
        return None

    with _lock:
        _retire_dead_taps()
        key = id(target)
        existing = _installed.get(key)
        if existing is not None:
            ref, tap = existing
            if ref() is target:
                return tap
            # Same address, different provider: the one this entry described has
            # been collected and its address handed to `target`. Returning `tap`
            # here is how a provider ends up with no processor at all -- the
            # defect this check exists for. Retire the dead tap's run-end
            # observer too, or it fires forever against an empty ledger and
            # reports every later run as costing nothing.
            unregister_run_end_observer(tap.on_run_end)
            del _installed[key]

        tap = OptioSpanTap(config)
        add_processor(tap)
        # Run end is driven by RunContext, not by the OTel SDK, so the tap has
        # to be told about it separately from being added as a processor.
        register_run_end_observer(tap.on_run_end)
        # Publish the sample rate once, here rather than per run: it is
        # configuration, and a quality score read months later is only
        # interpretable if the fraction it was drawn from is recorded alongside
        # it (Section 12). Registered only when the quality lane is on, so a
        # user who never enabled it does not get a gauge reading 0.1 for a
        # judge that never runs.
        if config.quality_lane:
            selfobs.record_sampling_rate(config.quality_sample_rate)
        # Stored with a weak reference so a later provider that lands on this
        # same address is detected as different rather than assumed to be
        # the same object.
        _installed[key] = (weakref.ref(target), tap)
        return tap


def installed_tap(provider: TracerProvider | None = None) -> OptioSpanTap | None:
    """Return the tap on a provider, if any.

    Args:
        provider: The provider to check. Defaults to the global one.

    Returns:
        The active tap, or ``None`` when nothing is installed.
    """
    target = provider if provider is not None else trace.get_tracer_provider()
    entry = _installed.get(id(target))
    if entry is None:
        return None
    ref, tap = entry
    # Same address check as `install_tap`: an entry under a recycled id belongs
    # to a provider that no longer exists, and reporting its tap as this
    # provider's would be the same mistake one level up.
    return tap if ref() is target else None


def reset_installations() -> None:
    """Forget tracked installs.

    Test-support only. Does not remove processors from a provider -- the OTel
    SDK has no supported way to do that -- so tests that need a clean slate
    should build a fresh provider as well.

    Run-end observers *are* unregistered, because those do leak across tests: a
    tap left registered would keep receiving run ends after its provider was
    discarded.
    """
    with _lock:
        for _ref, tap in _installed.values():
            unregister_run_end_observer(tap.on_run_end)
        _installed.clear()
