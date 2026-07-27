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
from typing import TYPE_CHECKING, Final

from opentelemetry import trace

from optio.runtime.run_context import (
    register_run_end_observer,
    unregister_run_end_observer,
)
from optio.runtime.span_tap import OptioSpanTap

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from optio.config import Config

_log: Final = logging.getLogger("optio")

_lock: Final = threading.Lock()

#: Taps by the ``id()`` of the provider they are installed on. Keyed by identity
#: because tracer providers are not reliably hashable across SDK versions.
_installed: Final[dict[int, OptioSpanTap]] = {}


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
        key = id(target)
        existing = _installed.get(key)
        if existing is not None:
            return existing

        tap = OptioSpanTap(config)
        add_processor(tap)
        # Run end is driven by RunContext, not by the OTel SDK, so the tap has
        # to be told about it separately from being added as a processor.
        register_run_end_observer(tap.on_run_end)
        _installed[key] = tap
        return tap


def installed_tap(provider: TracerProvider | None = None) -> OptioSpanTap | None:
    """Return the tap on a provider, if any.

    Args:
        provider: The provider to check. Defaults to the global one.

    Returns:
        The active tap, or ``None`` when nothing is installed.
    """
    target = provider if provider is not None else trace.get_tracer_provider()
    return _installed.get(id(target))


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
        for tap in _installed.values():
            unregister_run_end_observer(tap.on_run_end)
        _installed.clear()
