"""A recycled memory address must not be mistaken for an installed tap (ADR-043).

``install_tap`` remembers which providers it has already tapped in a dict keyed
by ``id(provider)``. ``id()`` is a memory address, and CPython reuses addresses
as soon as the object at one is collected. Measured here: **189 of 200**
short-lived ``TracerProvider`` instances landed on an address a previous one had
already used.

When that happens the lookup finds a stale entry, returns the old tap, and
**never adds a span processor to the new provider**. Nothing is tapped, so
nothing is priced, so the run span carries no ``gen_ai.run.actual_cost`` -- and
the failure surfaces as a ``KeyError`` in a test about caching, three layers
away from the cause.

It presented as a flake because address reuse depends on allocation history:
adding a ``print`` to the failing test changed the allocations and made it pass,
which is what made it look like a race rather than a lookup returning the wrong
object.

The second half is the leak. Each install also does
``register_run_end_observer(tap.on_run_end)`` and nothing ever unregisters, so
every provider a process creates adds another observer that fires forever --
each with its own empty ledger, reporting a run cost of nothing.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider

from optio.config import Config
from optio.runtime import run_context
from optio.runtime.installer import install_tap


def _tapped(provider: TracerProvider) -> int:
    """How many optio taps are registered as processors on ``provider``."""
    from optio.runtime.span_tap import OptioSpanTap

    multi = provider._active_span_processor
    return sum(1 for p in multi._span_processors if isinstance(p, OptioSpanTap))


class TestEveryProviderGetsItsOwnTap:
    def test_a_fresh_provider_is_tapped_even_after_many_before_it(self) -> None:
        """The bug, stated directly.

        Churning providers is what a test suite does and what any process that
        reconfigures tracing does. Without an address-reuse check this fails
        within a handful of iterations -- 189 of 200 addresses were recycled on
        this machine.
        """
        config = Config()
        for _ in range(50):
            provider = TracerProvider()
            install_tap(config, provider)
            assert _tapped(provider) == 1, "a new provider was left without a tap"
            provider.shutdown()

    def test_the_same_provider_is_tapped_only_once(self) -> None:
        """The property the id() cache exists for, which must survive the fix.

        Two taps on one provider means every span dispatched twice and every
        cost counted twice.
        """
        config = Config()
        provider = TracerProvider()

        first = install_tap(config, provider)
        second = install_tap(config, provider)

        assert first is second
        assert _tapped(provider) == 1
        provider.shutdown()


class TestTheIdentityCheckItself:
    """The sweep in :func:`_retire_dead_taps` clears dead entries before the
    lookup ever sees them, so it masks the identity check in ordinary use --
    removing the check breaks no test that goes through the front door.

    These reach in and forge the state instead. A recycled address whose old
    provider is *still alive somewhere else* is not reachable by churning
    providers, but it is exactly what the check is written for, and it is one
    weakref away from what garbage collection produces naturally.
    """

    def test_an_entry_belonging_to_another_provider_is_replaced(self) -> None:
        import weakref

        from optio.runtime import installer
        from optio.runtime.span_tap import OptioSpanTap

        config = Config()
        impostor = TracerProvider()
        target = TracerProvider()
        stale_tap = OptioSpanTap(config)

        # An entry filed under `target`'s address that describes a different,
        # still-living provider -- what a recycled id() looks like before the
        # weakref has been cleared.
        installer._installed[id(target)] = (weakref.ref(impostor), stale_tap)
        # Registered exactly as a real install would, so retiring it is
        # observable. Without this, the "was it unregistered?" assertion below
        # passes for a tap that was never registered in the first place -- which
        # is what it did until a mutation pointed it out.
        run_context.register_run_end_observer(stale_tap.on_run_end)
        try:
            tap = install_tap(config, target)

            assert tap is not stale_tap, "a foreign entry was returned as this provider's tap"
            assert _tapped(target) == 1, "the provider was left without a processor"
            # Displacing the entry is not enough: the dead tap's observer keeps
            # firing on every later run with an empty ledger, reporting a run
            # that "cost nothing" rather than one nobody priced.
            assert stale_tap.on_run_end not in run_context._run_end_observers
        finally:
            installer._installed.pop(id(target), None)
            run_context.unregister_run_end_observer(stale_tap.on_run_end)
            target.shutdown()
            impostor.shutdown()

    def test_installed_tap_does_not_report_a_foreign_entry(self) -> None:
        """The same mistake one level up: reporting another provider's tap as
        this one's would make ``instrument()`` look installed when it is not."""
        import weakref

        from optio.runtime import installer
        from optio.runtime.span_tap import OptioSpanTap

        config = Config()
        impostor = TracerProvider()
        target = TracerProvider()
        stale_tap = OptioSpanTap(config)

        installer._installed[id(target)] = (weakref.ref(impostor), stale_tap)
        try:
            assert installer.installed_tap(target) is None
        finally:
            installer._installed.pop(id(target), None)
            target.shutdown()
            impostor.shutdown()


class TestTheSweepItself:
    """The identity check handles a *recycled* address; the sweep handles the
    rest -- a provider that dies and whose address is never reused again.

    Churning providers in a loop cannot tell the two apart, because CPython
    recycles well over 90% of those addresses and the identity check quietly
    covers them. Holding the providers alive first forces distinct addresses,
    which is the only shape that isolates the sweep.
    """

    def test_taps_for_collected_providers_are_retired(self) -> None:
        config = Config()
        before = len(run_context._run_end_observers)

        # Alive simultaneously, so every one gets its own address and no
        # identity check can fire for them.
        providers = [TracerProvider() for _ in range(40)]
        for provider in providers:
            install_tap(config, provider)
        assert len(run_context._run_end_observers) - before == 40

        for provider in providers:
            provider.shutdown()
        providers.clear()
        # The loop variable still references the last one, and a live provider's
        # tap is not a leak. Dropping it is the difference between measuring the
        # sweep and measuring test scaffolding.
        del provider

        # Any install sweeps; this one is only a vehicle for it.
        trigger = TracerProvider()
        install_tap(config, trigger)
        after = len(run_context._run_end_observers)
        trigger.shutdown()

        assert after - before <= 1, f"{after - before - 1} dead taps were never retired"


class TestObserversDoNotAccumulate:
    def test_churning_providers_does_not_grow_the_observer_list(self) -> None:
        """Unbounded growth in a module-level list is a leak in a long-lived
        agent process, and each stale observer reports a run it never saw --
        reconciled_steps of zero, which reads as "this run cost nothing"."""
        config = Config()
        before = len(run_context._run_end_observers)

        for _ in range(200):
            provider = TracerProvider()
            install_tap(config, provider)
            provider.shutdown()

        # Drop the loop's last provider before measuring: it is still referenced
        # by the loop variable, so its entry is legitimately live and counting it
        # would measure test scaffolding rather than the leak.
        del provider

        # One more install so the sweep runs after that provider is gone.
        final = TracerProvider()
        install_tap(config, final)
        after = len(run_context._run_end_observers)
        final.shutdown()

        # Exactly one: `final`'s own, still live. The number that matters is
        # that it does not scale with the 200 -- a leak here is proportional to
        # every provider a process ever creates, and each stale observer reports
        # a run it never saw as costing nothing.
        assert after - before <= 1, f"observers grew by {after - before} across 200 providers"
