"""Fault injection against the fail-open guard (SC-4, R-TECH-4).

This is a blocking CI gate. It exists to falsify the claim "adding optio
cannot break your agent", so the tests are written adversarially: raise the
nastiest thing available at the guard boundary and assert the caller survives.

The suite covers four escape routes a naive guard would leave open:

1. The exception type is one we did not anticipate (``BLE001`` territory).
2. The failure happens inside the guard's own error handling.
3. The exception is control flow that must *not* be absorbed.
4. The failure repeats every step and drowns the user in logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from optio.errors import (
    LedgerInvariantError,
    OptioConfigError,
    OptioInternalError,
    SignalWriteError,
    StateStoreError,
    UnsupportedFrameworkError,
)
from optio.runtime import failopen

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.failinject


@pytest.fixture(autouse=True)
def _clean_activations() -> Iterator[None]:
    """Isolate activation counters and log-once state between tests."""
    failopen.reset_activations()
    yield
    failopen.reset_activations()


# Every internal exception type, plus builtins a buggy lane could plausibly
# raise, plus pathological user-defined ones. The guarantee is about the
# agent's safety, not our exception hygiene, so builtins matter as much as ours.
class _ExplodingReprError(Exception):
    """An exception whose repr and str both explode.

    Not hypothetical: a lane holding a broken object can produce this, and a
    guard that formats the exception eagerly would raise while logging.
    """

    def __str__(self) -> str:
        """Raise instead of returning a string."""
        raise RuntimeError("str() exploded")

    def __repr__(self) -> str:
        """Raise instead of returning a repr."""
        raise RuntimeError("repr() exploded")


class _ExplodingEqError(Exception):
    """An exception that raises on comparison and hashing."""

    def __eq__(self, other: object) -> bool:
        """Raise instead of comparing."""
        raise RuntimeError("__eq__ exploded")

    def __hash__(self) -> int:
        """Raise instead of hashing."""
        raise RuntimeError("__hash__ exploded")


ABSORBED_EXCEPTIONS: list[BaseException] = [
    OptioInternalError("internal"),
    LedgerInvariantError("double reconcile"),
    StateStoreError("redis gone"),
    SignalWriteError("span closed"),
    KeyError("missing"),
    ValueError("bad value"),
    TypeError("bad type"),
    AttributeError("no attr"),
    ZeroDivisionError("division by zero"),
    RecursionError("too deep"),
    MemoryError("out of memory"),
    OSError("disk gone"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
    _ExplodingReprError(),
    _ExplodingEqError(),
]

#: Control flow that belongs to the user's process. Absorbing these would make
#: the library un-killable -- its own way of breaking the agent.
PROPAGATED_BASE_EXCEPTIONS: list[type[BaseException]] = [
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
]


@pytest.mark.parametrize("exc", ABSORBED_EXCEPTIONS, ids=lambda e: type(e).__name__)
def test_guard_absorbs_any_exception(exc: BaseException) -> None:
    """No exception type escapes ``guard``; the fallback is returned instead."""

    def boom() -> str:
        raise exc

    assert failopen.guard(boom, "fallback", component="lane") == "fallback"
    assert failopen.activation_count("lane") == 1


@pytest.mark.parametrize("exc", ABSORBED_EXCEPTIONS, ids=lambda e: type(e).__name__)
def test_guarded_decorator_absorbs_any_exception(exc: BaseException) -> None:
    """The decorator form offers the same guarantee as the callable form."""

    empty: list[str] = []

    @failopen.guarded(fallback=empty, component="lane")
    def process_span() -> list[str]:
        raise exc

    assert process_span() == []
    assert failopen.activation_count("lane") == 1


@pytest.mark.parametrize("exc_type", PROPAGATED_BASE_EXCEPTIONS)
def test_control_flow_exceptions_propagate(exc_type: type[BaseException]) -> None:
    """``BaseException`` control flow is never swallowed.

    A Ctrl-C absorbed to protect a cost calculation would leave the user unable
    to stop their own process.
    """

    def boom() -> str:
        raise exc_type

    with pytest.raises(exc_type):
        failopen.guard(boom, "fallback")

    assert failopen.activation_count() == 0


def test_config_errors_are_not_absorbed() -> None:
    """Setup errors stay loud (Section 4.2).

    Failing open at runtime is safety; failing open at setup would silently ship
    a meter that measures nothing.
    """

    def bad_setup() -> str:
        raise UnsupportedFrameworkError("no adapter for object")

    with pytest.raises(OptioConfigError):
        failopen.guard(bad_setup, "fallback")

    assert failopen.activation_count() == 0


def test_guarded_decorator_does_not_absorb_config_errors() -> None:
    """The decorator form keeps setup errors loud, exactly like ``guard``.

    Both forms must offer identical guarantees; a divergence here would mean a
    lane wired with the decorator silently swallowed a misconfiguration.
    """

    @failopen.guarded(fallback=None, component="lane")
    def bad_setup() -> None:
        raise UnsupportedFrameworkError("no adapter for object")

    with pytest.raises(OptioConfigError):
        bad_setup()

    assert failopen.activation_count() == 0


@pytest.mark.parametrize("exc_type", PROPAGATED_BASE_EXCEPTIONS)
def test_guarded_decorator_propagates_control_flow(
    exc_type: type[BaseException],
) -> None:
    """The decorator form also refuses to swallow ``BaseException``."""

    @failopen.guarded(fallback=None, component="lane")
    def boom() -> None:
        raise exc_type

    with pytest.raises(exc_type):
        boom()

    assert failopen.activation_count() == 0


def test_guarded_decorator_forwards_arguments() -> None:
    """Arguments reach the wrapped function unchanged."""

    @failopen.guarded(fallback=-1, component="lane")
    def add(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    assert add(1, 2, c=3) == 6


def test_guard_survives_a_logger_that_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken logging handler cannot turn a dropped signal into an outage.

    Users configure their own logging. A handler that raises -- a full disk, a
    dead syslog socket -- must not escape through our error path.
    """

    def exploding_warning(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("logging handler is broken")

    monkeypatch.setattr(failopen._log, "warning", exploding_warning)

    def boom() -> str:
        raise ValueError("lane bug")

    assert failopen.guard(boom, "fallback", component="lane") == "fallback"


def test_guard_survives_broken_bookkeeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even if the activation counter itself fails, the agent proceeds."""

    class ExplodingDict(dict[Any, Any]):
        def get(self, *_args: object, **_kwargs: object) -> NoReturn:
            raise RuntimeError("counter is broken")

    monkeypatch.setattr(failopen, "_activations", ExplodingDict())

    def boom() -> str:
        raise ValueError("lane bug")

    assert failopen.guard(boom, "fallback", component="lane") == "fallback"


def test_repeated_failure_logs_once_per_component(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lane failing every step must not flood the user's logs (Section 12).

    Log-once keeps our bug from becoming the user's second incident, while the
    activation counter preserves the full picture.
    """

    def boom() -> str:
        raise ValueError("fails every span")

    with caplog.at_level(logging.WARNING, logger="optio"):
        for _ in range(100):
            failopen.guard(boom, "fallback", component="cost")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert failopen.activation_count("cost") == 100


def test_each_component_logs_its_own_first_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log-once is per component, so a second broken lane is still reported."""

    def boom() -> str:
        raise ValueError("bug")

    with caplog.at_level(logging.WARNING, logger="optio"):
        failopen.guard(boom, None, component="cost")
        failopen.guard(boom, None, component="behavior")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_log_never_contains_exception_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only the exception *type* is logged, never its message (Section 7.2).

    Exception messages can carry prompt or completion content. This library
    never logs that, so the message must not appear in the WARN line even
    though the traceback is attached for debugging.
    """
    secret = "PROMPT-CONTENT-THAT-MUST-NOT-LEAK"

    def boom() -> str:
        raise ValueError(secret)

    with caplog.at_level(logging.WARNING, logger="optio"):
        failopen.guard(boom, None, component="cost")

    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert secret not in record.getMessage()
    assert "ValueError" in record.getMessage()


def test_successful_calls_pass_through_untouched() -> None:
    """The guard is transparent when nothing fails."""
    assert failopen.guard(lambda: 42, 0) == 42
    assert failopen.activation_count() == 0


def test_guard_forwards_arguments() -> None:
    """Positional and keyword arguments reach the wrapped callable."""

    def add(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    assert failopen.guard(add, 0, 1, 2, c=3) == 6


def test_component_kwarg_is_not_forwarded_to_the_callable() -> None:
    """``component`` configures the guard and must not leak into ``fn``.

    A callable that happens to accept ``**kwargs`` would otherwise silently
    receive a bookkeeping argument it never asked for.
    """

    def capture(**kwargs: object) -> dict[str, object]:
        return kwargs

    empty: dict[str, object] = {}
    assert failopen.guard(capture, empty, x=1) == {"x": 1}


def test_guard_signals_returns_empty_on_failure() -> None:
    """The signal-shaped helper drops to no signals, never a fabricated value."""

    def boom() -> list[str]:
        raise StateStoreError("store gone")

    assert failopen.guard_signals(boom, component="cost") == []
    assert failopen.activation_count("cost") == 1


def test_guard_signals_passes_signals_through() -> None:
    """A working lane's signals reach the caller unchanged."""

    def works(value: str) -> list[str]:
        return [value]

    assert failopen.guard_signals(works, "signal", component="cost") == ["signal"]
    assert failopen.activation_count("cost") == 0


def test_guarded_preserves_function_metadata() -> None:
    """``functools.wraps`` keeps the wrapped function introspectable."""

    @failopen.guarded(fallback=None, component="lane")
    def process_span(span: object) -> None:
        """Do nothing."""

    assert process_span.__name__ == "process_span"
    assert process_span.__doc__ == "Do nothing."


def test_activation_count_is_scoped_by_component() -> None:
    """Per-component counts let a runbook point at the specific broken lane."""

    def boom() -> None:
        raise ValueError("bug")

    failopen.guard(boom, None, component="cost")
    failopen.guard(boom, None, component="cost")
    failopen.guard(boom, None, component="behavior")

    assert failopen.activation_count("cost") == 2
    assert failopen.activation_count("behavior") == 1
    assert failopen.activation_count() == 3
