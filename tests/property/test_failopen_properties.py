"""Property tests for the fail-open guard (M1-1 acceptance criterion).

The fault-injection suite enumerates exception types we thought of. These tests
assert the guarantee over exception types we did *not* -- which is the failure
mode that matters, since the guard's job is to survive tomorrow's lane bug.

The stated property from Section 5.1: *for any raised exception type, guard
returns fallback and agent proceeds*.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from agentmeter.runtime import failopen

# Arbitrary exception *types*, including ones constructed on the fly, to avoid
# the guard accidentally depending on a known class hierarchy.
exception_types = st.sampled_from(
    [
        Exception,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ArithmeticError,
        LookupError,
        BufferError,
        StopIteration,
        StopAsyncIteration,
        NotImplementedError,
    ]
)

fallbacks = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
)


@given(exc_type=exception_types, fallback=fallbacks, message=st.text())
def test_any_exception_yields_the_fallback(
    exc_type: type[Exception], fallback: object, message: str
) -> None:
    """For any exception type and any fallback, the fallback comes back."""
    failopen.reset_activations()

    def boom() -> object:
        raise exc_type(message)

    assert failopen.guard(boom, fallback, component="prop") == fallback


@given(
    exc_type=exception_types,
    # Valid identifiers only: ``type()`` rejects null bytes in a class name, so
    # anything else fails in test setup before the guard is ever reached.
    name=st.from_regex(r"\A[A-Za-z_][A-Za-z0-9_]{0,19}\Z"),
)
def test_dynamically_created_exception_types_are_absorbed(
    exc_type: type[Exception], name: str
) -> None:
    """Exception classes that did not exist when the guard was written."""
    failopen.reset_activations()
    dynamic = type(name, (exc_type,), {})

    def boom() -> str:
        raise dynamic("synthesised")

    assert failopen.guard(boom, "fallback", component="prop") == "fallback"


@given(failures=st.integers(min_value=0, max_value=50))
def test_activation_count_equals_failure_count(failures: int) -> None:
    """Every absorbed failure is counted exactly once.

    The count is the only evidence a silently-broken lane leaves behind
    (ADR-004), so under- or over-counting would make the runbook misleading.
    """
    failopen.reset_activations()

    def boom() -> None:
        raise ValueError("bug")

    for _ in range(failures):
        failopen.guard(boom, None, component="prop")

    assert failopen.activation_count("prop") == failures


@given(value=st.integers())
def test_successful_calls_are_never_counted(value: int) -> None:
    """A working lane leaves the activation counter at zero."""
    failopen.reset_activations()

    assert failopen.guard(lambda: value, 0, component="prop") == value
    assert failopen.activation_count("prop") == 0
