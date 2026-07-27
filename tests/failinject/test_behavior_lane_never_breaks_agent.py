"""SC-4 for the behavior lane: no internal failure reaches the agent.

The generic proof lives in `test_broken_lane_never_breaks_agent.py`. This file
covers the failure modes specific to *this* lane -- the ones that involve data
the agent controls rather than a lane deliberately raising.

That distinction matters. A tool argument is attacker-adjacent input in an agent
system: it can be a hostile object with an exploding `__repr__`, an unbounded
string, or a structure deep enough to blow the stack. The window hashes all of
it on the hot path, inside the agent's call stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from optio import semconv
from optio.config import default_config
from optio.lanes.base import Signal
from optio.lanes.behavior.lane import BehaviorLane
from optio.runtime import failopen
from optio.runtime.failopen import guard_signals

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

pytestmark = pytest.mark.failinject


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    failopen.reset_activations()
    yield
    failopen.reset_activations()


class Run:
    run_id = "run-1"
    budget = None


class TestHostileSpanData:
    """Malformed spans must degrade to no signal, never to an exception."""

    def _process(self, span: object) -> list[Signal]:
        # The span is deliberately malformed, which is the whole point; cast
        # rather than blanket-ignore so the *shape* of the lie stays visible.
        lane = BehaviorLane(default_config())
        return guard_signals(
            lane.process_span, cast("ReadableSpan", span), Run(), component="behavior"
        )

    def test_a_span_with_no_attributes(self) -> None:
        # No attributes is not an error: the step signs as "-" and is still
        # counted, so a run of unidentifiable steps can still be seen looping.
        span = Mock(name="s", attributes=None, status=None)

        assert self._process(span) != []
        assert failopen.activation_count("behavior") == 0

    def test_an_attribute_whose_repr_explodes(self) -> None:
        class ExplodingReprError:
            def __repr__(self) -> str:
                raise RuntimeError("boom")

        span = Mock()
        span.name = "s"
        span.attributes = {semconv.GEN_AI_TOOL_NAME: "t", "gen_ai.x": ExplodingReprError()}
        span.status = None

        assert self._process(span) != []

    def test_an_attribute_dict_that_raises_on_iteration(self) -> None:
        class HostileMappingError(dict):  # type: ignore[type-arg]
            def __iter__(self) -> Iterator[str]:
                raise RuntimeError("boom")

        span = Mock()
        span.name = "s"
        span.attributes = HostileMappingError({semconv.GEN_AI_TOOL_NAME: "t"})
        span.status = None

        # Signals are dropped; the agent is untouched.
        assert self._process(span) == []
        assert failopen.activation_count("behavior") == 1

    def test_a_gigantic_attribute_value(self) -> None:
        span = Mock()
        span.name = "s"
        span.attributes = {semconv.GEN_AI_TOOL_NAME: "t", "gen_ai.blob": "x" * 5_000_000}
        span.status = None

        assert self._process(span) != []

    def test_a_span_name_that_is_not_a_string(self) -> None:
        span = Mock()
        span.name = 12345
        span.attributes = {semconv.GEN_AI_TOOL_NAME: "t"}
        span.status = None

        assert self._process(span) != []

    def test_null_bytes_and_exotic_unicode(self) -> None:
        span = Mock()
        span.name = "s"
        span.attributes = {
            semconv.GEN_AI_TOOL_NAME: "tool\x00null",
            # Ambiguous-unicode lint is suppressed because the confusable
            # characters are the subject of the test, not an accident.
            "gen_ai.arg": "🔥 ＦＵＬＬＷＩＤＴＨ \ud800",  # noqa: RUF001
        }
        span.status = None

        assert self._process(span) != []


class TestTheAgentSurvives:
    def test_a_lane_that_always_raises_leaves_the_agent_running(self) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        from optio import meter
        from optio.runtime import installer
        from optio.runtime.span_tap import OptioSpanTap

        class BrokenBehaviorLane(BehaviorLane):
            def process_span(self, span: object, run: object) -> list[object]:  # type: ignore[override]
                raise RuntimeError("lane is broken")

            def on_run_end(self, run: object) -> list[object]:  # type: ignore[override]
                raise RuntimeError("lane is broken")

        installer.reset_installations()
        provider = TracerProvider()
        provider.add_span_processor(
            OptioSpanTap(default_config(), lanes=[BrokenBehaviorLane(default_config())])
        )
        tracer = provider.get_tracer("t")

        @meter(provider=provider)
        def agent() -> str:
            for _ in range(20):
                with tracer.start_as_current_span("tool") as span:
                    span.set_attribute(semconv.GEN_AI_TOOL_NAME, "search")
            return "agent finished"

        assert agent() == "agent finished"
        assert failopen.activation_count() > 0, "the guard should have absorbed the failures"
        installer.reset_installations()
