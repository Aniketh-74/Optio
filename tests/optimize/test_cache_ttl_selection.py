"""Choosing a one-hour cache entry over a five-minute one (ADR-021).

Anthropic writes a 5-minute entry for 1.25x base input and a one-hour entry for
2.0x, and reads either at 0.1x. Deciding the TTL for *this* write, with ``m``
further uses of the same prefix inside the hour and each after a gap long enough
to expire a five-minute entry:

* one hour: ``2.0 + 0.1m``
* five minutes: ``1.25(m + 1)``

The hour wins when ``0.75 < 1.15m``, so from **m >= 1**. One further use pays for
the upgrade.

The whole risk is the other direction. A one-hour write on a prefix that is never
re-used costs 0.75 rate-units more than the five-minute write would have -- and
this is the first lever in the package that can *raise* a bill, which ADR-013's
rule 1 forbids. So the stage asks for an hour only once expiry has been
**observed**: the same prefix seen again after a gap exceeding five minutes.
Nothing here predicts a gap. That is the difference between this and the weak
proxy ADR-018 rejected for reasoning budgets.

These tests drive the clock rather than waiting on it. The live script waits out
real five-minute windows, because only the provider can confirm the entry
actually expired.
"""

from __future__ import annotations

import pytest

from optio_optimize.config import OptimizeConfig
from optio_optimize.stages.base import StageContext
from optio_optimize.stages.caching import (
    FIVE_MINUTE_WINDOW_SECONDS,
    MAX_TRACKED_PREFIXES,
    PrefixCacheStage,
)
from optio_optimize.tokens import HeuristicCounter
from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


def _ctx(**overrides: object) -> StageContext:
    return StageContext(
        config=OptimizeConfig(**overrides),  # type: ignore[arg-type]
        counter=HeuristicCounter(),
    )


def _request(*, tail: str = "latest question") -> LLMRequest:
    """A request whose prefix clears the cacheable floor."""
    system = "You are a meticulous claims adjuster. Follow the schedule exactly. " * 600
    return LLMRequest(
        model="claude-haiku-4-5",
        messages=(
            Message(role="system", content=system),
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content=tail),
        ),
        temperature=0.0,
    )


def _marked_ttl(result: object) -> str | None:
    """The TTL on whichever message the stage marked."""
    request = result.request  # type: ignore[attr-defined]
    return next((m.cache_ttl for m in request.messages if m.cacheable), None)


class TestTheDefaultStaysFiveMinutes:
    def test_a_first_sighting_asks_for_no_particular_ttl(self) -> None:
        """``None`` means "say nothing", which the provider reads as five minutes.

        Requesting an hour on a prefix seen once would be a 60% premium on a
        write that may never be read -- a guaranteed cost increase for a
        conditional saving.
        """
        stage = PrefixCacheStage()

        result = stage.before(_request(), _ctx(cache_ttl_selection=True))

        assert _marked_ttl(result) is None

    def test_a_prompt_re_used_quickly_stays_on_five_minutes(self) -> None:
        # Inside the window the five-minute entry is still live, so its read is
        # already 0.1x and its write was cheaper. Upgrading buys nothing.
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx(cache_ttl_selection=True)

        stage.before(_request(), ctx)
        clock.advance(60.0)
        result = stage.before(_request(tail="second question"), ctx)

        assert _marked_ttl(result) is None

    def test_the_flag_off_means_the_ttl_is_never_set(self) -> None:
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx()  # cache_ttl_selection defaults off

        stage.before(_request(), ctx)
        clock.advance(FIVE_MINUTE_WINDOW_SECONDS + 60.0)
        result = stage.before(_request(tail="second question"), ctx)

        assert _marked_ttl(result) is None


class TestAnHourIsRequestedOnlyAfterObservedExpiry:
    def test_a_gap_past_the_window_upgrades_the_next_write(self) -> None:
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx(cache_ttl_selection=True)

        stage.before(_request(), ctx)
        clock.advance(FIVE_MINUTE_WINDOW_SECONDS + 1.0)
        result = stage.before(_request(tail="second question"), ctx)

        assert _marked_ttl(result) == "1h"

    def test_the_upgrade_survives_later_calls_in_the_same_loop(self) -> None:
        """Once a workload is known to have long gaps, it keeps the hour.

        Reverting to five minutes on the next quick call would re-write the
        prefix at 1.25x while a live one-hour entry was sitting there unread.
        """
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx(cache_ttl_selection=True)

        stage.before(_request(), ctx)
        clock.advance(FIVE_MINUTE_WINDOW_SECONDS + 1.0)
        stage.before(_request(tail="second"), ctx)
        clock.advance(30.0)
        result = stage.before(_request(tail="third"), ctx)

        assert _marked_ttl(result) == "1h"

    def test_a_different_prefix_does_not_inherit_the_upgrade(self) -> None:
        """Expiry is observed per prefix, not per process.

        One slow workload sharing a process with a fast one must not push the
        fast one onto 2x writes it will never read.
        """
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx(cache_ttl_selection=True)
        other = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(role="system", content="A completely different brief. " * 900),
                Message(role="user", content="q"),
                Message(role="assistant", content="a"),
                Message(role="user", content="q2"),
            ),
            temperature=0.0,
        )

        stage.before(_request(), ctx)
        clock.advance(FIVE_MINUTE_WINDOW_SECONDS + 1.0)
        stage.before(_request(tail="second"), ctx)

        assert _marked_ttl(stage.before(other, ctx)) is None


class TestTheBookkeepingIsSafeToKeep:
    def test_the_prefix_is_identified_by_hash_not_by_text(self) -> None:
        """Section 10's content rule reaches this map too.

        These keys sit in a long-lived process and end up in debug output the
        same way ``request_key``'s do, and a prompt must not be reconstructible
        from one.
        """
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx(cache_ttl_selection=True)
        secret = "the patient's diagnosis is confidential " * 1200
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(role="system", content=secret),
                Message(role="user", content="q"),
                Message(role="assistant", content="a"),
                Message(role="user", content="q2"),
            ),
            temperature=0.0,
        )

        stage.before(request, ctx)

        keys = list(stage._last_seen)
        assert keys, "nothing was recorded, so the observation cannot work at all"
        for key in keys:
            assert "diagnosis" not in key
            assert key.isalnum()

    def test_the_map_stays_bounded(self) -> None:
        """Section 11: this lives for the process lifetime, and an agent that
        rotates prompts would otherwise grow it without limit.

        The filler has to clear ``MIN_PREFIX_TOKENS`` or the stage declines
        before it records anything and this passes against an empty map. The first
        version of this test used 702 tokens and did exactly that -- it survived
        deleting the eviction loop entirely, which is how it was caught.
        """
        clock = _Clock(0.0)
        stage = PrefixCacheStage(clock=clock)
        ctx = _ctx(cache_ttl_selection=True)

        for index in range(3_000):
            stage.before(
                LLMRequest(
                    model="claude-haiku-4-5",
                    messages=(
                        Message(role="system", content=f"brief {index} " + "filler " * 4400),
                        Message(role="user", content="q"),
                        Message(role="assistant", content="a"),
                        Message(role="user", content="q2"),
                    ),
                    temperature=0.0,
                ),
                ctx,
            )

        assert stage._last_seen, "nothing was recorded, so this bound is not being tested"
        assert len(stage._last_seen) <= MAX_TRACKED_PREFIXES


class TestTheStageStillDoesItsOriginalJob:
    def test_marking_is_unchanged_when_the_ttl_feature_is_idle(self) -> None:
        stage = PrefixCacheStage()

        result = stage.before(_request(), _ctx())

        assert any(m.cacheable for m in result.request.messages)
        assert "prefix marked" in result.note

    def test_a_short_prompt_is_still_declined(self) -> None:
        stage = PrefixCacheStage()
        short = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="system", content="be terse"),),
            temperature=0.0,
        )

        assert stage.before(short, _ctx(cache_ttl_selection=True)).note == ""


class _Clock:
    """An injectable monotonic clock.

    The stage cannot be tested against ``time.monotonic`` without sleeping past a
    five-minute window, and a test that sleeps five minutes is a test nobody
    runs. The live script does the waiting, because only the provider can confirm
    the entry really expired.
    """

    def __init__(self, now: float) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds
