"""Eight findings from a static review of `main...HEAD`, verified (ADR-040).

Every one was reported without running a test -- `pytest` was not installed in
the reviewing environment -- and seven of the eight were real. That ratio is
worth recording: the suite was green at 2,110 tests while carrying all of them,
because each defect lives in a *combination* the tests exercised separately and
never together.

Four share one shape. `cache_write_tokens` and `cache_write_1h_tokens` were
added to :class:`~optio_optimize.types.LLMResponse` for ADR-021, and every
site that *reads* usage was updated. The sites that **copy, zero or re-price** a
response were not, and each defaults the new fields to ``0`` -- so they stayed
silent instead of failing. A field added to a dataclass with a default is a
field every existing constructor keeps compiling around.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from optio_optimize.types import LLMRequest, LLMResponse, Message

pytestmark = pytest.mark.optimize


class TestACacheHitDoesNotRebillTheOriginalWrite:
    """Finding 3. `exact_cache` and `semantic_cache` both route through here.

    ``served_from_cache`` zeroes the token counts because they describe what the
    *original* call cost, and leaving them would re-bill that spend every time
    the entry is served. It zeroed three fields and ADR-021 added two more, both
    of which are **premium** bands -- 1.25x and 2x base input. So every hit
    carried the original call's write tokens forward, and a cache that saves
    money reported itself spending at the most expensive rate in the table.
    """

    def test_the_write_bands_are_zeroed_with_the_rest(self) -> None:
        from optio_optimize.stages.caching import served_from_cache

        original = LLMResponse(
            content="answer",
            input_tokens=5_000,
            output_tokens=100,
            cached_input_tokens=4_000,
            cache_write_tokens=1_000,
            cache_write_1h_tokens=400,
        )

        served = served_from_cache(original, "exact_cache")

        assert served.cache_write_tokens == 0
        assert served.cache_write_1h_tokens == 0

    def test_every_billable_field_is_zeroed(self) -> None:
        """Written as a sweep rather than a list, because the list is what failed.

        The next billable band added to ``LLMResponse`` will default to ``0``
        and be copied straight through this function exactly as these two were.
        Enumerating the fields here means the test fails when that happens
        instead of the report quietly re-billing a new band.
        """
        from dataclasses import fields

        from optio_optimize.stages.caching import served_from_cache

        billable = {f.name for f in fields(LLMResponse) if f.name.endswith("_tokens")}
        original = LLMResponse(content="a", **dict.fromkeys(billable, 7))  # type: ignore[arg-type]

        served = served_from_cache(original, "exact_cache")

        assert {name: getattr(served, name) for name in billable} == dict.fromkeys(billable, 0)


class TestTheSpendGuardCountsWhatTheCallWillCost:
    """Finding 2. A live `--cap` is the only thing standing between a probe and
    a bill; ADR-037's probe overran by $7.60 on a wrong assumption.

    ``ABResult.cost_usd`` was taught the write bands in this same branch.
    ``_actual_cost``, which is what the guard records, was not -- so on Anthropic
    every cache write was recorded at base rate instead of 1.25x or 2x, and the
    guard's running total drifted below the real one on exactly the workload
    prefix caching is for.
    """

    def test_a_cache_write_is_priced_above_a_plain_input_token(self) -> None:
        from optio_optimize.bench.providers import _actual_cost

        plain = LLMResponse(content="x", input_tokens=10_000, model="claude-haiku-4-5")
        writing = LLMResponse(
            content="x",
            input_tokens=10_000,
            cache_write_tokens=10_000,
            model="claude-haiku-4-5",
        )

        assert _actual_cost(writing, "claude-haiku-4-5") > _actual_cost(plain, "claude-haiku-4-5")

    def test_the_one_hour_band_costs_more_than_the_five_minute_one(self) -> None:
        """2x base against 1.25x, and the guard has to see the difference."""
        from optio_optimize.bench.providers import _actual_cost

        five_min = LLMResponse(
            content="x", input_tokens=10_000, cache_write_tokens=10_000, model="claude-haiku-4-5"
        )
        one_hour = LLMResponse(
            content="x",
            input_tokens=10_000,
            cache_write_tokens=10_000,
            cache_write_1h_tokens=10_000,
            model="claude-haiku-4-5",
        )

        assert _actual_cost(one_hour, "claude-haiku-4-5") > _actual_cost(
            five_min, "claude-haiku-4-5"
        )

    def test_it_agrees_with_the_reported_cost(self) -> None:
        """The guard and the report must price one call identically.

        They diverged because two call sites computed the same thing and only
        one was updated. Asserting they agree is what stops that recurring.
        """
        from optio_optimize.bench.providers import _actual_cost
        from optio_optimize.config import pricing_for
        from optio_optimize.savings import _cost

        response = LLMResponse(
            content="x",
            input_tokens=10_000,
            output_tokens=500,
            cached_input_tokens=3_000,
            cache_write_tokens=2_000,
            cache_write_1h_tokens=800,
            model="claude-haiku-4-5",
        )
        pricing = pricing_for("claude-haiku-4-5")
        assert pricing is not None

        assert _actual_cost(response, "claude-haiku-4-5") == _cost(
            pricing,
            response.input_tokens,
            response.output_tokens,
            response.cached_input_tokens,
            response.cache_write_tokens,
            response.cache_write_1h_tokens,
        )


class TestStreamingReportsTheSameBandsAsANonStreamedCall:
    """Finding 1. The streaming accumulator reads
    ``cache_creation_input_tokens`` but never the ``cache_creation`` breakdown,
    so every streamed reply reported ``cache_write_1h_tokens=0``.

    ``wire.response_from_anthropic_message`` has read it since ADR-021. Two
    paths, one provider, different numbers -- and the streamed one under-bills
    the most expensive band by 37.5% exactly when ``cache_ttl_selection`` is
    doing its job.
    """

    def test_the_one_hour_band_survives_the_stream(self) -> None:
        from optio_optimize.adapters.anthropic_streaming import _Accumulator

        class _Creation:
            ephemeral_1h_input_tokens = 4_000

        class _Usage:
            input_tokens = 100
            output_tokens = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 6_000
            cache_creation = _Creation()

        class _Message:
            usage = _Usage()
            model = "claude-haiku-4-5"

        class _Start:
            type = "message_start"
            message = _Message()

        accumulator = _Accumulator()
        accumulator.feed(_Start())

        assert accumulator.response().cache_write_1h_tokens == 4_000

    def test_the_hour_band_never_exceeds_the_total(self) -> None:
        """It is a *subset* of ``cache_write_tokens`` (ADR-021), and a provider
        that reported otherwise must not make the two inconsistent here."""
        from optio_optimize.adapters.anthropic_streaming import _Accumulator

        class _Creation:
            ephemeral_1h_input_tokens = 9_999

        class _Usage:
            input_tokens = 100
            output_tokens = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 1_000
            cache_creation = _Creation()

        class _Message:
            usage = _Usage()
            model = "claude-haiku-4-5"

        class _Start:
            type = "message_start"
            message = _Message()

        accumulator = _Accumulator()
        accumulator.feed(_Start())
        response = accumulator.response()

        assert response.cache_write_1h_tokens <= response.cache_write_tokens

    def test_a_stream_with_no_cache_creation_block_still_works(self) -> None:
        """Anthropic omits the breakdown when nothing was written."""
        from optio_optimize.adapters.anthropic_streaming import _Accumulator

        class _Usage:
            input_tokens = 100
            output_tokens = 0
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Message:
            usage = _Usage()
            model = "claude-haiku-4-5"

        class _Start:
            type = "message_start"
            message = _Message()

        accumulator = _Accumulator()
        accumulator.feed(_Start())

        assert accumulator.response().cache_write_1h_tokens == 0


class TestToolSchemasAreCountedAtTheVendorsRate:
    """Finding 4. ``TOOL_SCHEMA_CALIBRATION = 0.65`` was fitted on gpt-4o-mini.
    ADR-036 measured Anthropic at **1.29** and applied it in ``minify_tools``,
    but ``count_tools`` kept the single global constant.

    The factor of two lands somewhere that matters: ``PrefixCacheStage`` checks
    a per-model minimum (512-4,096 tokens) before marking a prefix, and
    ``count_tools`` feeds that check. A tool-heavy Anthropic prefix genuinely
    above 4,096 measures ~2,000 here and is declined -- which is half of the
    exact failure ADR-036 was written to fix.
    """

    def test_anthropic_tool_schemas_are_not_counted_at_openais_rate(self) -> None:
        from optio_optimize.tokens import HeuristicCounter, count_tools

        tools = tuple(
            {"name": f"t{n}", "description": "reconcile " * 50, "parameters": {}} for n in range(20)
        )
        counter = HeuristicCounter()

        anthropic = count_tools(tools, counter, "claude-haiku-4-5")
        openai = count_tools(tools, counter, "gpt-4o-mini")

        assert anthropic > openai

    def test_an_unknown_vendor_keeps_the_lowest_measured_ratio(self) -> None:
        """Under-claiming is the safe direction, and the rule ADR-036 set.

        A ratio too high inflates every saving derived from it; too low declines
        a prefix that would have cached. Only one of those is a wrong number in
        a report.
        """
        from optio_optimize.tokens import HeuristicCounter, count_tools

        tools = ({"name": "t", "description": "reconcile " * 50, "parameters": {}},)
        counter = HeuristicCounter()

        unknown = count_tools(tools, counter, "some-future-model")
        anthropic = count_tools(tools, counter, "claude-haiku-4-5")
        openai = count_tools(tools, counter, "gpt-4o-mini")

        # Strictly below Anthropic's, not merely "no higher": an unrecognized
        # vendor inheriting the *highest* measured ratio also satisfies `<=`,
        # and that is the over-claiming direction ADR-036 forbids.
        assert unknown < anthropic
        assert unknown == openai


class TestTheCounterWarmUpCoversTheEncodingsWeUse:
    """Finding 8. ADR-038 moved ``tiktoken``'s 395 ms vocabulary load out of the
    first request's latency budget -- but warmed one encoding.

    ``_warm_counter`` calls ``count_text("warm", "")``. ``encoding_for_model("")``
    raises ``KeyError`` and falls back to ``o200k_base``, so ``cl100k_base``
    stays cold and ``gpt-4`` and ``gpt-3.5-turbo`` still pay the full load out of
    a 100 ms deadline on request one -- the precise bug ADR-038 exists to close,
    surviving for the models it did not name.
    """

    def test_both_encodings_are_warm_after_construction(self) -> None:
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.pipeline import Pipeline
        from optio_optimize.stages import build_stages

        counted: list[str] = []

        class _RecordingCounter:
            is_exact = True

            def count_text(self, text: str, model: str = "") -> int:
                counted.append(model)
                return len(text) // 4

        config = OptimizeConfig()
        Pipeline(config=config, stages=build_stages(config), counter=_RecordingCounter())

        # Named models rather than encoding names: this package must not depend
        # on tiktoken's internal naming to state which families it warms.
        assert any("gpt-4o" in m for m in counted)
        assert any(m in {"gpt-4", "gpt-3.5-turbo"} for m in counted)

    def test_warming_still_never_raises(self) -> None:
        """ADR-038's rule: this moves *when* a cost is paid and must never
        become a new way to fail. More encodings means more chances to."""
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.pipeline import Pipeline
        from optio_optimize.stages import build_stages

        class _AngryCounter:
            is_exact = True

            def count_text(self, text: str, model: str = "") -> int:
                raise RuntimeError("no vocabulary here")

        config = OptimizeConfig()
        pipeline = Pipeline(config=config, stages=build_stages(config), counter=_AngryCounter())

        assert pipeline is not None

    def test_one_unavailable_vocabulary_leaves_the_others_warm(self) -> None:
        """The loop must continue past a failure, not return.

        Aborting on the first would reintroduce exactly the bug being fixed --
        a cold encoding paying its load inside request one's deadline -- for
        whichever family happened to sort after the broken one.
        """
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.pipeline import Pipeline
        from optio_optimize.stages import build_stages

        warmed: list[str] = []

        class _PartlyAngryCounter:
            is_exact = True

            def count_text(self, text: str, model: str = "") -> int:
                if model == "gpt-4o":
                    raise RuntimeError("this vocabulary is unavailable")
                warmed.append(model)
                return len(text) // 4

        config = OptimizeConfig()
        Pipeline(config=config, stages=build_stages(config), counter=_PartlyAngryCounter())

        assert "gpt-4" in warmed


class TestTheBenchmarkPricesTheOneHourBandToo:
    """A ninth instance, found by sweeping for the pattern rather than reported.

    ``ArmResult`` carries ``cache_write_tokens`` and had **no field at all** for
    the one-hour band, so ``run_arm`` could not accumulate it and
    ``ABResult.cost_usd`` priced every 2x write at 1.25x.

    The comment directly above ``cost_usd`` describes removing this exact
    asymmetry for the five-minute band -- "the identical asymmetry ADR-021
    removed from ``SavingsReport``, reproduced inside the benchmark that
    measures it" -- and then leaves the hour band with it. Third occurrence of
    one mistake, each one layer further out.
    """

    def test_an_arm_records_the_hour_band(self) -> None:
        from optio_optimize.bench.metrics import ArmResult

        assert hasattr(ArmResult(name="x"), "cache_write_1h_tokens")

    def test_an_hour_write_is_priced_above_a_five_minute_one(self) -> None:
        from optio_optimize.bench.metrics import ABResult, ArmResult, QualityResult

        def _arm(hour: int) -> ArmResult:
            return ArmResult(
                name="optimized",
                input_tokens=10_000,
                output_tokens=100,
                cache_write_tokens=10_000,
                cache_write_1h_tokens=hour,
            )

        def _ab(hour: int) -> ABResult:
            return ABResult(
                workload="w",
                baseline=_arm(0),
                optimized=_arm(hour),
                quality=QualityResult(),
                model="claude-haiku-4-5",
            )

        result = _ab(0)
        hourly = _ab(10_000)

        assert hourly.cost_usd(hourly.optimized) > result.cost_usd(result.optimized)  # type: ignore[operator]


class TestTheReasoningBudgetConvergesRatherThanCollapsing:
    """Finding 6, and the one of the eight that is **wrong**.

    The report: ``after`` records outputs produced *under a reduced budget* into
    the same ``_lengths`` the ceiling is derived from, so the ceiling ratchets
    down toward ``MIN_THINKING_BUDGET`` over a long-lived process.

    The loop is real; the collapse is not. ``REASONING_CEILING_MULTIPLIER`` is
    **2.0**, and ``before`` lowers only when ``2 x p95 < budget`` -- so a budget
    can only fall while the model uses less than half of it, and it stops the
    moment that ceases. Simulated over 60 turns:

    ======================================  =======  ======
    model behaviour                         budget   ends
    ======================================  =======  ======
    needs a fixed 3,000 whatever it is given 32,000  6,000
    fills whatever budget it is given        32,000  32,000
    uses 90% of whatever it is given         32,000  32,000
    uses 40% of whatever it is given         32,000  25,600
    ======================================  =======  ======

    Converging on twice what the model actually uses is the stage working. The
    multiplier is the damping term, which is why these tests are written against
    the *behaviour* rather than the constant: lowering it to 1.0 would make the
    reported failure real, and that is what must fail here.
    """

    @staticmethod
    def _settle(need: object, turns: int = 60) -> list[int]:
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.output import ReasoningBudgetStage
        from optio_optimize.tokens import HeuristicCounter

        stage = ReasoningBudgetStage()
        ctx = StageContext(
            config=OptimizeConfig(reasoning_budget=True),
            counter=HeuristicCounter(),
        )
        applied: list[int] = []
        for _ in range(turns):
            request = LLMRequest(
                model="claude-haiku-4-5",
                messages=(Message(role="user", content="think hard"),),
                thinking_budget=32_000,
                temperature=0.0,
            )
            result = stage.before(request, ctx)
            budget = result.request.thinking_budget or 0
            applied.append(budget)
            stage.after(
                result.request,
                LLMResponse(content="x", output_tokens=need(budget)),  # type: ignore[operator]
                ctx,
            )
        return applied

    def test_a_model_that_needs_a_fixed_amount_settles_above_it(self) -> None:
        from optio_optimize.stages.output import MIN_THINKING_BUDGET

        settled = self._settle(lambda b: min(b, 3_000))[-1]

        assert settled >= 2 * 3_000
        assert settled > MIN_THINKING_BUDGET

    def test_a_model_that_fills_its_budget_is_never_lowered(self) -> None:
        """The regime where lowering would truncate real reasoning."""
        applied = self._settle(lambda b: b)

        assert applied[-1] == applied[0]

    def test_it_does_not_walk_down_to_the_floor(self) -> None:
        """The reported failure, stated directly. Fails if the multiplier stops
        exceeding 1.0 -- which is what would actually cause it."""
        from optio_optimize.stages.output import MIN_THINKING_BUDGET

        for need in (lambda b: b, lambda b: int(b * 0.9), lambda b: int(b * 0.4)):
            assert self._settle(need)[-1] > MIN_THINKING_BUDGET


class TestExpiryIsObservableInAGrowingConversation:
    """Finding 5. ``cache_ttl_selection`` reacts to an *observed* expiry: the
    same prefix seen again after longer than five minutes.

    It identified that prefix by ``messages[:boundary]``, where ``boundary`` is
    the stable-prefix length -- ``len(messages) - 2``. In an appending
    conversation that grows every turn, so the digest changed every turn,
    ``_last_seen`` never matched, ``_expires`` was never populated, and the
    one-hour TTL could never be emitted for the agent loop the feature is for.

    The existing tests missed it because they vary only the final message of a
    fixed four-message request, which holds ``messages[:2]`` constant. Real
    conversations append.

    What identifies the entry is the part that does **not** grow: the system
    block. Anthropic caches incrementally, so a longer prompt sharing that head
    still reads it back -- the head is the thing whose expiry can be observed.
    """

    @staticmethod
    def _turn(n: int) -> LLMRequest:
        system = "You are a meticulous claims adjuster. Follow the schedule exactly. " * 600
        history: list[Message] = [Message(role="system", content=system)]
        for i in range(n):
            history.append(Message(role="user", content=f"q{i}"))
            history.append(Message(role="assistant", content=f"a{i}"))
        history.append(Message(role="user", content=f"q{n}"))
        return LLMRequest(model="claude-haiku-4-5", messages=tuple(history), temperature=0.0)

    def test_an_expiry_is_noticed_when_the_conversation_has_grown(self) -> None:
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.caching import FIVE_MINUTE_WINDOW_SECONDS, PrefixCacheStage
        from optio_optimize.tokens import HeuristicCounter

        now = [1_000.0]
        stage = PrefixCacheStage(clock=lambda: now[0])
        ctx = StageContext(
            config=OptimizeConfig(cache_ttl_selection=True),
            counter=HeuristicCounter(),
        )

        stage.before(self._turn(0), ctx)
        now[0] += FIVE_MINUTE_WINDOW_SECONDS + 60
        result = stage.before(self._turn(1), ctx)

        assert any(m.cache_ttl == "1h" for m in result.request.messages if m.cacheable)

    def test_a_conversation_inside_the_window_still_gets_no_ttl(self) -> None:
        """The guard that keeps this from becoming a predictor.

        A one-hour write costs 2x base against 1.25x, so emitting it without an
        observed expiry raises the bill -- ADR-013 rule 1. Making the digest
        stable across turns must not also make it fire eagerly.
        """
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.caching import PrefixCacheStage
        from optio_optimize.tokens import HeuristicCounter

        now = [1_000.0]
        stage = PrefixCacheStage(clock=lambda: now[0])
        ctx = StageContext(
            config=OptimizeConfig(cache_ttl_selection=True),
            counter=HeuristicCounter(),
        )

        stage.before(self._turn(0), ctx)
        now[0] += 30.0
        result = stage.before(self._turn(1), ctx)

        assert all(m.cache_ttl is None for m in result.request.messages if m.cacheable)

    def test_a_different_system_prompt_is_a_different_entry(self) -> None:
        """Identity must still discriminate. Two tenants sharing an optimizer
        have separate provider entries, and one expiring says nothing about the
        other."""
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.caching import FIVE_MINUTE_WINDOW_SECONDS, PrefixCacheStage
        from optio_optimize.tokens import HeuristicCounter

        now = [1_000.0]
        stage = PrefixCacheStage(clock=lambda: now[0])
        ctx = StageContext(
            config=OptimizeConfig(cache_ttl_selection=True),
            counter=HeuristicCounter(),
        )
        other = self._turn(0)
        tenant_b = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(role="system", content="A different tenant's policy. " * 600),
                *other.messages[1:],
            ),
            temperature=0.0,
        )

        stage.before(other, ctx)
        now[0] += FIVE_MINUTE_WINDOW_SECONDS + 60
        result = stage.before(tenant_b, ctx)

        assert all(m.cache_ttl is None for m in result.request.messages if m.cacheable)


class TestAnthropicToolCallsReachTheCascadeVerifier:
    """Finding 7. ``response_from_anthropic_message`` joins only ``text`` blocks
    and sets no ``extra``, so a ``tool_use`` reply arrives with empty content and
    no ``tool_calls``.

    ``default_verifier`` then hits its "empty text answer" rule -- which is
    guarded by *"only when no tool call was proposed"*, and no call was visible
    to propose. So every tool-using step escalates: cheap model **and**
    expensive model, on every call, which is strictly worse than not routing at
    all. The config docstring promises the opposite.
    """

    def test_a_tool_use_reply_carries_its_calls(self) -> None:
        from optio_optimize.wire import response_from_anthropic_message

        class _ToolUse:
            type = "tool_use"
            id = "toolu_01"
            name = "search"
            input: ClassVar[dict[str, str]] = {"query": "weather"}

        class _Usage:
            input_tokens = 100
            output_tokens = 20
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Message:
            content: ClassVar[list[object]] = [_ToolUse()]
            usage = _Usage()
            model = "claude-haiku-4-5"
            stop_reason = "tool_use"

        response = response_from_anthropic_message(_Message())

        assert response.extra.get("tool_calls")

    def test_the_verifier_accepts_a_valid_tool_call(self) -> None:
        """The end the finding is about: no escalation on a good proposal."""
        from optio_optimize.cascade import default_verifier
        from optio_optimize.wire import response_from_anthropic_message

        class _ToolUse:
            type = "tool_use"
            id = "toolu_01"
            name = "search"
            input: ClassVar[dict[str, str]] = {"query": "weather"}

        class _Usage:
            input_tokens = 100
            output_tokens = 20
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Message:
            content: ClassVar[list[object]] = [_ToolUse()]
            usage = _Usage()
            model = "claude-haiku-4-5"
            stop_reason = "tool_use"

        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="what is the weather"),),
            tools=(
                {
                    "name": "search",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ),
            temperature=0.0,
        )

        assert default_verifier(request, response_from_anthropic_message(_Message())) is True

    def test_a_call_naming_an_unknown_tool_is_still_rejected(self) -> None:
        """Surfacing the calls must not weaken the check -- the point of ADR-023
        step 3 is that a cheap model's proposal gets vetted before it runs."""
        from optio_optimize.cascade import default_verifier
        from optio_optimize.wire import response_from_anthropic_message

        class _ToolUse:
            type = "tool_use"
            id = "toolu_01"
            name = "rm_rf"
            input: ClassVar[dict[str, str]] = {}

        class _Usage:
            input_tokens = 100
            output_tokens = 20
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Message:
            content: ClassVar[list[object]] = [_ToolUse()]
            usage = _Usage()
            model = "claude-haiku-4-5"
            stop_reason = "tool_use"

        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="hi"),),
            tools=({"name": "search", "input_schema": {"type": "object"}},),
            temperature=0.0,
        )

        assert default_verifier(request, response_from_anthropic_message(_Message())) is False
