"""Exact Anthropic token counts, and why they are not for the request path (ADR-048).

ADR-042 made the counter pluggable and shipped no implementation of one — an
extension point with no example. This is the example, and the first counter here
that is *exact* for a vendor rather than tiktoken applied to everybody.

``messages.count_tokens`` is the provider's own arithmetic and **free**: it bills
nothing, so re-measuring costs an API key and a round trip rather than money.
That is what makes it worth having, and it is also the whole problem — a round
trip per call, while :func:`~optio_optimize.tokens.count_request` calls
``count_text`` once per message and once per tool. A forty-turn conversation
with twenty tools is sixty network calls against a 100 ms latency budget.

So this is a **measurement instrument**, not a request-path counter, and the
tests below say so in the only way that survives someone not reading the
docstring: by counting the round trips a single ``count_request`` would make.
"""

from __future__ import annotations

from typing import Any

import pytest

from optio_optimize.types import LLMRequest, Message

pytestmark = pytest.mark.optimize


class _FakeMessages:
    """Stands in for ``client.messages``, recording every call."""

    def __init__(self, per_call: int = 7) -> None:
        self.calls: list[dict[str, Any]] = []
        self._per_call = per_call

    def count_tokens(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class _Result:
            input_tokens = self._per_call

        return _Result()


class _FakeClient:
    def __init__(self, per_call: int = 7) -> None:
        self.messages = _FakeMessages(per_call)


class TestItReturnsTheProvidersOwnNumber:
    def test_a_count_is_whatever_the_api_said(self) -> None:
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        counter = AnthropicCounter(client=_FakeClient(per_call=42))

        assert counter.count_text("some prose", "claude-haiku-4-5") == 42

    def test_it_reports_itself_as_exact(self) -> None:
        """``is_exact`` gates a safety margin elsewhere (``fits_in_window``).

        Claiming exactness that is not exact would remove a margin that exists
        to keep a prompt under a hard provider limit.
        """
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        assert AnthropicCounter(client=_FakeClient()).is_exact is True

    def test_empty_text_costs_no_round_trip(self) -> None:
        """Zero tokens is knowable without asking, and the request path counts
        plenty of empty strings."""
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        client = _FakeClient()
        counter = AnthropicCounter(client=client)

        assert counter.count_text("", "claude-haiku-4-5") == 0
        assert client.messages.calls == []


class TestItAsksOnce:
    def test_the_same_text_is_not_counted_twice(self) -> None:
        """Memoized because each miss is a network round trip, not a microsecond.

        The optimizer counts the same system prompt on every request, and
        ``trim_history`` re-counts messages it has already seen.
        """
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        client = _FakeClient()
        counter = AnthropicCounter(client=client)

        counter.count_text("identical prose", "claude-haiku-4-5")
        counter.count_text("identical prose", "claude-haiku-4-5")

        assert len(client.messages.calls) == 1

    def test_the_model_is_part_of_the_key(self) -> None:
        """Anthropic's tokenizer is not guaranteed identical across models, and
        the endpoint takes a model for that reason. Caching across models would
        answer a question nobody asked."""
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        client = _FakeClient()
        counter = AnthropicCounter(client=client)

        counter.count_text("identical prose", "claude-haiku-4-5")
        counter.count_text("identical prose", "claude-opus-4-5")

        assert len(client.messages.calls) == 2

    def test_it_sends_the_model_it_was_asked_about(self) -> None:
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        client = _FakeClient()
        AnthropicCounter(client=client).count_text("prose", "claude-opus-4-5")

        assert client.messages.calls[0]["model"] == "claude-opus-4-5"

    def test_an_unnamed_model_falls_back_to_a_real_one(self) -> None:
        """``count_text`` is called with ``""`` by callers that do not care --
        the ADR-038 warm-up is one -- and the endpoint requires a model."""
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        client = _FakeClient()
        AnthropicCounter(client=client).count_text("prose", "")

        assert client.messages.calls[0]["model"].startswith("claude-")


class TestWhyThisIsNotARequestPathCounter:
    """The constraint that shapes the whole design, asserted rather than
    described. A docstring warning is advice; a number is evidence.
    """

    def test_one_count_request_costs_a_round_trip_per_message(self) -> None:
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter
        from optio_optimize.tokens import count_request

        client = _FakeClient()
        counter = AnthropicCounter(client=client)
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=tuple(
                Message(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
                for i in range(40)
            ),
            temperature=0.0,
        )

        count_request(request, counter)

        # One per distinct message: the memo cannot help, because the whole
        # point of a conversation is that the turns differ. At even 50 ms of
        # latency this is two seconds against a 100 ms budget.
        assert len(client.messages.calls) >= 40

    def test_it_is_not_offered_as_the_default_counter(self) -> None:
        """``default_counter`` must stay offline. A default that reaches the
        network turns importing this package into a service dependency."""
        from optio_optimize.tokens import default_counter

        assert type(default_counter()).__name__ != "AnthropicCounter"


class TestFailuresAreLoudHere:
    def test_an_api_error_is_not_swallowed(self) -> None:
        """The opposite of the pipeline's rule, on purpose.

        ADR-013 rule 1 makes a stage failure invisible to the agent, because a
        cost optimization must never break a request. This is an instrument: a
        measurement that silently substituted an estimate would produce a number
        indistinguishable from an exact one, and the entire reason to reach for
        it is that it is exact.
        """
        from optio_optimize.adapters.anthropic_tokens import AnthropicCounter

        class _Broken:
            class messages:  # noqa: N801
                @staticmethod
                def count_tokens(**kwargs: Any) -> Any:
                    raise RuntimeError("rate limited")

        with pytest.raises(RuntimeError, match="rate limited"):
            AnthropicCounter(client=_Broken()).count_text("prose", "claude-haiku-4-5")
