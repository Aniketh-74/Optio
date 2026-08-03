"""Exact token counts from Anthropic's own arithmetic (ADR-048).

``messages.count_tokens`` returns the number the provider will bill, and bills
nothing to say so. That makes it the only free source of ground truth this
package has, and the reason ADR-036's per-vendor tool calibration could be
measured at all rather than estimated.

**This is an instrument, not a request-path counter.** Every
:meth:`AnthropicCounter.count_text` that misses the memo is a network round
trip, and :func:`~optio_optimize.tokens.count_request` calls it once per message
and once per tool — sixty calls for a forty-turn conversation with twenty tools,
against a 100 ms latency budget. Use it in scripts, benchmarks and calibration
runs; leave :func:`~optio_optimize.tokens.default_counter` on the request path.

The offline counters estimate what this measures. When the two disagree, this
one is right by definition, and the gap is a calibration constant somebody
should record with a date beside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

#: Sent when a caller names no model. The endpoint requires one, and callers
#: that pass ``""`` -- the ADR-038 warm-up among them -- are asking about text
#: rather than about a model. Haiku because it is the cheapest to have wrong:
#: `count_tokens` bills nothing on any model, so the only cost of the fallback
#: is that a caller who meant Opus gets Haiku's tokenizer, and the two have not
#: been observed to differ (ADR-036 measured identical ratios across Haiku 4.5,
#: Sonnet 4.5 and Opus 4.5).
DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicCounter:
    """Counts tokens by asking Anthropic, exactly and for free.

    Implements :class:`~optio_optimize.tokens.TokenCounter`, so it can be handed
    to :class:`~optio_optimize.optimizer.Optimizer` — but see the module
    docstring before doing that on a live request path.

    Args:
        client: An ``anthropic.Anthropic``. Constructed from the environment
            when omitted, which is the usual case for a script.
        default_model: Model to ask about when the caller names none.

    Raises:
        ImportError: If the ``anthropic`` package is not installed and no client
            was supplied. Loud on purpose: an instrument that silently degraded
            to an estimate would report an approximation as exact.
    """

    #: The provider's own number, so `fits_in_window` drops its safety margin.
    is_exact = True

    __slots__ = ("_cache", "_client", "_default_model")

    def __init__(self, client: Any | None = None, default_model: str = DEFAULT_MODEL) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self._client = client
        self._default_model = default_model
        # Memoized because a miss is a round trip rather than a microsecond, and
        # the same system prompt is counted on every request. Unbounded: a
        # measurement run is short-lived, and the alternative -- evicting an
        # entry whose replacement costs a network call -- is the wrong trade
        # here even though it is the right one in `MemoizingCounter`.
        self._cache: dict[tuple[str, str], int] = {}

    def count_text(self, text: str, model: str = "") -> int:
        """Return the exact number of tokens ``text`` costs on ``model``.

        Args:
            text: The text to count.
            model: Model to count against; :data:`DEFAULT_MODEL` when empty.

        Returns:
            The provider's own count.

        Raises:
            Exception: Whatever the SDK raises. Deliberately not swallowed --
                see the module docstring.
        """
        if not text:
            # Knowable without asking, and the request path counts plenty of
            # empty strings.
            return 0

        target = model or self._default_model
        key = (target, text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # The model is part of the key rather than folded away: the endpoint
        # takes one, so caching across models would answer a question nobody
        # asked, and a future model whose tokenizer differs would be silently
        # mis-counted from an entry measured on another.
        result = self._client.messages.count_tokens(
            model=target,
            messages=[{"role": "user", "content": text}],
        )
        counted = int(result.input_tokens)
        self._cache[key] = counted
        return counted
