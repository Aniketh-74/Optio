"""Token counting, and honesty about its accuracy.

Every savings number this package reports is a difference between two token
counts, so the counter is load-bearing: a biased estimator produces confident
savings figures that are wrong, which is worse than reporting nothing.

Two accuracy tiers, because the two uses have genuinely different requirements:

**Measuring savings** needs a *consistent* estimator, not an exact one. Savings
are a ratio between two counts produced by the same counter, so a uniform bias
largely cancels. The heuristic tier is adequate here and costs nothing.

**Budgeting against a context window** needs an *accurate* one. "Will this fit
in 128k?" answered by a heuristic that runs 15% low is a request that fails at
the provider. Use the exact tier for that, and note that the library degrades to
a conservative safety margin when it is unavailable.

The exact tier requires ``tiktoken``, which is an optional dependency: it is a
large wheel with a Rust extension, and forcing it on someone who only wants
history trimming is the kind of dependency creep §4.4 exists to prevent.
"""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest, Message

#: Mean characters per token for English prose under BPE tokenizers. The widely
#: repeated "4 chars per token" is close for prose and badly wrong for code and
#: JSON, which tokenize far denser. We correct for that below rather than
#: applying one ratio to everything.
_CHARS_PER_TOKEN_PROSE = 4.0

#: Code, JSON and markup fragment into many short tokens: punctuation, brackets
#: and indentation each cost one. Measured against tiktoken on JSON payloads,
#: the effective ratio sits near 3.
_CHARS_PER_TOKEN_DENSE = 3.0

#: Per-message overhead the provider adds for role framing. OpenAI documents ~3
#: for chat models; Anthropic is similar. Small per message, but a 40-turn
#: history makes it ~120 tokens, which is not noise when the point is accuracy.
_PER_MESSAGE_OVERHEAD = 3

#: Safety multiplier applied when a heuristic count is used for a *limit*
#: decision. Under-counting there means a provider-side failure, so the
#: heuristic is deliberately pessimistic when the answer gates a request.
_HEURISTIC_SAFETY_MARGIN = 1.15

_DENSE_MARKERS = re.compile(r"[{}\[\]<>:;=|\\/]|\s{2,}")


class TokenCounter(Protocol):
    """Counts tokens for a model.

    Implementations must be deterministic and side-effect free: counts feed
    cache keys and budget decisions, and a counter that varied between calls
    would make both incoherent.
    """

    @property
    def is_exact(self) -> bool:
        """Whether counts come from the real tokenizer."""
        ...

    def count_text(self, text: str, model: str) -> int:
        """Return the token count of a string."""
        ...


class HeuristicCounter:
    """Character-ratio estimator. No dependencies, no network, ~1 microsecond.

    Distinguishes prose from structured text, because one ratio across both is
    where naive estimators acquire most of their error: a JSON tool result
    counted at 4 chars/token can be under-counted by a third.
    """

    is_exact = False

    def count_text(self, text: str, model: str = "") -> int:
        """Estimate tokens in a string.

        Args:
            text: The text to measure.
            model: Ignored; the heuristic does not vary by model.

        Returns:
            Estimated token count, minimum 1 for non-empty input.
        """
        if not text:
            return 0
        markers = len(_DENSE_MARKERS.findall(text))
        # Treat the text as dense when structural characters are common enough
        # to indicate markup rather than prose containing occasional brackets.
        dense = markers > len(text) / 40
        ratio = _CHARS_PER_TOKEN_DENSE if dense else _CHARS_PER_TOKEN_PROSE
        return max(1, int(len(text) / ratio))


class TiktokenCounter:
    """Exact counts via ``tiktoken``, when it is installed.

    Encodings are cached per model: constructing one reads a vocabulary file and
    is far too slow to repeat per request.
    """

    is_exact = True

    def __init__(self) -> None:
        """Build a counter, verifying tiktoken is importable.

        Raises:
            ImportError: If tiktoken is not installed. Callers should use
                :func:`default_counter`, which falls back rather than raising.
        """
        import tiktoken  # noqa: F401 - imported for the availability check

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _encoding(model: str) -> object:
        """Return a cached encoding for a model, falling back to a general one."""
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # An unknown or newly released model. o200k_base is the current
            # frontier encoding; guessing it is much closer than refusing.
            return tiktoken.get_encoding("o200k_base")

    def count_text(self, text: str, model: str = "gpt-4o") -> int:
        """Return the exact token count of a string."""
        if not text:
            return 0
        encoding = self._encoding(model)
        return len(encoding.encode(text, disallowed_special=()))  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=1)
def default_counter() -> TokenCounter:
    """Return the most accurate counter available in this environment.

    Prefers tiktoken and falls back to the heuristic. The fallback is silent by
    design -- a hard failure here would make an optional dependency effectively
    required -- but :attr:`TokenCounter.is_exact` lets callers that need
    accuracy detect it, and the budget stage applies a safety margin when it is
    ``False`` rather than trusting an estimate it knows is approximate.
    """
    try:
        return TiktokenCounter()
    except ImportError:
        return HeuristicCounter()


def count_message(message: Message, counter: TokenCounter, model: str) -> int:
    """Return the token cost of one message, including role framing."""
    total = counter.count_text(message.content, model) + _PER_MESSAGE_OVERHEAD
    if message.name:
        total += counter.count_text(message.name, model)
    return total


def count_request(request: LLMRequest, counter: TokenCounter | None = None) -> int:
    """Return the total input tokens a request will be billed for.

    Includes tool schemas, which are easy to forget and frequently large -- a
    dozen tool definitions can outweigh the conversation itself, and they are
    resent on every single step.

    Args:
        request: The request to measure.
        counter: Counter to use; the best available one when omitted.

    Returns:
        Estimated or exact prompt tokens.
    """
    import json

    active = counter or default_counter()
    total = sum(count_message(m, active, request.model) for m in request.messages)
    for tool in request.tools:
        total += active.count_text(json.dumps(tool, separators=(",", ":")), request.model)
    return total


def fits_in_window(tokens: int, limit: int, counter: TokenCounter) -> bool:
    """Whether a token count fits a context window, allowing for estimator error.

    Applies a safety margin to inexact counts. Being wrong in the optimistic
    direction means a provider-side rejection the user sees as a crash; being
    wrong pessimistically means trimming slightly more history than strictly
    necessary. The asymmetry is the whole reason this function exists rather
    than a bare ``<`` at each call site.
    """
    effective = tokens if counter.is_exact else int(tokens * _HEURISTIC_SAFETY_MARGIN)
    return effective <= limit
