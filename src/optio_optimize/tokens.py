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
import hashlib
import json
import re
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol

from optio_optimize.images import message_image_tokens

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


class MemoizingCounter:
    """Caches counts by content digest, wrapping any other counter.

    Agent prompts are overwhelmingly repetitive: the system prompt and tool
    schemas are byte-identical on every step of a run, and each history turn is
    re-sent on every subsequent step. Counting them fresh each time is the
    single largest cost in this pipeline.

    Measured on the benchmark's realistic workloads: tiktoken takes **2.5 ms**
    on a 1387-token system prompt, and total per-request overhead was 1.7-1.9 ms
    -- essentially one tokenization. Hashing the same text costs a few
    microseconds, so the trade is roughly 500:1.

    Keyed on a digest rather than the text, which keeps entries at tens of bytes
    instead of holding megabytes of prompt. That also means **no prompt content
    is retained**, which matters here for the same reason it does in the core:
    cached strings show up in heap dumps and debugger sessions.
    """

    #: Entries retained. Each is a digest and an int, so 4096 costs well under
    #: a megabyte -- bounded because this lives for the process lifetime (§11).
    MAX_ENTRIES = 4096

    #: Below this length, hashing and a dict lookup cost about as much as just
    #: counting. Memoizing short strings adds entries that displace useful ones.
    MIN_LENGTH = 200

    def __init__(self, inner: TokenCounter) -> None:
        """Wrap ``inner`` with a bounded content-digest cache."""
        self._inner = inner
        self._counts: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def is_exact(self) -> bool:
        """Whether the wrapped counter is exact. Memoizing changes nothing."""
        return self._inner.is_exact

    def count_text(self, text: str, model: str = "") -> int:
        """Return the token count, from cache when the text was seen before."""
        if len(text) < self.MIN_LENGTH:
            return self._inner.count_text(text, model)

        key = (
            hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest(),
            model,
        )
        with self._lock:
            cached = self._counts.get(key)
            if cached is not None:
                self._counts.move_to_end(key)
                self.hits += 1
                return cached
            self.misses += 1

        # Counted outside the lock: tokenizing is milliseconds, and holding the
        # lock across it would serialise every thread in a concurrent agent on
        # the one operation this class exists to make cheap.
        count = self._inner.count_text(text, model)
        with self._lock:
            self._counts[key] = count
            self._counts.move_to_end(key)
            while len(self._counts) > self.MAX_ENTRIES:
                self._counts.popitem(last=False)
        return count

    @property
    def hit_rate(self) -> float | None:
        """Fraction of counts served from cache, or ``None`` before any."""
        total = self.hits + self.misses
        return self.hits / total if total else None


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
        return MemoizingCounter(TiktokenCounter())
    except ImportError:
        # The heuristic is already microseconds, so memoizing it would add a
        # hash and a lock for no gain.
        return HeuristicCounter()


def count_message(message: Message, counter: TokenCounter, model: str) -> int:
    """Return the token cost of one message, including role framing and images.

    Images are counted from block metadata rather than by the text counter, which
    has no way to see them: ``Message.content`` holds only extracted text, so a
    vision message used to cost 8 tokens against a real ~1,535 (ADR-022).
    """
    total = counter.count_text(message.content, model) + _PER_MESSAGE_OVERHEAD
    if message.name:
        total += counter.count_text(message.name, model)
    return total + message_image_tokens(message, model=model)


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
    active = counter or default_counter()
    total = sum(count_message(m, active, request.model) for m in request.messages)
    return total + count_tools(request.tools, active, request.model)


#: Multiplier applied to the raw-JSON token count of a tool schema.
#:
#: **Measured, not guessed.** Providers do not bill the JSON you hand them:
#: they re-render tool schemas into their own compact internal form, so
#: counting the serialized JSON overstates the bill. Against live
#: ``gpt-4o-mini`` on 2026-07-29, with the tool cost isolated by differencing
#: against a no-tools call:
#:
#: ===========  ==========  ==========  ======
#: tools        our JSON    OpenAI      ratio
#: ===========  ==========  ==========  ======
#: 1            138         103         1.34
#: 3            415         279         1.49
#: 5            695         458         1.52
#: 10           1395        898         1.55
#: ===========  ==========  ==========  ======
#:
#: The ratio rises with tool count and settles near 1.5, so ``0.65`` is its
#: reciprocal over the range that matters -- a handful of tools, not one.
#:
#: This is a correction for *overstating savings*, which is the direction this
#: project treats as the serious one: uncorrected, ``minify_tools`` reported
#: 3,240 tokens saved on ``mcp_agent`` where the provider billed 1,210 fewer.
#: It is calibrated against one provider and one model, which is why it is now
#: the *fallback* rather than the answer: ADR-036 measured Anthropic at **1.29**
#: -- billing more than the raw JSON rather than less -- so this value applies
#: only to models :data:`TOOL_SCHEMA_CALIBRATION_BY_MODEL` does not name. It is
#: the lowest measured ratio, so an unrecognized vendor is under-counted rather
#: than over-counted.
TOOL_SCHEMA_CALIBRATION = 0.65

#: The same ratio per vendor, because it is not the same per vendor -- and the
#: two vendors differ in **direction**, not just magnitude.
#:
#: The constant above was fitted on ``gpt-4o-mini``: OpenAI bills *less* than the
#: raw JSON tokenizes to. Anthropic bills *more*, because it re-renders the
#: schema into its own representation. Measured against
#: ``messages.count_tokens``, which is exact and free (ADR-036):
#:
#: ======  ======  ==========  ==========
#: tools   real    raw JSON    real/raw
#: ======  ======  ==========  ==========
#: 1       89      69          **1.290**
#: 5       445     345         **1.290**
#: 20      1,780   1,380       **1.290**
#: ======  ======  ==========  ==========
#:
#: Identical on Haiku 4.5, Sonnet 4.5 and Opus 4.5, so it is a property of the
#: API's tool rendering rather than of a model.
#:
#: ADR-036 applied this in ``minify_tools`` and left ``count_tools`` on the
#: global 0.65 -- a ~2x undercount that lands somewhere load-bearing:
#: :class:`~optio_optimize.stages.caching.PrefixCacheStage` checks a per-model
#: minimum (512-4,096 tokens) before marking a prefix, and this feeds it. A
#: tool-heavy Anthropic prefix genuinely over 4,096 measured about 2,000 and was
#: declined -- half of the failure ADR-036 was written to fix, still live.
TOOL_SCHEMA_CALIBRATION_BY_MODEL: dict[str, float] = {
    "gpt-": 0.65,
    "claude": 1.29,
}


def tool_schema_calibration_for(model: str) -> float:
    """The vendor's ratio of billed tool tokens to raw-JSON tokens.

    Longest-prefix match against :data:`TOOL_SCHEMA_CALIBRATION_BY_MODEL`,
    falling back to :data:`TOOL_SCHEMA_CALIBRATION` -- the lowest measured ratio
    -- so an unrecognized vendor is under-counted rather than over-counted
    (ADR-036). Under-counting declines a prefix that would have cached, which
    loses a saving; over-counting inflates every figure derived from it.

    Args:
        model: Model name, as a caller wrote it.

    Returns:
        The calibration factor to apply to a raw-JSON token count.
    """
    best = ""
    for name in TOOL_SCHEMA_CALIBRATION_BY_MODEL:
        if model.startswith(name) and len(name) > len(best):
            best = name
    return TOOL_SCHEMA_CALIBRATION_BY_MODEL[best] if best else TOOL_SCHEMA_CALIBRATION


def count_tools(
    tools: tuple[dict[str, Any], ...],
    counter: TokenCounter,
    model: str = "",
) -> int:
    """Estimate what a provider will bill for a tool schema set.

    Args:
        tools: Schemas in the provider's own shape.
        counter: Counter to use.
        model: Model name, for tokenizer selection **and** for the vendor's
            calibration -- the two differ by a factor of two between OpenAI and
            Anthropic, and in direction.

    Returns:
        Calibrated token estimate; see
        :data:`TOOL_SCHEMA_CALIBRATION_BY_MODEL` for the measurement.
    """
    raw = sum(counter.count_text(json.dumps(tool, separators=(",", ":")), model) for tool in tools)
    return int(raw * tool_schema_calibration_for(model))


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
