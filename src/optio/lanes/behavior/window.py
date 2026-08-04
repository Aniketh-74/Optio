"""Step-signature window (M3-1) -- the substrate for pathology detection.

A *signature* is a fingerprint of what a step did: which tool, with roughly
which arguments, and whether it errored. An agent stuck in a loop produces the
same signature over and over. That is the entire premise of the behavior lane,
and this module is the part that has to be right -- a signature that is unstable
across identical calls makes looping undetectable, and one that is too coarse
makes healthy work look pathological.

Two properties are load-bearing:

**Bounded memory.** The window is a ``deque`` with ``maxlen``. An agent run can
be arbitrarily long, and this state lives in the user's process for the run's
duration. Unbounded growth here would be a leak with the same shape as the one
stress testing found in the ledger -- so the bound is structural, not a policy
someone has to remember to enforce.

**No content retention.** Tool arguments routinely contain user prompts,
retrieved documents, and PII. Section 10 makes content privacy the primary
security control: optio never emits or logs prompt content. So arguments
are reduced to a truncated hash *at the boundary* and the original string is
never stored on the signature. A hash cannot be read back, cannot be logged by
accident, and cannot leak through a debugger session or a crash dump.

The hash is not a security primitive -- it is a fingerprint, and a collision
merely makes two different calls look alike. ``blake2b`` is used for speed
rather than for collision resistance against an adversary; nothing here defends
against a malicious agent trying to *hide* a loop.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from collections.abc import Mapping  # runtime use: isinstance in _canonical
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from optio import semconv
from optio.lanes.behavior.store import WindowState

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.sdk.trace import ReadableSpan

#: Bytes of digest retained per argument fingerprint. 8 bytes (16 hex chars)
#: gives a collision probability far below the point where it could affect a
#: verdict computed over a window of at most a few hundred steps.
_DIGEST_BYTES: Final = 8

#: Signature component used when a step names no tool. Model calls are steps
#: too, and a model called repeatedly with the same arguments is exactly the
#: loop we want to catch.
_NO_TOOL: Final = "-"

#: Prefix of the attributes we fingerprint. Derived from the semconv namespace
#: rather than written literally (Section 16 rule 5).
_GENAI_PREFIX: Final = f"{semconv.GENAI_NAMESPACE}."


@dataclass(frozen=True, slots=True)
class StepSignature:
    """A privacy-preserving fingerprint of one step.

    Frozen and hashable so that counting repeats is a plain
    :class:`collections.Counter` operation rather than a quadratic scan.

    Attributes:
        tool: Tool name, or ``"-"`` for a step that named no tool.
        args_digest: Truncated hash of the step's arguments. Never the
            arguments themselves (Section 10).
        errored: Whether the step ended in error. Kept separate from the digest
            so a retry of a failing call is recognisably *the same call*,
            which is what distinguishes a retry storm from ordinary repetition.
    """

    tool: str
    args_digest: str
    errored: bool

    @property
    def call(self) -> tuple[str, str]:
        """The call identity, ignoring outcome.

        Two attempts at the same failing operation share a ``call`` but differ
        in ``errored``. Detectors need both views.

        Returns:
            A ``(tool, args_digest)`` pair.
        """
        return (self.tool, self.args_digest)


def digest_args(value: object) -> str:
    """Reduce an argument value to a short, stable, non-reversible fingerprint.

    Stability across identical calls is the whole requirement: a signature that
    differs between two identical invocations makes a loop invisible. Mappings
    are therefore serialised in sorted-key order, since Python dict ordering
    reflects insertion order and a framework may well build the same argument
    dict differently on two passes.

    Args:
        value: The argument payload. Any type; unhashable and exotic values are
            handled by falling back to their repr.

    Returns:
        A truncated hex digest. Never the input content.
    """
    return _hash(_canonical(value))


def _canonical(value: object, depth: int = 0) -> str:
    """Render a value as a stable string for hashing.

    Args:
        value: The value to render.
        depth: Current recursion depth, bounded to keep a deeply nested or
            self-referential structure from blowing the stack on the hot path.

    Returns:
        A deterministic string rendering.
    """
    if depth > 6:
        return "..."

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        items = sorted((str(k), _canonical(v, depth + 1)) for k, v in value.items())
        return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v, depth + 1) for v in value) + "]"

    # repr covers the numeric and enum cases exactly, and gives *something*
    # deterministic for the long tail. A type whose repr embeds its memory
    # address degrades to "every call looks unique", which fails toward
    # `healthy` -- the correct direction (ADR-004).
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - a broken __repr__ must not break the agent
        return f"<unrepresentable {type(value).__name__}>"


def _hash(text: str) -> str:
    """Return the truncated digest of a string.

    Args:
        text: Text to fingerprint.

    Returns:
        Hex digest of :data:`_DIGEST_BYTES` bytes.
    """
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=_DIGEST_BYTES).hexdigest()


def signature_of(span: ReadableSpan) -> StepSignature:
    """Build the signature for one finished span.

    Args:
        span: The span to fingerprint.

    Returns:
        The step's signature.
    """
    attributes: Mapping[str, Any] = span.attributes or {}

    tool = attributes.get(semconv.GEN_AI_TOOL_NAME)
    if not isinstance(tool, str) or not tool:
        # Fall back to the model, so repeated identical model calls still form
        # a recognisable signature.
        model = attributes.get(semconv.GEN_AI_RESPONSE_MODEL) or attributes.get(
            semconv.GEN_AI_REQUEST_MODEL
        )
        tool = model if isinstance(model, str) and model else _NO_TOOL

    return StepSignature(
        tool=tool,
        args_digest=_span_args_digest(span, attributes),
        errored=_is_error(span),
    )


def _span_args_digest(span: ReadableSpan, attributes: Mapping[str, Any]) -> str:
    """Fingerprint the arguments a span represents.

    GenAI semconv has no single "tool arguments" attribute, and we deliberately
    do not read prompt content. What is available is the *shape* of the call:
    its non-content ``gen_ai.*`` attributes plus the span name. That is coarser
    than true argument equality, so it under-detects rather than over-detects --
    the safe direction, since a fabricated pathology could get a healthy run
    killed by a downstream policy.

    Args:
        span: The span being fingerprinted.
        attributes: Its attributes.

    Returns:
        A truncated digest of the call shape.
    """
    # `or ""` is not enough: a non-string name passes straight through it and
    # fails inside str.join. The guard would absorb that, but the behavior
    # signal would then be silently missing for every span of that shape --
    # exactly the quiet degradation Section 12 exists to make visible.
    parts = [span.name if isinstance(span.name, str) else ""]
    for key in sorted(attributes, key=str):
        if not isinstance(key, str) or not key.startswith(_GENAI_PREFIX):
            continue
        if key in semconv.EMITTED_SIGNALS:
            # Our own signals change every step (cost accumulates), which would
            # make every signature unique and defeat detection entirely.
            continue
        if key in _VOLATILE_ATTRIBUTES:
            continue
        parts.append(f"{key}={_canonical(attributes[key])}")
    return _hash("|".join(parts))


#: Attributes that differ between two otherwise-identical calls. Including any
#: of these would make every signature unique.
_VOLATILE_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        semconv.GEN_AI_TOOL_CALL_ID,
        semconv.GEN_AI_USAGE_INPUT_TOKENS,
        semconv.GEN_AI_USAGE_OUTPUT_TOKENS,
        semconv.RUN_ID,
    }
)


def _is_error(span: ReadableSpan) -> bool:
    """Whether a span ended in error.

    Args:
        span: The span to inspect.

    Returns:
        ``True`` when the span's status is ``ERROR``.
    """
    status = getattr(span, "status", None)
    status_code = getattr(status, "status_code", None)
    name = getattr(status_code, "name", None)
    return name == "ERROR"


class BehaviorWindow:
    """A bounded sliding window of recent step signatures for one run.

    Attributes:
        maxlen: Maximum signatures retained.
    """

    __slots__ = ("_call_counts", "_errors", "_steps", "_total", "maxlen")

    def __init__(self, maxlen: int) -> None:
        """Build an empty window.

        Args:
            maxlen: Maximum signatures to retain. Must be positive; the config
                layer validates this at setup (Section 4.2), and a non-positive
                value here would silently disable detection.

        Raises:
            ValueError: If ``maxlen`` is not positive.
        """
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self.maxlen = maxlen
        self._steps: deque[StepSignature] = deque(maxlen=maxlen)
        self._total = 0
        # Maintained incrementally by `add` rather than rebuilt per step. See
        # `add` for why that is worth the extra state.
        self._call_counts: Counter[tuple[str, str]] = Counter()
        self._errors = 0

    def add(self, signature: StepSignature) -> None:
        """Record one step, evicting the oldest when full.

        Call counts and the error tally are updated here instead of being
        derived on read. Deriving them meant every step re-counted the whole
        window, making per-step cost linear in ``maxlen``: measured at 53 us
        per step at the default 50, but 370 us at 1000. That is bounded, so it
        never becomes O(run) -- but it silently taxed anyone who widened the
        window to catch longer cycles, which is a reasonable thing to do.
        Maintaining the counts on append makes classification O(1) instead.

        Correctness rests on this window being append-only: signatures are
        never removed except by eviction here, so the counts cannot drift out
        of step with ``_steps``. ``test_counts_match_a_full_recount`` checks
        that equivalence directly.

        Args:
            signature: The step's signature.
        """
        # `deque(maxlen=...)` discards silently on overflow, so the evicted
        # signature has to be taken before appending or its counts leak.
        if len(self._steps) == self.maxlen:
            evicted = self._steps[0]
            call = evicted.call
            if self._call_counts[call] == 1:
                # Drop the key rather than leaving a zero behind: `classify`
                # reads `len(self._call_counts)` as the distinct-call count,
                # and a zeroed key would inflate it forever.
                del self._call_counts[call]
            else:
                self._call_counts[call] -= 1
            if evicted.errored:
                self._errors -= 1

        self._steps.append(signature)
        self._call_counts[signature.call] += 1
        if signature.errored:
            self._errors += 1
        self._total += 1

    def state(self, k: int) -> WindowState:
        """Summarise this window for a verdict.

        Reads the aggregates :meth:`add` already maintains, so this is
        O(distinct log k) rather than O(window) -- ``Counter.most_common(k)``
        uses a heap once ``k`` is given, instead of sorting everything.

        The return is a fixed shape on purpose. It is what a shared backend
        sends over a network per step, and a summary that grew with ``maxlen``
        would tax anyone who widened the window to catch longer cycles, which
        is the exact cost :meth:`add` exists to have already removed.

        Args:
            k: How many of the largest per-call counts to include.

        Returns:
            The state.
        """
        return WindowState(
            size=len(self._steps),
            errors=self._errors,
            distinct_calls=len(self._call_counts),
            top_counts=tuple(count for _, count in self._call_counts.most_common(k)),
        )

    @property
    def call_counts(self) -> Counter[tuple[str, str]]:
        """Live counts of each call identity currently in the window.

        Returned directly rather than copied -- copying would reintroduce the
        O(window) per-step cost this exists to remove. Callers must not mutate
        it; :func:`optio.lanes.behavior.detectors.classify` only reads.
        """
        return self._call_counts

    @property
    def error_count(self) -> int:
        """Number of retained steps that ended in error."""
        return self._errors

    @property
    def total_steps(self) -> int:
        """Steps ever added, including those evicted from the window."""
        return self._total

    def __len__(self) -> int:
        """Return the number of signatures currently retained."""
        return len(self._steps)

    def __iter__(self) -> Iterator[StepSignature]:
        """Iterate retained signatures, oldest first."""
        return iter(self._steps)

    def __repr__(self) -> str:
        """Return a debug representation.

        Deliberately reports only counts. A repr that dumped signatures could
        put fingerprinted call shapes into a log line or a crash report.
        """
        return f"<BehaviorWindow len={len(self._steps)} maxlen={self.maxlen} total={self._total}>"
