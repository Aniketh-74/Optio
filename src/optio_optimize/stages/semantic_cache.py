"""Serve a response from the nearest sufficiently-similar prior request.

Where :class:`~optio_optimize.stages.caching.ExactCacheStage` needs a
byte-identical prompt, this needs only a *close* one -- and that is exactly
why ADR-013 calls it lossy in the strongest sense: the returned text is not
what the model would have produced for *this* prompt, only for one judged
similar enough. Off by default; every other lossy stage in this package
drops or reorders content the model would still see something of, but this
one can replace the answer outright.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from optio_optimize.similarity import jaccard
from optio_optimize.stages.base import Fidelity, Stage, StageResult
from optio_optimize.stages.caching import served_from_cache
from optio_optimize.tokens import count_request

if TYPE_CHECKING:
    from optio_optimize.stages.base import StageContext
    from optio_optimize.types import LLMRequest, LLMResponse

#: ``(a, b) -> similarity in [0, 1]``. The stage's only real extension point:
#: swap in a real embedding-based cosine metric without touching anything
#: else here.
SimilarityFn = Callable[[str, str], float]

#: Entries retained. Lookup is a linear scan against every stored entry, so
#: this doubles as the latency/hit-rate control. Small on purpose: this is
#: already the highest-risk stage in the package, and a shallow store is
#: easier to reason about than a large one, not just cheaper.
DEFAULT_MAX_ENTRIES = 128

#: Entry lifetime, matching :class:`~optio_optimize.cache.MemoryCache`'s
#: reasoning: model outputs do not spoil, but the world they describe does.
DEFAULT_TTL_SECONDS = 3600.0

#: Scratch key under which ``before`` carries the text it looked up, so
#: ``after`` stores the entry under the same text rather than under whatever
#: later stages rewrote the request into. See :meth:`SemanticCacheStage.before`.
_LOOKUP_TEXT = "semantic_cache_lookup_text"


@dataclass(frozen=True, slots=True)
class _Entry:
    """One stored (prompt text, response) pair."""

    text: str
    model: str
    response: LLMResponse
    stored_at: float


def _prompt_text(request: LLMRequest) -> str:
    """The text a similarity function compares: every message, joined."""
    return "\n".join(m.content for m in request.messages)


class SemanticCacheStage(Stage):
    """Serve near-matching deterministic requests from an in-memory store.

    Only ``temperature == 0`` requests are ever stored or matched, for the
    same reason :class:`~optio_optimize.stages.caching.ExactCacheStage`
    restricts itself the same way -- and doubly so here, since the frozen
    answer being served was not even generated for this exact prompt.
    Matching is scoped to entries stored under the same ``model``: a
    response from one model answering for another substitutes a different
    capability and style, not just a cached fact.

    **The default similarity is lexical (word-set overlap), not embeddings**
    -- see :mod:`optio_optimize.similarity`. This class used to say that
    combining it with :data:`~optio_optimize.config.DEFAULT_SEMANTIC_THRESHOLD`'s
    severe 0.97 made the stage "deliberately conservative to the point of
    rarely firing on anything but near-identical wording", and that "the
    threshold does the entire job of keeping this safe".

    **That was wrong, and it was measured wrong on 2026-07-29.** Against a
    workload of long shared contexts differing in one embedded fact -- the
    ordinary shape of a RAG prompt -- the stage fired on 7 of 8 near-duplicates
    at 0.97 and served a demonstrably wrong answer on 6 of the 7 probes that
    could distinguish one (85.7%). The reasoning above has the mechanism
    backwards: a *long shared prefix* is what drives Jaccard similarity, so
    the more context a prompt carries, the less the one word that changes the
    answer moves the score. Negation is the extreme -- dropping ``not`` from a
    ~100-word prompt scored 0.9888.

    No threshold fixes this. Sweeping 0.90 through 1.00 live, the rate of
    legitimate paraphrase hits fell to zero (at 0.98) *before* the wrong-answer
    rate did (at 0.99), so every setting is either unsafe or inert, and the
    inert end is territory :class:`~optio_optimize.stages.caching.ExactCacheStage`
    already covers losslessly and by default. That is a property of the metric,
    not of the number: a word-set representation has no axis on which "covered"
    and "not covered" are far apart.

    So the guidance is not "raise the threshold", it is: **this stage is only
    defensible with an embedding-based ``similarity_fn`` supplied by the
    caller**, and remains off by default. See
    ``docs/optimize-benchmarks.md`` for the full sweep and ADR-015 for the
    resolution.
    """

    fidelity = Fidelity.ALTERED

    def __init__(
        self,
        similarity_fn: SimilarityFn | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        """Build the stage.

        Args:
            similarity_fn: ``(a, b) -> [0, 1]``. Defaults to
                :func:`~optio_optimize.similarity.jaccard` -- free, no
                embeddings, no network call.
            max_entries: Bound on stored entries.
            ttl_seconds: Entry lifetime.
        """
        self._similarity_fn: SimilarityFn = similarity_fn if similarity_fn is not None else jaccard
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "semantic_cache"

    def before(self, request: LLMRequest, ctx: StageContext) -> StageResult:
        """Serve the nearest stored response, if any clears the threshold."""
        if not request.is_deterministic:
            return self.declines(request)

        # Carried to `after` rather than recomputed there. `after` receives the
        # request *as sent* -- every later stage's rewrites included -- so
        # recomputing would store an entry under text no lookup will ever
        # produce, and the cache would silently never hit again. `concision`
        # made that concrete: it appends an instruction to plain chat requests,
        # which took this stage's hit rate to zero on workloads it had been
        # serving. `ExactCacheStage` has always stashed its key for the same
        # reason; this stage did not, and the defect was latent only because
        # every other message-rewriting stage happened to decline on the
        # workloads that exercised it.
        ctx.scratch[_LOOKUP_TEXT] = _prompt_text(request)

        threshold = ctx.config.semantic_threshold
        match = self._best_match(_prompt_text(request), request.model, threshold)
        if match is None:
            return self.declines(request)
        entry, score = match

        saved_input = entry.response.input_tokens or count_request(request, ctx.counter)
        return StageResult(
            request=request,
            response=served_from_cache(entry.response, self.name),
            saved_input_tokens=saved_input,
            saved_output_tokens=entry.response.output_tokens,
            note=f"semantic hit (similarity {score:.2f})",
        )

    def after(self, request: LLMRequest, response: LLMResponse, ctx: StageContext) -> None:
        """Store a fresh, real response for future near-matches."""
        if response.served_from is not None or not request.is_deterministic:
            return
        stored = ctx.scratch.get(_LOOKUP_TEXT)
        text = stored if isinstance(stored, str) else _prompt_text(request)
        with self._lock:
            self._entries.append(
                _Entry(
                    text=text,
                    model=request.model,
                    response=response,
                    stored_at=time.monotonic(),
                )
            )
            while len(self._entries) > self._max_entries:
                self._entries.pop(0)

    def _best_match(self, text: str, model: str, threshold: float) -> tuple[_Entry, float] | None:
        """Return the closest live entry for ``model`` clearing ``threshold``."""
        now = time.monotonic()
        with self._lock:
            live = [e for e in self._entries if (now - e.stored_at) < self._ttl_seconds]
            self._entries[:] = live  # opportunistic expiry, same as MemoryCache.get

            best_entry: _Entry | None = None
            best_score = 0.0
            for entry in live:
                if entry.model != model:
                    continue
                score = self._similarity_fn(text, entry.text)
                if score > best_score:
                    best_entry, best_score = entry, score

        if best_entry is not None and best_score >= threshold:
            return best_entry, best_score
        return None
