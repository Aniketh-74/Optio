"""Case types for the ALTERED-stage quality gate.

Three shapes, because the four lossy stages break down into three different
questions about what "quality" even means for them -- forcing one format
onto all of them would hide that rather than answer it (see the package
docstring for which half of the problem each one covers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest, LLMResponse


@dataclass(frozen=True, slots=True)
class FactPreservationCase:
    """A stage must not remove text needed to answer a specific question.

    For ``compress_prompt``: does a fact-bearing sentence survive
    near-duplicate removal. Checked against whatever the stage actually sends
    onward -- the transformed request, or a short-circuited response's
    content if the stage served one itself.

    Attributes:
        name: Identifier, used in failure messages.
        request: The request to run the stage against.
        required_facts: Case-insensitive substrings that must appear
            *somewhere* in the surviving text. Each is checked independently.
    """

    name: str
    request: LLMRequest
    required_facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheBehaviorCase:
    """A cache-style stage must hit a close paraphrase and refuse a stranger.

    For ``semantic_cache``: populates the cache with one request/response,
    then checks a near-duplicate request hits it and an unrelated request
    does not. Tests the *decision boundary*, not whether the served answer
    would actually have been correct for the near request -- that is exactly
    the risk ADR-013 calls "lossy in the strongest sense", and no
    model-free check can rule it out; the threshold and this case both exist
    to make false hits rare, not impossible.

    Attributes:
        name: Identifier, used in failure messages.
        stored_request: The request that populates the cache.
        stored_response: What gets stored against it.
        near_request: A request close enough that it should hit.
        far_request: A request unrelated enough that it must not hit.
    """

    name: str
    stored_request: LLMRequest
    stored_response: LLMResponse
    near_request: LLMRequest
    far_request: LLMRequest


@dataclass(frozen=True, slots=True)
class DecisionBoundaryCase:
    """A stage's *choice* to act must match what its own stated rule says.

    For ``route_models``: no model-free check can confirm a cheap model
    would answer as well as the requested one -- that is a model-capability
    question, out of scope here (``bench/harness.py``'s live judge path is
    where it belongs). What this checks instead: does the stage route
    exactly the requests its own documented heuristic says it should, and
    decline exactly the ones it says it should not.

    Attributes:
        name: Identifier, used in failure messages.
        request: The request to evaluate.
        should_act: Whether the stage is expected to transform this request.
    """

    name: str
    request: LLMRequest
    should_act: bool
