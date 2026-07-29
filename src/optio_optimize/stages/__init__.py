"""Stage registry and pipeline ordering.

Order is a correctness property, not a preference, so it lives here rather than
being left to the caller. The sequence below is derived from six rules -- the
fourth joined the original three once the ``ALTERED``-tier stages landed, the
fifth once tool schemas became something a stage rewrites, and the sixth once
a stage existed that reads the request rather than changing it:

1. **Cheapest exit first.** A cache hit makes every later stage's work wasted,
   so lookups run before any transformation -- ``exact_cache`` before
   ``semantic_cache``, since a byte-exact match is strictly cheaper and safer
   to check than a similarity one, and both are pointless to run once a
   `structured_output`/history transform has already changed the text a
   lookup would key on.
2. **Shrink before you mark.** Prefix marking must see the final message list,
   or it marks a boundary that trimming then moves.
3. **Lossy last, within its own constraint.** A lossy shrinking stage runs on
   the smallest prompt the lossless ones already produced, so it has the
   least left to damage -- but still before prefix marking (rule 2 outranks
   this one, not the other way round).
4. **``summarize_history`` precedes ``trim_history``, not the reverse.** Both
   target the same aged-out window. If trimming ran first it would already
   have deleted everything there is to summarize, so summarizing would find
   nothing and silently no-op forever -- the exact "looks configured, does
   nothing" trap ``config.py`` exists to prevent. Trimming still runs
   afterward as a backstop; by then the window is already within
   ``recent_turns`` and it correctly declines.
5. **``minify_tools`` precedes ``prune_tools``.** Both rewrite ``tools``, and
   running the lossless one first means the lossy one scores and reports
   against schemas that are already as small as they can safely be -- so its
   claimed saving is what pruning itself bought, not what minifying had
   already banked. Reversed, the same tokens would be counted twice, once by
   each stage, and the report would overstate the pair.
6. **``unstable_prefix`` runs before everything.** It diagnoses the caller's
   prompt assembly, so it has to see the request as the caller built it. Run
   after any transform it would report *our* rewriting as their instability --
   ``trim_history`` changes the message list on most turns, which is precisely
   the pattern the diagnostic looks for. A false finding here is worse than no
   finding: it sends someone hunting for a timestamp in a prompt that has none.

Getting this wrong is subtle rather than loud: marking a prefix before trimming
still *works*, it just silently never hits the provider cache. The same is true
of rule 4 -- a misordered summarize_history doesn't error, it just never fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio_optimize.stages.base import Stage, StageContext, StageResult
from optio_optimize.stages.caching import ExactCacheStage, PrefixCacheStage
from optio_optimize.stages.compress import CompressPromptStage
from optio_optimize.stages.diagnostics import PrefixFinding, UnstablePrefixStage
from optio_optimize.stages.history import TrimHistoryStage
from optio_optimize.stages.output import (
    AdaptiveMaxTokensStage,
    ConcisionStage,
    StructuredOutputStage,
)
from optio_optimize.stages.retrieval import DeduplicateStage, PruneRetrievalStage
from optio_optimize.stages.routing import RouteModelsStage
from optio_optimize.stages.semantic_cache import SemanticCacheStage
from optio_optimize.stages.summarize import SummarizeHistoryStage
from optio_optimize.stages.tools import (
    CapToolResultsStage,
    MinifyToolsStage,
    PruneToolsStage,
)

if TYPE_CHECKING:
    from optio_optimize.config import OptimizeConfig
    from optio_optimize.stages.semantic_cache import SimilarityFn
    from optio_optimize.stages.summarize import Summarizer

__all__ = [
    "AdaptiveMaxTokensStage",
    "CapToolResultsStage",
    "CompressPromptStage",
    "ConcisionStage",
    "DeduplicateStage",
    "ExactCacheStage",
    "MinifyToolsStage",
    "PrefixCacheStage",
    "PrefixFinding",
    "PruneRetrievalStage",
    "PruneToolsStage",
    "RouteModelsStage",
    "SemanticCacheStage",
    "Stage",
    "StageContext",
    "StageResult",
    "StructuredOutputStage",
    "SummarizeHistoryStage",
    "TrimHistoryStage",
    "UnstablePrefixStage",
    "build_stages",
]


def build_stages(
    config: OptimizeConfig,
    *,
    summarizer: Summarizer | None = None,
    similarity_fn: SimilarityFn | None = None,
) -> list[Stage]:
    """Assemble the enabled stages in execution order.

    Args:
        config: Validated configuration.
        summarizer: Passed to :class:`SummarizeHistoryStage`. ``None`` means
            that stage always declines, even if ``config.summarize_history``
            is set -- see :class:`Optimizer`'s own docstring for why that is
            still a safe default rather than a trap.
        similarity_fn: Passed to :class:`SemanticCacheStage`. ``None`` uses
            its own lexical default.

    Returns:
        Stages to run, ordered per this module's rules.
    """
    stages: list[Stage] = []

    # 0. Diagnosis, and it must be first (rule 6). It reports on the caller's
    #    own prompt assembly, so it has to see the request as the caller built
    #    it. Placed after any transform it would diagnose *our* rewriting as
    #    their unstable prefix -- trim_history changes the message list on most
    #    turns, which is exactly the pattern this looks for.
    if config.detect_unstable_prefix:
        stages.append(UnstablePrefixStage())

    # 1. Exits. Nothing below matters if one of these hits. Exact before
    #    semantic: cheaper to check and strictly safer when both would hit.
    if config.exact_cache:
        stages.append(ExactCacheStage())
    if config.semantic_cache:
        stages.append(SemanticCacheStage(similarity_fn=similarity_fn))

    # 2. Output-side shaping. Independent of prompt size, so position is free;
    #    placed before trimming to keep the prompt-shrinking stages adjacent.
    #    structured_output and concision are mutually exclusive by their own
    #    guards -- one requires a schema or tools, the other declines when
    #    either is present -- so their relative order is free too.
    if config.structured_output:
        stages.append(StructuredOutputStage())
    if config.concision:
        stages.append(ConcisionStage())
    if config.adaptive_max_tokens:
        stages.append(AdaptiveMaxTokensStage())

    # 3. Model routing. Touches only `.model`, never messages -- position
    #    relative to the shrink block below is free; grouped here with the
    #    other stage that doesn't touch prompt content.
    if config.route_models:
        stages.append(RouteModelsStage())

    # 4. Shrink. Must run before prefix marking sees the message list (rule 2
    #    in the module docstring). summarize_history precedes trim_history
    #    (rule 4) so it sees the aged-out window before trimming deletes it;
    #    trim_history then runs as a backstop on whatever remains.
    #    compress_prompt runs last in this block (rule 3): it is lossy and
    #    sentence-level, so it should see the smallest prompt the lossless
    #    and summarizing stages already produced.
    if config.summarize_history:
        stages.append(SummarizeHistoryStage(summarizer=summarizer))
    if config.trim_history:
        stages.append(TrimHistoryStage())
    if config.deduplicate:
        stages.append(DeduplicateStage())
    if config.prune_retrieval:
        stages.append(PruneRetrievalStage())
    if config.cap_tool_results:
        stages.append(CapToolResultsStage(max_tokens=config.max_tool_result_tokens))
    if config.minify_tools:
        stages.append(MinifyToolsStage())
    if config.compress_prompt:
        stages.append(CompressPromptStage())
    if config.prune_tools:
        stages.append(PruneToolsStage())

    # 5. Prefix marking. Must be last so it sees the final message list.
    if config.prefix_cache:
        stages.append(PrefixCacheStage())

    return stages
