"""Stage registry and pipeline ordering.

Order is a correctness property, not a preference, so it lives here rather than
being left to the caller. The sequence below is derived from three rules:

1. **Cheapest exit first.** A cache hit makes every later stage's work wasted,
   so lookups run before any transformation.
2. **Shrink before you mark.** Prefix marking must see the final message list,
   or it marks a boundary that trimming then moves.
3. **Lossy last.** Compression runs on the smallest prompt that lossless stages
   could produce, so it has the least left to damage.

Getting this wrong is subtle rather than loud: marking a prefix before trimming
still *works*, it just silently never hits the provider cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optio_optimize.stages.base import Stage, StageContext, StageResult
from optio_optimize.stages.caching import ExactCacheStage, PrefixCacheStage
from optio_optimize.stages.output import AdaptiveMaxTokensStage, StructuredOutputStage

if TYPE_CHECKING:
    from optio_optimize.config import OptimizeConfig

__all__ = [
    "AdaptiveMaxTokensStage",
    "ExactCacheStage",
    "PrefixCacheStage",
    "Stage",
    "StageContext",
    "StageResult",
    "StructuredOutputStage",
    "build_stages",
]


def build_stages(config: OptimizeConfig) -> list[Stage]:
    """Assemble the enabled stages in execution order.

    Args:
        config: Validated configuration.

    Returns:
        Stages to run, ordered per this module's rules.
    """
    stages: list[Stage] = []

    # 1. Exits. Nothing below matters if one of these hits.
    if config.exact_cache:
        stages.append(ExactCacheStage())

    # 2. Output-side shaping. Independent of prompt size, so position is free;
    #    placed before trimming to keep the prompt-shrinking stages adjacent.
    if config.structured_output:
        stages.append(StructuredOutputStage())
    if config.adaptive_max_tokens:
        stages.append(AdaptiveMaxTokensStage())

    # 3. Prefix marking. Must be last so it sees the final message list.
    if config.prefix_cache:
        stages.append(PrefixCacheStage())

    return stages
