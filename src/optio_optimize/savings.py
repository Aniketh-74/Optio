"""Savings accounting — the number this package exists to produce.

An optimization nobody measured is indistinguishable from a bug that happens not
to crash. So every stage reports what it saved, the pipeline accumulates it, and
the result is available per request and in aggregate.

Two rules keep the numbers honest:

**Only count what was avoided, never what was hoped for.** A trimming stage
reports the tokens it actually removed from the request it was handed, measured
by the same counter on both sides. No stage reports a saving it did not cause.

**Report the counter's accuracy alongside the total.** With the heuristic
counter these are estimates, and a report that hides that invites decisions the
data cannot support. :attr:`SavingsReport.exact` carries it through.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from optio_optimize.config import pricing_for

if TYPE_CHECKING:
    from optio_optimize.config import ModelPricing


@dataclass(frozen=True, slots=True)
class StageSaving:
    """What one stage saved on one request.

    Attributes:
        stage: The stage's name.
        input_tokens: Prompt tokens avoided.
        output_tokens: Completion tokens avoided.
        note: Short reason, never containing prompt content.
        elapsed_ms: Wall time the stage took, so a stage that costs more
            latency than it saves money is visible rather than inferred.
    """

    stage: str
    input_tokens: int = 0
    output_tokens: int = 0
    note: str = ""
    elapsed_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Tokens avoided in total."""
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class SavingsReport:
    """Accumulated savings across one or many requests.

    Attributes:
        stages: Per-stage running totals.
        baseline_input_tokens: What the unoptimized requests would have cost.
        baseline_output_tokens: Likewise for completions.
        actual_input_tokens: What was really billed.
        actual_output_tokens: Likewise.
        provider_cached_tokens: Prompt tokens the provider served from *its*
            prefix cache. Discounted, not free, so tracked separately.
        provider_written_tokens: Prompt tokens the provider charged a **premium**
            to write into that cache. Tracked separately from the discounted
            reads for the obvious reason and one less obvious one: it is the only
            quantity here that makes this package's reported saving smaller, so
            leaving it out could never be caught by a report that looked wrong.
        provider_written_1h_tokens: The **subset** of the above that went into a
            one-hour cache entry, billed at 2x base input rather than 1.25x. A
            subset, matching how the provider reports it -- adding the two would
            double-count the most expensive tokens in the request (ADR-021).
        requests: How many requests contributed.
        short_circuits: How many never reached the provider.
        exact: Whether counts came from a real tokenizer.
    """

    stages: dict[str, StageSaving] = field(default_factory=dict)
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    provider_cached_tokens: int = 0
    provider_written_tokens: int = 0
    provider_written_1h_tokens: int = 0
    requests: int = 0
    short_circuits: int = 0
    exact: bool = False

    def record(self, saving: StageSaving) -> None:
        """Fold one stage's contribution into the totals."""
        existing = self.stages.get(saving.stage)
        if existing is None:
            self.stages[saving.stage] = saving
            return
        self.stages[saving.stage] = StageSaving(
            stage=saving.stage,
            input_tokens=existing.input_tokens + saving.input_tokens,
            output_tokens=existing.output_tokens + saving.output_tokens,
            note=saving.note or existing.note,
            elapsed_ms=existing.elapsed_ms + saving.elapsed_ms,
        )

    @property
    def total_saved_tokens(self) -> int:
        """Tokens avoided across every stage."""
        return sum(s.total_tokens for s in self.stages.values())

    @property
    def reduction_ratio(self) -> float | None:
        """Fraction of baseline tokens avoided, or ``None`` with no baseline.

        ``None`` rather than ``0.0`` when nothing has run: a zero would read as
        "measured, saved nothing", which is a different and much more damning
        claim than "no data yet". Absence is not zero -- the same rule the core
        applies to its signals.
        """
        baseline = self.baseline_input_tokens + self.baseline_output_tokens
        if baseline <= 0:
            return None
        return self.total_saved_tokens / baseline

    def estimated_cost_usd(self, model: str) -> float | None:
        """What the actual usage cost, if the model's price is known.

        Returns ``None`` for an unpriced model rather than assuming a rate --
        the same absence-is-not-zero rule the cost lane follows, and for the
        same reason: a fabricated zero would be read as "this was free".
        """
        pricing = pricing_for(model)
        if pricing is None:
            return None
        return _cost(
            pricing,
            self.actual_input_tokens,
            self.actual_output_tokens,
            self.provider_cached_tokens,
            self.provider_written_tokens,
            self.provider_written_1h_tokens,
        )

    def estimated_saved_usd(self, model: str) -> float | None:
        """Money not spent, if the model's price is known.

        The baseline is priced with no cache activity at all -- neither reads nor
        writes -- because the baseline is what the caller would have paid without
        this package, and without a breakpoint Anthropic neither discounts nor
        charges a premium. Pricing the baseline's writes too would cancel the
        premium out of both sides and hide it.
        """
        pricing = pricing_for(model)
        if pricing is None:
            return None
        baseline = _cost(pricing, self.baseline_input_tokens, self.baseline_output_tokens, cached=0)
        actual = _cost(
            pricing,
            self.actual_input_tokens,
            self.actual_output_tokens,
            self.provider_cached_tokens,
            self.provider_written_tokens,
            self.provider_written_1h_tokens,
        )
        return baseline - actual

    def summary_lines(self, model: str = "") -> list[str]:
        """Render a human-readable report.

        Args:
            model: Model to price against; omit to report tokens only.

        Returns:
            Lines suitable for printing or logging.
        """
        ratio = self.reduction_ratio
        accuracy = "exact" if self.exact else "estimated"
        lines = [
            f"requests: {self.requests}  (served without a provider call: {self.short_circuits})",
            f"tokens:   baseline {self.baseline_input_tokens + self.baseline_output_tokens:,} "
            f"-> actual {self.actual_input_tokens + self.actual_output_tokens:,}  [{accuracy}]",
        ]
        lines.append(
            f"saved:    {self.total_saved_tokens:,} tokens"
            + (f"  ({ratio:.1%})" if ratio is not None else "  (no baseline yet)")
        )
        if model:
            saved = self.estimated_saved_usd(model)
            lines.append(
                f"cost:     ${saved:.4f} not spent on {model}"
                if saved is not None
                else f"cost:     unpriced model {model!r}; token counts only"
            )
        if self.stages:
            lines.append("by stage:")
            ordered = sorted(self.stages.values(), key=lambda s: -s.total_tokens)
            for saving in ordered:
                lines.append(
                    f"  {saving.stage:<22} {saving.total_tokens:>9,} tokens"
                    f"   {saving.elapsed_ms:>7.1f} ms"
                    + (f"   {saving.note}" if saving.note else "")
                )
        return lines


def _cost(
    pricing: ModelPricing,
    input_tokens: int,
    output_tokens: int,
    cached: int,
    written: int = 0,
    written_1h: int = 0,
) -> float:
    """Return USD for a token usage against a price table.

    Prompt tokens fall into **four** price bands. ``cached`` are reads and cost
    roughly a tenth; ``written`` populate a 5-minute cache entry at a premium;
    ``written_1h`` populate a one-hour entry at a larger premium; everything else
    pays the base rate. All three are subsets of ``input_tokens``, and
    ``written_1h`` is a subset of ``written`` -- the provider reports the total in
    ``cache_creation_input_tokens`` and breaks it down in ``cache_creation``, so
    treating the hour figure as *additional* would double-count the most
    expensive tokens in the request.

    Both write arguments default to ``0`` so callers with no write data are
    priced exactly as before. That default is also the shape of the bug they fix:
    with writes folded into base-rate input, a cached Anthropic call looked
    cheaper than it was, in the one direction that inflates this package's
    headline saving. ``written_1h`` was added before anything could request a
    one-hour TTL, so the second occurrence of that error was never possible
    (ADR-021).
    """
    written = max(0, min(written, max(0, input_tokens - cached)))
    written_1h = max(0, min(written_1h, written))
    written_5m = written - written_1h
    full_input = max(0, input_tokens - cached - written)
    cached_rate = (
        pricing.cached_input_usd_per_m
        if pricing.cached_input_usd_per_m is not None
        else pricing.input_usd_per_m
    )
    write_rate = (
        pricing.cache_write_usd_per_m
        if pricing.cache_write_usd_per_m is not None
        else pricing.input_usd_per_m
    )
    # Falls back to the 5-minute rate, never to the base rate: a one-hour write
    # is certainly not cheaper than a five-minute one, and guessing low is the
    # flattering direction.
    write_1h_rate = (
        pricing.cache_write_1h_usd_per_m
        if pricing.cache_write_1h_usd_per_m is not None
        else write_rate
    )
    return (
        full_input * pricing.input_usd_per_m / 1_000_000
        + cached * cached_rate / 1_000_000
        + written_5m * write_rate / 1_000_000
        + written_1h * write_1h_rate / 1_000_000
        + output_tokens * pricing.output_usd_per_m / 1_000_000
    )


def merge(reports: list[SavingsReport]) -> SavingsReport:
    """Combine reports, e.g. from parallel workers."""
    merged = SavingsReport(exact=all(r.exact for r in reports) if reports else False)
    per_stage: dict[str, list[StageSaving]] = defaultdict(list)
    for report in reports:
        merged.baseline_input_tokens += report.baseline_input_tokens
        merged.baseline_output_tokens += report.baseline_output_tokens
        merged.actual_input_tokens += report.actual_input_tokens
        merged.actual_output_tokens += report.actual_output_tokens
        merged.provider_cached_tokens += report.provider_cached_tokens
        merged.provider_written_tokens += report.provider_written_tokens
        merged.provider_written_1h_tokens += report.provider_written_1h_tokens
        merged.requests += report.requests
        merged.short_circuits += report.short_circuits
        for name, saving in report.stages.items():
            per_stage[name].append(saving)
    for name, savings in per_stage.items():
        merged.stages[name] = StageSaving(
            stage=name,
            input_tokens=sum(s.input_tokens for s in savings),
            output_tokens=sum(s.output_tokens for s in savings),
            elapsed_ms=sum(s.elapsed_ms for s in savings),
        )
    return merged
