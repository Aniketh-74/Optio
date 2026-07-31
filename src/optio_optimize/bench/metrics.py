"""What an A/B run measures, and which parts of it are trustworthy.

The metrics split into three groups by how much a benchmark can actually know,
and conflating them is how optimization libraries end up publishing numbers that
do not survive contact with a real workload.

**Exact regardless of provider.** Token counts we avoided sending, cache hit
rates, memory, our own overhead, throughput. These are properties of this code
and a simulator measures them as well as a live API does.

**Exact only against a real provider.** Output-token reduction and end-to-end
latency. A simulator's completion length and response time are whatever we told
it to produce, so measuring them proves nothing about a real model. They are
reported as ``None`` in simulated mode rather than as a number, because a
plausible-looking fabricated figure is worse than an admitted gap.

**Only meaningful when a stage is lossy.** Output quality. With lossless stages
the correct result is byte-identical responses, which is checkable and worth
checking. With semantic caching or compression on, quality needs a real model
and a judgement about what counts as equivalent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest


def sent_fingerprint(request: LLMRequest) -> str:
    """Identify one request by everything that can change what it costs.

    Built on :func:`~optio_optimize.cache.request_key` rather than beside it, so
    a field added to the cache key is covered here automatically. Two fields
    that key deliberately excludes are added back, because the two functions
    answer different questions -- one asks "may a stored reply be reused", this
    one asks "did we send something different":

    * ``max_tokens``, which ``request_key`` omits so a cached completion can be
      reused across differing limits by checking ``finish_reason``.
      ``adaptive_max_tokens`` is on by default and changes that field and no
      other, so borrowing the key unmodified would file a run where only that
      stage fired as a no-op and discard a real saving as noise (ADR-028).
    * ``Message.cacheable``, which ``request_key`` omits as "a marker this
      library placed, not caller input". True, and irrelevant here: the marker
      moves prompt tokens onto the write and read rates, which is the largest
      lossless saving this package has.

    Returns:
        A hex digest. Never the prompt: §10's content rule binds the benchmark
        as it binds everything else.
    """
    from optio_optimize.cache import request_key

    payload = json.dumps(
        {
            "key": request_key(request),
            "max_tokens": request.max_tokens,
            "cacheable": [m.cacheable for m in request.messages],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", "replace")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


@dataclass(slots=True)
class ArmResult:
    """Measurements from one side of the A/B.

    Attributes:
        name: ``"baseline"`` or ``"optimized"``.
        requests: Requests issued by the workload.
        provider_calls: Requests that actually reached the provider.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        cache_write_tokens: Prompt tokens the provider charged a **premium** to
            write into its cache. Tracked separately from reads because
            reads-0 means two different things and only this tells them
            apart: reads 0 / writes 0 is a breakpoint the provider ignored,
            reads 0 / writes N is one that worked over a prefix that then
            changed.
        cached_input_tokens: Prompt tokens served by the provider's own prefix
            cache -- discounted, not free.
        wall_seconds: Total elapsed time, including our overhead.
        provider_seconds: Time spent inside the provider call. Subtracting this
            from ``wall_seconds`` isolates our overhead, which is the number a
            user cares about when deciding whether this library is worth adding.
        peak_memory_bytes: Peak tracked allocation during the run.
        responses: Response texts, kept for quality comparison.
        errors: Failures encountered.
        live: Whether a real provider served this arm.
        sent_digest: A running digest of every request this arm handed the
            provider. Compared against the other arm's, it answers the question
            a cost delta is meaningless without: did the optimizer change
            anything at all? See :attr:`ABResult.cost_is_attributable`.
    """

    name: str
    requests: int = 0
    provider_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    wall_seconds: float = 0.0
    provider_seconds: float = 0.0
    peak_memory_bytes: int = 0
    responses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    live: bool = False
    sent_digest: str = ""

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def overhead_seconds(self) -> float:
        """Time spent in this library rather than waiting on the provider."""
        return max(0.0, self.wall_seconds - self.provider_seconds)

    @property
    def throughput_rps(self) -> float | None:
        """Requests per second, or ``None`` if nothing ran."""
        if self.requests == 0 or self.wall_seconds <= 0:
            return None
        return self.requests / self.wall_seconds

    @property
    def cache_hit_rate(self) -> float | None:
        """Fraction of requests served without a provider call."""
        if self.requests == 0:
            return None
        return (self.requests - self.provider_calls) / self.requests


@dataclass(slots=True)
class QualityResult:
    """How the optimized arm's outputs compare with the baseline's.

    Attributes:
        compared: Response pairs examined.
        identical: Pairs that matched exactly.
        equivalent: Pairs judged equivalent though not identical. Requires a
            judge; ``0`` when none was supplied.
        divergent: Pairs that differ and were not judged equivalent.
        deterministic_baseline: Whether the baseline was sampled at
            ``temperature=0``. Without that, differing responses prove nothing
            -- two identical calls to a sampled model also differ -- and every
            quality figure here is uninterpretable.
        divergent_pairs: The actual ``(baseline, optimized)`` text of every
            divergence, kept so they can be *read*. ADR-015 requires this for
            ``ALTERED``-tier stages specifically: "10/10 diverged" cannot
            distinguish harmless rewording from the model getting something
            wrong it had right, and only the pairs themselves can. Empty for
            an identical run, so the ordinary lossless path stores nothing.
    """

    compared: int = 0
    identical: int = 0
    equivalent: int = 0
    divergent: int = 0
    deterministic_baseline: bool = True
    divergent_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def identical_rate(self) -> float | None:
        """Fraction of responses that matched byte-for-byte."""
        if self.compared == 0:
            return None
        return self.identical / self.compared

    @property
    def preserved_rate(self) -> float | None:
        """Fraction identical or judged equivalent."""
        if self.compared == 0:
            return None
        return (self.identical + self.equivalent) / self.compared

    @property
    def is_interpretable(self) -> bool:
        """Whether these numbers support a conclusion.

        A sampled baseline makes the comparison meaningless: divergence would
        measure the model's randomness, not the optimizer's fidelity.
        """
        return self.deterministic_baseline and self.compared > 0


@dataclass(slots=True)
class ABResult:
    """A complete A/B comparison.

    Attributes:
        workload: Name of the workload run.
        baseline: The unoptimized arm.
        optimized: The optimized arm.
        quality: Output comparison.
        model: Model used, for pricing.
        stage_savings: Per-stage token attribution from the optimizer.
        cache_lookups: Cache reads attempted, for a true hit rate.
        cache_hits: Cache reads that hit.
        models_latency: Whether the provider spent a realistic amount of time
            per call -- live, or a simulator given a deliberate delay. Declared
            by the provider rather than inferred from the clock, because our own
            tokenizer runs inside the timed region and is easily mistaken for
            provider latency.
    """

    workload: str
    baseline: ArmResult
    optimized: ArmResult
    quality: QualityResult
    model: str
    stage_savings: dict[str, int] = field(default_factory=dict)
    cache_lookups: int = 0
    cache_hits: int = 0
    models_latency: bool = False

    @property
    def live(self) -> bool:
        """Whether a real provider served both arms."""
        return self.baseline.live and self.optimized.live

    @property
    def input_token_reduction(self) -> float | None:
        """Fraction of prompt tokens avoided."""
        return _reduction(self.baseline.input_tokens, self.optimized.input_tokens)

    @property
    def output_token_reduction(self) -> float | None:
        """Fraction of completion tokens avoided.

        ``None`` in simulated mode. A simulator emits whatever completion length
        it was told to, so a reduction here would measure the simulator's
        configuration rather than any effect of the optimizer.
        """
        if not self.live:
            return None
        return _reduction(self.baseline.output_tokens, self.optimized.output_tokens)

    @property
    def total_token_reduction(self) -> float | None:
        """Fraction of all tokens avoided."""
        return _reduction(self.baseline.total_tokens, self.optimized.total_tokens)

    def cost_usd(self, arm: ArmResult) -> float | None:
        """Cost of one arm, or ``None`` for an unpriced model."""
        from optio_optimize.config import pricing_for
        from optio_optimize.savings import _cost

        pricing = pricing_for(self.model)
        if pricing is None:
            return None
        # Writes are passed through, not dropped. Until 2026-07-31 this
        # priced every prompt token at the base or cached rate and never at
        # the 1.25x write premium, so the arm that places breakpoints was
        # billed 1.25x tokens at 1.0x -- the identical asymmetry ADR-021
        # removed from SavingsReport, reproduced inside the benchmark that
        # measures it, and flattering the optimized arm every time
        # prefix_cache wrote.
        return _cost(
            pricing,
            arm.input_tokens,
            arm.output_tokens,
            arm.cached_input_tokens,
            arm.cache_write_tokens,
        )

    @property
    def cost_is_attributable(self) -> bool:
        """Whether this run's cost delta measures the library or the provider.

        False when both arms sent identical requests **and** made the same
        number of provider calls -- there is then nothing the optimizer did that
        could have moved spend, so whatever :attr:`cost_reduction` reports is the
        provider's own output nondeterminism.

        That is not a hypothetical. Five of twelve workloads were no-ops on the
        live run of 2026-07-31, and three of them printed a number large enough
        to read as a finding: ``timestamped_agent`` -1.6% and
        ``sampled_creative`` -4.7% look like ADR-013 rule 1 violations, and
        ``unique_questions`` +2.8% claimed a saving on the workload that exists
        to report this suite's limits.

        A cache hit changes the call count; every other stage changes the
        digest. Nothing else this package does can alter spend.
        """
        return not (
            self.baseline.sent_digest == self.optimized.sent_digest
            and self.baseline.provider_calls == self.optimized.provider_calls
        )

    @property
    def cost_reduction(self) -> float | None:
        """Fraction of spend avoided.

        Not the same as token reduction, and usually larger: completion tokens
        cost three to five times prompt tokens, and provider prefix-cache hits
        are billed at roughly a tenth. A library reporting only token reduction
        understates itself on some workloads and overstates on others.
        """
        base = self.cost_usd(self.baseline)
        opt = self.cost_usd(self.optimized)
        if base is None or opt is None or base <= 0:
            return None
        return (base - opt) / base

    @property
    def latency_change(self) -> float | None:
        """Fractional change in wall time; negative means faster.

        ``None`` unless provider time was meaningful -- see
        :attr:`throughput_change` for why a zero-latency provider makes this
        number actively misleading rather than merely imprecise.
        """
        if not self._provider_time_is_meaningful:
            return None
        return _change(self.baseline.wall_seconds, self.optimized.wall_seconds)

    @property
    def added_overhead_ms_per_request(self) -> float | None:
        """Milliseconds of our own work added per request.

        Measurable without a live provider, because it is time spent in this
        library rather than waiting on a network.
        """
        if self.optimized.requests == 0:
            return None
        delta = self.optimized.overhead_seconds - self.baseline.overhead_seconds
        return delta / self.optimized.requests * 1000.0

    @property
    def throughput_change(self) -> float | None:
        """Fractional change in requests per second; positive means faster.

        ``None`` unless the provider took a realistic amount of time, live or
        simulated. Against a zero-latency provider this measures nothing but our
        own overhead against a free call, which produced a headline
        ``-28.5%`` on the first run of this suite -- a true number about a
        situation that does not exist, since a real model takes hundreds of
        milliseconds and our overhead is a fraction of one.

        Set ``SimulatedProvider.latency_ms`` to model a provider, or use
        ``--live``.
        """
        if not self._provider_time_is_meaningful:
            return None
        base = self.baseline.throughput_rps
        opt = self.optimized.throughput_rps
        if base is None or opt is None or base <= 0:
            return None
        return (opt - base) / base

    @property
    def _provider_time_is_meaningful(self) -> bool:
        """Whether provider time was real enough to reason about.

        True for a live run, or a simulated one where the provider was given a
        deliberate delay. A provider that answers instantly makes every
        wall-clock comparison a measurement of this library alone.
        """
        # Declared by the provider, never inferred from the clock. Two earlier
        # attempts inferred it and both were wrong: total elapsed time let 12
        # instant calls pass as meaningful, and per-call time then passed
        # because the simulator's own tokenizer took 109 ms, which is our work
        # being counted as the provider's. A provider knows whether it models
        # latency; nothing else reliably does.
        return self.models_latency

    @property
    def memory_overhead_bytes(self) -> int:
        """Extra peak memory the optimized arm held.

        Caches trade memory for money, so this is the denominator of that
        trade rather than an incidental statistic.
        """
        return max(0, self.optimized.peak_memory_bytes - self.baseline.peak_memory_bytes)

    @property
    def true_cache_hit_rate(self) -> float | None:
        """Hits over lookups.

        Distinct from :attr:`ArmResult.cache_hit_rate`, which is hits over all
        requests. A request the cache declined to consider -- a sampled one, say
        -- was never a lookup, and counting it as a miss understates a cache
        that is behaving correctly.
        """
        if self.cache_lookups == 0:
            return None
        return self.cache_hits / self.cache_lookups

    @property
    def cost_performance_ratio(self) -> float | None:
        """Cost saved per millisecond of overhead added.

        The summary trade this library asks a user to accept. A large positive
        number means cheap savings; a small one means we are charging latency
        for very little money.
        """
        saved = self.cost_reduction
        overhead = self.added_overhead_ms_per_request
        if saved is None or overhead is None:
            return None
        if overhead <= 0:
            return float("inf")  # Savings at no latency cost.
        base_cost = self.cost_usd(self.baseline) or 0.0
        return (saved * base_cost) / overhead


def _reduction(baseline: int, optimized: int) -> float | None:
    """Fraction of ``baseline`` avoided, or ``None`` with no baseline."""
    if baseline <= 0:
        return None
    return (baseline - optimized) / baseline


def _change(baseline: float, optimized: float) -> float | None:
    """Fractional change from ``baseline`` to ``optimized``."""
    if baseline <= 0:
        return None
    return (optimized - baseline) / baseline
