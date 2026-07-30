"""Warm-up ordering for a concurrent fan-out (ADR-020).

N concurrent calls over a shared prompt prefix each pay to populate the
provider's cache, because none of them can see another's write. Issue one first
and the remaining N-1 read what it wrote. On Anthropic a 5-minute write costs
1.25x the base input rate and a read costs 0.1x, so for a fan-out of five the
shared prefix goes from ``5 x 1.25 = 6.25`` to ``1.25 + 4 x 0.1 = 1.65`` -- **74%
off**, with no request altered and no answer changed.

It helps on OpenAI too, which is unusual here: OpenAI caches long prefixes
automatically at a 50% input discount, but something still has to go first.

**The cost is latency**, one round trip prepended to the batch. That is why this
is opt-in and why the warm-up is skipped whenever it cannot pay: below a
provider's minimum cacheable length nothing is cached at all, so warming there
buys a full round trip of delay for exactly zero discount, and buys it silently.

This is not a :class:`~optio_optimize.stages.base.Stage`. A stage answers *what
should this request look like*, and nothing here changes a request -- only the
order they are dispatched in, which is ADR-017's test for something that is not a
stage.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, cast

from optio_optimize.stages.caching import MIN_PREFIX_TOKENS
from optio_optimize.tokens import count_message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optio_optimize.pipeline import AsyncProviderCall, Pipeline, PreparedRequest
    from optio_optimize.types import LLMRequest, LLMResponse, Message

_log = logging.getLogger("optio_optimize")

#: Shared prefix required before a warm-up call is worth its latency.
#:
#: Deliberately :data:`~optio_optimize.stages.caching.MIN_PREFIX_TOKENS` itself
#: rather than a second number beside it. That constant is known to be wrong in a
#: specific way -- the real floor spans 512 tokens on Opus 5 to 4,096 on Haiku 4.5
#: and this is one figure for all of them -- and one wrong constant that gets
#: fixed once is better than two that drift apart.
WARM_UP_MIN_PREFIX_TOKENS = MIN_PREFIX_TOKENS


def shared_prefix_length(items: Sequence[PreparedRequest]) -> int:
    """Count the leading messages identical across every request.

    Compared on ``(role, content, name)`` rather than by message equality:
    ``cacheable`` and ``extra`` are annotations this package added, and the
    provider hashes the serialized prompt. Two requests whose prefix differs only
    in a marker we placed share a prefix as far as the cache is concerned.
    """
    if not items:
        return 0
    heads = [item.request.messages for item in items]
    shortest = min(len(h) for h in heads)
    for index in range(shortest):
        if any(_identity(h[index]) != _identity(heads[0][index]) for h in heads):
            return index
    return shortest


def _identity(message: Message) -> tuple[str, str, str | None]:
    """What the provider's cache would key this message on."""
    return (message.role, message.content, message.name)


def should_warm_up(items: Sequence[PreparedRequest]) -> bool:
    """Whether one call should go first, given the requests actually being sent.

    Args:
        items: Only the requests that will reach the provider. A short-circuited
            request must never appear here: a cached answer that counted toward
            "is a warm-up worth it" would let a hit justify a real call's
            latency, and the hit is not waiting for anything.

    Returns:
        ``True`` when at least two requests share a prefix above
        :data:`WARM_UP_MIN_PREFIX_TOKENS`.
    """
    if len(items) < 2:
        # Nothing to warm the cache *for*. With one call the warm-up is the whole
        # job, done at twice the latency, read by nobody.
        return False

    shared = shared_prefix_length(items)
    if shared == 0:
        return False

    counter = items[0].ctx.counter
    model = items[0].request.model
    tokens = sum(count_message(m, counter, model) for m in items[0].request.messages[:shared])
    return tokens >= WARM_UP_MIN_PREFIX_TOKENS


async def dispatch(
    pipeline: Pipeline,
    requests: Sequence[LLMRequest],
    provider: AsyncProviderCall,
    *,
    run_id: str | None = None,
) -> list[LLMResponse]:
    """Optimize every request, then dispatch in the cheaper order.

    Args:
        pipeline: The pipeline whose stages run and whose report is updated.
        requests: The fan-out, in the caller's order.
        provider: Async function that performs one real provider call.
        run_id: optio run these belong to, for attributing savings.

    Returns:
        One response per request, **in the caller's order** -- they correlate a
        fan-out by index, so completion order would be a correctness bug wearing
        a performance feature's clothes.

    Raises:
        Exception: Only whatever ``provider`` raises. A provider failing is a real
            error the caller must see; fail-open covers failures in *this*
            package, and swallowing this one would return a response nobody
            produced.
    """
    prepared = [pipeline.prepare(request, run_id=run_id) for request in requests]
    results: list[LLMResponse | None] = [None] * len(requests)
    live: list[int] = []

    for index, item in enumerate(prepared):
        if item.short_circuit is not None and not item.bypassed:
            results[index] = pipeline.complete(item, item.short_circuit, run_id=run_id)
        else:
            live.append(index)

    async def send(index: int) -> None:
        item = prepared[index]
        # A bypassed request never went through a stage, so the original is what
        # must be sent -- `item.request` is the same object, but relying on that
        # would make this correct by coincidence.
        outgoing = requests[index] if item.bypassed else item.request
        response = await provider(outgoing)
        results[index] = pipeline.complete(item, response, run_id=run_id)

    if live and _warm_up_wanted([prepared[index] for index in live]):
        await send(live[0])
        rest = live[1:]
    else:
        rest = live

    if rest:
        await asyncio.gather(*(send(index) for index in rest))

    return cast("list[LLMResponse]", results)


def _warm_up_wanted(items: Sequence[PreparedRequest]) -> bool:
    """:func:`should_warm_up`, but a failure means "dispatch normally".

    ADR-013 rule 1 applied to a decision rather than a transformation: if working
    out whether warming pays goes wrong, every request goes out concurrently,
    which is what the caller would have done unaided.
    """
    try:
        return should_warm_up(items)
    except Exception as exc:  # noqa: BLE001 - never break the caller
        # Type only, never the message (§10).
        _log.warning(
            "optio_optimize: could not decide whether to warm the cache (%s); "
            "dispatching the fan-out concurrently",
            type(exc).__name__,
        )
        return False
