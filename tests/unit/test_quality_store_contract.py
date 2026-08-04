"""One suite, every quality backend, so "interchangeable" is a checked claim.

Parametrised over the backends: a rule that holds in memory and not in Redis
fails here rather than in production.

The contract is smaller than the other two lanes' because the lane asks less of
it. ``record`` returns nothing -- quality is a run-scoped property and cannot be
judged from one step -- and the only question asked at run end is "how many
steps, and what was the last one". So the load-bearing cases are the two that
answer it: the count must be a count, and *last* must mean last.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from optio.lanes.quality.store import QualityStep, QualityStore
from optio.lanes.quality.store_memory import InMemoryQualityStore


@pytest.fixture(params=["memory"])
def store(request: pytest.FixtureRequest) -> Iterator[QualityStore]:
    """A backend under test."""
    assert request.param == "memory"
    yield InMemoryQualityStore()


def _step(
    *, errored: bool = False, tokens: int | None = 50, reasons: tuple[str, ...] = ()
) -> QualityStep:
    """A projection, distinguishable by its token count."""
    return QualityStep(errored=errored, finish_reasons=reasons, output_tokens=tokens)


class TestRecording:
    def test_the_summary_counts_every_step(self, store: QualityStore) -> None:
        """A count, not a buffer size. This is the number a user's own evaluator
        receives, and it reported a retention cap until 0.3.1."""
        for _ in range(100):
            store.record("run", _step())

        summary = store.close_run("run")

        assert summary is not None
        assert summary.step_count == 100

    def test_the_summary_keeps_the_last_step(self, store: QualityStore) -> None:
        """The heuristic scores the final answer, so "last" is the whole
        contract -- keeping the first would score the run's opening move."""
        store.record("run", _step(tokens=10))
        store.record("run", _step(tokens=99))

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last is not None
        assert summary.last.output_tokens == 99

    def test_the_last_step_is_kept_whole(self, store: QualityStore) -> None:
        """Every field survives the round trip. A backend that dropped one
        would silently disable whichever check reads it -- and a check that
        stopped running looks exactly like a run with nothing wrong."""
        store.record("run", _step(errored=True, tokens=0, reasons=("length", "stop")))

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last == _step(errored=True, tokens=0, reasons=("length", "stop"))

    def test_an_absent_token_count_survives_as_absent(self, store: QualityStore) -> None:
        """``None`` must not arrive as ``0``. Absence is unknown; zero is
        positive evidence the model produced nothing, and the two lead to
        opposite verdicts."""
        store.record("run", _step(tokens=None))

        summary = store.close_run("run")

        assert summary is not None
        assert summary.last is not None
        assert summary.last.output_tokens is None

    def test_runs_do_not_bleed(self, store: QualityStore) -> None:
        store.record("a", _step(tokens=1))
        store.record("b", _step(tokens=2))

        first = store.close_run("a")

        assert first is not None
        assert first.last is not None
        assert first.last.output_tokens == 1
        assert first.step_count == 1


class TestLifecycle:
    def test_closing_releases_the_run(self, store: QualityStore) -> None:
        store.record("run", _step())
        store.close_run("run")

        assert store.run_count() == 0

    def test_closing_twice_reports_nothing_the_second_time(self, store: QualityStore) -> None:
        """Run end fires more than once (M1-2). A second close reporting an
        empty summary would let the lane score from no evidence and emit a
        weaker verdict over the first -- the failure the behavior lane hit, and
        here it would overwrite a judge result the user paid for."""
        store.record("run", _step())
        store.close_run("run")

        assert store.close_run("run") is None

    def test_closing_an_unknown_run_is_not_an_error(self, store: QualityStore) -> None:
        assert store.close_run("never") is None
        assert store.run_count() == 0

    def test_closing_one_run_leaves_the_others(self, store: QualityStore) -> None:
        store.record("a", _step())
        store.record("b", _step())

        store.close_run("a")

        assert store.run_count() == 1

    def test_recording_after_a_close_starts_a_fresh_run(self, store: QualityStore) -> None:
        """Unlike the ledger, closing is not final here. A re-opened run counts
        from zero, so it reports what it can actually account for rather than
        resuming a total whose steps have already been scored."""
        for _ in range(5):
            store.record("run", _step())
        store.close_run("run")

        store.record("run", _step())
        summary = store.close_run("run")

        assert summary is not None
        assert summary.step_count == 1
