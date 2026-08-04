"""The projection is defined as *what the scorer reads*, and that is pinned here.

The design spec guessed this wrong, which is why it asked for the claim to be
verified against the code rather than trusted. It supposed the retained spans
were read for their count and by the tier decision, so the projection would need
the span name, its timestamps, and the attributes that decision consults. But
``sampling.decide`` takes the run and the config and no spans at all, and the
only consumer is ``heuristic.score``, which reads the last span alone -- three
fields off it.

Two properties matter enough to assert directly rather than leave implied by the
scoring tests: the projection carries no run content (Section 10), and it
survives leaving the process (ADR-050). Neither is visible in a verdict, so
neither would fail a behavioural test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from opentelemetry.trace import StatusCode

from optio import semconv
from optio.lanes.quality.heuristic import project
from optio.lanes.quality.store import QualityStep

if TYPE_CHECKING:
    from collections.abc import Mapping


def span(attributes: Mapping[str, object] | None = None, *, errored: bool = False) -> Mock:
    """Build a stand-in for a finished ReadableSpan."""
    mock = Mock()
    mock.name = "step"
    mock.attributes = attributes or {}
    mock.status = Mock(status_code=StatusCode.ERROR if errored else StatusCode.OK)
    return mock


class TestTheProjectionIsExactlyWhatTheScorerReads:
    def test_it_captures_the_three_fields(self) -> None:
        step = project(
            span(
                {
                    semconv.GEN_AI_RESPONSE_FINISH_REASONS: ["stop"],
                    semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 42,
                }
            )
        )

        assert step == QualityStep(errored=False, finish_reasons=("stop",), output_tokens=42)

    def test_the_field_set_does_not_grow_by_accident(self) -> None:
        """A field added without a reader is dead weight on every step of every
        run; a field added *with* one is a deliberate act. Either way it should
        be a decision, not a discovery -- so the set is written down."""
        assert {f.name for f in fields(QualityStep)} == {
            "errored",
            "finish_reasons",
            "output_tokens",
        }

    def test_an_error_status_is_captured(self) -> None:
        assert project(span({}, errored=True)).errored is True


class TestUnreadableAttributesDegradeToAbsence:
    """Every guard lives at the boundary, so the scorer is logic over clean
    types. Absence is unknown, never zero (``docs/signals.md``)."""

    def test_a_flattened_finish_reason_becomes_a_tuple(self) -> None:
        # Upstream types this as an array; several instrumentations flatten it.
        assert project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: "length"})).finish_reasons == (
            "length",
        )

    @pytest.mark.parametrize("value", [42, {"a": 1}, None, object()])
    def test_an_unreadable_finish_reason_becomes_empty(self, value: object) -> None:
        assert project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: value})).finish_reasons == ()

    def test_non_string_entries_are_dropped_rather_than_coerced(self) -> None:
        """A reason that is not a string is not a reason. Coercing would invent
        a token no provider emitted and compare it against the truncation set."""
        step = project(span({semconv.GEN_AI_RESPONSE_FINISH_REASONS: [None, 7, "length"]}))

        assert step.finish_reasons == ("length",)

    @pytest.mark.parametrize("value", ["fifty", None, 3.5, object()])
    def test_a_non_integer_token_count_is_absent(self, value: object) -> None:
        assert project(span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: value})).output_tokens is None

    def test_a_boolean_token_count_is_not_read_as_a_number(self) -> None:
        """``bool`` is a subclass of ``int``, so ``True`` would pass as "1
        token" -- turning an instrumentation quirk into evidence that the model
        answered."""
        assert project(span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: True})).output_tokens is None

    def test_a_span_with_no_attributes_projects_cleanly(self) -> None:
        mock = span()
        mock.attributes = None

        assert project(mock) == QualityStep(errored=False, finish_reasons=(), output_tokens=None)


class TestTheProjectionCanLeaveTheProcess:
    def test_it_carries_no_run_content(self) -> None:
        """Section 10. The span holds the prompt and the completion; the
        projection must hold neither, and this is the only place that would
        notice -- a leaked field changes no verdict."""
        secret = "patient SSN 123-45-6789"
        step = project(
            span(
                {
                    semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 10,
                    "gen_ai.completion": secret,
                    "gen_ai.prompt": secret,
                    "gen_ai.response.text": secret,
                }
            )
        )

        assert secret not in repr(step)
        assert secret not in json.dumps(asdict(step))

    def test_it_survives_serialisation(self) -> None:
        """A store may be another process. A span cannot cross that boundary at
        all, which is what kept this lane process-local; the projection has to,
        so it is asserted rather than assumed."""
        step = project(
            span(
                {
                    semconv.GEN_AI_RESPONSE_FINISH_REASONS: ["length"],
                    semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 7,
                }
            )
        )

        restored = json.loads(json.dumps(asdict(step)))

        assert (
            QualityStep(
                errored=restored["errored"],
                finish_reasons=tuple(restored["finish_reasons"]),
                output_tokens=restored["output_tokens"],
            )
            == step
        )

    def test_it_holds_no_reference_to_the_span(self) -> None:
        """The retained span was the reason this lane could not be shared. A
        projection that quietly kept one would serialise today and fail the
        moment a backend actually crossed a process."""
        original = span({semconv.GEN_AI_USAGE_OUTPUT_TOKENS: 1})

        step = project(original)

        assert original not in asdict(step).values()
        assert all(isinstance(v, (bool, int, tuple, type(None))) for v in asdict(step).values())


class TestTheStepIsInert:
    def test_it_is_frozen(self) -> None:
        step = QualityStep(errored=False, finish_reasons=(), output_tokens=1)
        try:
            step.errored = True  # type: ignore[misc]
        except AttributeError:
            return
        raise AssertionError("QualityStep should be immutable")
