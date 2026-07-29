"""What makes two requests the same request.

``request_key`` decides whether ``exact_cache`` -- which is **on by default** --
may serve one request's stored answer for another. Getting it wrong is not a
missed saving, it is a wrong answer returned confidently, and the report shows a
cache hit either way.

This file exists because that function had **no tests at all**, and the gap had
already produced a live defect: its docstring documented ``stop`` and
``thinking_budget`` as part of the key, with a paragraph of reasoning for each,
and neither was in the payload. Two requests differing only in their stop
sequence hashed identically, so a caller who asked generation to halt at a
delimiter could be served an answer produced without one.

The rule here mirrors ``test_wire.py``'s: **every field on ``LLMRequest`` is
either in the key or named in ``cache.UNKEYED_FIELDS`` with a reason.** Adding a
field to the request type fails this file until someone decides which it is. A
prose list in a docstring is not a decision anything can check -- that is the
whole lesson of the bug above.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from optio_optimize import LLMRequest, Message
from optio_optimize.cache import UNKEYED_FIELDS, request_key

pytestmark = pytest.mark.optimize

_MESSAGES = (Message(role="user", content="hi"),)


def _request(**overrides: Any) -> LLMRequest:
    # `model` and `messages` are overridable like any other field, so they go
    # through the same dict rather than being passed separately -- otherwise the
    # parametrized test below collides on them and cannot check the two fields
    # most obviously required to split the cache.
    return LLMRequest(**{"model": "gpt-4o", "messages": _MESSAGES, **overrides})


#: One distinguishing value per keyed field. Anything absent from here and from
#: ``UNKEYED_FIELDS`` is a field nobody has ruled on.
_DISTINCT: dict[str, tuple[Any, Any]] = {
    "model": ("gpt-4o", "gpt-4o-mini"),
    "messages": (_MESSAGES, (Message(role="user", content="different"),)),
    "tools": ((), ({"type": "function", "function": {"name": "f"}},)),
    "temperature": (0.0, 0.7),
    "response_format": (None, {"type": "json_object"}),
    "stop": ((), ("<END>",)),
    "thinking_budget": (None, 4_000),
    "reasoning_effort": (None, "high"),
}


class TestEveryFieldIsRuledOn:
    def test_no_field_is_silently_ignored(self):
        for spec in fields(LLMRequest):
            if spec.name in UNKEYED_FIELDS:
                continue
            assert spec.name in _DISTINCT, (
                f"LLMRequest.{spec.name} is new: either give it a distinguishing "
                f"pair here so the key is proven to notice it, or name it in "
                f"cache.UNKEYED_FIELDS with a reason. A field the key ignores "
                f"lets exact_cache serve the wrong answer."
            )

    def test_unkeyed_fields_are_real_fields(self):
        names = {spec.name for spec in fields(LLMRequest)}
        assert set(UNKEYED_FIELDS) <= names

    @pytest.mark.parametrize("name", sorted(_DISTINCT))
    def test_changing_a_keyed_field_changes_the_key(self, name: str) -> None:
        low, high = _DISTINCT[name]
        assert request_key(_request(**{name: low})) != request_key(_request(**{name: high})), (
            f"two requests differing in {name} share a cache key, so exact_cache "
            f"will serve one for the other"
        )


class TestTheFieldsDeliberatelyIgnored:
    """Excluding a field is a real decision; these lock the two that exist."""

    def test_max_tokens_does_not_split_the_cache(self):
        # A ceiling truncates a reply rather than changing it. The stage
        # compensates by refusing to serve a stored response that was truncated.
        assert request_key(_request(max_tokens=100)) == request_key(_request(max_tokens=9_000))

    def test_a_marker_this_library_placed_does_not_split_the_cache(self):
        marked = (Message(role="user", content="hi", cacheable=True),)
        assert request_key(_request()) == request_key(LLMRequest(model="gpt-4o", messages=marked))

    def test_provider_transport_details_do_not_split_the_cache(self):
        assert request_key(_request()) == request_key(_request(extra={"extra_headers": {"a": "b"}}))


class TestTheKeyLeaksNothing:
    def test_the_prompt_is_not_in_the_key(self):
        # Keys reach logs and metrics; IMPLEMENTATION.md section 10 applies here too.
        secret = "the patient's diagnosis is confidential"
        key = request_key(LLMRequest(model="gpt-4o", messages=(Message("user", secret),)))
        assert secret not in key
        assert key.isalnum()
