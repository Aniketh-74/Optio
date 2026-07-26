"""Step-signature window unit tests (M3-1).

Two guarantees carry the weight here: the window is bounded (memory safety in a
long-lived agent process) and signatures are stable across identical calls
(without which looping is undetectable). Both also have property tests; these
cover the specific cases worth naming.

The privacy tests are not decoration. Section 10 makes "never retain prompt or
completion content" the primary security control, and tool arguments routinely
carry user prompts, retrieved documents, and PII. A regression that started
storing raw arguments would pass every functional test in this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from opentelemetry.trace import StatusCode

from agentmeter import semconv
from agentmeter.lanes.behavior.window import (
    BehaviorWindow,
    StepSignature,
    digest_args,
    signature_of,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def span(
    name: str = "step",
    attributes: Mapping[str, object] | None = None,
    *,
    errored: bool = False,
) -> Mock:
    """Build a stand-in for a finished ReadableSpan."""
    mock = Mock()
    mock.name = name
    mock.attributes = attributes or {}
    mock.status = Mock(status_code=StatusCode.ERROR if errored else StatusCode.OK)
    return mock


class TestBounds:
    def test_window_never_exceeds_maxlen(self) -> None:
        window = BehaviorWindow(10)
        for n in range(1000):
            window.add(StepSignature(f"t{n}", "d", errored=False))

        assert len(window) == 10

    def test_total_steps_counts_beyond_the_window(self) -> None:
        window = BehaviorWindow(5)
        for _ in range(100):
            window.add(StepSignature("t", "d", errored=False))

        assert len(window) == 5
        assert window.total_steps == 100

    def test_oldest_signatures_are_evicted_first(self) -> None:
        window = BehaviorWindow(3)
        for n in range(5):
            window.add(StepSignature(f"t{n}", "d", errored=False))

        assert [s.tool for s in window] == ["t2", "t3", "t4"]

    def test_a_non_positive_bound_is_rejected_loudly(self) -> None:
        # Setup fails loudly (Section 4.2). A zero-length window would silently
        # disable detection rather than error.
        for bad in (0, -1):
            try:
                BehaviorWindow(bad)
            except ValueError:
                continue
            raise AssertionError(f"maxlen={bad} should have been rejected")


class TestSignatureStability:
    def test_identical_spans_produce_identical_signatures(self) -> None:
        attributes = {
            semconv.GEN_AI_TOOL_NAME: "search",
            semconv.GEN_AI_REQUEST_MODEL: "gpt-4o",
        }
        assert signature_of(span("s", attributes)) == signature_of(span("s", dict(attributes)))

    def test_token_counts_do_not_change_the_signature(self) -> None:
        # Two identical calls almost never burn identical token counts. If
        # usage fed the digest, every signature would be unique and no loop
        # could ever be detected.
        base = {semconv.GEN_AI_TOOL_NAME: "search"}
        first = signature_of(span("s", {**base, semconv.GEN_AI_USAGE_INPUT_TOKENS: 100}))
        second = signature_of(span("s", {**base, semconv.GEN_AI_USAGE_INPUT_TOKENS: 999}))

        assert first == second

    def test_tool_call_id_does_not_change_the_signature(self) -> None:
        base = {semconv.GEN_AI_TOOL_NAME: "search"}
        first = signature_of(span("s", {**base, semconv.GEN_AI_TOOL_CALL_ID: "call_1"}))
        second = signature_of(span("s", {**base, semconv.GEN_AI_TOOL_CALL_ID: "call_2"}))

        assert first == second

    def test_our_own_signals_do_not_change_the_signature(self) -> None:
        # Cost accumulates every step. Feeding emitted signals back into the
        # digest would make each signature unique -- the same feedback loop the
        # span tap guards against.
        base = {semconv.GEN_AI_TOOL_NAME: "search"}
        first = signature_of(span("s", {**base, semconv.RUN_ACTUAL_COST: 0.01}))
        second = signature_of(span("s", {**base, semconv.RUN_ACTUAL_COST: 0.99}))

        assert first == second

    def test_non_genai_attributes_are_ignored(self) -> None:
        # Real spans carry http.*, service.*, and framework-specific keys
        # alongside gen_ai.*. Anything volatile among them -- a request id, a
        # timestamp -- would make every signature unique if it were hashed.
        base = {semconv.GEN_AI_TOOL_NAME: "search"}
        bare = signature_of(span("s", base))
        noisy = signature_of(
            span(
                "s",
                {
                    **base,
                    "http.request.id": "req-abc-123",
                    "service.name": "my-agent",
                    "custom.timestamp": 1234567890,
                },
            )
        )

        assert bare == noisy

    def test_a_non_string_attribute_key_is_skipped(self) -> None:
        # OTel forbids these, but the tap accepts whatever the framework
        # produced. `sorted` on mixed key types raises without the guard.
        signature = signature_of(span("s", {semconv.GEN_AI_TOOL_NAME: "t", 42: "x"}))  # type: ignore[dict-item]
        assert signature.tool == "t"

    def test_different_tools_differ(self) -> None:
        a = signature_of(span("s", {semconv.GEN_AI_TOOL_NAME: "search"}))
        b = signature_of(span("s", {semconv.GEN_AI_TOOL_NAME: "write"}))

        assert a != b

    def test_error_status_is_recorded(self) -> None:
        ok = signature_of(span("s", {semconv.GEN_AI_TOOL_NAME: "x"}))
        bad = signature_of(span("s", {semconv.GEN_AI_TOOL_NAME: "x"}, errored=True))

        assert not ok.errored
        assert bad.errored
        assert ok.call == bad.call, "a retry is the same call, differing only in outcome"

    def test_a_model_call_falls_back_to_the_model_name(self) -> None:
        signature = signature_of(span("chat", {semconv.GEN_AI_REQUEST_MODEL: "gpt-4o"}))
        assert signature.tool == "gpt-4o"

    def test_a_span_with_no_identity_still_signs(self) -> None:
        signature = signature_of(span("mystery", {}))
        assert signature.tool == "-"
        assert signature.args_digest


class TestArgumentDigestIsStable:
    def test_dict_key_order_does_not_matter(self) -> None:
        # Python dicts preserve insertion order, and a framework may build the
        # same argument dict differently on two passes.
        assert digest_args({"a": 1, "b": 2}) == digest_args({"b": 2, "a": 1})

    def test_nested_structures_are_stable(self) -> None:
        first = {"outer": {"z": [1, 2], "a": "x"}}
        second = {"outer": {"a": "x", "z": [1, 2]}}

        assert digest_args(first) == digest_args(second)

    def test_different_values_differ(self) -> None:
        assert digest_args({"q": "cats"}) != digest_args({"q": "dogs"})

    def test_deep_nesting_terminates(self) -> None:
        # A pathological structure must not blow the stack on the hot path.
        deep: dict[str, object] = {}
        node = deep
        for _ in range(500):
            child: dict[str, object] = {}
            node["next"] = child
            node = child

        assert digest_args(deep)

    def test_a_broken_repr_does_not_propagate(self) -> None:
        class ExplodingReprError:
            def __repr__(self) -> str:
                raise RuntimeError("boom")

        # Fails toward a usable digest rather than breaking the agent.
        assert digest_args(ExplodingReprError())


class TestContentIsNeverRetained:
    """Section 10: the library never retains prompt or completion content."""

    def test_the_digest_does_not_contain_the_input(self) -> None:
        secret = "patient SSN 123-45-6789"
        assert secret not in digest_args({"query": secret})

    def test_the_signature_holds_no_raw_argument_text(self) -> None:
        secret = "my private prompt"
        signature = signature_of(
            span("s", {semconv.GEN_AI_TOOL_NAME: "search", "gen_ai.prompt": secret})
        )

        assert secret not in repr(signature)
        assert secret not in signature.args_digest
        # Every field, not just the digest: content must not survive anywhere
        # on the retained object.
        for field in (signature.tool, signature.args_digest, str(signature.errored)):
            assert secret not in field

    def test_the_window_repr_leaks_nothing(self) -> None:
        window = BehaviorWindow(4)
        window.add(StepSignature("tool", digest_args("sensitive"), errored=False))

        text = repr(window)
        assert "sensitive" not in text
        assert "tool" not in text
