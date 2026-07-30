"""Two different images must not share a cache key (ADR-022, decision 1).

`request_key` keys `[role, content, name]` per message. An image block never
reaches ``Message.content`` -- ``_text_from_content`` extracts text blocks and
ignores the rest -- so it rode through in ``extra[_RAW]``, which
``UNKEYED_FIELDS`` excludes as "provider transport details, not semantics".

For an image that classification is wrong: the image **is** the semantics. And
``exact_cache`` is on by default and caches at ``temperature == 0``, which is
exactly the setting deterministic vision work uses -- OCR, screenshot analysis,
document extraction. So "describe this image" over two different images returned
the first image's description for the second.

That is a wrong answer, not a mis-measurement. It is also the same defect class
as the ``stop`` bug already recorded in ``UNKEYED_FIELDS``: a field whose keying
was settled in prose that the payload did not implement. The difference is that
``stop`` was documented as keyed and absent, while ``extra`` was documented as
*un*keyed on a rationale that does not survive an image block.

The digest is of a normalized identity, never of the bytes: section 10's content
rule covers images at least as squarely as prose, an image is more identifying
than most text, and cache keys reach logs and metrics.
"""

from __future__ import annotations

import pytest

from optio_optimize.cache import non_text_digest, request_key
from optio_optimize.types import LLMRequest, Message
from optio_optimize.wire import RAW_CONTENT_KEY as _RAW

pytestmark = pytest.mark.optimize


def _vision(image_data: str, *, text: str = "Describe this image.") -> LLMRequest:
    """A vision request in the shape the Anthropic adapter normalizes to."""
    raw = {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_data},
            },
        ],
    }
    return LLMRequest(
        model="claude-haiku-4-5",
        messages=(Message(role="user", content=text, extra={_RAW: raw}),),
        temperature=0.0,
    )


class TestTheImageReachesTheKey:
    def test_two_different_images_do_not_collide(self) -> None:
        """The bug, stated as the assertion that used to fail.

        Both keys were ``277f0c84dc88772471a95b2f9cbe7846`` before this landed.
        """
        cat = _vision("iVBORw0KGgoAAAANSUhEUgAAAcat")
        dog = _vision("iVBORw0KGgoAAAANSUhEUgAAAdog")

        assert request_key(cat) != request_key(dog)

    def test_the_same_image_still_hits(self) -> None:
        # The fix must not simply make every vision request unique, which would
        # "fix" the collision by destroying the cache. A repeated identical
        # screenshot is a legitimate hit and the reason exact_cache exists.
        first = _vision("iVBORw0KGgoAAAANSUhEUgAAAcat")
        second = _vision("iVBORw0KGgoAAAANSUhEUgAAAcat")

        assert request_key(first) == request_key(second)

    def test_a_text_only_raw_payload_keys_identically_to_no_payload(self) -> None:
        """The digest must contribute *nothing* when there is no non-text content.

        Stated as an equality between two requests rather than against a
        hardcoded digest, because a literal would only record whatever the
        implementation currently produces. This version fails if the digest is
        computed over the whole raw payload -- role framing, block wrappers and
        all -- which is the obvious wrong implementation: the text is already
        keyed through ``content``, so keying it twice makes a text-only
        conversation's key depend on which adapter normalized it.
        """
        bare = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="What is 2 + 2?"),),
            temperature=0.0,
        )
        via_adapter = LLMRequest(
            model="claude-haiku-4-5",
            messages=(
                Message(
                    role="user",
                    content="What is 2 + 2?",
                    extra={
                        _RAW: {
                            "role": "user",
                            "content": [{"type": "text", "text": "What is 2 + 2?"}],
                        }
                    },
                ),
            ),
            temperature=0.0,
        )

        assert request_key(bare) == request_key(via_adapter)

    def test_an_image_added_to_a_prompt_changes_its_key(self) -> None:
        # The asymmetric case: same text, one call with an image and one without.
        # A shared key here would serve a text-only answer to a vision question.
        text_only = LLMRequest(
            model="claude-haiku-4-5",
            messages=(Message(role="user", content="Describe this image."),),
            temperature=0.0,
        )

        assert request_key(text_only) != request_key(_vision("iVBORw0KGgoAAAAcat"))


class TestEveryNonTextBlockCounts:
    """Not just images.

    The bug came from judging a block semantically irrelevant. Repeating that
    judgement for ``tool_use`` inputs, or for a block type nobody has seen yet,
    would reproduce it in a year. Anything that is not a text block contributes.
    """

    def test_differing_tool_use_arguments_do_not_collide(self) -> None:
        def with_args(city: str) -> LLMRequest:
            raw = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "weather",
                        "input": {"city": city},
                    },
                ],
            }
            return LLMRequest(
                model="claude-haiku-4-5",
                messages=(Message(role="assistant", content="Let me check.", extra={_RAW: raw}),),
                temperature=0.0,
            )

        assert request_key(with_args("Paris")) != request_key(with_args("Tokyo"))

    def test_differing_tool_results_do_not_collide(self) -> None:
        # Anthropic sends tool results as role "user" with a tool_result block,
        # so the text of the turn is empty and the payload is entirely non-text.
        def with_result(value: str) -> LLMRequest:
            raw = {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": value},
                ],
            }
            return LLMRequest(
                model="claude-haiku-4-5",
                messages=(Message(role="user", content="", extra={_RAW: raw}),),
                temperature=0.0,
            )

        assert request_key(with_result("18C")) != request_key(with_result("31C"))

    def test_an_unknown_future_block_type_still_counts(self) -> None:
        def with_block(payload: str) -> LLMRequest:
            raw = {
                "role": "user",
                "content": [{"type": "some_future_thing", "opaque": payload}],
            }
            return LLMRequest(
                model="claude-haiku-4-5",
                messages=(Message(role="user", content="", extra={_RAW: raw}),),
                temperature=0.0,
            )

        assert request_key(with_block("a")) != request_key(with_block("b"))


class TestTheDigestNeverCarriesTheImage:
    """Asserted against ``non_text_digest``, not against ``request_key``.

    The first version of this class checked ``request_key``'s output, which
    hashes its whole payload at the outer layer -- so the result is
    unconditionally 32 hex characters and the assertions could not fail whatever
    the digest did. Mutation testing found it: replacing the digest with the raw
    base64 payload passed all 23 tests.

    ``non_text_digest`` is where the guarantee actually lives. It is public, its
    docstring promises "never the content itself", and it is the value that could
    reach a log line or a debug dump on its own.
    """

    def test_the_base64_payload_is_not_in_the_digest(self) -> None:
        """Section 10 covers images. A digest holding one would log a screenshot."""
        secret = "iVBORw0KGgoSECRETMEDICALSCANPAYLOAD"

        digest = non_text_digest(_vision(secret).messages[0])

        assert digest is not None
        assert "SECRET" not in digest
        assert secret[:16] not in digest
        assert all(c in "0123456789abcdef" for c in digest)

    def test_the_digest_stays_short_regardless_of_image_size(self) -> None:
        # A megabyte of base64 must not become a megabyte of cache key.
        big = non_text_digest(_vision("A" * 1_000_000).messages[0])
        small = non_text_digest(_vision("A" * 10).messages[0])

        assert big is not None and small is not None
        assert len(big) == len(small) <= 64
        assert big != small

    def test_a_message_with_no_non_text_content_has_no_digest(self) -> None:
        # None rather than a digest of nothing, so a text-only request's key is
        # byte-identical to what it was before ADR-022.
        plain = Message(role="user", content="hello")

        assert non_text_digest(plain) is None
