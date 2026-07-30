"""Counting what an image costs (ADR-022, decision 2).

An image contributed **zero** tokens. ``count_request`` returned 8 for a request
that really bills ~1,535, and the consequence is not the one that first comes to
mind:

* ``reduction_ratio`` is **not** inflated. ``pipeline`` takes
  ``response.input_tokens`` on a live call -- the provider's own number, which
  counts images -- so image tokens sit on both sides of the ratio.
* ``fits_in_window`` *is* unsafe, and that is the real defect. It applies a 1.15x
  margin to guard a few percent of estimator error, while a vision request was
  off by two orders of magnitude. Under-counting a limit decision is the
  provider-side rejection that function exists to prevent.
* A short-circuited vision request reports ~8 tokens avoided instead of ~1,535,
  because no provider number exists on that path.

**The numbers here were measured, not taken from the documentation.** Anthropic's
``messages.count_tokens`` is exact and free, so the estimator is calibrated
against synthetic PNGs of known size with the image cost differenced out against
a text-only baseline. That mattered: the published ``(w*h)/750`` formula is good
in the mid-range and then breaks badly, overstating a 1568x1568 image by 2.15x
and a 400x3000 one by 3.6x, because two caps apply that it does not mention.
"""

from __future__ import annotations

import base64
import struct
import zlib

import pytest

from optio_optimize.images import (
    MAX_IMAGE_TOKENS,
    UNKNOWN_IMAGE_TOKENS,
    anthropic_image_tokens,
    image_dimensions,
    message_image_tokens,
    openai_image_tokens,
)
from optio_optimize.tokens import HeuristicCounter, count_message, count_request
from optio_optimize.types import LLMRequest, Message
from optio_optimize.wire import RAW_CONTENT_KEY

pytestmark = pytest.mark.optimize


def _png(width: int, height: int) -> bytes:
    """A real PNG of exactly these dimensions, built without Pillow.

    The suite must not gain an image dependency to test a module that exists
    specifically to avoid one (§4.4).
    """

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes(width * 3) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows, 1))
        + chunk(b"IEND", b"")
    )


def _gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x80\x00\x00"


def _jpeg(width: int, height: int, *, huffman_first: bool = False) -> bytes:
    """A JPEG header with one SOF0 segment; enough to carry dimensions.

    ``huffman_first`` puts a ``DHT`` (``0xC4``) segment before the frame header.
    ``0xC4`` sits inside the ``0xC0``-``0xCF`` range that holds the start-of-frame
    markers but is a Huffman table, so a parser that accepts the whole range
    reads two bytes of coding table as the image size. Real JPEGs almost always
    carry a DHT, which is why this fixture exists: without it the synthetic file
    is unrealistically simple and the exclusion is untested.
    """
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + bytes(9)
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + bytes(14)
    # A DHT whose payload bytes would decode to plainly wrong dimensions.
    dht = b"\xff\xc4" + struct.pack(">H", 20) + bytes([0x00, *([0xFF] * 17)])
    return b"\xff\xd8" + app0 + (dht if huffman_first else b"") + sof


def _webp(width: int, height: int) -> bytes:
    """A VP8X (extended) WebP, whose canvas size is stored minus one."""
    body = b"VP8X" + struct.pack("<I", 10) + b"\x00" + bytes(3)
    body += struct.pack("<I", width - 1)[:3] + struct.pack("<I", height - 1)[:3]
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


def _vision_message(payload: bytes, *, media_type: str = "image/png") -> Message:
    raw = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(payload).decode(),
                },
            },
        ],
    }
    return Message(role="user", content="Describe this image.", extra={RAW_CONTENT_KEY: raw})


class TestDimensionsComeOffTheHeader:
    """No Pillow. Four formats, each with dimensions in a fixed early field.

    Pillow is a large wheel with native code, and §4.4 exists to stop it being
    pulled in for four integers.
    """

    @pytest.mark.parametrize(("width", "height"), [(1, 1), (64, 48), (1568, 1568), (4000, 3000)])
    def test_png(self, width: int, height: int) -> None:
        assert image_dimensions(_png(width, height)) == (width, height)

    @pytest.mark.parametrize(("width", "height"), [(1, 1), (800, 600), (1920, 1080)])
    def test_gif(self, width: int, height: int) -> None:
        assert image_dimensions(_gif(width, height)) == (width, height)

    @pytest.mark.parametrize(("width", "height"), [(1, 1), (640, 480), (2048, 1536)])
    def test_jpeg(self, width: int, height: int) -> None:
        assert image_dimensions(_jpeg(width, height)) == (width, height)

    @pytest.mark.parametrize(("width", "height"), [(1, 1), (512, 512), (1600, 900)])
    def test_webp(self, width: int, height: int) -> None:
        assert image_dimensions(_webp(width, height)) == (width, height)

    def test_a_jpeg_huffman_table_is_not_mistaken_for_a_frame_header(self) -> None:
        """``0xC4`` shares the start-of-frame range and is not one.

        Found by mutation: accepting the whole ``0xC0``-``0xCF`` range passed
        every other test in this file, because the plain fixture carries no DHT
        while a real JPEG nearly always does. Reading a Huffman table as
        dimensions would then produce a garbage size and a garbage token count on
        most real photographs.
        """
        assert image_dimensions(_jpeg(640, 480, huffman_first=True)) == (640, 480)

    def test_an_unrecognized_format_is_absent_not_zero(self) -> None:
        assert image_dimensions(b"this is not an image at all") is None

    def test_a_truncated_header_is_absent_rather_than_a_crash(self) -> None:
        # A partially uploaded or corrupt payload must not raise: this runs
        # inside a stage, and ADR-013 rule 1 says the caller's request survives.
        assert image_dimensions(_png(100, 100)[:12]) is None
        assert image_dimensions(b"") is None
        assert image_dimensions(b"\x89PNG\r\n\x1a\n") is None


class TestTheEstimatorMatchesWhatAnthropicBills:
    """Calibrated against ``messages.count_tokens``, which is exact and free.

    Every expected value in this table is a real measurement on
    ``claude-haiku-4-5``, with the text-only baseline differenced out. The band
    is +/-6%, which is the estimator's honest accuracy above 512px -- stated
    rather than implied, because the alternative on offer is not exactness, it is
    the zero this replaces.
    """

    @pytest.mark.parametrize(
        ("width", "height", "measured"),
        [
            (512, 512, 365),
            (800, 600, 642),
            (784, 784, 788),
            (1000, 1000, 1300),
            (1200, 958, 1509),
            (1092, 1092, 1525),
            (1568, 784, 1572),
        ],
    )
    def test_within_six_percent_of_the_measurement(
        self, width: int, height: int, measured: int
    ) -> None:
        estimate = anthropic_image_tokens(width, height)

        assert estimate == pytest.approx(measured, rel=0.06), (
            f"{width}x{height}: estimated {estimate} against a measured {measured}"
        )

    @pytest.mark.parametrize(
        ("width", "height", "measured"),
        [
            (1568, 1568, 1525),
            (2000, 1000, 1572),
            (400, 3000, 452),
            (3000, 400, 452),
        ],
    )
    def test_the_capped_sizes_are_where_the_bare_formula_breaks(
        self, width: int, height: int, measured: int
    ) -> None:
        """``(w*h)/750`` predicts 3,278 for 1568x1568 against a real 1,525.

        A max edge of 1568 explains the thin images and an area cap holds the
        rest near 1,600. Without both, a full-page screenshot is overstated by
        more than 2x -- and this project treats a wrong denominator as seriously
        as a wrong numerator.
        """
        naive = width * height / 750

        estimate = anthropic_image_tokens(width, height)

        assert estimate == pytest.approx(measured, rel=0.06)
        if naive > MAX_IMAGE_TOKENS:
            assert naive > estimate * 1.5, "this size is supposed to exercise a cap"

    def test_nothing_exceeds_the_measured_ceiling(self) -> None:
        # The provider downscales; no single image can bill more than this.
        for width, height in [(10_000, 10_000), (99_999, 1), (5000, 4000)]:
            assert anthropic_image_tokens(width, height) <= MAX_IMAGE_TOKENS

    @pytest.mark.parametrize(
        ("width", "height", "measured"),
        [
            (128, 96, 24),
            (320, 240, 112),
            (640, 480, 418),
            (900, 700, 829),
            (1024, 768, 1_040),
            (1366, 768, 1_376),
            (1440, 900, 1_554),
            (1920, 1080, 1_564),
            (2560, 1440, 1_564),
            (3840, 2160, 1_564),
            (600, 2400, 788),
            (2400, 600, 788),
            (100, 1500, 220),
        ],
    )
    def test_held_out_sizes_the_constants_were_not_fitted_to(
        self, width: int, height: int, measured: int
    ) -> None:
        """The only honest test of an estimator: data it has never seen.

        The class above asserts against the sizes the constants were chosen
        from, which proves they were copied correctly and nothing else. These
        thirteen are common real screen and camera dimensions, measured
        afterwards against ``count_tokens`` and none of them used to pick a
        constant.

        Worst absolute error **5.5%**, mean signed error **+1.4%** -- biased
        high, which is the safe direction in both places this number is used:
        window checks stay conservative and reported savings get smaller rather
        than larger. Ten of the thirteen over-estimate.

        Note 1920x1080, 2560x1440 and 3840x2160 all measuring 1,564: that is the
        ceiling, independently confirmed on three sizes spanning a factor of four
        in area.
        """
        assert anthropic_image_tokens(width, height) == pytest.approx(measured, rel=0.06)

    def test_a_tiny_image_costs_more_than_its_area_suggests(self) -> None:
        # Measured: 64x64 bills 13 tokens where area/750 predicts 5. Returning 5
        # would be the same understating error as returning 0, just smaller.
        assert anthropic_image_tokens(64, 64) == pytest.approx(13, abs=4)

    def test_a_degenerate_size_does_not_divide_by_zero(self) -> None:
        assert anthropic_image_tokens(0, 0) >= 0
        assert anthropic_image_tokens(1, 1) > 0


class TestOpenAIUsesItsPublishedFormula:
    """Unmeasured here, and it says so -- the ``TOOL_SCHEMA_CALIBRATION`` precedent.

    There is no free exact counter on that side, so this is the vendor's tile
    formula rather than a calibration. It is still far closer than the zero it
    replaces.
    """

    def test_low_detail_is_a_flat_cost(self) -> None:
        assert openai_image_tokens(2000, 2000, detail="low") == 85
        assert openai_image_tokens(64, 64, detail="low") == 85

    def test_high_detail_scales_with_tile_count(self) -> None:
        small = openai_image_tokens(512, 512, detail="high")
        large = openai_image_tokens(2048, 2048, detail="high")

        assert large > small
        assert small == 85 + 170  # one 512x512 tile

    def test_low_detail_is_the_cheaper_option(self) -> None:
        # The whole basis of the deferred reduction lever in decision 3.
        assert openai_image_tokens(1024, 1024, detail="low") < openai_image_tokens(
            1024, 1024, detail="high"
        )


class TestAnUncountableImageIsNotFree:
    def test_a_url_sourced_image_gets_the_documented_constant(self) -> None:
        """Absence is not zero -- the rule the core applies to its signals.

        An ``http`` URL carries no bytes to parse. Those are overwhelmingly
        photographs and screenshots, which sit at the cap, so the constant is a
        central estimate for that population rather than a worst case.
        """
        raw = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        }
        message = Message(role="user", content="What is this?", extra={RAW_CONTENT_KEY: raw})

        assert message_image_tokens(message, model="gpt-4o") == UNKNOWN_IMAGE_TOKENS

    def test_an_undecodable_payload_gets_the_constant_too(self) -> None:
        message = _vision_message(b"not really a png")

        assert message_image_tokens(message, model="claude-haiku-4-5") == UNKNOWN_IMAGE_TOKENS

    def test_the_constant_is_not_zero(self) -> None:
        # The assertion that would have caught the original bug.
        assert UNKNOWN_IMAGE_TOKENS > 0


class TestTheCountReachesTheRequest:
    def test_a_vision_message_no_longer_counts_eight_tokens(self) -> None:
        """The bug, as a number. ``count_request`` returned 8 for this.

        A data URL is used rather than a bare base64 blob so the assertion does
        not depend on the constant for unknown dimensions -- this must be the
        *parsed* count.
        """
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=(_vision_message(_png(1092, 1092)),),
            temperature=0.0,
        )

        total = count_request(request, HeuristicCounter())

        assert total > 1_400, f"a 1092x1092 image should cost ~1,525 tokens, counted {total}"

    def test_a_text_only_message_is_unchanged(self) -> None:
        plain = Message(role="user", content="hello there")

        assert count_message(plain, HeuristicCounter(), "claude-haiku-4-5") == count_message(
            plain, HeuristicCounter(), "gpt-4o"
        )

    def test_two_images_cost_about_twice_one(self) -> None:
        one = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(_png(512, 512)).decode(),
            },
        }
        raw = {"role": "user", "content": [{"type": "text", "text": "Compare."}, one, one]}
        pair = Message(role="user", content="Compare.", extra={RAW_CONTENT_KEY: raw})

        assert message_image_tokens(pair, model="claude-haiku-4-5") == pytest.approx(
            2 * anthropic_image_tokens(512, 512)
        )

    def test_the_window_check_now_sees_the_image(self) -> None:
        """The defect that actually mattered.

        ``fits_in_window`` guards a few percent of estimator error with a 1.15x
        margin while a vision request was under-counted by two orders of
        magnitude, so it would happily declare a request fits a window it cannot
        fit -- the provider-side rejection that function exists to prevent.
        """
        from optio_optimize.tokens import fits_in_window

        counter = HeuristicCounter()
        request = LLMRequest(
            model="claude-haiku-4-5",
            messages=tuple(_vision_message(_png(1092, 1092)) for _ in range(10)),
            temperature=0.0,
        )

        assert not fits_in_window(count_request(request, counter), 5_000, counter)


class TestNoImageContentIsRetained:
    def test_the_counter_returns_only_a_number(self) -> None:
        """Section 10 again. Counting must not stash or log the payload.

        Weak as a test and worth keeping as a statement of the contract: the
        risk here is a future edit that memoizes on the raw base64, which is
        exactly what ``MemoizingCounter`` deliberately does not do.
        """
        message = _vision_message(_png(256, 256))

        result = message_image_tokens(message, model="claude-haiku-4-5")

        assert isinstance(result, int)
