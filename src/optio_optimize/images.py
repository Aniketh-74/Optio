"""What an image costs, measured rather than taken from the documentation.

An image contributed **zero** tokens to every count this package made. The
consequence is narrower than it first looks and worth stating precisely, because
the obvious version of the claim is wrong: ``pipeline`` uses
``response.input_tokens`` on a live call -- the provider's own number, which
counts images -- and ``baseline = actual + saved``, so image tokens sit on both
sides of ``reduction_ratio`` and the headline percentage was never inflated by
this.

What it did break:

* :func:`~optio_optimize.tokens.fits_in_window` applies a 1.15x safety margin to
  guard a few percent of estimator error, on the stated grounds that
  under-counting a *limit* decision causes a provider-side rejection the user
  sees as a crash. A vision request was under-counted by two orders of magnitude,
  so that guard was answering the wrong question.
* A short-circuited request never reaches a provider, so this estimate is the
  only number available -- a vision cache hit reported ~8 tokens avoided against
  a true ~1,535.
* Nothing in the package could reason about image cost at all.

**The Anthropic numbers here are measurements.** ``messages.count_tokens`` is
exact and free, so the estimator was calibrated against synthetic PNGs of known
size with a text-only baseline differenced out, on ``claude-haiku-4-5``:

===========  ==========  =============  ============  ======
w x h        pixels      image tokens   ``w*h/750``   ratio
===========  ==========  =============  ============  ======
64 x 64      4,096       13             5             2.38
200 x 200    40,000      68             53            1.27
512 x 512    262,144     365            350           1.04
800 x 600    480,000     642            640           1.00
784 x 784    614,656     788            820           0.96
1000 x 1000  1,000,000   1,300          1,333         0.98
1200 x 958   1,149,600   1,509          1,533         0.98
1092 x 1092  1,192,464   1,525          1,590         0.96
1568 x 1568  2,458,624   1,525          3,278         0.47
1568 x 784   1,229,312   1,572          1,639         0.96
2000 x 1000  2,000,000   1,572          2,667         0.59
400 x 3000   1,200,000   452            1,600         0.28
===========  ==========  =============  ============  ======

Measuring rather than trusting the published ``(w*h)/750`` is what keeps this
module from being wrong: the bare formula is accurate to +/-4% from 512x512
through 1200x958 and then **breaks**, overstating a full-size square image by
2.15x and a thin one by 3.6x, because two caps apply that it does not mention.
Small images cost *more* than it says.

The residual error is quantized rather than smooth -- ``1092x1092 -> 1,525`` and
``784x1568 -> 1,572`` both land on exactly ``w*h/782`` while 800x600 and 512x512
do not, which is the signature of patch-based processing. Chasing that with a
curve fit would be false precision, so this errs high and says so.

**Validated on held-out sizes**, because a table the constants were fitted to
proves only that they were copied correctly. Thirteen common screen and camera
dimensions (128x96 through 3840x2160, plus three extreme aspect ratios), measured
afterwards and none of them used to choose a constant: worst absolute error
**5.5%**, mean signed error **+1.4%**, ten of thirteen over-estimating. Biased
high is the safe direction in both places this number is used -- window checks
stay conservative, and a larger denominator can only make a reported saving
smaller. ``scripts/measure_image_tokens.py`` re-runs it for free.

**No image library.** Dimensions come off the header, which every format in use
puts in a fixed early field. Pillow is a large wheel with native code and §4.4
exists to keep it out of a dependency tree for four integers.
"""

from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Any

from optio_optimize.wire import RAW_CONTENT_KEY, is_text_block

if TYPE_CHECKING:
    from optio_optimize.types import Message

#: Pixels per token in Anthropic's own published formula, and accurate to +/-4%
#: in the mid-range of the table above.
PIXELS_PER_TOKEN = 750

#: Longest edge the provider keeps. Above this the image is downscaled before it
#: is tokenized, which is the only thing that explains the thin images:
#: ``400x3000`` bills 452 tokens where its raw area predicts 1,600.
MAX_EDGE_PIXELS = 1568

#: Ceiling for one image, from the measured table -- the largest observed value
#: is 1,572 and nothing exceeded it however large the input. Rounded up rather
#: than down: over-estimating is safe in both directions that matter here, since
#: it makes window checks conservative and savings ratios smaller.
MAX_IMAGE_TOKENS = 1600

#: Flat per-image cost on top of area. Supported by the two smallest
#: measurements, which area alone under-predicts: ``64x64`` bills 13 against a
#: predicted 5, and ``200x200`` bills 68 against 53. Plausibly the block framing
#: itself. Immaterial above a few hundred pixels, and the alternative is
#: under-counting small images by a fifth.
IMAGE_BLOCK_OVERHEAD_TOKENS = 8

#: What an image whose dimensions cannot be read is counted as. **Not zero** --
#: that is the bug this module fixes, and absence is not zero (the same rule the
#: core applies to its signals).
#:
#: Sized at the cap deliberately. Unreadable dimensions means a URL-sourced
#: image essentially every time, and those are photographs and screenshots
#: rather than icons, so the cap is a *central* estimate for that population and
#: not a worst case. Erring high is also the safe direction twice over: window
#: checks stay conservative, and a larger denominator can only make this
#: package's reported saving smaller.
UNKNOWN_IMAGE_TOKENS = MAX_IMAGE_TOKENS

#: OpenAI's published figures: a flat cost at ``detail: "low"``, and a base plus
#: per-tile cost at ``detail: "high"`` after fitting the image to 2048x2048 and
#: then its shortest side to 768.
#:
#: **Unmeasured here, unlike the Anthropic side**, because there is no free exact
#: counter to calibrate against. Stated the way
#: :data:`~optio_optimize.tokens.TOOL_SCHEMA_CALIBRATION` states its own limits:
#: this is the vendor's formula rather than an observation, it is not right for
#: every model in their range, and it is still far closer than the zero it
#: replaces.
OPENAI_BASE_TOKENS = 85
OPENAI_TILE_TOKENS = 170
_OPENAI_MAX_SQUARE = 2048
_OPENAI_SHORT_SIDE = 768
_OPENAI_TILE = 512


def anthropic_image_tokens(width: int, height: int) -> int:
    """Estimate what Anthropic bills for one image of these dimensions.

    Applies the two caps the published formula omits: the longest edge is scaled
    down to :data:`MAX_EDGE_PIXELS` first, and the result is held under
    :data:`MAX_IMAGE_TOKENS`. Accurate to within 6% of every measurement in this
    module's table above 512px, and a few tokens out below it.
    """
    if width <= 0 or height <= 0:
        return 0
    scale = min(1.0, MAX_EDGE_PIXELS / max(width, height))
    area = (width * scale) * (height * scale)
    return min(MAX_IMAGE_TOKENS, int(area / PIXELS_PER_TOKEN) + IMAGE_BLOCK_OVERHEAD_TOKENS)


def openai_image_tokens(width: int, height: int, *, detail: str = "high") -> int:
    """Estimate what OpenAI bills for one image, by their published formula.

    ``detail: "low"`` is a flat cost regardless of size, which is the basis of
    the reduction lever ADR-022 deliberately does **not** ship: it degrades what
    the model can see, making it ``ALTERED``, and ADR-015 wants a vision accuracy
    probe before anything like that is turned on.
    """
    if detail == "low":
        return OPENAI_BASE_TOKENS
    if width <= 0 or height <= 0:
        return 0
    scale = min(1.0, _OPENAI_MAX_SQUARE / max(width, height))
    width, height = int(width * scale), int(height * scale)
    shortest = min(width, height)
    if shortest > _OPENAI_SHORT_SIDE:
        shrink = _OPENAI_SHORT_SIDE / shortest
        width, height = int(width * shrink), int(height * shrink)
    tiles = -(-width // _OPENAI_TILE) * -(-height // _OPENAI_TILE)
    return OPENAI_BASE_TOKENS + OPENAI_TILE_TOKENS * tiles


def image_dimensions(payload: bytes) -> tuple[int, int] | None:
    """Read ``(width, height)`` off an image header, or ``None``.

    PNG, GIF, JPEG and WebP -- every format the vision APIs accept. Each keeps
    its dimensions in a fixed early field or an early marker, so this needs no
    decoder and no dependency (§4.4).

    ``None`` for anything unrecognized, truncated or corrupt. Never raises: this
    runs inside a stage, and ADR-013 rule 1 says a caller's request survives
    whatever this package fails to understand.
    """
    try:
        return _dimensions(payload)
    except (struct.error, IndexError, ValueError):
        # A truncated or malformed header, which is a normal thing to be handed
        # and not a reason to lose the request.
        return None


def _dimensions(payload: bytes) -> tuple[int, int] | None:
    """The per-format header reads. Wrapped by :func:`image_dimensions`."""
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(payload) < 24:
            return None
        width, height = struct.unpack(">II", payload[16:24])
        return (width, height) if width and height else None

    if payload.startswith((b"GIF87a", b"GIF89a")):
        if len(payload) < 10:
            return None
        width, height = struct.unpack("<HH", payload[6:10])
        return (width, height) if width and height else None

    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return _webp_dimensions(payload)

    if payload.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(payload)

    return None


def _webp_dimensions(payload: bytes) -> tuple[int, int] | None:
    """WebP keeps its size in whichever of three chunk types comes first."""
    kind = payload[12:16]
    if kind == b"VP8X" and len(payload) >= 30:
        # Canvas size is stored as 24-bit little-endian, minus one.
        width = int.from_bytes(payload[24:27], "little") + 1
        height = int.from_bytes(payload[27:30], "little") + 1
        return (width, height)
    if kind == b"VP8 " and len(payload) >= 30:
        # 14 significant bits each, after a 3-byte start code.
        width = int.from_bytes(payload[26:28], "little") & 0x3FFF
        height = int.from_bytes(payload[28:30], "little") & 0x3FFF
        return (width, height) if width and height else None
    if kind == b"VP8L" and len(payload) >= 25:
        bits = int.from_bytes(payload[21:25], "little")
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    return None


#: Start-of-frame markers carrying dimensions. ``0xC4``, ``0xC8`` and ``0xCC``
#: share the range and are Huffman/arithmetic tables rather than frame headers,
#: so reading dimensions from one would return two bytes of a coding table.
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg_dimensions(payload: bytes) -> tuple[int, int] | None:
    """Walk JPEG segments to the first frame header.

    JPEG has no fixed dimension offset -- a file may carry any number of
    application, comment and table segments first -- so this is a scan rather
    than a slice. Bounded by the payload length, and only the header needs to be
    present.
    """
    offset = 2
    end = len(payload)
    while offset + 4 <= end:
        if payload[offset] != 0xFF:
            # Not on a marker boundary; a padded or malformed file.
            return None
        marker = payload[offset + 1]
        if marker in _JPEG_SOF:
            if offset + 9 > end:
                return None
            height, width = struct.unpack(">HH", payload[offset + 5 : offset + 9])
            return (width, height) if width and height else None
        length = struct.unpack(">H", payload[offset + 2 : offset + 4])[0]
        if length < 2:
            return None
        offset += 2 + length
    return None


def message_image_tokens(message: Message, *, model: str) -> int:
    """Total image tokens carried by one message's content blocks.

    Reads the adapter's stashed original param, since ``Message.content`` holds
    only extracted text -- the same place :func:`~optio_optimize.cache
    .non_text_digest` reads, and for the same reason.

    Args:
        message: The message to measure.
        model: Model name, which selects the provider's formula. Anthropic's side
            is calibrated against measurements; OpenAI's is their published tile
            formula and is not.

    Returns:
        Estimated image tokens, ``0`` when the message carries no image. An image
        whose dimensions cannot be read counts :data:`UNKNOWN_IMAGE_TOKENS`
        rather than zero.
    """
    raw = message.extra.get(RAW_CONTENT_KEY)
    if not isinstance(raw, dict):
        return 0
    content = raw.get("content")
    if not isinstance(content, list):
        return 0
    anthropic = not model.startswith(("gpt", "o1", "o3", "o4"))
    return sum(
        _block_image_tokens(block, anthropic=anthropic)
        for block in content
        if not is_text_block(block)
    )


def _block_image_tokens(block: Any, *, anthropic: bool) -> int:
    """One block's image cost, or ``0`` when it is not an image."""
    payload = _image_payload(block)
    if payload is None:
        return 0
    if not payload:
        # An image we can see but cannot measure -- a URL, or a format with no
        # readable header. Absence is not zero.
        return UNKNOWN_IMAGE_TOKENS
    size = image_dimensions(payload)
    if size is None:
        return UNKNOWN_IMAGE_TOKENS
    width, height = size
    if anthropic:
        return anthropic_image_tokens(width, height)
    return openai_image_tokens(width, height, detail=_openai_detail(block))


def _image_payload(block: Any) -> bytes | None:
    """Decoded image bytes, ``b""`` for an unreadable image, ``None`` for a non-image.

    The three-way return distinguishes "not an image" from "an image this cannot
    measure", which must be counted rather than skipped.
    """
    data = block if isinstance(block, dict) else _as_dict(block)
    if not isinstance(data, dict):
        return None
    kind = data.get("type")
    if kind == "image":
        source = data.get("source")
        if not isinstance(source, dict):
            return b""
        if source.get("type") == "base64":
            return _decode(source.get("data"))
        # A URL source, or Anthropic's `file` source: no bytes to read here.
        return b""
    if kind in {"image_url", "input_image"}:
        url = data.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str) and url.startswith("data:"):
            _, _, encoded = url.partition(",")
            return _decode(encoded)
        return b""
    return None


def _openai_detail(block: Any) -> str:
    """The ``detail`` a caller asked for, defaulting to OpenAI's own default."""
    data = block if isinstance(block, dict) else _as_dict(block)
    if isinstance(data, dict):
        url = data.get("image_url")
        if isinstance(url, dict) and isinstance(url.get("detail"), str):
            return str(url["detail"])
        if isinstance(data.get("detail"), str):
            return str(data["detail"])
    return "high"


#: Base64 characters decoded when looking for a header. A multiple of 4, so the
#: prefix decodes cleanly on its own.
#:
#: Only the header is wanted, and decoding a 5 MB screenshot to read four
#: integers would put the cost of this module on every request. 12 KB is
#: generous: PNG needs 24 bytes and GIF 10, while JPEG has no fixed offset and
#: must be scanned past any number of application and comment segments -- an
#: EXIF block alone can run to 64 KB, so a JPEG carrying an unusually large one
#: yields no dimensions and is counted as unknown rather than as zero.
_HEADER_BASE64_CHARS = 16_384


def _decode(encoded: object) -> bytes:
    """Base64 to bytes, or ``b""`` when it will not decode.

    Decodes only a prefix, so a payload truncated in transit still yields
    dimensions.
    """
    if not isinstance(encoded, str):
        return b""
    try:
        return base64.b64decode(encoded[:_HEADER_BASE64_CHARS], validate=False)
    except (ValueError, TypeError):
        return b""


def _as_dict(block: object) -> dict[str, Any] | None:
    """An SDK block object as a plain dict, when it offers one."""
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        result = dump()
        return result if isinstance(result, dict) else None
    return None
