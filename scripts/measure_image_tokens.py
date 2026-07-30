"""Held-out validation: sizes the estimator was NOT calibrated on.

Asserting against the calibration table only proves the constants were copied
correctly. count_tokens is free, so there is no excuse for not testing the
estimator on data it has never seen.
"""

import base64
import os
import pathlib
import struct
import sys
import zlib

ROOT = pathlib.Path("c:/Users/Hp/Desktop/Side Project")
sys.path.insert(0, str(ROOT / "src"))

for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
    if "=" in line:
        name, value = line.split("=", 1)
        if name.strip() in {"ANTHROPIC_API_KEY", "ANTHROPIC_KEY"}:
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("\"'")

from anthropic import Anthropic  # noqa: E402

from optio_optimize.images import anthropic_image_tokens  # noqa: E402

MODEL = "claude-haiku-4-5"

# None of these appear in the calibration table.
HELD_OUT = [
    (128, 96),
    (320, 240),
    (640, 480),
    (900, 700),
    (1024, 768),
    (1366, 768),
    (1440, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
    (600, 2400),
    (2400, 600),
    (100, 1500),
]


def png(w: int, h: int) -> bytes:
    """A real PNG of exactly these dimensions, with no image library."""

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes((x * 7 + y * 13) % 256 for x in range(w * 3)) for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(rows, 1))
        + chunk(b"IEND", b"")
    )


client = Anthropic()
base = client.messages.count_tokens(
    model=MODEL, messages=[{"role": "user", "content": [{"type": "text", "text": "Describe."}]}]
).input_tokens

print(f"{'w':>6} {'h':>6} {'measured':>9} {'estimate':>9} {'error':>8}")
print("-" * 44)
worst = 0.0
errors = []
for w, h in HELD_OUT:
    data = base64.b64encode(png(w, h)).decode()
    measured = (
        client.messages.count_tokens(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": data,
                            },
                        },
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
        ).input_tokens
        - base
    )
    estimate = anthropic_image_tokens(w, h)
    error = (estimate - measured) / measured
    errors.append(error)
    worst = max(worst, abs(error))
    print(f"{w:>6} {h:>6} {measured:>9,} {estimate:>9,} {error:>+7.1%}")

print("-" * 44)
print(f"worst absolute error: {worst:.1%}")
print(f"mean signed error:    {sum(errors) / len(errors):+.1%}  (positive = over-estimates)")
print(f"over-estimates: {sum(1 for e in errors if e >= 0)}/{len(errors)}")
