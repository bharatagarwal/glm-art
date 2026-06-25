"""Rasterize a word into a blocky 2D bitmap — the raw stone before the river."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    # A heavy, blocky face reads best as "raw stone": thick strokes, minimal
    # finesse, so the erosion has real mass to chew through.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rasterize_word(
    word: str,
    width: int,
    height: int,
    invert: bool = False,
) -> np.ndarray:
    """Render ``word`` into a boolean array of shape ``(height, width)``.

    Stone cells are ``True`` (the glyph), air is ``False``. The word is fit
    horizontally with a margin so the current has something to flow into before
    it meets the leading edge.
    """
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    # Pick the largest font size that still fits the word horizontally.
    lo, hi = 8, height
    best_font = _load_font(lo)
    for _ in range(24):
        mid = (lo + hi) // 2
        font = _load_font(mid)
        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= width and h <= height:
            best_font = font
            lo = mid
        else:
            hi = mid

    bbox = draw.textbbox((0, 0), word, font=best_font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (width - w) // 2 - bbox[0]
    y = (height - h) // 2 - bbox[1]
    draw.text((x, y), word, fill=255, font=best_font)

    arr = np.array(img, dtype=np.uint8) > 127
    if invert:
        arr = ~arr
    return arr
