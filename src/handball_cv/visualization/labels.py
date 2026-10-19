"""Player labels under each box, drawn where they land rather than over the whole frame.

The picture is `sv.RichLabelAnnotator`'s at `Position.BOTTOM_CENTER`: a box in
the player's palette colour, `TEXT_PADDING` around white text, hung `OFFSET`
below the bottom-centre of the player box. The annotator gets there by turning
the whole frame into a PIL image and back on every call -- ~4 of its 5 ms a
1080p frame, for ~13 labels that cover a fraction of a percent of it. This draws
each label with the same PIL calls on a crop just big enough to hold it, so the
pixels are the same and only the conversion is gone. Labels are drawn in order,
each onto the frame as the previous ones left it, so overlaps resolve as before.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

__all__ = ["LabelDrawer"]

TEXT_PADDING = 10
OFFSET = (0, 10)
TEXT_BGR = (255, 255, 255)


class _Layout(NamedTuple):
    """One label text measured once: everything but where it goes."""
    width: int        # background box, as RichLabelAnnotator sizes it
    height: int
    lines: tuple      # (line, y offset from the text origin)
    ink: tuple        # (left, top, right, bottom) of the glyphs, from the text origin


class LabelDrawer:
    """`draw(frame_bgr, boxes_xyxy, labels, color_lookup)` puts one label under each box."""

    def __init__(self, font_path: Path, palette: sv.ColorPalette, frame_width: int) -> None:
        font_path = Path(font_path)
        if not font_path.is_file():
            # RichLabelAnnotator fell back to PIL's default font here, silently.
            raise FileNotFoundError(f"font not found: {font_path}")
        self.font = ImageFont.truetype(str(font_path), round(34 * max(frame_width / 1920.0, 1.0)))
        self.palette = palette
        self._measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        # Every label starts from the "Tg" origin, as in RichLabelAnnotator.
        self._origin = self._measure.textbbox((0, 0), "Tg", font=self.font)[:2]
        # The same few hundred texts recur every frame; lay each out once.
        self._layout = lru_cache(maxsize=4096)(self._lay_out)

    def draw(self, frame_bgr: np.ndarray, boxes_xyxy, labels, color_lookup) -> None:
        """Draw in place, in order, so a later label overlaps an earlier one."""
        for box, text, color in zip(np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4),
                                    labels, color_lookup, strict=True):
            self._draw_one(frame_bgr, box, text, self.palette.by_idx(int(color)).as_bgr())

    def _lay_out(self, text: str) -> _Layout:
        """RichLabelAnnotator's measurement: lines split as `wrap_text` splits
        them with no line length, stacked `TEXT_PADDING` apart."""
        lines = text.splitlines() or [""]
        boxes = [self._measure.textbbox((0, 0), line, font=self.font) for line in lines]
        width = int(max(right - left for left, _, right, _ in boxes) + 2 * TEXT_PADDING)
        height = int(sum(bottom - top for _, top, _, bottom in boxes)
                     + (len(lines) - 1) * TEXT_PADDING + 2 * TEXT_PADDING)
        placed, ink, y = [], [], 0
        for line, (left, top, right, bottom) in zip(lines, boxes):
            placed.append((line, y))
            ink.append((left, y + top, right, y + bottom))
            y += (bottom - top) + TEXT_PADDING
        ink_box = (min(i[0] for i in ink), min(i[1] for i in ink),
                   max(i[2] for i in ink), max(i[3] for i in ink))
        return _Layout(width, height, tuple(placed), ink_box)

    def _draw_one(self, frame_bgr: np.ndarray, box, text: str, fill_bgr: tuple) -> None:
        layout = self._layout(text)
        # RichLabelAnnotator's anchor and background rectangle, inclusive corners.
        cx = int((box[0] + box[2]) / 2) + OFFSET[0]
        cy = int(box[3]) + OFFSET[1]
        rect = (cx - layout.width // 2, cy, cx + layout.width // 2, cy + layout.height)
        tx = rect[0] + TEXT_PADDING - self._origin[0]
        ty = rect[1] + TEXT_PADDING - self._origin[1]

        # Everything this label can touch, clipped to the frame. A margin costs
        # nothing -- untouched pixels round-trip unchanged -- and covers glyph
        # anti-aliasing that strays a pixel past its measured box.
        frame_h, frame_w = frame_bgr.shape[:2]
        x0 = max(0, int(np.floor(min(rect[0], tx + layout.ink[0]))) - 2)
        y0 = max(0, int(np.floor(min(rect[1], ty + layout.ink[1]))) - 2)
        x1 = min(frame_w, int(np.ceil(max(rect[2], tx + layout.ink[2]))) + 3)
        y1 = min(frame_h, int(np.ceil(max(rect[3], ty + layout.ink[3]))) + 3)
        if x0 >= x1 or y0 >= y1:
            return
        # Drawn in BGR channel order throughout: every channel is computed on
        # its own, so swapping the fill colours instead of the pixels is exact.
        crop = Image.fromarray(np.ascontiguousarray(frame_bgr[y0:y1, x0:x1]))
        draw = ImageDraw.Draw(crop)
        draw.rounded_rectangle((rect[0] - x0, rect[1] - y0, rect[2] - x0, rect[3] - y0),
                               radius=0, fill=fill_bgr, outline=None)
        for line, dy in layout.lines:
            draw.text((tx - x0, ty + dy - y0), line, font=self.font, fill=TEXT_BGR)
        frame_bgr[y0:y1, x0:x1] = np.asarray(crop)
