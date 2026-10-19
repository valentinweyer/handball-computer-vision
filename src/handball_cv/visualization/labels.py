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

from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

__all__ = ["LabelDrawer"]

TEXT_PADDING = 10
OFFSET = (0, 10)
TEXT_BGR = (255, 255, 255)


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
        # Every line starts from the "Tg" origin, as in RichLabelAnnotator.
        self._origin = self._measure.textbbox((0, 0), "Tg", font=self.font)[:2]

    def draw(self, frame_bgr: np.ndarray, boxes_xyxy, labels, color_lookup) -> None:
        """Draw in place, in order, so a later label overlaps an earlier one."""
        for box, text, color in zip(np.asarray(boxes_xyxy, dtype=float), labels, color_lookup):
            self._draw_one(frame_bgr, box, text, self.palette.by_idx(int(color)).as_bgr())

    def _draw_one(self, frame_bgr: np.ndarray, box, text: str, fill_bgr: tuple) -> None:
        if len(text.splitlines()) > 1:
            raise ValueError(f"labels are one line; got {text!r}")
        left, top, right, bottom = self._measure.textbbox((0, 0), text, font=self.font)
        width = int((right - left) + 2 * TEXT_PADDING)
        height = int((bottom - top) + 2 * TEXT_PADDING)
        # RichLabelAnnotator's anchor and background rectangle, inclusive corners.
        cx = int((box[0] + box[2]) / 2) + OFFSET[0]
        cy = int(box[3]) + OFFSET[1]
        rect = (cx - width // 2, cy, cx + width // 2, cy + height)
        text_xy = (rect[0] + TEXT_PADDING - self._origin[0], rect[1] + TEXT_PADDING - self._origin[1])
        ink = self._measure.textbbox(text_xy, text, font=self.font)

        # Everything this label can touch, clipped to the frame. A margin costs
        # nothing -- untouched pixels round-trip unchanged -- and covers glyph
        # anti-aliasing that strays a pixel past its measured box.
        frame_h, frame_w = frame_bgr.shape[:2]
        x0 = max(0, int(np.floor(min(rect[0], ink[0]))) - 2)
        y0 = max(0, int(np.floor(min(rect[1], ink[1]))) - 2)
        x1 = min(frame_w, int(np.ceil(max(rect[2], ink[2]))) + 3)
        y1 = min(frame_h, int(np.ceil(max(rect[3], ink[3]))) + 3)
        if x0 >= x1 or y0 >= y1:
            return
        # Drawn in BGR channel order throughout: every channel is computed on
        # its own, so swapping the fill colours instead of the pixels is exact.
        crop = Image.fromarray(np.ascontiguousarray(frame_bgr[y0:y1, x0:x1]))
        draw = ImageDraw.Draw(crop)
        draw.rounded_rectangle((rect[0] - x0, rect[1] - y0, rect[2] - x0, rect[3] - y0),
                               radius=0, fill=fill_bgr, outline=None)
        draw.text((text_xy[0] - x0, text_xy[1] - y0), text, font=self.font, fill=TEXT_BGR)
        frame_bgr[y0:y1, x0:x1] = np.asarray(crop)
