"""A top-down court with every tracked player on it, for compositing into a frame.

Each player is a dot in their palette colour carrying their jersey number, at
the court position the bundle recorded for them. The camera's view -- the frame
border pushed through that frame's court fit -- is drawn as an outline, the same
polygon `scripts.render_court_view` draws. When the court fit abstains the
minimap says so and draws no players: a guessed position is worse than none.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["CourtMinimap"]

# How far outside the court lines a player may still be drawn, in cm. The bench
# and the zone behind the goal lines are a couple of metres out; anything
# further is an extrapolation of the fit rather than a place someone stands.
DRAW_MARGIN_CM = 400
LINE_COLOUR = (255, 220, 0)       # BGR, the court-view render's line colour
DIM = 0.35


class CourtMinimap:
    """Background drawn once; `draw` returns a fresh BGR image per frame."""

    def __init__(self, width_px: int, palette_bgr: list[tuple[int, int, int]], padding_cm: float = 300.0):
        from sports import MeasurementUnit
        from sports.handball import CourtConfiguration, League, draw_court

        config = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)
        self.length, self.width_cm = float(config.court_length), float(config.court_width)
        self.scale = width_px / (self.length + 2 * padding_cm)
        self.pad = padding_cm * self.scale
        background = draw_court(config, scale=self.scale, padding=round(self.pad),
                                line_thickness=max(1, round(width_px / 400)))
        self.pad = float(round(self.pad))  # draw_court takes integer padding
        self.background = background
        self.palette = palette_bgr
        self.radius = max(4, round(width_px / 70))
        self.font_scale = max(0.4, width_px / 950)

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.background.shape[:2]
        return width, height

    def to_pixels(self, court_xy: np.ndarray) -> np.ndarray:
        """Court cm -> minimap px; y is flipped, as `draw_court` draws it."""
        court_xy = np.asarray(court_xy, dtype=np.float64).reshape(-1, 2)
        x = court_xy[:, 0] * self.scale + self.pad
        y = (self.width_cm - court_xy[:, 1]) * self.scale + self.pad
        return np.stack([x, y], axis=1)

    def draw(self, fit, frame_size: tuple[int, int], players) -> np.ndarray:
        """`players`: iterable of `(court_x, court_y, palette_index, label_or_None)`."""
        image = self.background.copy()
        if fit is None or not fit.usable:
            image = (image * DIM).astype(np.uint8)
            cv2.putText(image, "no court fit", (10, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale * 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            return image

        self._draw_view(image, fit, frame_size)
        for court_x, court_y, color, label in players:
            if not (np.isfinite(court_x) and np.isfinite(court_y)):
                continue
            if not (-DRAW_MARGIN_CM <= court_x <= self.length + DRAW_MARGIN_CM
                    and -DRAW_MARGIN_CM <= court_y <= self.width_cm + DRAW_MARGIN_CM):
                continue
            x, y = (int(round(v)) for v in self.to_pixels([(court_x, court_y)])[0])
            cv2.circle(image, (x, y), self.radius, self.palette[int(color)], -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), self.radius, (20, 20, 20), 1, cv2.LINE_AA)
            if label:
                cv2.putText(image, label, (x + self.radius + 1, y - self.radius), cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(image, label, (x + self.radius + 1, y - self.radius), cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def _draw_view(self, image: np.ndarray, fit, frame_size: tuple[int, int]) -> None:
        """The frame border on the court: what the camera can see right now."""
        width, height = frame_size
        steps = np.linspace(0.0, 1.0, 40)
        border = np.concatenate([
            np.stack([steps * width, np.full_like(steps, height)], 1),
            np.stack([np.full_like(steps, width), (1 - steps) * height], 1),
            np.stack([(1 - steps) * width, np.zeros_like(steps)], 1),
            np.stack([np.zeros_like(steps), steps * height], 1),
        ])
        on_court = fit.to_court(border)
        sane = np.all(np.abs(on_court - (self.length / 2, self.width_cm / 2)) < 3 * self.length, axis=1)
        if sane.sum() > 3:
            polygon = self.to_pixels(on_court[sane]).round().astype(np.int32)
            cv2.polylines(image, [polygon], True, LINE_COLOUR, 1, cv2.LINE_AA)

    def composite(self, frame_bgr: np.ndarray, minimap: np.ndarray, corner: str = "top-right",
                  margin: int = 24) -> None:
        """Paste `minimap` into a corner of `frame_bgr`, in place.

        Opaque on purpose: broadcast graphics (score, clock) often sit in the
        same corner, and blending lets them bleed through the court.
        """
        h, w = minimap.shape[:2]
        frame_h, frame_w = frame_bgr.shape[:2]
        x0 = margin if corner.endswith("left") else frame_w - w - margin
        y0 = margin if corner.startswith("top") else frame_h - h - margin
        frame_bgr[y0:y0 + h, x0:x0 + w] = minimap
        cv2.rectangle(frame_bgr, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (20, 20, 20), 1)
