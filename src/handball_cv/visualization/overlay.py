"""The per-player overlay every render draws: tinted masks, labels, the legend.

Shared by `scripts.render_full_pipeline` (McByte, SAM2 and the redraw) and
`scripts.render_result`. It lives here rather than in either script so that
drawing a finished bundle does not import the trackers, torch and the team
model only to reach these few functions -- ~5 s of every render's start-up.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from handball_cv.tracking.geometry_cache import as_crop
from handball_cv.visualization.labels import LabelDrawer
from handball_cv.visualization.mask_overlay import tint_masks
from handball_cv.visualization.palette import LEGEND, TEAM_A, TEAM_B

__all__ = [
    "DEFAULT_FONT", "MASK_OPACITY", "annotate_frame", "draw_legend",
    "load_roster", "number_label", "roster_name",
]

DEFAULT_FONT = Path(__file__).resolve().parents[3] / "notebooks" / "fonts" / "ConcertOne-Regular.ttf"
MASK_OPACITY = 0.5


def load_roster(path: Path | None) -> dict:
    """-> {team_index_or_None: {number: name}}. Empty dict when unavailable."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if all(isinstance(v, dict) for v in data.values()):
        by_team = {"A": TEAM_A, "B": TEAM_B, "0": TEAM_A, "1": TEAM_B}
        return {
            by_team.get(str(key).upper(), None): {str(n): v for n, v in value.items()}
            for key, value in data.items()
        }
    return {None: {str(k): v for k, v in data.items()}}


def roster_name(roster: dict, team_index: int, number: str) -> str | None:
    if not roster:
        return None
    for key in (team_index, None):
        if key in roster and number in roster[key]:
            return roster[key][number]
    return None


def number_label(number, color: int, roster: dict, player_id: int) -> str:
    """The one place a drawn label is spelled, shared by both drawing passes."""
    if number is None:
        return f"P{player_id}"
    name = roster_name(roster, color, number)
    return f"#{number} {name}" if name else f"#{number}"


def draw_legend(frame: np.ndarray, palette: sv.ColorPalette) -> None:
    height, width = frame.shape[:2]
    scale = max(width / 1920.0, 1.0)
    x, y = round(18 * scale), height - round(18 * scale)
    chip = round(14 * scale)
    for index, label in reversed(LEGEND):
        color = palette.by_idx(index).as_bgr()
        cv2.rectangle(frame, (x, y - chip), (x + chip, y), color, -1)
        cv2.rectangle(frame, (x, y - chip), (x + chip, y), (20, 20, 20), 1)
        cv2.putText(
            frame, label, (x + chip + round(6 * scale), y - round(2 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255),
            max(1, round(scale)), cv2.LINE_AA,
        )
        y -= chip + round(10 * scale)


def annotate_frame(
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    color_lookup,
    masks,
    labels: list,
    palette: sv.ColorPalette,
    label_drawer: LabelDrawer,
) -> np.ndarray:
    """Team-tinted mask fill plus one already-decided label per player.

    Shared by all three drawing paths -- McByte, SAM2, and the redraw -- so it
    takes the palette index directly rather than a PlayerRecord, and the label
    text rather than a voter. Deciding what to draw is the caller's job; this
    only draws it, which is what lets the redraw pass reuse it with verdicts
    replayed from a cache and no live tracker state at all.

    `masks` holds one entry per player, anything `as_crop` takes: a `MaskCrop`
    (what the geometry cache stores, and the cheap form to draw), a full-frame
    mask (SAM2's `(N, H, W)` stack, McByte's mask manager), or None where
    McByte's mask manager has no mask for that tracklet this frame. Every
    argument is per player, and lengths that disagree are refused rather than
    drawn with the wrong player's colour or without a label.
    """
    counts = {"boxes": len(boxes_xyxy), "colors": len(color_lookup),
              "masks": len(masks), "labels": len(labels)}
    if len(set(counts.values())) > 1:
        raise ValueError(f"one entry per player in each of {counts}")
    annotated = frame_bgr.copy()
    if not len(boxes_xyxy):
        return annotated

    tint_masks(annotated, [as_crop(mask) for mask in masks],
               [palette.by_idx(int(c)).as_bgr() for c in color_lookup], opacity=MASK_OPACITY)
    label_drawer.draw(annotated, boxes_xyxy, labels, color_lookup)
    return annotated
