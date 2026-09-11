"""Per-frame drawable geometry, so a render can be redrawn without tracking again.

The overlay needs four things per player per frame -- box, stable id, palette
colour and mask -- and none of them depend on what text is drawn. They cost a
full SAM2 pass (~0.46 s/frame) to produce and are identical no matter how the
labels are decided, so a run that keeps only the finished pixels has to redo
that pass to change a single label. This is the same bargain the reads cache
already makes one layer up: cache the expensive stage that does not change, so
iterating on the cheap stage that does costs seconds.

Why not `MaskCache`, which also stores per-frame masks: it flattens a frame to
one uint8 label map, so where two players overlap only the last one written
keeps those pixels. That is fine for sampling a torso colour, which is what it
was built for, but an overlay draws every player's mask and the redraw has to
match the original pixel for pixel -- and handball players overlap constantly.
Masks are therefore kept per player here, cropped to their own bounding box and
bit-packed, which is both exact and smaller: ~1.1 KB per player-frame, ~21 MB
for a 1499-frame clip with ~12.7 players live, against 2.07 MB per player-frame
unpacked (~40 GB for the same clip).

Layout is the flat-array-plus-offsets form `load_detection_cache` already uses:
`offsets[i]:offsets[i+1]` selects frame `i`'s rows out of the player arrays, and
`mask_offsets[r]:mask_offsets[r+1]` selects row `r`'s packed mask bytes.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = 1


@dataclass(frozen=True)
class FrameGeometry:
    """One frame's drawable rows, in the order the tracker returned them."""
    frame_idx: int
    player_ids: np.ndarray   # (N,) int
    boxes: np.ndarray        # (N, 4) xyxy float
    color_index: np.ndarray  # (N,) palette index the run drew this player with
    team_id: np.ndarray      # (N,) voted team, -1 where the player had none
    masks: np.ndarray        # (N, H, W) bool


def _crop_rect(mask: np.ndarray) -> tuple:
    """(x0, y0, w, h) tight around `mask`, or a zero rect if it is empty.

    `any` along each axis rather than `nonzero` over the whole mask: the same
    reason the centroid stopped materialising every pixel coordinate, and this
    runs once per player per frame.
    """
    rows = mask.any(axis=1)
    if not rows.any():
        return 0, 0, 0, 0
    cols = mask.any(axis=0)
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


class GeometryCache:
    """Accumulated by a tracking pass, then replayed by a drawing pass."""

    def __init__(self, width: int, height: int, total_frames: int):
        self.width = int(width)
        self.height = int(height)
        self.total_frames = int(total_frames)
        self._rows: dict = {}

    def add(self, frame_idx: int, player_ids, boxes, masks, color_index, team_id) -> None:
        """Record one frame. Rows must be parallel and in draw order."""
        packed = []
        rects = []
        for mask in masks:
            x0, y0, w, h = _crop_rect(mask)
            rects.append((x0, y0, w, h))
            packed.append(
                np.packbits(mask[y0:y0 + h, x0:x0 + w]) if w and h
                else np.zeros(0, dtype=np.uint8)
            )
        self._rows[int(frame_idx)] = (
            np.asarray(player_ids, dtype=np.int32),
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(color_index, dtype=np.uint8),
            np.asarray(team_id, dtype=np.int8),
            np.asarray(rects, dtype=np.int32).reshape(-1, 4),
            packed,
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        order = sorted(self._rows)
        counts = np.zeros(self.total_frames + 1, dtype=np.int64)
        for frame_idx in order:
            counts[frame_idx + 1] = len(self._rows[frame_idx][0])
        offsets = np.cumsum(counts)

        player_ids, boxes, colors, teams, rects, bits = [], [], [], [], [], []
        mask_sizes = [0]
        for frame_idx in order:
            ids, box, color, team, rect, packed = self._rows[frame_idx]
            player_ids.append(ids)
            boxes.append(box)
            colors.append(color)
            teams.append(team)
            rects.append(rect)
            for chunk in packed:
                bits.append(chunk)
                mask_sizes.append(len(chunk))

        def cat(parts, dtype, width=None):
            if not parts:
                return np.zeros((0, width) if width else 0, dtype=dtype)
            return np.concatenate(parts).astype(dtype)

        np.savez_compressed(
            path,
            schema=SCHEMA,
            width=self.width, height=self.height, total_frames=self.total_frames,
            offsets=offsets,
            player_ids=cat(player_ids, np.int32),
            boxes=cat(boxes, np.float32, width=4).reshape(-1, 4),
            color_index=cat(colors, np.uint8),
            team_id=cat(teams, np.int8),
            crop_rect=cat(rects, np.int32, width=4).reshape(-1, 4),
            mask_offsets=np.cumsum(mask_sizes, dtype=np.int64),
            mask_bits=cat(bits, np.uint8),
            # Which frames the pass recorded, empty ones included: a frame with
            # no live player is still a frame the render wrote, and a redraw
            # that skipped it would come out one frame short and desynced.
            visited=np.asarray(order, dtype=np.int64),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "_LoadedGeometry":
        with np.load(Path(path)) as data:
            store = {k: data[k] for k in data.files}
        if int(store["schema"]) != SCHEMA:
            raise ValueError(
                f"geometry cache schema {int(store['schema'])}, expected {SCHEMA}: "
                f"{path}. Re-run the tracking pass to rebuild it."
            )
        return _LoadedGeometry(store)


class _LoadedGeometry:
    """Read side of a saved cache."""

    def __init__(self, store: dict):
        self._s = store
        self.width = int(store["width"])
        self.height = int(store["height"])
        self.total_frames = int(store["total_frames"])

    @property
    def frame_indices(self) -> list:
        """Every frame the tracking pass recorded, in order, empty ones included."""
        return [int(i) for i in self._s["visited"]]

    def frame(self, frame_idx: int) -> FrameGeometry:
        s = self._s
        offsets = s["offsets"]
        if not 0 <= frame_idx < self.total_frames:
            lo = hi = 0
        else:
            lo, hi = int(offsets[frame_idx]), int(offsets[frame_idx + 1])
        shape = (self.height, self.width)
        masks = np.zeros((hi - lo, *shape), dtype=bool)
        for row, index in enumerate(range(lo, hi)):
            x0, y0, w, h = (int(v) for v in s["crop_rect"][index])
            if not (w and h):
                continue
            start, stop = int(s["mask_offsets"][index]), int(s["mask_offsets"][index + 1])
            crop = np.unpackbits(s["mask_bits"][start:stop], count=h * w)
            masks[row, y0:y0 + h, x0:x0 + w] = crop.reshape(h, w).astype(bool)
        return FrameGeometry(
            frame_idx=frame_idx,
            player_ids=s["player_ids"][lo:hi],
            boxes=s["boxes"][lo:hi].astype(float),
            color_index=s["color_index"][lo:hi],
            team_id=s["team_id"][lo:hi],
            masks=masks,
        )
