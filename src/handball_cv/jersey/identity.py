"""Jersey-number identity signal.

Number boxes are class 4 of the player-detection model already being called for
players -- no separate detector, no extra inference call to find them. OCR is a
fine-tuned SmolVLM2 VLM trained on basketball jerseys; digits are digits, so it
still reads handball numbers, but treat its output as noisy and vote it over time
rather than trusting a single read.

Numbers are matched to player masks by mask IoS (intersection over the *smaller*
area), not IoU: a number crop is small and, when correctly matched, fully
contained within its player's mask, so IoS saturates to 1.0 while IoU would stay
low regardless of match quality. Ported from the notebook this pipeline never
wired in (handball_ai_how_to_..._players.ipynb, cells 83-95).

Two things are deliberately NOT ported from that notebook:

  - `sports.common.temporal.ConsecutiveValueTracker` locks a value permanently
    with no un-lock, which is wrong once re-ID or a hard reset can legitimately
    move a physical player to a new tracker-level id.
  - its `player_idx = [i + 1 for i in player_idx]` assumes
    `index + 1 == tracker_id`, true only because SAM2 was seeded once with
    contiguous ids from `np.arange(1, N+1)`. That assumption is false once
    TrackManager (or McByte) allocates and retires ids out of order.

`NumberVoter` here keys on the caller's own stable identity (obj_id / player_id)
and can revise its answer as evidence accumulates.
"""
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

NUMBER_CLASS_ID   = 4
NUMBER_MODEL_ID   = "basketball-jersey-numbers-ocr/3"
NUMBER_PROMPT     = "Read the number."
NUMBER_CROP_PAD   = 10
NUMBER_CROP_SIZE  = (224, 224)  # matches how the OCR model was trained (notebook cell 86)
MATCH_IOS_MIN     = 0.9
OCR_EVERY_N_FRAMES = 5          # SmolVLM2 is the expensive step; sample, don't run every frame


def extract_number_boxes(detections: sv.Detections) -> sv.Detections:
    """Split class-4 (jersey number) detections out of a player-model result."""
    return detections[detections.class_id == NUMBER_CLASS_ID]


def match_numbers_to_players(
    player_masks: np.ndarray,
    number_xyxy: np.ndarray,
    frame_shape: tuple,
) -> list:
    """[(player_row, number_row), ...] pairs above MATCH_IOS_MIN, best match first.

    frame_shape: (height, width) -- note sv.xyxy_to_mask wants resolution_wh as
    (width, height), the reverse order.
    """
    if len(player_masks) == 0 or len(number_xyxy) == 0:
        return []
    h, w = frame_shape[:2]
    number_masks = sv.xyxy_to_mask(boxes=np.asarray(number_xyxy).reshape(-1, 4), resolution_wh=(w, h))
    iou = sv.mask_iou_batch(
        masks_true=player_masks, masks_detection=number_masks, overlap_metric=sv.OverlapMetric.IOS
    )
    rows, cols = np.where(iou > MATCH_IOS_MIN)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    pairs.sort(key=lambda rc: iou[rc[0], rc[1]], reverse=True)
    return pairs


def read_numbers(model, frame_rgb: np.ndarray, number_xyxy: np.ndarray) -> list:
    """OCR each number crop. Returns raw model strings; '' for an empty/failed read."""
    number_xyxy = np.asarray(number_xyxy).reshape(-1, 4)
    if len(number_xyxy) == 0:
        return []
    h, w = frame_rgb.shape[:2]
    padded = sv.clip_boxes(
        sv.pad_boxes(xyxy=number_xyxy, px=NUMBER_CROP_PAD, py=NUMBER_CROP_PAD), (w, h)
    )
    crops = [
        sv.resize_image(sv.crop_image(frame_rgb, box), resolution_wh=NUMBER_CROP_SIZE)
        for box in padded
    ]
    out = []
    for crop in crops:
        try:
            out.append(model.infer(crop, prompt=NUMBER_PROMPT)[0].response)
        except Exception:
            out.append("")
    return out


@dataclass
class _Votes:
    counts: dict = field(default_factory=dict)  # normalized value -> count

    def add(self, value: str) -> None:
        self.counts[value] = self.counts.get(value, 0) + 1

    def best(self):
        """(value, count, margin) for the leading value, or (None, 0, 0.0)."""
        if not self.counts:
            return None, 0, 0.0
        total = sum(self.counts.values())
        value, count = max(self.counts.items(), key=lambda kv: kv[1])
        rest = sorted(self.counts.values(), reverse=True)[1:2]
        runner_up = rest[0] if rest else 0
        margin = (count - runner_up) / total
        return value, count, margin


class NumberVoter:
    """Per-identity jersey-number histogram that can change its mind.

    Unlike ConsecutiveValueTracker this never locks permanently -- necessary once
    re-ID or a hard reset can legitimately reassign what obj_id/player_id a
    physical player is tracked under, and a stale locked number would then be
    silently wrong for the rest of the clip.
    """

    def __init__(self, min_votes: int = 3, min_margin: float = 0.5):
        self.min_votes = min_votes
        self.min_margin = min_margin
        self._votes: dict = {}

    def observe(self, identity_id: int, raw_value: str) -> None:
        value = self._normalize(raw_value)
        if value is None:
            return
        self._votes.setdefault(identity_id, _Votes()).add(value)

    @staticmethod
    def _normalize(raw_value):
        s = (raw_value or "").strip()
        return s if s.isdigit() else None

    def best(self, identity_id: int):
        """(number, votes, margin). number is None until min_votes/min_margin are met."""
        v = self._votes.get(identity_id)
        if v is None:
            return None, 0, 0.0
        value, count, margin = v.best()
        if count < self.min_votes or margin < self.min_margin:
            return None, count, margin
        return value, count, margin

    def reset(self, identity_id: int) -> None:
        self._votes.pop(identity_id, None)

    def merge(self, from_id: int, into_id: int) -> None:
        """Fold one identity's votes into another's -- call this on a re-ID hit
        so accumulated evidence isn't discarded when a tracker-level id changes."""
        src = self._votes.pop(from_id, None)
        if src is None:
            return
        dst = self._votes.setdefault(into_id, _Votes())
        for value, count in src.counts.items():
            dst.counts[value] = dst.counts.get(value, 0) + count
