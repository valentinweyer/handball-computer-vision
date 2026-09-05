"""Jersey-number identity signal.

Number boxes are class 4 of the player-detection model already being called for
players -- no separate detector, no extra inference call to find them. OCR is a
EasyOCR recognizes tight native-size grayscale crops locally. The original
basketball-tuned SmolVLM2 path systematically confused small handball ``13``
crops with ``8``/``18``, so reads remain confidence-gated and voted over time.

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

import cv2
import numpy as np
import supervision as sv

NUMBER_CLASS_ID   = 4
NUMBER_PROMPT     = "Read the number."  # retained for notebook/API compatibility
NUMBER_CROP_PAD   = 0
EASY_OCR_MIN_CONFIDENCE = 0.5
MATCH_IOS_MIN     = 0.9
OCR_EVERY_N_FRAMES = 5          # number detection is expensive; sample, don't run every frame
# IHF numbering: 1-99. "0"/"00" are not legal, and a leading zero ("07") is not
# how a jersey is printed, so any of those coming back from the reader is a
# misread or a false number detection rather than a player.
MIN_JERSEY_NUMBER, MAX_JERSEY_NUMBER = 1, 99


def is_valid_number(value: str) -> bool:
    """Whether a read could be a real handball jersey number."""
    if not value.isdigit() or len(value) > 2:
        return False
    if len(value) == 2 and value[0] == "0":
        return False
    return MIN_JERSEY_NUMBER <= int(value) <= MAX_JERSEY_NUMBER


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
    """Recognize tight, native-size number crops with EasyOCR.

    The number detector already supplies a tight box. Padding and square
    stretching made the 25-40 px handball digits less legible, while the
    basketball-tuned VLM systematically read a labeled ``13`` as ``8``/``18``.
    Returns ``''`` for a low-confidence or failed read.
    """
    number_xyxy = np.asarray(number_xyxy).reshape(-1, 4)
    if len(number_xyxy) == 0:
        return []
    h, w = frame_rgb.shape[:2]
    padded = sv.clip_boxes(
        sv.pad_boxes(xyxy=number_xyxy, px=NUMBER_CROP_PAD, py=NUMBER_CROP_PAD), (w, h)
    )
    crops_grey = [
        cv2.cvtColor(sv.crop_image(frame_rgb, box), cv2.COLOR_RGB2GRAY)
        for box in padded
    ]
    out = []
    for crop in crops_grey:
        try:
            crop_h, crop_w = crop.shape
            result = model.recognize(
                crop,
                horizontal_list=[[0, crop_w, 0, crop_h]],
                free_list=[],
                allowlist="0123456789",
                detail=1,
                decoder="beamsearch",
                beamWidth=10,
                reformat=False,
            )
            value, confidence = result[0][1], float(result[0][2])
            out.append(value if confidence >= EASY_OCR_MIN_CONFIDENCE else "")
        except Exception:
            out.append("")
    return out


@dataclass
class _Votes:
    counts: dict = field(default_factory=dict)  # normalized value -> count

    def add(self, value: str) -> None:
        self.counts[value] = self.counts.get(value, 0) + 1

    def resolved_counts(self, min_promote_votes: int, min_promote_ratio: float = 0.5) -> dict:
        """Counts with single-digit reads folded into the number they partially show.

        A crop that catches only the trailing digit of a two-digit number reads as that
        digit, so "7" is exactly what a partial view of "17" looks like; the reverse
        cannot happen. Those are corroborating observations of one number, not rival
        candidates, and leaving them to compete splits a player's evidence across both
        -- measured on FelixClaar, where a player wearing 17 accumulated {17: 4, 7: 4}
        and the tie suppressed any verdict.

        Folding is deliberately conservative:
          - only a 1-digit read folds, and only into a 2-digit read (never the reverse),
          - the longer reading needs both `min_promote_votes` of its own support AND
            `min_promote_ratio` of the support of the read it is absorbing, so a handful
            of misreads cannot capture a well-established shorter number. The absolute
            floor alone was not enough: measured on BHC-FAG, a player visibly wearing 7
            read {'7': 32, '77': 2} and two stray "77"s captured all 32, flipping a
            correct verdict to a wrong one.
          - if two equally-supported extensions exist ("17" and "27" both at 3), the
            short read cannot choose between them, so nothing folds.
        """
        merged = dict(self.counts)
        for short in [value for value in self.counts if len(value) == 1]:
            floor = max(min_promote_votes, min_promote_ratio * self.counts[short])
            extensions = sorted(
                (
                    (count, value) for value, count in self.counts.items()
                    if len(value) == 2
                    and value.endswith(short)
                    and count >= floor
                ),
                reverse=True,
            )
            if not extensions:
                continue
            if len(extensions) > 1 and extensions[0][0] == extensions[1][0]:
                continue
            target = extensions[0][1]
            merged[target] = merged.get(target, 0) + merged.pop(short, 0)
        return merged

    def best(self, min_promote_votes: int = 2, min_promote_ratio: float = 0.5):
        """(value, count, margin) for the leading valid number, or (None, 0, 0.0).

        Filtering happens *after* folding, not at `observe()`, and the ordering
        is load-bearing. A crop catching only the trailing digit of 10/20/40
        reads as "0", which is not itself a legal jersey number but is real
        corroboration for one -- measured on BHC-FAG, a stray "0" folded into
        p11's "40" and took it from 5 votes to 6. Dropping "0" on arrival would
        discard that. Barring it only from *winning* keeps the evidence and
        still prevents the verdict: on the same clip p3 read {"0": 13, ...} off
        a shorts logo the number detector mistook for a digit, and without this
        it reported 0 as that player's number.

        Invalid values are also excluded from `total`, so they cannot dilute a
        real value's margin -- a false detection is evidence about nothing.
        """
        counts = self.resolved_counts(min_promote_votes, min_promote_ratio)
        counts = {v: n for v, n in counts.items() if is_valid_number(v)}
        if not counts:
            return None, 0, 0.0
        total = sum(counts.values())
        value, count = max(counts.items(), key=lambda kv: kv[1])
        rest = sorted(counts.values(), reverse=True)[1:2]
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

    def __init__(
        self, min_votes: int = 3, min_margin: float = 0.2, min_promote_votes: int = 2,
        min_promote_ratio: float = 0.5,
    ):
        self.min_votes = min_votes
        self.min_margin = min_margin
        self.min_promote_votes = min_promote_votes
        self.min_promote_ratio = min_promote_ratio
        self._votes: dict = {}

    def observe(self, identity_id: int, raw_value: str) -> None:
        value = self._normalize(raw_value)
        if value is None:
            return
        self._votes.setdefault(identity_id, _Votes()).add(value)

    @staticmethod
    def _normalize(raw_value):
        s = (raw_value or "").strip()
        return s if s.isdigit() and len(s) <= 2 else None

    def best(self, identity_id: int):
        """(number, votes, margin). number is None until min_votes/min_margin are met."""
        v = self._votes.get(identity_id)
        if v is None:
            return None, 0, 0.0
        value, count, margin = v.best(self.min_promote_votes, self.min_promote_ratio)
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
