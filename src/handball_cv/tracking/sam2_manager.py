"""Track lifecycle policy for the SAM2 offline predictor.

SAM2 seeded once on frame 0 never recovers from a track jumping onto the
wrong player or a player entering after the seed frame. This module drives
periodic detector checkpoints against the live tracks and issues the
add / remove / re-prompt calls SAM2 needs to correct itself, keyed by
`obj_id` (never by index -- `remove_object` renumbers internal indices).

Two different fixes look identical from the outside (a matched track with
poor IoU or a collapsed mask) but need opposite handling: a mask that has
merely drifted off a still-correctly-identified player should be reprompted
in place (SAM2 keeps its memory and treats the click as a correction), while
a mask that has jumped onto a different player should have its memory torn
down and be re-added fresh under the same `obj_id` (`reset`) -- reprompting
in place would seed the "correction" from the wrong player's mask and keep
their appearance in memory. `PlayerRecord.embedding` (EMA-refreshed at every
checkpoint) versus the checkpoint's own detection embedding is what tells
the two cases apart.

Identity, team evidence, goalkeeper role, and re-ID are owned by
`handball_cv.tracking.identity.PlayerRegistry`, the same shared layer McByte's
`IdentityManager` uses -- SAM2 has no separate short-lived id to translate
away, so this module uses the registry's `player_id` directly as the SAM2
`obj_id`. `Track` here holds only the mask-geometry state (centroid/area
history, pending-action confirmation counters) that is specific to reasoning
about SAM2 propagation, not identity.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from handball_cv.tracking.identity import TEAM_SWITCH_OBSERVATIONS, PlayerRegistry

# tunables
HISTORY_LEN = 30
MIN_CONFIRM_CHECKPOINTS = 2          # consecutive checkpoints before acting
DUPLICATE_IOU_MIN = 0.6              # mask intersection-over-min-area
DROPOUT_AREA_FRAC = 0.2              # of the track's own running median
# Checkpoints a track may go unmatched by the detector before it is retired,
# however healthy its mask looks. Rule 2 only catches a *collapsing* mask, so a
# player who walks off and sits on the bench keeps a perfect mask and is tracked
# for the rest of the clip. Measured on Melsungen_Berlin_2min_1: 17.8 live tracks
# against 14.0 detections per frame, live > detections on 97% of frames, and 34%
# of frames pinned at MAX_LIVE_OBJECTS -- so stale tracks were not merely clutter,
# they were consuming the budget a genuinely new player needed.
#
# The threshold is the cost of being wrong in the other direction. Linking raw
# detections across the three 10-minute windows (n=14232 dropouts of a person who
# returns) gives p90 = 65 frames and p95 = 124: a real on-court player does vanish
# from the detector, for seconds at a time. At CHECK_EVERY=10 this is 8 seconds of
# footage, past which 1.6% of genuine dropouts would be retired early -- and those
# recover through re-ID, while a bench-sitter never leaves on its own.
UNMATCHED_CHECKPOINTS_MAX = 20
DRIFT_IOU_LOW, DRIFT_IOU_HIGH = 0.1, 0.5
MATCH_IOU_MIN = 0.1                  # below this, a detection counts as unmatched
# A candidate new player is followed between checkpoints by proximity, not by a
# fixed spatial grid. Measured on a 25fps 1080p clip at CHECK_EVERY=10, people move
# a median 30px, p90 93px, so grid bins of any fixed size systematically admit
# stationary players and reject running ones. IoU is the wrong metric too: a player
# box is ~40px wide, so a 30px sideways step -- the median -- already drops IoU to
# 0.14. Centre distance scaled by box height tolerates real movement while staying
# scale-aware as players change distance from the camera.
PENDING_MAX_CENTRE_FRAC = 0.6
REID_COS_SIM_MIN = 0.7
# Units bug fix: this was previously compared against a *frame* index while
# named/commented as a checkpoint count ("~300 frames at CHECK_EVERY=10"), so
# the real re-ID window was only 30 frames (~1.25s @ 24fps) instead of the
# intended ~300. Renamed to make the unit explicit and set to the value the
# original comment intended.
REID_MAX_GAP_FRAMES = 300
MAX_LIVE_OBJECTS = 20
# Blends each checkpoint's detection embedding into the track's own, so re-ID
# and the drift/body-swap check below compare against roughly-current
# appearance instead of a frame-0 (or last-reprompt) snapshot that may no
# longer resemble the player. Low weight: one noisy read should not overwrite
# a good running appearance estimate.
EMBEDDING_EMA_ALPHA = 0.3
# Rule 3 fires on poor box IoU or a collapsed mask, which is ambiguous: either
# the mask is still on the right player and has just drifted (reprompting in
# place is correct -- SAM2 keeps its memory and treats the click as a
# correction), or the mask has jumped onto a different player entirely
# (reprompting in place would seed the correction from contaminated memory --
# see `reset` below, which tears memory down and re-adds fresh). Appearance
# similarity against the track's own (EMA-refreshed) embedding distinguishes
# the two. Reuses REID_COS_SIM_MIN: same question -- "is this the same
# person" -- just asked in-place against a live track instead of against the
# retired gallery.
BODY_SWAP_COS_SIM_MIN = REID_COS_SIM_MIN


def _centre_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Centre distance between two boxes, as a fraction of their mean height."""
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    scale = ((a[3] - a[1]) + (b[3] - b[1])) / 2.0
    if scale <= 0:
        return float("inf")
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / scale)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def mask_area_and_centroid(mask: np.ndarray) -> tuple[float, tuple[float, float] | None]:
    """Pixel count and centroid of a boolean mask, computed on its bounding box.

    The obvious spelling -- `m.sum()` then `np.nonzero(m)` and two `.mean()`
    calls -- scans the whole 1080p frame and allocates two int64 index arrays
    sized to the true-pixel count (~576 KB per player, 13.4 players a frame)
    purely to take two means. A synchronized profile put it at 56.6ms of
    `update_from_propagation`'s 57.6ms per frame, second only to the model
    itself. A player covers a few tens of thousands of those two million
    pixels, so its bounding box is ~50x less area to touch.

    `cv2.moments(binaryImage=True)` returns exactly the quantities wanted:
    `m00` is the pixel count, and `m10`, `m01` are the summed x and y of the
    true pixels -- so dividing once by the count reproduces `xs.mean()` and
    `ys.mean()` bit for bit. Measured 44x faster on 826 real SAM2 masks with no
    differing result; equivalence is pinned by
    `tests/unit/test_mask_area_and_centroid.py`.

    Returns `(0.0, None)` for an empty mask, so callers keep their existing
    "no centroid unless there is area" rule. Both outputs feed lifecycle
    decisions -- `Track.areas` drives mask-collapse detection and
    `Track.centroids` drives duplicate removal's centroid-jump test -- so this
    has to stay a speed change and nothing else.
    """
    rows = mask.any(axis=1)
    if not rows.any():
        return 0.0, None
    cols = mask.any(axis=0)
    y0 = int(np.argmax(rows))
    y1 = len(rows) - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = len(cols) - int(np.argmax(cols[::-1]))

    moments = cv2.moments(mask[y0:y1, x0:x1].view(np.uint8), binaryImage=True)
    area = float(moments["m00"])
    if area <= 0:
        return 0.0, None
    # Fold the crop offset into the numerator, not onto the quotient. Both are
    # exact integers in float64 here (coordinate sums stay far below 2**53), so
    # dividing once reproduces `xs.mean()` bit for bit; dividing on the crop and
    # adding x0 afterwards rounds twice and drifts by an ulp.
    return area, ((moments["m10"] + area * x0) / area,
                  (moments["m01"] + area * y0) / area)


@dataclass
class Track:
    """Mask-geometry state for one live SAM2 object.

    Identity, team, goalkeeper, and embedding live on the matching
    `PlayerRecord` in `TrackManager.registry`, keyed by the same `obj_id`.
    """
    obj_id: int
    centroids: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    areas: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    # pending-action confirmation counters, reset whenever the condition lapses
    _pending: dict = field(default_factory=dict)

    def median_area(self) -> Optional[float]:
        return float(np.median(self.areas)) if self.areas else None

    def max_recent_jump(self) -> float:
        if len(self.centroids) < 2:
            return 0.0
        pts = np.array(self.centroids)
        return float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def confirm(self, key: str, threshold: int = MIN_CONFIRM_CHECKPOINTS) -> bool:
        """Increment a named condition counter; True once it hits threshold."""
        self._pending[key] = self._pending.get(key, 0) + 1
        return self._pending[key] >= threshold

    def clear(self, key: str) -> None:
        self._pending.pop(key, None)

    def clear_all_but(self, keys: set) -> None:
        for k in list(self._pending):
            if k not in keys:
                self._pending.pop(k)


class TrackManager:
    """Owns per-track geometry and decides add/remove/reprompt actions.

    Identity (team, goalkeeper, re-ID) is delegated to a `PlayerRegistry`
    shared with McByte's `IdentityManager`, so both trackers apply the same
    team-evidence decay/hysteresis and running-majority goalkeeper logic.

    Does not call the SAM2 predictor itself -- `checkpoint()` returns a list
    of actions for the caller to apply, since the caller owns the predictor
    session and frame cache.
    """

    def __init__(
        self, team_model, court_test_fn, next_obj_id_start=1,
        team_switch_observations=TEAM_SWITCH_OBSERVATIONS,
        reid_encoder=None,
    ):
        self.team_model = team_model  # a team_model.TeamModel
        # Optional `crops_rgb -> (N, D)` callable describing people for identity
        # only; team classification always stays on the team model's features.
        # See `_appearance` for what it buys.
        self.reid_encoder = reid_encoder
        self.court_test_fn = court_test_fn  # (xyxy) -> bool, inside playing surface
        self.registry = PlayerRegistry(
            reid_cos_sim_min=REID_COS_SIM_MIN,
            reid_max_gap_frames=REID_MAX_GAP_FRAMES,
            team_switch_observations=team_switch_observations,
            next_id_start=next_obj_id_start,
        )
        self.tracks: dict[int, Track] = {}
        self.events: list[dict] = []
        # confirmation counters for not-yet-seen players, keyed by a coarse
        # spatial slot (no Track object exists for these yet)
        # [box, consecutive_checkpoint_count] per unconfirmed candidate
        self._pending_new_boxes: list = []

    def _appearance(self, frame: np.ndarray, boxes_xyxy: np.ndarray, fallback):
        """Identity vectors for these boxes: the re-ID encoder's, or `fallback`.

        The team model describes a torso well enough to read shirt colour, which
        is all team classification asks. Identity asks it to tell two people in
        the *same* shirt apart, and measured on the labelled 1080p set it cannot:
        different teammates sit as close as two views of one player (0.822 vs
        0.802 median cosine on Melsungen), for 0.35 rank-1 within a team against
        a 0.19 chance floor. A person-reID encoder scores 0.55 on the same
        queries and wins on every clip. Keeping the two feature spaces separate
        also keeps an identity error from becoming a team error.

        A box too small to crop keeps its fallback vector rather than becoming a
        zero vector, which would sit at cosine 0 from everything and re-ID as
        nothing.
        """
        if self.reid_encoder is None or not len(boxes_xyxy):
            return fallback
        height, width = frame.shape[:2]
        crops, rows = [], []
        for row, (x1, y1, x2, y2) in enumerate(
            np.asarray(boxes_xyxy, dtype=float).round().astype(int)
        ):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                crops.append(frame[y1:y2, x1:x2])
                rows.append(row)
        if not crops:
            return fallback
        encoded = np.asarray(self.reid_encoder(crops), dtype=float)
        out = np.zeros((len(boxes_xyxy), encoded.shape[1]), dtype=float)
        for slot, row in enumerate(rows):
            out[row] = encoded[slot]
        for row in set(range(len(boxes_xyxy))) - set(rows):
            out[row] = np.resize(np.asarray(fallback[row], dtype=float), out.shape[1])
        return out

    @property
    def retired(self) -> list:
        return self.registry.retired

    def _alloc_id(self) -> int:
        return self.registry.alloc_id()

    def seed(
        self, frame_idx: int, boxes_xyxy: np.ndarray, frame: np.ndarray,
        is_goalkeeper: np.ndarray,
    ) -> list[int]:
        """Initial frame-0 seeding. Returns assigned obj_ids in input order."""
        boxes_xyxy = np.asarray(boxes_xyxy).reshape(-1, 4)
        embeddings, teams, confidence, quality = self.team_model.observe(
            frame, boxes_xyxy
        )
        embeddings = self._appearance(frame, boxes_xyxy, embeddings)
        obj_ids = []
        for emb, team, conf, crop_quality, is_gk in zip(
            embeddings, teams, confidence, quality, is_goalkeeper
        ):
            player = self.registry.create(
                frame_idx, None, int(team), emb, bool(is_gk),
                confidence=float(conf), quality=float(crop_quality),
            )
            oid = player.player_id
            self.tracks[oid] = Track(obj_id=oid)
            obj_ids.append(oid)
        return obj_ids

    def update_from_propagation(self, frame_idx: int, obj_ids, masks: np.ndarray):
        """Record centroid/area history for every live track from tracker output.

        masks: (N, H, W) bool, ordered to match obj_ids.
        """
        for oid, m in zip(obj_ids, masks):
            t = self.tracks.get(int(oid))
            player = self.registry.live.get(int(oid))
            if t is None:
                continue
            area, centroid = mask_area_and_centroid(m)
            t.areas.append(area)
            if centroid is not None:
                t.centroids.append(centroid)
                if player is not None:
                    player.last_seen = frame_idx

    def _match_pending(self, box) -> int | None:
        """Index of the pending candidate this detection continues, if any."""
        best, best_gap = None, PENDING_MAX_CENTRE_FRAC
        for index, (pending_box, _count) in enumerate(self._pending_new_boxes):
            gap = _centre_gap(box, pending_box)
            if gap < best_gap:
                best, best_gap = index, gap
        return best

    def checkpoint(
        self, frame_idx, frame, live_obj_ids, live_masks, det_boxes_xyxy,
        det_is_goalkeeper=None,
    ):
        """One detector checkpoint. Returns a list of action dicts:

          {"type": "remove", "obj_id": int}
          {"type": "reprompt", "obj_id": int, "box": xyxy}
          {"type": "reset", "obj_id": int, "box": xyxy}  # same obj_id, fresh memory
          {"type": "add", "obj_id": int, "box": xyxy}   # obj_id is new
        """
        actions = []
        live_obj_ids = list(live_obj_ids)
        masks_by_id = {int(oid): m for oid, m in zip(live_obj_ids, live_masks)}

        # ---- match detections to live tracks by mask IoU ----
        det_boxes_xyxy = np.asarray(det_boxes_xyxy).reshape(-1, 4)
        det_is_goalkeeper = (
            np.zeros(len(det_boxes_xyxy), dtype=bool)
            if det_is_goalkeeper is None else np.asarray(det_is_goalkeeper)
        )
        if len(live_obj_ids) and len(det_boxes_xyxy):
            track_boxes = np.array([
                sv.mask_to_xyxy(masks_by_id[oid][None])[0]
                if masks_by_id[oid].any() else np.array([0, 0, 0, 0])
                for oid in live_obj_ids
            ])
            iou = sv.box_iou_batch(track_boxes, det_boxes_xyxy)  # (n_tracks, n_dets)
            cost = 1.0 - iou
            row, col = linear_sum_assignment(cost)
            track_to_det = {
                live_obj_ids[r]: c for r, c in zip(row, col) if iou[r, c] >= MATCH_IOU_MIN
            }
        else:
            iou = np.zeros((len(live_obj_ids), len(det_boxes_xyxy)))
            track_to_det = {}

        matched_det_idx = set(track_to_det.values())
        active_keys_per_track = {oid: set() for oid in live_obj_ids}

        # ---- one team/re-ID observation batch for every detection ----
        # This sees the full scene for overlap rejection and is reused below for
        # both matched tracks and newly confirmed detections.
        if len(det_boxes_xyxy):
            det_embeddings, det_teams, det_confidence, det_quality = (
                self.team_model.observe(frame, det_boxes_xyxy)
            )
            det_embeddings = self._appearance(frame, det_boxes_xyxy, det_embeddings)
        else:
            det_embeddings = np.empty((0, 0), dtype=float)
            det_teams = np.empty(0, dtype=int)
            det_confidence = np.empty(0, dtype=float)
            det_quality = np.empty(0, dtype=float)

        # Snapshot embeddings before the refresh below rebinds them. Rule 3's
        # body-swap check needs the track's appearance as it stood BEFORE this
        # checkpoint's observation was blended in -- comparing the post-refresh
        # embedding against the very sample it was just blended with would
        # understate a real swap (it would already be ~30% that sample).
        pre_refresh_embedding = {
            oid: self.registry.live[oid].embedding for oid in track_to_det
        }

        for oid, detection_index in track_to_det.items():
            confidence = float(det_confidence[detection_index])
            quality = float(det_quality[detection_index])
            player = self.registry.live[oid]
            if quality > 0:
                # Gated on crop legibility alone (not team-vote confidence,
                # not goalkeeper status) -- this is an appearance estimate,
                # not a team-color read, and goalkeepers need their embedding
                # refreshed too so they remain re-ID-able.
                player.embedding = (
                    EMBEDDING_EMA_ALPHA * det_embeddings[detection_index]
                    + (1 - EMBEDDING_EMA_ALPHA) * player.embedding
                )
            player.record_role(bool(det_is_goalkeeper[detection_index]))
            if not bool(det_is_goalkeeper[detection_index]):
                self.registry.observe_team(
                    oid, frame_idx, int(det_teams[detection_index]),
                    confidence, quality, fragment_id=oid,
                )

        # ---- rule 1: duplicate tracks (mask overlap) ----
        acted_ids = set()
        for i, oid_a in enumerate(live_obj_ids):
            for oid_b in live_obj_ids[i + 1:]:
                ma, mb = masks_by_id[oid_a], masks_by_id[oid_b]
                if not ma.any() or not mb.any():
                    continue
                inter = float((ma & mb).sum())
                min_area = min(ma.sum(), mb.sum())
                if min_area == 0:
                    continue
                if inter / min_area > DUPLICATE_IOU_MIN:
                    key = "dup"
                    active_keys_per_track[oid_a].add(key)
                    active_keys_per_track[oid_b].add(key)
                    ta, tb = self.tracks[oid_a], self.tracks[oid_b]
                    # evaluate both unconditionally -- `and` short-circuits and
                    # would leave tb's counter permanently one round behind
                    a_confirmed = ta.confirm(key)
                    b_confirmed = tb.confirm(key)
                    if a_confirmed and b_confirmed:
                        jumper = ta if ta.max_recent_jump() >= tb.max_recent_jump() else tb
                        if jumper.obj_id not in acted_ids:
                            actions.append({"type": "remove", "obj_id": jumper.obj_id})
                            acted_ids.add(jumper.obj_id)
                            # Unlike rule 2's dropout, duplicate resolution is a
                            # guess (larger recent jump) that can pick the wrong
                            # one of the pair -- retiring it, like rule 2 does,
                            # means a wrong guess is still recoverable by re-ID
                            # instead of permanently destroying that identity.
                            self.registry.retire(jumper.obj_id, frame_idx)
                            kept = tb.obj_id if jumper is ta else ta.obj_id
                            self.events.append({
                                "frame": frame_idx, "type": "remove_duplicate",
                                "obj_id": jumper.obj_id, "kept": kept,
                            })

        # ---- rule 2: dropout (empty / collapsed mask, no nearby detection) ----
        for oid in live_obj_ids:
            if oid in acted_ids:
                continue
            t = self.tracks[oid]
            m = masks_by_id[oid]
            area = float(m.sum())
            med = t.median_area()
            collapsed = (area == 0) or (med and med > 0 and area < DROPOUT_AREA_FRAC * med)
            has_match = oid in track_to_det
            if collapsed and not has_match:
                key = "gone"
                active_keys_per_track[oid].add(key)
                if t.confirm(key):
                    actions.append({"type": "remove", "obj_id": oid})
                    acted_ids.add(oid)
                    self.registry.retire(oid, frame_idx)
                    self.events.append({"frame": frame_idx, "type": "remove_gone", "obj_id": oid})
                continue
            if not has_match:
                # The mask is healthy and SAM2 is happily tracking *something*
                # the detector no longer calls a player -- a substitute on the
                # bench, or a track that has drifted onto furniture. Nothing else
                # ends these, so they accumulate until the object budget is full.
                key = "undetected"
                active_keys_per_track[oid].add(key)
                if t.confirm(key, threshold=UNMATCHED_CHECKPOINTS_MAX):
                    actions.append({"type": "remove", "obj_id": oid})
                    acted_ids.add(oid)
                    self.registry.retire(oid, frame_idx)
                    self.events.append({
                        "frame": frame_idx, "type": "remove_undetected", "obj_id": oid,
                    })

        # ---- rule 3: drift (matched but poor IoU, or collapsed with a good detection) ----
        for oid in live_obj_ids:
            if oid in acted_ids or oid not in track_to_det:
                continue
            t = self.tracks[oid]
            di = track_to_det[oid]
            track_iou = iou[live_obj_ids.index(oid), di]
            area = float(masks_by_id[oid].sum())
            med = t.median_area()
            collapsed = med and med > 0 and area < DROPOUT_AREA_FRAC * med
            if (DRIFT_IOU_LOW <= track_iou <= DRIFT_IOU_HIGH) or collapsed:
                key = "drift"
                active_keys_per_track[oid].add(key)
                if t.confirm(key):
                    box = det_boxes_xyxy[di]
                    same_player = (
                        _cosine_similarity(pre_refresh_embedding[oid], det_embeddings[di])
                        >= BODY_SWAP_COS_SIM_MIN
                    )
                    if same_player:
                        actions.append({"type": "reprompt", "obj_id": oid, "box": box})
                        self.events.append({
                            "frame": frame_idx, "type": "reprompt", "obj_id": oid,
                            "iou": float(track_iou),
                        })
                    else:
                        # Appearance no longer matches: the mask likely jumped
                        # to a different player. Reprompting in place would
                        # correct from that wrong mask and keep the wrong
                        # appearance in memory -- ask the caller to tear the
                        # object down and re-add it fresh under the same id.
                        actions.append({"type": "reset", "obj_id": oid, "box": box})
                        self.events.append({
                            "frame": frame_idx, "type": "reset", "obj_id": oid,
                            "iou": float(track_iou),
                        })

        # decay confirmation counters for conditions that didn't fire this round
        for oid in live_obj_ids:
            self.tracks[oid].clear_all_but(active_keys_per_track[oid])

        # ---- rule 4: new player (unmatched detection, inside court, confirmed) ----
        n_live_after = len(live_obj_ids) - len([a for a in actions if a["type"] == "remove"])
        # Claimed retired identities within this checkpoint, so two unmatched
        # detections cannot both revive the same one.
        taken: set = set()
        # Candidates seen at THIS checkpoint. Anything pending that is not seen
        # again is dropped below, so MIN_CONFIRM_CHECKPOINTS really does mean
        # consecutive -- previously a counter survived arbitrary absences, so
        # detection / gap / detection confirmed a player that was never
        # continuously present.
        confirmed_this_round: set = set()
        for di, box in enumerate(det_boxes_xyxy):
            if di in matched_det_idx or n_live_after >= MAX_LIVE_OBJECTS:
                continue
            if not self.court_test_fn(box):
                continue
            # A genuinely new player should be seen again at the next checkpoint --
            # but "the same place" has to mean "the same person", not "the same
            # square of the image". Follow the candidate by box overlap so a
            # running player confirms as readily as a standing one.
            slot = self._match_pending(box)
            if slot is None:
                self._pending_new_boxes.append([box.copy(), 1])
                confirmed_this_round.add(len(self._pending_new_boxes) - 1)
                continue
            self._pending_new_boxes[slot][0] = box.copy()
            self._pending_new_boxes[slot][1] += 1
            confirmed_this_round.add(slot)
            if self._pending_new_boxes[slot][1] < MIN_CONFIRM_CHECKPOINTS:
                continue
            self._pending_new_boxes[slot][1] = 0   # consumed; drop below

            is_gk = bool(det_is_goalkeeper[di])
            emb = det_embeddings[di]
            team = int(det_teams[di])
            team_conf = float(det_confidence[di])
            team_quality = float(det_quality[di])

            match = self.registry.reid_match(
                emb, frame_idx, taken, team_id=team,
                confidence=team_conf, quality=team_quality, is_goalkeeper=is_gk,
            )
            if match is not None:
                oid = match.player_id
                taken.add(oid)
                self.registry.revive(
                    match, frame_idx, oid, is_goalkeeper=is_gk,
                    team=team, confidence=team_conf, quality=team_quality,
                )
                self.tracks[oid] = Track(obj_id=oid)
                self.events.append({"frame": frame_idx, "type": "reid", "obj_id": oid})
            else:
                player = self.registry.create(
                    frame_idx, None, team, emb, is_gk,
                    confidence=team_conf, quality=team_quality,
                )
                oid = player.player_id
                self.tracks[oid] = Track(obj_id=oid)
                self.events.append({"frame": frame_idx, "type": "add_new", "obj_id": oid})
            actions.append({"type": "add", "obj_id": oid, "box": box})
            n_live_after += 1

        # Prune: a candidate not seen at this checkpoint has broken its run, and
        # one already consumed into a track has count 0.
        self._pending_new_boxes = [
            entry for index, entry in enumerate(self._pending_new_boxes)
            if index in confirmed_this_round and entry[1] > 0
        ]

        return actions
