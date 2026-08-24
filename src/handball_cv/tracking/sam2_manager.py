"""Track lifecycle policy for the SAM2 offline predictor.

SAM2 seeded once on frame 0 never recovers from a track jumping onto the
wrong player or a player entering after the seed frame. This module drives
periodic detector checkpoints against the live tracks and issues the
add / remove / re-prompt calls SAM2 needs to correct itself, keyed by
`obj_id` (never by index -- `remove_object` renumbers internal indices).

Team is tracked as a running vote (`Track.voted_team_id`), not frozen at
creation: a single frame-0 read can be wrong (occlusion, an off-colour crop),
and freezing it means a re-ID hit inherits a stale, possibly incorrect label
forever. Team and goalkeeper status also gate re-ID candidates -- a retired
player should never be revived onto the opposing team or across the
keeper/field-player boundary.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from handball_cv.teams.model import (
    MIN_STABLE_TEAM_CONFIDENCE, MIN_TEAM_VOTE_CONFIDENCE,
    record_team_vote, team_vote_confidence, voted_team_id,
)

# tunables
HISTORY_LEN = 30
MIN_CONFIRM_CHECKPOINTS = 2          # consecutive checkpoints before acting
DUPLICATE_IOU_MIN = 0.6              # mask intersection-over-min-area
DROPOUT_AREA_FRAC = 0.2              # of the track's own running median
DRIFT_IOU_LOW, DRIFT_IOU_HIGH = 0.1, 0.5
MATCH_IOU_MIN = 0.1                  # below this, a detection counts as unmatched
REID_COS_SIM_MIN = 0.7
# Units bug fix: this was previously compared against a *frame* index while
# named/commented as a checkpoint count ("~300 frames at CHECK_EVERY=10"), so
# the real re-ID window was only 30 frames (~1.25s @ 24fps) instead of the
# intended ~300. Renamed to make the unit explicit and set to the value the
# original comment intended.
REID_MAX_GAP_FRAMES = 300
MAX_LIVE_OBJECTS = 20


@dataclass
class Track:
    obj_id: int
    team_id: int            # creation-time guess; prefer `voted_team_id`
    embedding: np.ndarray
    is_goalkeeper: bool      # from the detector's own class, never re-guessed
    created_at: int
    last_seen: int
    centroids: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    areas: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    miss_count: int = 0
    team_votes: dict = field(default_factory=dict)
    # pending-action confirmation counters, reset whenever the condition lapses
    _pending: dict = field(default_factory=dict)

    @property
    def voted_team_id(self) -> int:
        return voted_team_id(self.team_votes, self.team_id)

    @property
    def team_confidence(self) -> float:
        return team_vote_confidence(self.team_votes)

    def record_team_vote(
        self, team_id: int, confidence: float = 1.0, quality: float = 1.0,
    ) -> None:
        record_team_vote(self.team_votes, team_id, confidence, quality)

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
    """Owns per-track state and decides add/remove/reprompt actions.

    Does not call the SAM2 predictor itself -- `checkpoint()` returns a list
    of actions for the caller to apply, since the caller owns the predictor
    session and frame cache.
    """

    def __init__(self, team_model, court_test_fn, next_obj_id_start=1):
        self.team_model = team_model  # a team_model.TeamModel
        self.court_test_fn = court_test_fn  # (xyxy) -> bool, inside playing surface
        self.tracks: dict[int, Track] = {}
        self.retired: list[Track] = []  # re-ID pool
        self._next_id = next_obj_id_start
        self.events: list[dict] = []
        # confirmation counters for not-yet-seen players, keyed by a coarse
        # spatial slot (no Track object exists for these yet)
        self._pending_new: dict[str, int] = {}

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def seed(
        self, frame_idx: int, boxes_xyxy: np.ndarray, frame: np.ndarray,
        is_goalkeeper: np.ndarray,
    ) -> list[int]:
        """Initial frame-0 seeding. Returns assigned obj_ids in input order."""
        boxes_xyxy = np.asarray(boxes_xyxy).reshape(-1, 4)
        embeddings, teams, confidence, quality = self.team_model.observe(
            frame, boxes_xyxy
        )
        obj_ids = []
        for emb, team, conf, crop_quality, is_gk in zip(
            embeddings, teams, confidence, quality, is_goalkeeper
        ):
            oid = self._alloc_id()
            track = Track(
                obj_id=oid, team_id=int(team), embedding=emb,
                is_goalkeeper=bool(is_gk), created_at=frame_idx,
                last_seen=frame_idx,
            )
            if (
                not bool(is_gk)
                and conf >= MIN_TEAM_VOTE_CONFIDENCE
                and crop_quality > 0
            ):
                track.record_team_vote(int(team), float(conf), float(crop_quality))
            self.tracks[oid] = track
            obj_ids.append(oid)
        return obj_ids

    def update_from_propagation(self, frame_idx: int, obj_ids, masks: np.ndarray):
        """Record centroid/area history for every live track from tracker output.

        masks: (N, H, W) bool, ordered to match obj_ids.
        """
        for oid, m in zip(obj_ids, masks):
            t = self.tracks.get(int(oid))
            if t is None:
                continue
            area = float(m.sum())
            t.areas.append(area)
            if area > 0:
                ys, xs = np.nonzero(m)
                t.centroids.append((float(xs.mean()), float(ys.mean())))
                t.last_seen = frame_idx

    def _reid_match(
        self, crop_embedding: np.ndarray, frame_idx: int,
        team_id: Optional[int] = None, team_conf: float = 0.0,
        is_goalkeeper: Optional[bool] = None,
    ):
        """Best retired player above the similarity floor, or None.

        Never revives a track across the goalkeeper/field-player boundary --
        that comes straight from the detector's own class, so it is always
        trusted. A team mismatch only excludes a candidate when the new
        detection's own team prediction is confident enough to trust; a
        low-confidence read must not permanently forbid the correct match.
        """
        best, best_sim = None, REID_COS_SIM_MIN
        for cand in self.retired:
            if frame_idx - cand.last_seen > REID_MAX_GAP_FRAMES:
                continue
            if is_goalkeeper is not None and cand.is_goalkeeper != is_goalkeeper:
                continue
            if (
                team_id is not None
                and team_conf >= MIN_TEAM_VOTE_CONFIDENCE
                and cand.team_confidence >= MIN_STABLE_TEAM_CONFIDENCE
                and cand.voted_team_id != team_id
            ):
                continue
            sim = float(
                np.dot(crop_embedding, cand.embedding)
                / (np.linalg.norm(crop_embedding) * np.linalg.norm(cand.embedding) + 1e-8)
            )
            if sim > best_sim:
                best, best_sim = cand, sim
        return best

    def checkpoint(
        self, frame_idx, frame, live_obj_ids, live_masks, det_boxes_xyxy,
        det_is_goalkeeper=None,
    ):
        """One detector checkpoint. Returns a list of action dicts:

          {"type": "remove", "obj_id": int}
          {"type": "reprompt", "obj_id": int, "box": xyxy}
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
        else:
            det_embeddings = np.empty((0, 0), dtype=float)
            det_teams = np.empty(0, dtype=int)
            det_confidence = np.empty(0, dtype=float)
            det_quality = np.empty(0, dtype=float)

        for oid, detection_index in track_to_det.items():
            confidence = float(det_confidence[detection_index])
            quality = float(det_quality[detection_index])
            if (
                not bool(det_is_goalkeeper[detection_index])
                and confidence >= MIN_TEAM_VOTE_CONFIDENCE
                and quality > 0
            ):
                self.tracks[oid].record_team_vote(
                    int(det_teams[detection_index]), confidence, quality
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
                            self.events.append({
                                "frame": frame_idx, "type": "remove_duplicate",
                                "obj_id": jumper.obj_id,
                                "kept": tb.obj_id if jumper is ta else ta.obj_id,
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
                    self.retired.append(t)
                    self.events.append({"frame": frame_idx, "type": "remove_gone", "obj_id": oid})

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
                    actions.append({"type": "reprompt", "obj_id": oid, "box": box})
                    self.events.append({
                        "frame": frame_idx, "type": "reprompt", "obj_id": oid,
                        "iou": float(track_iou),
                    })

        # decay confirmation counters for conditions that didn't fire this round
        for oid in live_obj_ids:
            self.tracks[oid].clear_all_but(active_keys_per_track[oid])

        # ---- rule 4: new player (unmatched detection, inside court, confirmed) ----
        n_live_after = len(live_obj_ids) - len([a for a in actions if a["type"] == "remove"])
        for di, box in enumerate(det_boxes_xyxy):
            if di in matched_det_idx or n_live_after >= MAX_LIVE_OBJECTS:
                continue
            if not self.court_test_fn(box):
                continue
            # coarse spatial slot: a genuinely new player should re-appear in
            # roughly the same place across consecutive checkpoints
            key = f"{round(box[0] / 50)}_{round(box[1] / 50)}"
            cnt = self._pending_new.get(key, 0) + 1
            self._pending_new[key] = cnt
            if cnt < MIN_CONFIRM_CHECKPOINTS:
                continue
            del self._pending_new[key]

            is_gk = bool(det_is_goalkeeper[di])
            emb = det_embeddings[di]
            team = int(det_teams[di])
            team_conf = float(det_confidence[di])
            team_quality = float(det_quality[di])

            match = self._reid_match(
                emb, frame_idx, team_id=team,
                team_conf=team_conf * team_quality, is_goalkeeper=is_gk,
            )
            if match is not None:
                oid = match.obj_id
                self.retired.remove(match)
                self.tracks[oid] = match
                self.tracks[oid].last_seen = frame_idx
                if (
                    not is_gk
                    and team_conf >= MIN_TEAM_VOTE_CONFIDENCE
                    and team_quality > 0
                ):
                    self.tracks[oid].record_team_vote(
                        team, team_conf, team_quality
                    )
                self.events.append({"frame": frame_idx, "type": "reid", "obj_id": oid})
            else:
                oid = self._alloc_id()
                track = Track(
                    obj_id=oid, team_id=team, embedding=emb,
                    is_goalkeeper=is_gk, created_at=frame_idx,
                    last_seen=frame_idx,
                )
                if (
                    not is_gk
                    and team_conf >= MIN_TEAM_VOTE_CONFIDENCE
                    and team_quality > 0
                ):
                    track.record_team_vote(team, team_conf, team_quality)
                self.tracks[oid] = track
                self.events.append({"frame": frame_idx, "type": "add_new", "obj_id": oid})
            actions.append({"type": "add", "obj_id": oid, "box": box})
            n_live_after += 1

        return actions
