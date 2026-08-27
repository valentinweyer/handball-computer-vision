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
their appearance in memory. `Track.embedding` (EMA-refreshed at every
checkpoint) versus the checkpoint's own detection embedding is what tells
the two cases apart.

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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


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
        is_goalkeeper: Optional[bool] = None, taken: Optional[set] = None,
    ):
        """Best retired player above the similarity floor, or None.

        Never revives a track across the goalkeeper/field-player boundary --
        that comes straight from the detector's own class, so it is always
        trusted. A team mismatch only excludes a candidate when the new
        detection's own team prediction is confident enough to trust; a
        low-confidence read must not permanently forbid the correct match.
        `taken` excludes candidates already claimed by another detection in
        this same checkpoint, so two new players cannot both revive the same
        retired identity.
        """
        best, best_sim = None, REID_COS_SIM_MIN
        for cand in self.retired:
            if taken is not None and cand.obj_id in taken:
                continue
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
            sim = _cosine_similarity(crop_embedding, cand.embedding)
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
            oid: self.tracks[oid].embedding for oid in track_to_det
        }

        for oid, detection_index in track_to_det.items():
            confidence = float(det_confidence[detection_index])
            quality = float(det_quality[detection_index])
            track = self.tracks[oid]
            if quality > 0:
                # Gated on crop legibility alone (not team-vote confidence,
                # not goalkeeper status) -- this is an appearance estimate,
                # not a team-color read, and goalkeepers need their embedding
                # refreshed too so they remain re-ID-able.
                track.embedding = (
                    EMBEDDING_EMA_ALPHA * det_embeddings[detection_index]
                    + (1 - EMBEDDING_EMA_ALPHA) * track.embedding
                )
            if (
                not bool(det_is_goalkeeper[detection_index])
                and confidence >= MIN_TEAM_VOTE_CONFIDENCE
                and quality > 0
            ):
                track.record_team_vote(
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
                            # Unlike rule 2's dropout, duplicate resolution is a
                            # guess (larger recent jump) that can pick the wrong
                            # one of the pair -- retiring it, like rule 2 does,
                            # means a wrong guess is still recoverable by re-ID
                            # instead of permanently destroying that identity.
                            self.retired.append(jumper)
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
                taken=taken,
            )
            if match is not None:
                oid = match.obj_id
                taken.add(oid)
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
