"""Stable player identity on top of a tracking-by-detection tracker.

McByte assigns a `tracker_id` per tracklet and retires it once the detector has
missed it for `lost_track_buffer` frames. A player who walks behind the goal and
reappears therefore comes back under a fresh `tracker_id`. This module maps
`tracker_id -> player_id`, stitching those fragments back together by SigLIP
appearance similarity against a gallery of retired players.

Team is sampled continuously at a bounded cadence. A tracked label is not
permanent: three consecutive qualified observations for the opposite team
replace it. This hysteresis corrects an unlucky initial crop without allowing
one noisy frame to flip the label.

That mechanism corrects *classifier* error, and it must not silently absorb
*tracker* error too. When McByte swaps two crossing players the observations are
right about the pixels and wrong about the identity, and they look exactly like
a correction. Evidence therefore decays instead of accumulating, so confidence
falls as soon as a label is contested and the identity stops vetoing re-ID
candidates on a claim it can no longer support; a flip on a settled label is
reported as `suspected_id_switch` rather than as a clean correction.

Team and detector-provided goalkeeper status gate re-ID only from a position of
evidence -- the incoming observation qualified, the candidate's own label
settled. Neither is frozen at creation.

Replaces the SAM2-era `track_manager.TrackManager`. The add/remove/reprompt rules
there existed only to correct a one-time-prompt propagator; McByte re-anchors on
detector boxes every frame, so all that remains is identity.
"""
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from team_model import (
    MIN_STABLE_TEAM_CONFIDENCE, MIN_TEAM_VOTE_CONFIDENCE,
    record_team_vote,
)

# tunables
REID_COS_SIM_MIN = 0.7          # cosine similarity floor for reviving a retired player
REID_MAX_GAP_FRAMES = 300       # don't re-ID against players gone longer than this
TEAM_OBSERVATION_INTERVAL = 5
TEAM_SWITCH_OBSERVATIONS = 3
TEAM_SWITCH_MIN_QUALITY = 0.40
# Evidence decays rather than accumulating without bound. A running total
# saturates `team_confidence` near 1.0 within a few hundred frames, after which
# the label reads as stable no matter what later frames say -- so a player whose
# team is actively contested still vetoes re-ID candidates. Decay bounds the
# mass at roughly weight/(1 - decay) and lets confidence fall again.
# 0.85 caps a saturated label at ~0.93 confidence, from which two opposing
# observations drop it under MIN_STABLE_TEAM_CONFIDENCE -- one clear of the
# third that fires the switch. At 0.9 the ceiling is 0.95 and a fully saturated
# label was still vetoing re-ID at the moment it flipped.
TEAM_EVIDENCE_DECAY = 0.85
# Qualified observations behind a label before a flip is more likely to mean the
# tracker changed person than that the classifier was wrong all along.
TEAM_SETTLED_OBSERVATIONS = 5
# Detector-class observations before goalkeeper status may veto an appearance
# match. Mirrors MIN_STABLE_TEAM_CONFIDENCE: a role needs evidence to gate.
GOALKEEPER_EVIDENCE_MIN = 5
GOALKEEPER_EVIDENCE_MAJORITY = 0.70


def is_qualified(confidence: float, quality: float) -> bool:
    """Whether one observation is allowed to move tracked state.

    The team vote and the re-ID team gate read the same two numbers and used to
    disagree about what counted -- the gate compared `confidence * quality`
    against a threshold meant for `confidence` alone, making it silently much
    stricter. One predicate so they cannot drift apart again.
    """
    return (
        confidence >= MIN_TEAM_VOTE_CONFIDENCE
        and quality >= TEAM_SWITCH_MIN_QUALITY
    )


@dataclass
class Player:
    player_id: int
    team_id: int             # creation-time guess; prefer `voted_team_id`
    embedding: np.ndarray
    created_goalkeeper: bool  # detector class at creation; only a tie-breaker
    created_at: int
    last_seen: int
    team_switch_observations: int = TEAM_SWITCH_OBSERVATIONS
    # every tracker_id this player has been seen under; len() > 1 means McByte
    # fragmented the identity and re-ID stitched it back
    tracker_ids: list = field(default_factory=list)
    team_votes: dict = field(default_factory=dict)
    team_observations: int = 0
    # observations that passed both gates; `team_observations` also counts the
    # weak ones, which may not settle a label
    qualified_observations: int = 0
    last_team_observation: int = -1_000_000
    pending_team_id: int | None = None
    pending_team_observations: int = 0
    pending_team_weight: float = 0.0
    # `team_confidence` as it stood when the current opposition began, before
    # those opposing votes dragged it down
    pending_start_confidence: float = 0.0
    team_switches: int = 0
    goalkeeper_observations: int = 0
    field_observations: int = 0

    @property
    def voted_team_id(self) -> int:
        return self.team_id

    @property
    def team_confidence(self) -> float:
        total = sum(self.team_votes.values()) + 1.0
        return float((self.team_votes.get(self.team_id, 0.0) + 0.5) / total)

    @property
    def team_is_provisional(self) -> bool:
        """True while `team_id` rests on no qualified observation at all.

        `update()` seeds `team_id` from whatever the first frame said, whether or
        not that read was confident enough to vote.
        """
        return self.qualified_observations == 0

    @property
    def is_goalkeeper(self) -> bool:
        """Running majority of the detector's own class.

        Never re-guessed by us -- but not frozen to one frame either. The class
        flickers, and freezing it at creation permanently forked an identity
        from its own fragments across the re-ID role gate.
        """
        if self.goalkeeper_observations == self.field_observations:
            return self.created_goalkeeper
        return self.goalkeeper_observations > self.field_observations

    @property
    def goalkeeper_settled(self) -> bool:
        """Whether the role has enough agreeing evidence to veto a match."""
        total = self.goalkeeper_observations + self.field_observations
        if total < GOALKEEPER_EVIDENCE_MIN:
            return False
        majority = max(self.goalkeeper_observations, self.field_observations)
        return majority >= GOALKEEPER_EVIDENCE_MAJORITY * total

    def record_role(self, goalkeeper: bool) -> None:
        """One frame of detector class. Free -- no model call -- so every frame counts."""
        if goalkeeper:
            self.goalkeeper_observations += 1
        else:
            self.field_observations += 1

    def record_team_vote(
        self, team_id: int, confidence: float = 1.0, quality: float = 1.0,
    ) -> str | None:
        """Record evidence and report what it did to the label.

        Only a qualified observation touches `team_votes`. A routine
        low-quality read (partial occlusion, blur, a small crop) must not
        erode trust in an established, correctly-tracked label -- confidence
        should fall only in response to real opposing evidence, never as a
        side effect of noise passing through. Callers pre-filter on a looser
        confidence/quality floor before reaching here (there is no point
        recording a read with zero confidence or zero quality at all), so
        `team_observations` -- a diagnostic count of attempted reads -- still
        counts every call; only qualification gates the vote itself.

        Returns None for ordinary evidence, "adopt" when a provisional label is
        replaced by its first qualified observation, "switch" when sustained
        opposition replaces a weakly-evidenced label, and "id_switch" when it
        replaces a settled one -- which more likely means the tracker changed
        person than that the classifier was wrong all along.
        """
        self.team_observations += 1
        if not is_qualified(confidence, quality):
            return None

        # Decay first, then add: evidence is a bounded recency-weighted
        # accumulator, so confidence can fall when a label becomes contested.
        for other in self.team_votes:
            self.team_votes[other] *= TEAM_EVIDENCE_DECAY
        before = sum(self.team_votes.values())
        record_team_vote(self.team_votes, team_id, confidence, quality)
        weight = sum(self.team_votes.values()) - before
        if weight <= 0:
            return None
        provisional = self.team_is_provisional
        self.qualified_observations += 1

        # A label backed by nothing has no claim to inertia. Requiring three
        # opposing observations to shed a guess that was never itself qualified
        # left players displaying an unevidenced team for ~15 frames.
        if provisional and team_id != self.team_id:
            self.team_id = team_id
            self._clear_pending()
            return "adopt"

        if team_id == self.team_id:
            self._clear_pending()
            return None
        if self.pending_team_id != team_id:
            self.pending_team_id = team_id
            self.pending_team_observations = 1
            self.pending_team_weight = weight
            self.pending_start_confidence = self.team_confidence
        else:
            self.pending_team_observations += 1
            self.pending_team_weight += weight
        if self.pending_team_observations < self.team_switch_observations:
            return None

        settled = (
            self.pending_start_confidence >= MIN_STABLE_TEAM_CONFIDENCE
            and self.qualified_observations >= TEAM_SETTLED_OBSERVATIONS
        )
        # Prior evidence is kept, only decayed. Discarding it here left a
        # just-flipped player reading as stable (~0.78, above the gate) and
        # still vetoing re-ID candidates on a label that had just proven
        # unreliable -- a tracking error hardening into a re-ID error.
        self.team_id = team_id
        self._clear_pending()
        self.team_switches += 1
        return "id_switch" if settled else "switch"

    def _clear_pending(self) -> None:
        self.pending_team_id = None
        self.pending_team_observations = 0
        self.pending_team_weight = 0.0
        self.pending_start_confidence = 0.0


class IdentityManager:
    """Maps McByte tracker_ids to stable player_ids.

    Does not touch the tracker. `update()` is called with the tracker's output
    for the current frame; `retire_missing()` is called with the set of
    tracker_ids the tracker still considers alive.
    """

    def __init__(
        self,
        team_model,
        reid_cos_sim_min: float = REID_COS_SIM_MIN,
        reid_max_gap_frames: int = REID_MAX_GAP_FRAMES,
        goalkeeper_class_id: int = 1,
        team_switch_observations: int = TEAM_SWITCH_OBSERVATIONS,
    ):
        self.team_model = team_model  # a team_model.TeamModel
        self.reid_cos_sim_min = reid_cos_sim_min
        self.reid_max_gap_frames = reid_max_gap_frames
        self.goalkeeper_class_id = goalkeeper_class_id
        self.team_switch_observations = max(1, int(team_switch_observations))

        self.players: dict[int, Player] = {}
        self.tracker_to_player: dict[int, int] = {}
        self.retired: list[Player] = []
        self.events: list[dict] = []
        self._next_id = 1

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _embed(
        self, frame_rgb: np.ndarray, boxes_xyxy: np.ndarray,
        context_boxes_xyxy: np.ndarray | None = None,
    ):
        """One batched SigLIP pass plus color and crop-quality evidence."""
        return self.team_model.observe(
            frame_rgb, boxes_xyxy, context_boxes_xyxy
        )

    def _reid_match(
        self, embedding: np.ndarray, frame_idx: int, taken: set,
        team_id: int = None, confidence: float = 0.0, quality: float = 0.0,
        is_goalkeeper: bool = None,
    ):
        """Best retired player above the similarity floor, or None.

        Team and goalkeeper role may both veto a match, but only from a position
        of evidence: the incoming observation must itself be qualified, and the
        candidate's own label must be settled. One flickered detector class or
        one weak colour read must not permanently forbid the correct match --
        and a label whose team is currently contested has stopped being a
        trustworthy veto, which falling `team_confidence` now expresses.
        """
        best, best_sim = None, self.reid_cos_sim_min
        incoming_qualified = is_qualified(confidence, quality)
        for cand in self.retired:
            if cand.player_id in taken:
                continue
            if frame_idx - cand.last_seen > self.reid_max_gap_frames:
                continue
            if (
                is_goalkeeper is not None
                and cand.goalkeeper_settled
                and cand.is_goalkeeper != is_goalkeeper
            ):
                continue
            if (
                team_id is not None
                and incoming_qualified
                and not cand.team_is_provisional
                and cand.team_confidence >= MIN_STABLE_TEAM_CONFIDENCE
                and cand.voted_team_id != team_id
            ):
                continue
            sim = float(
                np.dot(embedding, cand.embedding)
                / (np.linalg.norm(embedding) * np.linalg.norm(cand.embedding) + 1e-8)
            )
            if sim > best_sim:
                best, best_sim = cand, sim
        return best

    def update(
        self, frame_idx: int, frame_rgb: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Assign stable player ids and bootstrap team evidence over a tracklet."""
        if len(detections) == 0:
            return np.empty(0, dtype=int)

        tracker_ids = detections.tracker_id.astype(int)
        player_ids = np.empty(len(tracker_ids), dtype=int)
        class_ids = detections.class_id
        is_goalkeeper = (
            class_ids == self.goalkeeper_class_id
            if class_ids is not None
            else np.zeros(len(detections), dtype=bool)
        )
        unseen_pos = [
            i for i, tracker_id in enumerate(tracker_ids)
            if int(tracker_id) not in self.tracker_to_player
        ]
        unseen_set = set(unseen_pos)

        for i, tracker_id in enumerate(tracker_ids):
            player_id = self.tracker_to_player.get(int(tracker_id))
            if player_id is not None:
                player_ids[i] = player_id
                player = self.players[player_id]
                player.last_seen = frame_idx
                player.record_role(bool(is_goalkeeper[i]))

        observation_pos = list(unseen_pos)
        for i, tracker_id in enumerate(tracker_ids):
            if i in unseen_set or is_goalkeeper[i]:
                continue
            player = self.players[self.tracker_to_player[int(tracker_id)]]
            cadence_ready = (
                frame_idx - player.last_team_observation
                >= TEAM_OBSERVATION_INTERVAL
            )
            if cadence_ready:
                observation_pos.append(i)

        observations = {}
        if observation_pos:
            embeddings, teams, confidence, quality = self._embed(
                frame_rgb, detections.xyxy[observation_pos], detections.xyxy
            )
            observations = {
                position: (
                    embeddings[slot],
                    int(teams[slot]),
                    float(confidence[slot]),
                    float(quality[slot]),
                )
                for slot, position in enumerate(observation_pos)
            }

        for i in observation_pos:
            if i in unseen_set:
                continue
            player_id = self.tracker_to_player[int(tracker_ids[i])]
            player = self.players[player_id]
            _embedding, team, confidence, quality = observations[i]
            player.last_team_observation = frame_idx
            if confidence >= MIN_TEAM_VOTE_CONFIDENCE and quality > 0:
                outcome = player.record_team_vote(team, confidence, quality)
                if outcome in ("switch", "id_switch"):
                    self.events.append({
                        "frame": frame_idx,
                        "type": (
                            "suspected_id_switch" if outcome == "id_switch"
                            else "team_switch"
                        ),
                        "player_id": player_id,
                        "tracker_id": int(tracker_ids[i]),
                        "team_id": player.voted_team_id,
                    })

        if unseen_pos:
            taken = set()
            for i in unseen_pos:
                tracker_id = int(tracker_ids[i])
                embedding, team, confidence, quality = observations[i]
                goalkeeper = bool(is_goalkeeper[i])
                match = self._reid_match(
                    embedding,
                    frame_idx,
                    taken,
                    team_id=team,
                    confidence=confidence,
                    quality=quality,
                    is_goalkeeper=goalkeeper,
                )
                if match is not None:
                    self.retired.remove(match)
                    self.players[match.player_id] = match
                    match.last_seen = frame_idx
                    match.last_team_observation = frame_idx
                    match.tracker_ids.append(tracker_id)
                    match.record_role(goalkeeper)
                    if (
                        not goalkeeper
                        and confidence >= MIN_TEAM_VOTE_CONFIDENCE
                        and quality > 0
                    ):
                        match.record_team_vote(team, confidence, quality)
                    taken.add(match.player_id)
                    player_id = match.player_id
                    self.events.append({
                        "frame": frame_idx,
                        "type": "reid",
                        "player_id": player_id,
                        "tracker_id": tracker_id,
                    })
                else:
                    player_id = self._alloc_id()
                    player = Player(
                        player_id=player_id,
                        team_id=team,
                        embedding=embedding,
                        created_goalkeeper=goalkeeper,
                        created_at=frame_idx,
                        last_seen=frame_idx,
                        team_switch_observations=self.team_switch_observations,
                        tracker_ids=[tracker_id],
                        last_team_observation=frame_idx,
                    )
                    player.record_role(goalkeeper)
                    if (
                        not goalkeeper
                        and confidence >= MIN_TEAM_VOTE_CONFIDENCE
                        and quality > 0
                    ):
                        player.record_team_vote(team, confidence, quality)
                    self.players[player_id] = player
                    self.events.append({
                        "frame": frame_idx,
                        "type": "add_new",
                        "player_id": player_id,
                        "tracker_id": tracker_id,
                    })
                self.tracker_to_player[tracker_id] = player_id
                player_ids[i] = player_id

        return player_ids

    def retire_missing(self, frame_idx: int, alive_tracker_ids) -> None:
        """Move players whose tracker_id the tracker has dropped into the re-ID gallery.

        McByte has already applied `lost_track_buffer` before a tracker_id
        disappears from `tracked_objects`, so no extra grace period is needed here.
        """
        alive = {int(t) for t in alive_tracker_ids}
        for tid in [t for t in self.tracker_to_player if t not in alive]:
            pid = self.tracker_to_player.pop(tid)
            player = self.players.pop(pid, None)
            if player is not None:
                self.retired.append(player)
                self.events.append({
                    "frame": frame_idx, "type": "retire",
                    "player_id": pid, "tracker_id": tid,
                })

    def team_by_player_id(self) -> dict[int, int]:
        """Current best (voted) team label for every player_id ever allocated."""
        return {p.player_id: p.voted_team_id for p in list(self.players.values()) + self.retired}

    def summary(self) -> dict:
        everyone = list(self.players.values()) + self.retired
        fragments = {p.player_id: len(p.tracker_ids) for p in everyone}
        return {
            "players": len(everyone),
            "tracker_ids_consumed": sum(fragments.values()),
            "reid_hits": sum(1 for e in self.events if e["type"] == "reid"),
            "team_switches": sum(
                1 for e in self.events if e["type"] == "team_switch"
            ),
            "suspected_id_switches": sum(
                1 for e in self.events if e["type"] == "suspected_id_switch"
            ),
            "new_allocations": sum(1 for e in self.events if e["type"] == "add_new"),
            "max_fragments_per_player": max(fragments.values()) if fragments else 0,
            "fragmented_players": sum(1 for n in fragments.values() if n > 1),
        }
