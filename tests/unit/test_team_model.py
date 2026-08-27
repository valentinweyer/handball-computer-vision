import unittest
from types import SimpleNamespace

import numpy as np
import supervision as sv

from handball_cv.tracking.identity import (
    MIN_STABLE_TEAM_CONFIDENCE,
    MIN_TEAM_VOTE_CONFIDENCE,
    TEAM_SWITCH_MIN_QUALITY,
    IdentityManager,
    Player,
    is_qualified,
)
from handball_cv.teams.masks import guarded_torso_masks
from handball_cv.tracking.sam2_manager import (
    BODY_SWAP_COS_SIM_MIN,
    EMBEDDING_EMA_ALPHA,
    Track,
    TrackManager,
)
from handball_cv.teams.model import (
    crop_quality,
    jersey_color_features,
    masked_jersey_color_features,
    record_team_vote,
    team_vote_confidence,
    torso_contamination,
    voted_team_id,
)


class TeamFeatureTests(unittest.TestCase):
    def test_color_features_are_finite_and_separate_distinct_kits(self):
        blue = np.zeros((64, 32, 3), dtype=np.uint8)
        blue[:] = (20, 30, 220)
        white = np.full((64, 32, 3), 235, dtype=np.uint8)
        features = jersey_color_features([blue, white])
        self.assertEqual(features.shape, (2, 62))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(np.linalg.norm(features[0] - features[1])), 0.5)

    def test_crop_quality_rejects_empty_and_flat_crops(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        flat = np.full((64, 32, 3), 128, dtype=np.uint8)
        textured = np.zeros((64, 32, 3), dtype=np.uint8)
        textured[::2] = 255
        self.assertFalse(crop_quality(empty).accepted)
        self.assertFalse(crop_quality(flat).accepted)
        self.assertTrue(crop_quality(textured).accepted)
        self.assertGreater(crop_quality(textured).score, 0.0)

    def test_masked_color_features_ignore_the_other_half_of_a_crop(self):
        red = np.zeros((64, 32, 3), dtype=np.uint8)
        red[:] = (220, 20, 20)
        blue = np.zeros_like(red)
        blue[:] = (20, 30, 220)
        mixed = red.copy()
        mixed[:, 16:] = blue[:, 16:]
        left = np.zeros(mixed.shape[:2], dtype=bool)
        left[:, :16] = True
        right = ~left

        references = jersey_color_features([red, blue])
        selected = masked_jersey_color_features([mixed, mixed], [left, right])
        self.assertEqual(selected.shape, (2, 62))
        self.assertLess(
            np.linalg.norm(selected[0] - references[0]),
            np.linalg.norm(selected[0] - references[1]),
        )
        self.assertLess(
            np.linalg.norm(selected[1] - references[1]),
            np.linalg.norm(selected[1] - references[0]),
        )

    def test_masked_color_features_validate_masks_and_handle_empty_selection(self):
        crop = np.full((20, 10, 3), 127, dtype=np.uint8)
        features = masked_jersey_color_features(
            [crop], [np.zeros((20, 10), dtype=bool)]
        )
        np.testing.assert_array_equal(features, np.zeros((1, 62)))
        with self.assertRaises(ValueError):
            masked_jersey_color_features([crop], [])
        with self.assertRaises(ValueError):
            masked_jersey_color_features(
                [crop], [np.ones((10, 20), dtype=bool)]
            )

    def test_guarded_masks_use_spatial_assignment_and_remove_overlap_boundary(self):
        masks = np.zeros((2, 120, 120), dtype=bool)
        masks[0, 15:105, 20:50] = True
        masks[1, 15:105, 50:80] = True
        output = SimpleNamespace(
            masks=masks,
            tracklet_mask_dict={100: 0, 101: 1},
            mask_avg_prob_dict={100: 0.9, 101: 0.9},
        )
        boxes = np.array([[10, 10, 70, 110], [30, 10, 90, 110]])
        evidence = guarded_torso_masks(boxes, [100, 101], output)
        self.assertTrue(all(item.valid for item in evidence))
        self.assertEqual([item.mask_index for item in evidence], [0, 1])
        self.assertTrue(all(item.tracker_agrees for item in evidence))
        self.assertTrue(all(item.torso_coverage >= 0.15 for item in evidence))

    def test_guarded_masks_abstain_when_tracker_and_spatial_mask_disagree(self):
        masks = np.zeros((2, 120, 120), dtype=bool)
        masks[0, 15:105, 20:50] = True
        masks[1, 15:105, 50:80] = True
        output = SimpleNamespace(
            masks=masks,
            tracklet_mask_dict={100: 0, 101: 1},
            mask_avg_prob_dict={100: 0.9, 101: 0.9},
        )
        boxes = np.array([[10, 10, 70, 110], [30, 10, 90, 110]])
        evidence = guarded_torso_masks(boxes, [101, 100], output)
        self.assertEqual(
            [item.reason for item in evidence],
            ["tracker_mask_disagreement", "tracker_mask_disagreement"],
        )
        self.assertFalse(any(item.valid for item in evidence))

    def test_torso_contamination_detects_another_player(self):
        separated = np.array([[0, 0, 100, 200], [150, 0, 250, 200]])
        overlapping = np.array([[0, 0, 100, 200], [40, 50, 90, 150]])
        np.testing.assert_allclose(torso_contamination(separated), 0.0)
        self.assertGreater(torso_contamination(overlapping)[0], 0.25)

    def test_weighted_evidence_recovers_from_a_weak_wrong_first_read(self):
        evidence = {}
        record_team_vote(evidence, 0, confidence=0.2, quality=0.5)
        record_team_vote(evidence, 1, confidence=0.9, quality=1.0)
        record_team_vote(evidence, 1, confidence=0.9, quality=1.0)
        self.assertEqual(voted_team_id(evidence, fallback=0), 1)
        self.assertGreater(team_vote_confidence(evidence), 0.70)

    def test_rejected_evidence_does_not_change_state(self):
        evidence = {}
        record_team_vote(evidence, 1, confidence=0.9, quality=0.0)
        self.assertEqual(evidence, {})
        self.assertEqual(team_vote_confidence(evidence), 0.5)


class FakeTeamModel:
    def __init__(self):
        self.observed_frames = []

    def observe(self, frame_rgb, boxes_xyxy, context_boxes_xyxy=None):
        self.observed_frames.append(int(frame_rgb[0, 0, 0]))
        count = len(boxes_xyxy)
        return (
            np.ones((count, 4), dtype=float),
            np.ones(count, dtype=int),
            np.full(count, 0.9, dtype=float),
            np.ones(count, dtype=float),
        )

class SequenceTeamModel(FakeTeamModel):
    def __init__(self, team_by_frame, default_team=0):
        super().__init__()
        self.team_by_frame = team_by_frame
        self.default_team = default_team

    def observe(self, frame_rgb, boxes_xyxy, context_boxes_xyxy=None):
        frame_index = int(frame_rgb[0, 0, 0])
        self.observed_frames.append(frame_index)
        count = len(boxes_xyxy)
        team = self.team_by_frame.get(frame_index, self.default_team)
        return (
            np.ones((count, 4), dtype=float),
            np.full(count, team, dtype=int),
            np.full(count, 0.9, dtype=float),
            np.ones(count, dtype=float),
        )



class IdentityBootstrapTests(unittest.TestCase):
    def test_team_evidence_is_sampled_at_bounded_bootstrap_cadence(self):
        model = FakeTeamModel()
        manager = IdentityManager(model)
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 50, 100]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
            tracker_id=np.array([42]),
        )
        for frame_index in range(13):
            frame = np.full((2, 2, 3), frame_index, dtype=np.uint8)
            manager.update(frame_index, frame, detections)
        player = manager.players[1]
        self.assertEqual(model.observed_frames, [0, 5, 10])
        self.assertEqual(player.team_observations, 3)
        self.assertEqual(player.voted_team_id, 1)
        self.assertGreater(player.team_confidence, 0.70)

    def test_sustained_opposite_raw_predictions_replace_tracked_team(self):
        model = SequenceTeamModel({0: 0}, default_team=1)
        manager = IdentityManager(model, team_switch_observations=3)
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 50, 100]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
            tracker_id=np.array([42]),
        )
        labels = {}
        for frame_index in range(17):
            frame = np.full((2, 2, 3), frame_index, dtype=np.uint8)
            manager.update(frame_index, frame, detections)
            if frame_index in (0, 5, 10, 15):
                labels[frame_index] = manager.players[1].voted_team_id
        self.assertEqual(labels, {0: 0, 5: 0, 10: 0, 15: 1})
        self.assertEqual(manager.players[1].team_switches, 1)
        self.assertTrue(any(
            event["type"] == "team_switch" for event in manager.events
        ))


    def test_one_opposite_prediction_does_not_flip_tracked_team(self):
        model = SequenceTeamModel({5: 1}, default_team=0)
        manager = IdentityManager(model, team_switch_observations=3)
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 50, 100]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
            tracker_id=np.array([42]),
        )
        for frame_index in range(17):
            frame = np.full((2, 2, 3), frame_index, dtype=np.uint8)
            manager.update(frame_index, frame, detections)
        self.assertEqual(manager.players[1].voted_team_id, 0)
        self.assertEqual(manager.players[1].team_switches, 0)



class TrackManagerTeamTests(unittest.TestCase):
    def test_goalkeepers_do_not_contribute_to_two_team_evidence(self):
        model = FakeTeamModel()
        manager = TrackManager(model, court_test_fn=lambda _box: True)
        boxes = np.array(
            [[10, 10, 50, 100], [60, 10, 100, 100]], dtype=float
        )
        obj_ids = manager.seed(
            0,
            boxes,
            np.zeros((120, 120, 3), dtype=np.uint8),
            np.array([False, True]),
        )
        self.assertNotEqual(manager.tracks[obj_ids[0]].team_votes, {})
        self.assertEqual(manager.tracks[obj_ids[1]].team_votes, {})


if __name__ == "__main__":
    unittest.main()


class ScriptedTeamModel(FakeTeamModel):
    """Per-frame (team, confidence, quality), so weak reads can be scripted."""

    def __init__(self, script, default):
        super().__init__()
        self.script = script
        self.default = default

    def observe(self, frame_rgb, boxes_xyxy, context_boxes_xyxy=None):
        frame_index = int(frame_rgb[0, 0, 0])
        self.observed_frames.append(frame_index)
        count = len(boxes_xyxy)
        team, confidence, quality = self.script.get(frame_index, self.default)
        return (
            np.ones((count, 4), dtype=float),
            np.full(count, team, dtype=int),
            np.full(count, confidence, dtype=float),
            np.full(count, quality, dtype=float),
        )


def _one_detection(goalkeeper=False):
    return sv.Detections(
        xyxy=np.array([[10, 10, 50, 100]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([1 if goalkeeper else 2]),
        tracker_id=np.array([42]),
    )


def _run(manager, frames, detections):
    for frame_index in range(frames):
        manager.update(
            frame_index,
            np.full((2, 2, 3), frame_index, dtype=np.uint8),
            detections,
        )


class TrackerErrorIsolationTests(unittest.TestCase):
    """A tracking error must not quietly become a settled team label."""

    def _settled_candidate(self, switch_observations=4, observations=10):
        player = Player(
            player_id=1,
            team_id=0,
            embedding=np.array([1.0, 0.0, 0.0, 0.0]),
            created_goalkeeper=False,
            created_at=0,
            last_seen=0,
            team_switch_observations=switch_observations,
        )
        for _ in range(observations):
            player.record_team_vote(0, 0.9, 1.0)
        return player

    def test_flip_on_a_settled_label_is_reported_as_a_suspected_id_switch(self):
        # Ten agreeing reads, then sustained opposition -- what a McByte swap
        # between crossing players looks like from the classifier's side.
        model = ScriptedTeamModel(
            {f: (0, 0.9, 1.0) for f in range(0, 50)}, default=(1, 0.9, 1.0)
        )
        manager = IdentityManager(model, team_switch_observations=3)
        _run(manager, 62, _one_detection())

        player = manager.players[1]
        self.assertEqual(player.voted_team_id, 1)
        types = [event["type"] for event in manager.events]
        self.assertIn("suspected_id_switch", types)
        self.assertNotIn("team_switch", types)
        # The identity is now in doubt, and says so.
        self.assertLess(player.team_confidence, MIN_STABLE_TEAM_CONFIDENCE)

    def test_flip_on_a_weakly_evidenced_label_is_still_a_plain_correction(self):
        # Two agreeing reads is a bad first crop, not a lost identity.
        model = ScriptedTeamModel(
            {0: (0, 0.9, 1.0), 5: (0, 0.9, 1.0)}, default=(1, 0.9, 1.0)
        )
        manager = IdentityManager(model, team_switch_observations=3)
        _run(manager, 27, _one_detection())

        self.assertEqual(manager.players[1].voted_team_id, 1)
        types = [event["type"] for event in manager.events]
        self.assertIn("team_switch", types)
        self.assertNotIn("suspected_id_switch", types)

    def test_a_contested_label_stops_vetoing_reid_candidates(self):
        manager = IdentityManager(FakeTeamModel(), team_switch_observations=4)
        candidate = self._settled_candidate()
        manager.retired.append(candidate)
        query = np.array([1.0, 0.0, 0.0, 0.0])

        # Settled and unchallenged: team disagreement vetoes the match.
        self.assertIsNone(manager._reid_match(
            query, 5, set(), team_id=1, confidence=0.9, quality=1.0,
            is_goalkeeper=False,
        ))

        # Three opposing reads, one short of a switch. The label has not
        # changed, but it is no longer trustworthy enough to exclude anyone.
        for _ in range(3):
            candidate.record_team_vote(1, 0.9, 1.0)
        self.assertEqual(candidate.voted_team_id, 0)
        self.assertEqual(candidate.team_switches, 0)
        self.assertIsNotNone(manager._reid_match(
            query, 5, set(), team_id=1, confidence=0.9, quality=1.0,
            is_goalkeeper=False,
        ))

    def test_a_long_agreeing_run_can_still_be_contested(self):
        # The unbounded accumulator reached ~0.99 after a few hundred frames and
        # could never express doubt again. Decay caps the mass at
        # weight/(1 - decay), so even a 400-observation player stays reachable.
        player = self._settled_candidate(observations=400)
        saturated = player.team_confidence
        self.assertLessEqual(sum(player.team_votes.values()), 10.0)

        for _ in range(3):
            player.record_team_vote(1, 0.9, 1.0)
        self.assertLess(player.team_confidence, saturated)
        self.assertLess(player.team_confidence, MIN_STABLE_TEAM_CONFIDENCE)

    def test_unqualified_same_team_noise_does_not_erode_a_settled_label(self):
        # Decay must respond to real opposing evidence, not to every call --
        # a run of low-quality reads (partial occlusion, blur, a small crop)
        # that still agree on the team must leave confidence untouched.
        # Before this was fixed, ~20 such reads alone dragged a 0.93-confidence
        # player under the stable gate with zero genuine opposition.
        player = self._settled_candidate(observations=20)
        settled = player.team_confidence
        for _ in range(30):
            outcome = player.record_team_vote(0, 0.5, 0.1)
            self.assertIsNone(outcome)
        self.assertEqual(player.team_confidence, settled)
        self.assertGreaterEqual(player.team_confidence, MIN_STABLE_TEAM_CONFIDENCE)

    def test_unqualified_opposite_team_noise_does_not_erode_a_settled_label(self):
        # Same guarantee when the noisy reads disagree -- an unqualified read
        # must never nudge the pending-switch counter either.
        player = self._settled_candidate(observations=20)
        settled = player.team_confidence
        for _ in range(30):
            outcome = player.record_team_vote(1, 0.5, 0.1)
            self.assertIsNone(outcome)
        self.assertEqual(player.team_confidence, settled)
        self.assertIsNone(player.pending_team_id)

    def test_genuine_qualified_opposition_still_erodes_confidence(self):
        # The fix must not blunt real contest detection -- three qualified
        # opposing reads still fire the switch, exactly as before.
        player = self._settled_candidate(switch_observations=3, observations=20)
        for _ in range(2):
            self.assertIsNone(player.record_team_vote(1, 0.9, 1.0))
        self.assertLess(player.team_confidence, MIN_STABLE_TEAM_CONFIDENCE)
        self.assertEqual(player.record_team_vote(1, 0.9, 1.0), "id_switch")


class ProvisionalTeamLabelTests(unittest.TestCase):
    def test_unqualified_creation_label_yields_to_first_qualified_read(self):
        # Frame 0 is too weak to vote, but still seeds `team_id`.
        model = ScriptedTeamModel(
            {0: (0, 0.2, 1.0)}, default=(1, 0.9, 1.0)
        )
        manager = IdentityManager(model, team_switch_observations=3)
        manager.update(
            0, np.zeros((2, 2, 3), dtype=np.uint8), _one_detection()
        )
        player = manager.players[1]
        self.assertTrue(player.team_is_provisional)
        self.assertEqual(player.voted_team_id, 0)

        # One qualified read replaces it outright -- no three-observation wait.
        _run(manager, 6, _one_detection())
        self.assertFalse(player.team_is_provisional)
        self.assertEqual(player.voted_team_id, 1)
        self.assertEqual(player.team_switches, 0)
        self.assertNotIn(
            "team_switch", [event["type"] for event in manager.events]
        )


class QualifiedObservationTests(unittest.TestCase):
    def test_reid_gate_and_team_vote_agree_on_the_same_observation(self):
        # The re-ID gate used to test `confidence * quality` against a threshold
        # meant for confidence alone, so this pair voted but could not gate.
        confidence, quality = 0.4, 0.5
        self.assertTrue(is_qualified(confidence, quality))
        self.assertLess(confidence * quality, MIN_TEAM_VOTE_CONFIDENCE)

        player = Player(
            player_id=1, team_id=0, embedding=np.array([1.0, 0.0]),
            created_goalkeeper=False, created_at=0, last_seen=0,
        )
        self.assertEqual(player.record_team_vote(1, confidence, quality), "adopt")
        self.assertEqual(player.qualified_observations, 1)

    def test_quality_below_the_switch_floor_votes_but_cannot_settle(self):
        player = Player(
            player_id=1, team_id=0, embedding=np.array([1.0, 0.0]),
            created_goalkeeper=False, created_at=0, last_seen=0,
        )
        weak_quality = TEAM_SWITCH_MIN_QUALITY / 2
        self.assertIsNone(player.record_team_vote(1, 0.9, weak_quality))
        self.assertEqual(player.team_observations, 1)
        self.assertEqual(player.qualified_observations, 0)
        self.assertTrue(player.team_is_provisional)


class GoalkeeperRoleTests(unittest.TestCase):
    def _candidate(self, created_goalkeeper):
        return Player(
            player_id=1, team_id=0, embedding=np.array([1.0, 0.0, 0.0, 0.0]),
            created_goalkeeper=created_goalkeeper, created_at=0, last_seen=0,
        )

    def test_one_flickered_goalkeeper_class_does_not_block_reid(self):
        manager = IdentityManager(FakeTeamModel())
        candidate = self._candidate(created_goalkeeper=True)
        candidate.record_role(True)   # the single bad frame at creation
        manager.retired.append(candidate)

        self.assertIsNotNone(manager._reid_match(
            np.array([1.0, 0.0, 0.0, 0.0]), 5, set(), is_goalkeeper=False,
        ))

    def test_a_settled_goalkeeper_still_vetoes_a_field_player_match(self):
        manager = IdentityManager(FakeTeamModel())
        candidate = self._candidate(created_goalkeeper=True)
        for _ in range(10):
            candidate.record_role(True)
        manager.retired.append(candidate)

        self.assertTrue(candidate.goalkeeper_settled)
        self.assertIsNone(manager._reid_match(
            np.array([1.0, 0.0, 0.0, 0.0]), 5, set(), is_goalkeeper=False,
        ))

    def test_role_follows_the_running_majority_not_the_creation_frame(self):
        candidate = self._candidate(created_goalkeeper=True)
        candidate.record_role(True)
        for _ in range(9):
            candidate.record_role(False)
        self.assertFalse(candidate.is_goalkeeper)
        self.assertTrue(candidate.goalkeeper_settled)


def _box_mask(box, shape=(120, 120)):
    x1, y1, x2, y2 = [int(v) for v in box]
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class EmbeddingScriptedTeamModel(FakeTeamModel):
    """Returns caller-supplied embeddings, one array per observe() call, in
    order -- for tests that need control over appearance, not just team/conf/quality.
    """

    def __init__(self, embeddings_by_call, team=0, confidence=0.9, quality=1.0):
        super().__init__()
        self.embeddings_by_call = list(embeddings_by_call)
        self.calls = 0
        self.team = team
        self.confidence = confidence
        self.quality = quality

    def observe(self, frame_rgb, boxes_xyxy, context_boxes_xyxy=None):
        count = len(boxes_xyxy)
        embeddings = np.asarray(self.embeddings_by_call[self.calls], dtype=float)
        self.calls += 1
        assert len(embeddings) == count, (
            f"call {self.calls}: expected {count} embeddings, got {len(embeddings)}"
        )
        return (
            embeddings,
            np.full(count, self.team, dtype=int),
            np.full(count, self.confidence, dtype=float),
            np.full(count, self.quality, dtype=float),
        )


class TrackEmbeddingRefreshTests(unittest.TestCase):
    """Track.embedding must track current appearance, not a frame-0 snapshot --
    that snapshot is what re-ID and the drift/body-swap check compare against.
    """

    def test_matched_track_embedding_moves_toward_the_new_observation(self):
        seed_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        det_embedding = np.array([0.0, 1.0, 0.0, 0.0])
        model = EmbeddingScriptedTeamModel([
            np.array([seed_embedding]),  # seed()
            np.array([det_embedding]),   # checkpoint()
        ])
        manager = TrackManager(model, court_test_fn=lambda _box: True)
        box = np.array([10.0, 10.0, 50.0, 100.0])
        obj_ids = manager.seed(
            0, np.array([box]), np.zeros((120, 120, 3), dtype=np.uint8),
            np.array([False]),
        )
        oid = obj_ids[0]

        manager.checkpoint(
            5, np.zeros((120, 120, 3), dtype=np.uint8),
            [oid], np.array([_box_mask(box)]), np.array([box]), np.array([False]),
        )

        expected = EMBEDDING_EMA_ALPHA * det_embedding + (1 - EMBEDDING_EMA_ALPHA) * seed_embedding
        np.testing.assert_allclose(manager.tracks[oid].embedding, expected)

    def test_low_quality_observation_does_not_refresh_the_embedding(self):
        seed_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        model = EmbeddingScriptedTeamModel(
            [np.array([seed_embedding]), np.array([np.array([0.0, 1.0, 0.0, 0.0])])],
            quality=0.0,
        )
        manager = TrackManager(model, court_test_fn=lambda _box: True)
        box = np.array([10.0, 10.0, 50.0, 100.0])
        obj_ids = manager.seed(
            0, np.array([box]), np.zeros((120, 120, 3), dtype=np.uint8),
            np.array([False]),
        )
        oid = obj_ids[0]

        manager.checkpoint(
            5, np.zeros((120, 120, 3), dtype=np.uint8),
            [oid], np.array([_box_mask(box)]), np.array([box]), np.array([False]),
        )

        np.testing.assert_array_equal(manager.tracks[oid].embedding, seed_embedding)


class DuplicateRetirementTests(unittest.TestCase):
    def test_duplicate_removed_track_is_retired_not_destroyed(self):
        # Two heavily overlapping tracks, confirmed over two checkpoints --
        # whichever one rule 1 removes must land in `retired`, recoverable by
        # re-ID, rather than being permanently lost the way it was before.
        model = FakeTeamModel()
        manager = TrackManager(model, court_test_fn=lambda _box: True)
        box_a = np.array([10.0, 10.0, 60.0, 100.0])
        box_b = np.array([15.0, 10.0, 65.0, 100.0])
        obj_ids = manager.seed(
            0, np.array([box_a, box_b]), np.zeros((120, 120, 3), dtype=np.uint8),
            np.array([False, False]),
        )
        oid_a, oid_b = obj_ids
        masks = np.array([_box_mask(box_a), _box_mask(box_b)])

        for frame_idx in (10, 20):
            manager.checkpoint(
                frame_idx, np.zeros((120, 120, 3), dtype=np.uint8),
                [oid_a, oid_b], masks, np.array([box_a, box_b]),
                np.array([False, False]),
            )

        remove_events = [e for e in manager.events if e["type"] == "remove_duplicate"]
        self.assertEqual(len(remove_events), 1)
        removed_id = remove_events[0]["obj_id"]
        self.assertIn(removed_id, [t.obj_id for t in manager.retired])


class ReidTakenGuardTests(unittest.TestCase):
    def test_two_new_detections_cannot_claim_the_same_retired_identity(self):
        # Two unmatched detections that both look like the one retired player
        # must not both revive it -- exactly one reid, one fresh allocation.
        # `self.retired.remove()` already enforces this by mutating the
        # candidate pool as each match is claimed within the loop; `taken`
        # makes that invariant explicit so a future change to the matching
        # order (e.g. batching `_reid_match` calls before applying removals)
        # can't silently reintroduce a double-claim.
        model = FakeTeamModel()  # embeddings are all-ones -- identical for every det
        manager = TrackManager(model, court_test_fn=lambda _box: True)
        retired = Track(
            obj_id=99, team_id=1, embedding=np.array([1.0, 1.0, 1.0, 1.0]),
            is_goalkeeper=False, created_at=0, last_seen=0,
        )
        manager.retired.append(retired)

        box_1 = np.array([10.0, 10.0, 50.0, 100.0])
        box_2 = np.array([500.0, 10.0, 540.0, 100.0])  # far away -> different spatial slot
        det_boxes = np.array([box_1, box_2])
        det_is_gk = np.array([False, False])

        for frame_idx in (10, 20):
            manager.checkpoint(
                frame_idx, np.zeros((600, 600, 3), dtype=np.uint8),
                [], np.empty((0, 0, 0), dtype=bool), det_boxes, det_is_gk,
            )

        reid_events = [e for e in manager.events if e["type"] == "reid"]
        add_events = [e for e in manager.events if e["type"] == "add_new"]
        self.assertEqual(len(reid_events), 1)
        self.assertEqual(len(add_events), 1)
        self.assertEqual(manager.retired, [])


class DriftVsBodySwapTests(unittest.TestCase):
    """Rule 3 fires on the same signal (poor IoU / collapsed mask) for two
    different failure modes; appearance similarity must tell them apart.
    """

    def _seed_single_track(self, model, box):
        manager = TrackManager(model, court_test_fn=lambda _box: True)
        obj_ids = manager.seed(
            0, np.array([box]), np.zeros((120, 120, 3), dtype=np.uint8),
            np.array([False]),
        )
        return manager, obj_ids[0]

    def test_matched_low_iou_with_consistent_appearance_is_a_reprompt(self):
        seed_box = np.array([10.0, 10.0, 50.0, 100.0])
        det_box = np.array([30.0, 10.0, 70.0, 100.0])  # ~0.33 IoU -- drift band
        same_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        model = EmbeddingScriptedTeamModel([
            np.array([same_embedding]),  # seed()
            np.array([same_embedding]),  # checkpoint 1 (below confirm threshold)
            np.array([same_embedding]),  # checkpoint 2 (confirms and acts)
        ])
        manager, oid = self._seed_single_track(model, seed_box)
        mask = _box_mask(seed_box)

        actions = []
        for frame_idx in (10, 20):
            actions = manager.checkpoint(
                frame_idx, np.zeros((120, 120, 3), dtype=np.uint8),
                [oid], np.array([mask]), np.array([det_box]), np.array([False]),
            )

        types = [a["type"] for a in actions]
        self.assertIn("reprompt", types)
        self.assertNotIn("reset", types)

    def test_matched_low_iou_with_a_different_player_triggers_a_reset(self):
        seed_box = np.array([10.0, 10.0, 50.0, 100.0])
        det_box = np.array([30.0, 10.0, 70.0, 100.0])
        seed_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        different_embedding = np.array([0.0, 1.0, 0.0, 0.0])  # orthogonal -> sim 0.0
        self.assertLess(BODY_SWAP_COS_SIM_MIN, 1.0)  # sanity: threshold is meaningful
        model = EmbeddingScriptedTeamModel([
            np.array([seed_embedding]),
            np.array([different_embedding]),
            np.array([different_embedding]),
        ])
        manager, oid = self._seed_single_track(model, seed_box)
        mask = _box_mask(seed_box)

        actions = []
        for frame_idx in (10, 20):
            actions = manager.checkpoint(
                frame_idx, np.zeros((120, 120, 3), dtype=np.uint8),
                [oid], np.array([mask]), np.array([det_box]), np.array([False]),
            )

        types = [a["type"] for a in actions]
        self.assertIn("reset", types)
        self.assertNotIn("reprompt", types)
