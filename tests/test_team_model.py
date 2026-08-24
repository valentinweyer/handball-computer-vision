import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from identity_manager import IdentityManager
from mask_team_features import guarded_torso_masks
from track_manager import TrackManager
from team_model import (
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
