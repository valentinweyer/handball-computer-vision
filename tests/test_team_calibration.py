import sys
import unittest
from pathlib import Path

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from team_aware_tracker import (
    TEAM_PROBABILITY_KEY,
    TEAM_QUALITY_KEY,
    TeamGatedByteTrackTracker,
)
from team_calibration import (
    ROLE_FIELD,
    ROLE_REFEREE,
    PrototypeTeamCalibrator,
)


def textured_crop(rgb, offset=0):
    crop = np.empty((72, 44, 3), dtype=np.uint8)
    crop[:] = np.clip(np.asarray(rgb) + offset, 0, 255)
    crop[::4] = np.clip(np.asarray(rgb) + 25 + offset, 0, 255)
    crop[:, ::7] = np.clip(np.asarray(rgb) - 20 + offset, 0, 255)
    return crop


class PrototypeCalibrationTests(unittest.TestCase):
    def test_two_team_seeds_and_referee_seed_define_semantic_roles(self):
        blue = [textured_crop((25, 50, 210), offset) for offset in range(-8, 9, 2)]
        white = [textured_crop((220, 220, 220), offset) for offset in range(-8, 9, 2)]
        red = [textured_crop((210, 35, 35), offset) for offset in range(-8, 9, 2)]
        model = PrototypeTeamCalibrator.fit(
            blue + white + red,
            {0: [blue[4]], 1: [white[4]], 2: [red[4]]},
        )
        probability_b, certainty, role, role_confidence = model.predict_crops(
            [blue[2], white[6], red[5]]
        )
        self.assertLess(probability_b[0], 0.5)
        self.assertGreater(probability_b[1], 0.5)
        self.assertEqual(role[0], ROLE_FIELD)
        self.assertEqual(role[1], ROLE_FIELD)
        self.assertEqual(role[2], ROLE_REFEREE)
        self.assertGreater(certainty[:2].min(), 0.2)
        self.assertGreater(role_confidence[2], 0.15)


class TeamGatedTrackerTests(unittest.TestCase):
    @staticmethod
    def detection(probability_b):
        return sv.Detections(
            xyxy=np.array([[10.0, 10.0, 50.0, 100.0]]),
            confidence=np.array([0.95]),
            class_id=np.array([2]),
            data={
                TEAM_PROBABILITY_KEY: np.array([probability_b]),
                TEAM_QUALITY_KEY: np.array([1.0]),
            },
        )

    def test_stable_team_contradiction_cannot_reuse_tracker_id(self):
        tracker = TeamGatedByteTrackTracker(
            frame_rate=25,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.1,
        )
        original_id = None
        for _ in range(4):
            tracked = tracker.update(self.detection(0.01))
            valid = tracked.tracker_id[tracked.tracker_id >= 0]
            if len(valid):
                original_id = int(valid[0])
        self.assertIsNotNone(original_id)
        team, confidence = tracker.association_team(original_id)
        self.assertEqual(team, 0)
        self.assertGreaterEqual(confidence, 0.80)

        contradictory = tracker.update(self.detection(0.99))
        self.assertNotIn(original_id, contradictory.tracker_id.tolist())
        self.assertGreater(tracker.team_gate_rejections, 0)

    def test_uncertain_detection_remains_associable(self):
        tracker = TeamGatedByteTrackTracker(
            frame_rate=25,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.1,
        )
        original_id = None
        for _ in range(4):
            tracked = tracker.update(self.detection(0.01))
            valid = tracked.tracker_id[tracked.tracker_id >= 0]
            if len(valid):
                original_id = int(valid[0])
        uncertain = tracker.update(self.detection(0.52))
        self.assertIn(original_id, uncertain.tracker_id.tolist())


if __name__ == "__main__":
    unittest.main()
