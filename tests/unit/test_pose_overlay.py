"""The score gate on skeleton drawing.

ViTPose returns all 17 keypoints for every box, always, including the ones it
could not see -- an occluded ankle comes back with a low score at a plausible
guessed position, not as a gap. Drawing those unconditionally turns a guess
into something that looks exactly like a measurement, and in a demo that is the
whole claim being made. So the gate, and the rule that a limb needs both ends,
are what these tests pin down.
"""
import unittest

import numpy as np

from handball_cv.pose.vitpose import KEYPOINT_COUNT, SKELETON
from handball_cv.visualization.pose_overlay import draw_pose


def _person(score: float) -> np.ndarray:
    keypoints = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    # A column of distinct, well-separated points inside a 200x200 frame.
    keypoints[:, 0] = np.linspace(20, 180, KEYPOINT_COUNT)
    keypoints[:, 1] = np.linspace(20, 180, KEYPOINT_COUNT)
    keypoints[:, 2] = score
    return keypoints


class PoseOverlayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_skeleton_indices_are_in_range(self) -> None:
        for start, end in SKELETON:
            self.assertLess(start, KEYPOINT_COUNT)
            self.assertLess(end, KEYPOINT_COUNT)
            self.assertNotEqual(start, end)

    def test_confident_keypoints_are_drawn(self) -> None:
        draw_pose(self.frame, _person(0.9), min_score=0.3)
        self.assertGreater(int(self.frame.sum()), 0)

    def test_low_scoring_keypoints_leave_no_mark(self) -> None:
        draw_pose(self.frame, _person(0.1), min_score=0.3)
        self.assertEqual(int(self.frame.sum()), 0)

    def test_a_limb_needs_both_ends(self) -> None:
        """One confident end must not draw a limb out to a guessed one."""
        start, end = SKELETON[0]
        keypoints = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
        keypoints[start, :2], keypoints[end, :2] = (20.0, 20.0), (180.0, 180.0)
        keypoints[[start, end], 2] = 0.9
        midpoint = (100, 100)

        draw_pose(self.frame, keypoints, min_score=0.3)
        self.assertGreater(int(self.frame[midpoint].sum()), 0)

        keypoints[end, 2] = 0.0
        half = np.zeros((200, 200, 3), dtype=np.uint8)
        draw_pose(half, keypoints, min_score=0.3)
        self.assertEqual(int(half[midpoint].sum()), 0)
        self.assertGreater(int(half.sum()), 0, "the seen end should still show")


if __name__ == "__main__":
    unittest.main()
