"""Decoding the RF-DETR court-keypoint ONNX output, and the slot names it ships with."""
import unittest

import numpy as np

from handball_cv.court.keypoints import KEYPOINT_COUNT, KEYPOINT_TO_VERTEX
from handball_cv.court.rfdetr_keypoints import decode_court_keypoints

COURT_CLASS = 1  # Roboflow layout: synthetic background class first

# Roboflow's ``keypoints_metadata`` for versions 3 and 4 of keypointv333-uwois-xprdi:
# the KP name the annotators typed for each output slot. Fetched read-only from
# the /ort endpoint on 2026-10-14; identical for both versions.
ROBOFLOW_SLOT_NAMES = (
    "02", "05", "08", "01", "06", "14", "17", "20", "33", "32", "29", "28", "31", "30",
    "03", "04", "16", "18", "19", "15", "07", "10", "27", "24", "25", "12", "13", "11",
    "21", "22", "23", "09", "26", "34", "35", "36", "37",
)


class SlotNameTests(unittest.TestCase):
    def test_mapping_agrees_with_the_annotators_names_on_every_slot(self):
        # The half-turned twin of the mapping fits every image equally well; the
        # annotators' own names are what tell the two apart.
        self.assertEqual(
            [f"{KEYPOINT_TO_VERTEX[s] + 1:02d}" for s in range(KEYPOINT_COUNT)],
            list(ROBOFLOW_SLOT_NAMES),
        )


class LocalLayoutTests(unittest.TestCase):
    """A local rfdetr export: Court is logit 0 and the block holds only its 37 slots."""

    def test_court_is_class_zero_with_a_single_block(self):
        logits = np.full((4, 2), -5.0)
        logits[2, 0] = 3.0
        logits[1, 1] = 9.0               # the unused second logit must not win
        keypoints = np.zeros((4, KEYPOINT_COUNT, 8))
        keypoints[2, 5] = (0.5, 0.25, 10.0, 0, 0, 0, 0, 0)
        points, score = decode_court_keypoints(logits, keypoints, 1000, 800)
        self.assertEqual(points[5][:3], (5, 500.0, 200.0))
        self.assertAlmostEqual(score, 1 / (1 + np.exp(-3.0)))

    def test_rejects_a_block_that_is_not_whole_groups_of_37(self):
        with self.assertRaises(ValueError):
            decode_court_keypoints(np.zeros((2, 2)), np.zeros((2, 40, 8)), 10, 10)


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.queries = 5
        self.logits = np.full((self.queries, 2), -5.0)
        self.keypoints = np.zeros((self.queries, 2 * KEYPOINT_COUNT, 8))

    def test_takes_the_best_court_query_and_its_court_block(self):
        self.logits[3, COURT_CLASS] = 4.0
        self.logits[1, 0] = 9.0          # a strong *background* query must not win
        block = self.keypoints[3].reshape(2, KEYPOINT_COUNT, 8)
        block[COURT_CLASS, 7] = (0.25, 0.5, 10.0, 0, 0, 0, 0, 0)
        block[0, 7] = (0.9, 0.9, 10.0, 0, 0, 0, 0, 0)   # background-class slots ignored
        points, score = decode_court_keypoints(self.logits, self.keypoints, 1920, 1080)
        self.assertEqual(len(points), KEYPOINT_COUNT)
        slot, x, y, conf = points[7]
        self.assertEqual((slot, x, y), (7, 480.0, 540.0))
        self.assertGreater(conf, 0.99)
        self.assertAlmostEqual(score, 1 / (1 + np.exp(-4.0)))

    def test_visibility_is_the_sigmoid_of_the_third_value(self):
        self.logits[0, COURT_CLASS] = 1.0
        self.keypoints[0].reshape(2, KEYPOINT_COUNT, 8)[COURT_CLASS, :, 2] = -10.0
        points, _ = decode_court_keypoints(self.logits, self.keypoints, 100, 100)
        self.assertTrue(all(conf < 0.001 for _, _, _, conf in points))


if __name__ == "__main__":
    unittest.main()
