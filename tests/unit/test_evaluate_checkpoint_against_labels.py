import unittest

import numpy as np

from scripts.evaluate_checkpoint_against_labels import box_iou, center_distance, match_sample


class BoxIouTests(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


class CenterDistanceTests(unittest.TestCase):
    def test_same_box_is_zero(self):
        self.assertEqual(center_distance([0, 0, 10, 10], [0, 0, 10, 10]), 0.0)

    def test_shifted_box(self):
        self.assertEqual(center_distance([0, 0, 10, 10], [3, 4, 13, 14]), 5.0)


class MatchSampleTests(unittest.TestCase):
    def test_picks_highest_iou_prediction(self):
        gt = [0, 0, 10, 10]
        preds = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        scores = np.array([0.9, 0.3])

        result = match_sample(gt, preds, scores)

        self.assertEqual(result["best_iou"], 1.0)
        self.assertEqual(result["matched_confidence"], 0.9)
        self.assertIsNone(result["nearest_center_distance"])
        self.assertEqual(result["n_predictions_on_image"], 2)

    def test_no_overlap_records_nearest_center_distance(self):
        gt = [0, 0, 10, 10]
        preds = np.array([[100, 100, 110, 110]], dtype=float)
        scores = np.array([0.5])

        result = match_sample(gt, preds, scores)

        self.assertEqual(result["best_iou"], 0.0)
        self.assertIsNone(result["matched_confidence"])
        self.assertIsNotNone(result["nearest_center_distance"])

    def test_no_predictions_on_image(self):
        gt = [0, 0, 10, 10]
        preds = np.zeros((0, 4))
        scores = np.zeros((0,))

        result = match_sample(gt, preds, scores)

        self.assertEqual(result["best_iou"], 0.0)
        self.assertIsNone(result["matched_confidence"])
        self.assertIsNone(result["nearest_center_distance"])
        self.assertEqual(result["n_predictions_on_image"], 0)


if __name__ == "__main__":
    unittest.main()
