import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

from team_evaluation import (
    accuracy_at_coverages,
    align_anonymous_binary_labels,
    evaluate_manifest_embeddings,
    spherical_two_means,
)


class AnonymousTeamMetricTests(unittest.TestCase):
    def test_cluster_ids_are_permutation_aligned_for_scoring(self):
        prediction = np.array([1, 1, 0, 0])
        truth = np.array([0, 0, 1, 1])
        aligned, mapping = align_anonymous_binary_labels(prediction, truth)
        np.testing.assert_array_equal(aligned, truth)
        self.assertEqual(mapping, {0: 1, 1: 0})

    def test_spherical_two_means_separates_two_directions(self):
        features = np.array([
            [1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1],
        ])
        result = spherical_two_means(features)
        self.assertEqual(result.labels[0], result.labels[1])
        self.assertEqual(result.labels[2], result.labels[3])
        self.assertNotEqual(result.labels[0], result.labels[2])
        self.assertTrue((result.confidence > 0).all())

    def test_accuracy_at_lower_coverage_keeps_high_confidence_samples(self):
        result = accuracy_at_coverages(
            np.array([True, True, False, False]),
            np.array([.9, .8, .2, .1]),
            coverages=(1.0, .5),
        )
        self.assertEqual(result["1.00"]["accuracy"], .5)
        self.assertEqual(result["0.50"]["accuracy"], 1.0)

    def test_manifest_fit_does_not_require_tracker_ids(self):
        samples = []
        for index, team in enumerate(["A", "A", "B", "B"]):
            samples.append({
                "sample_id": str(index),
                "detector_class_id": 2,
                "torso_contamination": 0.0,
                "crop_quality": {"accepted": True},
                "annotation": {
                    "code": team,
                    "team": team,
                    "role": "field",
                    "quality": "clean",
                },
            })
        manifest = {"video_id": "test", "samples": samples}
        features = np.array([
            [1.0, 0.0], [.9, .1], [-1.0, 0.0], [-.9, -.1],
        ])
        result = evaluate_manifest_embeddings(manifest, features)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["fit_count"], 4)


if __name__ == "__main__":
    unittest.main()
