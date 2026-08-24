import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.label_team_detections import migrate_legacy_labels
from handball_cv.teams.dataset import (
    annotation_from_code,
    build_sample_records,
    evenly_spaced_detection_frames,
    frame_arrays,
    load_detection_cache,
    new_manifest,
    validate_annotation,
    validate_manifest,
)


def synthetic_cache():
    return {
        "schema": 1,
        "detector_id": "test-detector",
        "total_frames": 5,
        "offsets": np.array([0, 1, 1, 3, 4, 5]),
        "boxes": np.array([
            [0, 0, 20, 40],
            [10, 10, 40, 80],
            [60, 10, 90, 80],
            [15, 15, 45, 90],
            [20, 20, 50, 100],
        ], dtype=np.float32),
        "confidence": np.array([.9, .8, .7, .95, .85], dtype=np.float32),
        "class_id": np.array([2, 2, 1, 2, 2], dtype=np.int16),
    }


class AnnotationContractTests(unittest.TestCase):
    def test_team_codes_expand_to_relative_field_labels(self):
        annotation = annotation_from_code("a", quality="occluded")
        self.assertEqual(annotation, {
            "code": "A", "team": "A", "role": "field", "quality": "occluded",
        })
        validate_annotation(annotation)

    def test_mixed_crop_is_explicitly_not_forced_into_a_team(self):
        annotation = annotation_from_code("M")
        self.assertIsNone(annotation["team"])
        self.assertEqual(annotation["quality"], "mixed")
        validate_annotation(annotation)


class DetectionCacheTests(unittest.TestCase):
    def test_frame_arrays_return_local_and_stable_flat_indices(self):
        boxes, confidence, classes, flat = frame_arrays(synthetic_cache(), 2)
        self.assertEqual(boxes.shape, (2, 4))
        np.testing.assert_array_equal(classes, [2, 1])
        np.testing.assert_array_equal(flat, [1, 2])
        self.assertAlmostEqual(float(confidence[0]), .8, places=6)

    def test_sampling_is_even_and_skips_empty_frames(self):
        self.assertEqual(evenly_spaced_detection_frames(synthetic_cache(), 3), [0, 2, 4])

    def test_cache_loader_rejects_inconsistent_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            cache = synthetic_cache()
            cache["offsets"] = np.array([0, 1, 1, 3, 4, 99])
            np.savez(path, **cache)
            with self.assertRaisesRegex(ValueError, "offsets do not cover"):
                load_detection_cache(path)

    def test_sample_records_have_no_tracker_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "match.mp4"
            video.touch()
            samples = build_sample_records(
                video, synthetic_cache(), [2], class_ids={1, 2}
            )
        self.assertEqual(len(samples), 2)
        self.assertTrue(all("track_id" not in sample for sample in samples))
        self.assertEqual({sample["detection_index"] for sample in samples}, {0, 1})


class ManifestAndMigrationTests(unittest.TestCase):
    def test_manifest_accepts_partial_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "match.mp4"
            cache_path = Path(directory) / "detections.npz"
            video.touch()
            cache_path.touch()
            samples = build_sample_records(video, synthetic_cache(), [0])
            samples[0]["annotation"] = annotation_from_code("B")
            manifest = new_manifest(video, cache_path, synthetic_cache(), samples)
            validate_manifest(manifest)

    def test_legacy_box_is_migrated_by_same_frame_iou_not_track_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "match.mp4"
            video.touch()
            cache = synthetic_cache()
            samples = build_sample_records(video, cache, [2])
            run_boxes = np.full((5, 1, 4), np.nan, dtype=np.float32)
            run_boxes[2, 0] = cache["boxes"][1]
            run_path = root / "run.npz"
            np.savez(run_path, boxes=run_boxes)
            label_path = root / "labels.json"
            label_path.write_text(json.dumps({"labels": {"2_0": "TEAM A"}}))
            result = migrate_legacy_labels(
                samples, cache, run_path, label_path, minimum_iou=.99
            )
        self.assertEqual(result["migrated"], 1)
        matched = next(sample for sample in samples if sample["detection_index"] == 0)
        self.assertEqual(matched["annotation"]["team"], "A")
        self.assertEqual(matched["annotation_source"]["iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
