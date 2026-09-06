import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.request import Request, urlopen

import numpy as np

from scripts.label_jersey_numbers import (
    LabelStore,
    clipped_box,
    create_server,
    filter_samples_by_player_overlap,
    score_model,
    validate_label,
)


DATASET = {
    "video": "/tmp/video.mp4",
    "samples": [
        {"index": 0, "crop_path": "crops/0.jpg", "context_path": "contexts/0.jpg"},
        {"index": 1, "crop_path": "crops/1.jpg", "context_path": "contexts/1.jpg"},
    ],
}


class LabelValidationTests(unittest.TestCase):
    def test_filters_numbers_not_contained_by_player_predictions(self):
        with TemporaryDirectory() as raw_dir:
            cache_path = Path(raw_dir) / "detections.npz"
            np.savez(
                cache_path,
                offsets=np.array([0, 2]),
                boxes=np.array([[0, 0, 20, 20], [50, 50, 70, 70]], dtype=float),
                class_id=np.array([2, 2]),
            )
            samples = [
                {"index": 0, "frame": 0, "box": [5, 5, 10, 10]},
                {"index": 1, "frame": 0, "box": [30, 30, 35, 35]},
            ]

            kept = filter_samples_by_player_overlap(samples, cache_path)

        self.assertEqual([sample["index"] for sample in kept], [0])
        self.assertEqual(kept[0]["player_containment"], 1.0)

    def test_clipped_box_returns_json_serializable_integers(self):
        box = clipped_box([1.2, 2.7, 8.8, 9.1], 10, 10)

        self.assertEqual(box, (1, 3, 9, 9))
        self.assertTrue(all(type(value) is int for value in box))

    def test_accepts_one_or_two_digit_readable_label(self):
        self.assertEqual(
            validate_label(0, {"status": "readable", "value": "13"}, {0})["value"],
            "13",
        )

    def test_rejects_invalid_or_unknown_label(self):
        with self.assertRaises(ValueError):
            validate_label(0, {"status": "readable", "value": "113"}, {0})
        with self.assertRaises(ValueError):
            validate_label(2, {"status": "unreadable"}, {0})

    def test_box_status_is_optional_and_travels_with_the_label(self):
        label = validate_label(
            0, {"status": "unreadable", "box_status": "partial"}, {0}
        )
        self.assertEqual(label["box_status"], "partial")

    def test_missing_box_status_is_not_recorded(self):
        label = validate_label(0, {"status": "unreadable"}, {0})
        self.assertNotIn("box_status", label)

    def test_rejects_unknown_box_status(self):
        with self.assertRaises(ValueError):
            validate_label(0, {"status": "unreadable", "box_status": "bogus"}, {0})

    def test_scoring_counts_abstentions_as_incorrect(self):
        result = score_model(
            {0: "13", 1: "24"}, {0: "13", 1: "", 2: ""}, {2}
        )

        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["selective_accuracy"], 1.0)
        self.assertEqual(result["unreadable_abstention_rate"], 1.0)


class LabelStoreDatasetPathTests(unittest.TestCase):
    def test_defaults_dataset_path_next_to_labels_file(self):
        with TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "labels.json"
            store = LabelStore(path, DATASET)

        self.assertEqual(store.document()["dataset"], str((Path(raw_dir) / "dataset.json").resolve()))

    def test_review_command_can_point_labels_at_a_separate_tracked_path(self):
        # scripts.label_jersey_numbers review keeps generated crops under runs/ (ignored)
        # but writes human labels under data/annotations/ (tracked) - the two live in
        # different directories, so LabelStore must not assume co-location.
        with TemporaryDirectory() as raw_dir:
            generated_dir = Path(raw_dir) / "runs" / "jersey_audit"
            tracked_dir = Path(raw_dir) / "data" / "annotations" / "jersey"
            tracked_dir.mkdir(parents=True)
            dataset_path = generated_dir / "dataset.json"
            store = LabelStore(
                tracked_dir / "audit_labels.json", DATASET, dataset_path=dataset_path
            )

        self.assertEqual(store.document()["dataset"], str(dataset_path.resolve()))


class LabelServerTests(unittest.TestCase):
    def test_preserves_but_excludes_labels_outside_filtered_dataset(self):
        with TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "labels.json"
            path.write_text(json.dumps({
                "labels": {
                    "0": {"status": "readable", "value": "13"},
                    "9": {"status": "readable", "value": "8"},
                }
            }))
            store = LabelStore(path, DATASET)
            migrated = json.loads(path.read_text())
            store.save(1, {"status": "unreadable"})
            saved = json.loads(path.read_text())

        self.assertEqual(set(migrated["labels"]), {"0"})
        self.assertEqual(set(migrated["excluded_labels"]), {"9"})
        self.assertEqual(set(saved["labels"]), {"0", "1"})
        self.assertEqual(saved["excluded_labels"]["9"]["value"], "8")

    def test_api_autosaves_and_restores_label(self):
        with TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            store = LabelStore(directory / "labels.json", DATASET)
            server = create_server(directory, DATASET, store, "127.0.0.1", 0)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                request = Request(
                    f"http://{host}:{port}/api/label",
                    data=json.dumps({"index": 0, "status": "readable", "value": "13"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)["value"], "13")
                saved = json.loads((directory / "labels.json").read_text())
                self.assertEqual(saved["labels"]["0"]["value"], "13")
                with urlopen(f"http://{host}:{port}/api/state") as response:
                    self.assertEqual(json.load(response)["labels"]["0"]["status"], "readable")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
