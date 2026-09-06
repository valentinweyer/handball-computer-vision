import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.build_jersey_audit_set import (
    HEIGHT_BANDS,
    height_band_label,
    load_jersey_annotations,
    render_dataset,
    source_clip,
    stratified_sample,
)
from scripts.label_jersey_numbers import LabelStore


class HeightBandTests(unittest.TestCase):
    def test_bands_partition_without_gaps_or_overlap(self):
        boundaries = [9.5, 15.999, 16, 19.999, 20, 23.999, 24, 100]
        labels = [height_band_label(value) for value in boundaries]
        self.assertEqual(
            labels,
            ["<16", "<16", "16-19", "16-19", "20-23", "20-23", ">=24", ">=24"],
        )

    def test_every_band_label_is_reachable(self):
        self.assertEqual({label for label, _, _ in HEIGHT_BANDS}, {"<16", "16-19", "20-23", ">=24"})


class SourceClipTests(unittest.TestCase):
    def test_collapses_frame_suffix(self):
        self.assertEqual(
            source_clip("Barsa_Kielce_mp4-0009_jpg.rf.deadbeef.jpg"), "Barsa_Kielce_mp4"
        )

    def test_groups_stills_under_one_label(self):
        self.assertEqual(source_clip("image1032_jpg.rf.deadbeef.jpg"), "image_stills")
        self.assertEqual(source_clip("image9_jpg.rf.deadbeef.jpg"), "image_stills")

    def test_non_digit_suffix_is_kept_as_is(self):
        self.assertEqual(source_clip("Clip_mp4_jpg.rf.deadbeef.jpg"), "Clip_mp4")


def _write_split(split_dir: Path, annotations: list[dict]) -> None:
    split_dir.mkdir(parents=True)
    images = [
        {"id": 0, "file_name": "clipA-0001_jpg.rf.aaa.jpg", "height": 100, "width": 100},
        {"id": 1, "file_name": "clipB-0007_jpg.rf.bbb.jpg", "height": 100, "width": 100},
    ]
    for image in images:
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(split_dir / image["file_name"]), canvas)
    (split_dir / "_annotations.coco.json").write_text(json.dumps({
        "categories": [
            {"id": 3, "name": "Player"},
            {"id": 5, "name": "jersey number"},
        ],
        "images": images,
        "annotations": annotations,
    }))


def _make_coco_fixture(root: Path) -> None:
    _write_split(root / "train", [
        {"id": 1, "image_id": 0, "category_id": 5, "bbox": [10, 10, 8, 12], "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 5, "bbox": [20, 20, 20, 30], "iscrowd": 0},
        {"id": 3, "image_id": 0, "category_id": 3, "bbox": [0, 0, 50, 50], "iscrowd": 0},
    ])
    _write_split(root / "valid", [
        {"id": 4, "image_id": 1, "category_id": 5, "bbox": [50, 50, 5, 5], "iscrowd": 0},
    ])


class LoadJerseyAnnotationsTests(unittest.TestCase):
    def test_keeps_only_jersey_number_category_and_resolves_it_by_name(self):
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            _make_coco_fixture(root)
            records = load_jersey_annotations(root)

        self.assertTrue(all(record["box_height"] > 0 for record in records))
        # 3 jersey-number annotations across train+valid; the Player one is excluded.
        self.assertEqual(len(records), 3)
        heights = sorted(record["box_height"] for record in records)
        self.assertEqual(heights, [5, 12, 30])

    def test_missing_split_directory_is_skipped_not_fatal(self):
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "train").mkdir()
            (root / "train" / "_annotations.coco.json").write_text(json.dumps({
                "categories": [{"id": 5, "name": "jersey number"}],
                "images": [],
                "annotations": [],
            }))
            records = load_jersey_annotations(root)

        self.assertEqual(records, [])


class StratifiedSampleTests(unittest.TestCase):
    def _records(self, n: int, height: float, clip_count: int) -> list[dict]:
        return [
            {
                "annotation_id": i,
                "image_id": i,
                "split": "train",
                "file_name": f"clip{i % clip_count}-{i:04d}_jpg.rf.x.jpg",
                "box": [0, 0, 10, height],
                "box_height": height,
                "source_clip": f"clip{i % clip_count}",
            }
            for i in range(n)
        ]

    def test_respects_per_band_quota(self):
        records = self._records(30, height=10, clip_count=3)  # all land in "<16"
        selected = stratified_sample(records, per_band=5, seed=0)
        self.assertEqual(len(selected), 5)

    def test_does_not_exceed_available_candidates(self):
        records = self._records(3, height=10, clip_count=3)
        selected = stratified_sample(records, per_band=50, seed=0)
        self.assertEqual(len(selected), 3)

    def test_deterministic_under_fixed_seed(self):
        records = self._records(40, height=10, clip_count=4)
        first = stratified_sample(records, per_band=10, seed=42)
        second = stratified_sample(records, per_band=10, seed=42)
        self.assertEqual(
            [r["annotation_id"] for r in first], [r["annotation_id"] for r in second]
        )

    def test_spreads_across_clips_rather_than_exhausting_one_first(self):
        # One clip dominates the pool (mirrors Barsa_Kielce/stills skew in the real data).
        records = (
            self._records(100, height=10, clip_count=1)  # all "clip0"
        )
        for i, record in enumerate(self._records(5, height=10, clip_count=1)):
            record["annotation_id"] = 1000 + i
            record["source_clip"] = "rare_clip"
            records.append(record)

        selected = stratified_sample(records, per_band=10, seed=0)
        clips_selected = {r["source_clip"] for r in selected}
        self.assertIn("rare_clip", clips_selected)

    def test_output_is_sorted_by_band_then_clip_then_annotation_id(self):
        records = self._records(6, height=10, clip_count=2) + [
            {**r, "box_height": 30, "box": [0, 0, 10, 30], "annotation_id": r["annotation_id"] + 100}
            for r in self._records(4, height=30, clip_count=2)
        ]
        selected = stratified_sample(records, per_band=50, seed=0)
        bands = [height_band_label(r["box_height"]) for r in selected]
        self.assertEqual(bands, sorted(bands, key=lambda b: {"<16": 0, "16-19": 1, "20-23": 2, ">=24": 3}[b]))


class RenderDatasetTests(unittest.TestCase):
    def test_renders_crops_and_emits_loader_compatible_dataset(self):
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir) / "source"
            output_dir = Path(raw_dir) / "out"
            _make_coco_fixture(root)
            records = load_jersey_annotations(root)
            selected = stratified_sample(records, per_band=50, seed=0)

            dataset = render_dataset(selected, root, output_dir)

            for sample in dataset["samples"]:
                self.assertTrue((output_dir / sample["crop_path"]).is_file())
                self.assertTrue((output_dir / sample["context_path"]).is_file())
                self.assertEqual(sample["crop_width"], sample["box"][2] - sample["box"][0])

            # LabelStore only requires dataset["samples"] carry unique "index" values;
            # constructing one is the same compatibility check the review server does.
            store = LabelStore(output_dir / "labels.json", dataset)
            self.assertEqual(store.valid_indices, {s["index"] for s in dataset["samples"]})

    def test_indices_are_dense_and_zero_based(self):
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir) / "source"
            output_dir = Path(raw_dir) / "out"
            _make_coco_fixture(root)
            records = load_jersey_annotations(root)
            dataset = render_dataset(records, root, output_dir)

        self.assertEqual(
            [s["index"] for s in dataset["samples"]], list(range(len(records)))
        )


if __name__ == "__main__":
    unittest.main()
