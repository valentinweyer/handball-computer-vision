import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.generate_jersey_candidates import (
    build_report,
    box_iou,
    distinct_images,
    is_center_inside_any,
    load_coco_index,
    match_proposals_to_image,
    sample_index_for,
)


class BoxIouTests(unittest.TestCase):
    def test_identical_boxes_are_one(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes_are_zero(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partial_overlap(self):
        # 5x10 intersection over (10*10 + 10*10 - 50) = 150 union
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)


class IsCenterInsideAnyTests(unittest.TestCase):
    def test_center_inside_one_target(self):
        self.assertTrue(is_center_inside_any([4, 4, 6, 6], [[0, 0, 10, 10]]))

    def test_center_outside_all_targets(self):
        self.assertFalse(is_center_inside_any([100, 100, 110, 110], [[0, 0, 10, 10]]))

    def test_empty_target_list(self):
        self.assertFalse(is_center_inside_any([4, 4, 6, 6], []))


def _write_split(split_dir: Path, categories: list[dict], images: list[dict], annotations: list[dict]) -> None:
    split_dir.mkdir(parents=True)
    (split_dir / "_annotations.coco.json").write_text(json.dumps({
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }))


class LoadCocoIndexTests(unittest.TestCase):
    def test_separates_jersey_and_referee_boxes_per_image(self):
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            _write_split(
                root / "train",
                categories=[
                    {"id": 4, "name": "Referee"},
                    {"id": 5, "name": "jersey number"},
                ],
                images=[{"id": 0, "file_name": "a.jpg"}],
                annotations=[
                    {"id": 1, "image_id": 0, "category_id": 5, "bbox": [1, 1, 2, 2]},
                    {"id": 2, "image_id": 0, "category_id": 4, "bbox": [10, 10, 5, 5]},
                ],
            )
            index = load_coco_index(root)

        entry = index["train"][0]
        self.assertEqual(entry["jersey"], [(1, [1, 1, 3, 3])])
        self.assertEqual(entry["referee"], [[10, 10, 15, 15]])

    def test_missing_category_names_are_tolerated(self):
        with TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            _write_split(
                root / "train",
                categories=[{"id": 5, "name": "jersey number"}],  # no Referee category
                images=[{"id": 0, "file_name": "a.jpg"}],
                annotations=[{"id": 1, "image_id": 0, "category_id": 5, "bbox": [0, 0, 1, 1]}],
            )
            index = load_coco_index(root)

        self.assertEqual(index["train"][0]["referee"], [])


class DistinctImagesTests(unittest.TestCase):
    def test_deduplicates_by_split_and_image_id_preserving_order(self):
        dataset = {"samples": [
            {"source_split": "train", "source_image_id": 5},
            {"source_split": "train", "source_image_id": 5},
            {"source_split": "valid", "source_image_id": 5},
            {"source_split": "train", "source_image_id": 9},
        ]}
        self.assertEqual(
            distinct_images(dataset),
            [("train", 5), ("valid", 5), ("train", 9)],
        )


class SampleIndexForTests(unittest.TestCase):
    def test_links_split_and_annotation_id_to_audit_sample_index(self):
        dataset = {"samples": [
            {"index": 0, "source_split": "train", "source_annotation_id": 12},
            {"index": 1, "source_split": "valid", "source_annotation_id": 12},
        ]}
        mapping = sample_index_for(dataset)
        self.assertEqual(mapping[("train", 12)], 0)
        self.assertEqual(mapping[("valid", 12)], 1)


class MatchProposalsToImageTests(unittest.TestCase):
    def test_matches_best_iou_annotation_above_threshold(self):
        entry = {"file_name": "a.jpg", "jersey": [(1, [0, 0, 10, 10]), (2, [100, 100, 110, 110])]}
        proposals = [{"box": [1, 1, 11, 11], "confidence": 0.9}]

        candidates, matched = match_proposals_to_image(proposals, entry, "train", 0, {("train", 1): 5})

        self.assertEqual(candidates[0]["matched_annotation_id"], 1)
        self.assertEqual(candidates[0]["matched_sample_index"], 5)
        self.assertEqual(matched, {("train", 1)})

    def test_below_threshold_iou_is_unmatched_new_candidate(self):
        entry = {"file_name": "a.jpg", "jersey": [(1, [0, 0, 10, 10])]}
        # tiny overlap, IoU well under IOU_MATCH_THRESHOLD (0.3)
        proposals = [{"box": [9, 9, 30, 30], "confidence": 0.5}]

        candidates, matched = match_proposals_to_image(proposals, entry, "train", 0, {})

        self.assertIsNone(candidates[0]["matched_annotation_id"])
        self.assertIsNone(candidates[0]["matched_sample_index"])
        self.assertEqual(matched, set())

    def test_unlinked_match_has_no_sample_index(self):
        # matches an existing annotation that wasn't part of the audit sample itself
        # (linked_index has no entry for it) -- a real, common case: most images carry
        # several jersey numbers but the audit set only samples one of them.
        entry = {"file_name": "a.jpg", "jersey": [(7, [0, 0, 10, 10])]}
        proposals = [{"box": [0, 0, 10, 10], "confidence": 0.9}]

        candidates, matched = match_proposals_to_image(proposals, entry, "train", 0, {})

        self.assertEqual(candidates[0]["matched_annotation_id"], 7)
        self.assertIsNone(candidates[0]["matched_sample_index"])
        self.assertEqual(matched, {("train", 7)})


class BuildReportTests(unittest.TestCase):
    def test_recovered_count_never_exceeds_audit_sample_count(self):
        # Regression: matched_annotation_ids spans every jersey annotation on an image
        # (sampled or not), so a naive len(matched_annotation_ids) can exceed the audit
        # set's own sample count when an image carries several unsampled numbers too.
        audit_dataset = {"samples": [
            {"index": 0, "source_split": "train", "source_file_name": "a.jpg",
             "source_annotation_id": 1, "height_band": "<16"},
        ]}
        # 5 distinct annotations matched on this image, only 1 of which is the sampled one.
        matched_annotation_ids = {("train", i) for i in range(1, 6)}

        report = build_report(
            Path("dataset.json"), Path("source"), audit_dataset, [("train", 0)],
            candidates=[], matched_annotation_ids=matched_annotation_ids,
        )

        self.assertEqual(report["existing_annotations"], 1)
        self.assertEqual(report["existing_annotations_recovered"], 1)
        self.assertLessEqual(
            report["existing_annotations_recovered"], report["existing_annotations"]
        )
        self.assertEqual(report["existing_annotations_missed_by_sam3"], 0)

    def test_unmatched_sample_is_a_recall_miss(self):
        audit_dataset = {"samples": [
            {"index": 0, "source_split": "train", "source_file_name": "a.jpg",
             "source_annotation_id": 1, "height_band": "<16"},
        ]}

        report = build_report(
            Path("dataset.json"), Path("source"), audit_dataset, [("train", 0)],
            candidates=[], matched_annotation_ids=set(),
        )

        self.assertEqual(report["existing_annotations_recovered"], 0)
        self.assertEqual(report["existing_annotations_missed_by_sam3"], 1)
        self.assertEqual(report["recall_misses"][0]["sample_index"], 0)

    def test_new_candidates_counts_unmatched_proposals(self):
        audit_dataset = {"samples": []}
        candidates = [
            {"matched_annotation_id": 1},
            {"matched_annotation_id": None},
            {"matched_annotation_id": None},
        ]

        report = build_report(
            Path("dataset.json"), Path("source"), audit_dataset, [],
            candidates=candidates, matched_annotation_ids={("train", 1)},
        )

        self.assertEqual(report["new_candidates"], 2)
        self.assertEqual(report["total_candidates"], 3)


if __name__ == "__main__":
    unittest.main()
