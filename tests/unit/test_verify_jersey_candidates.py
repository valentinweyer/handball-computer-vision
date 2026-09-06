from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.verify_jersey_candidates import append_new_sam3_candidates, compute_agreement


def _dataset(samples: list[dict]) -> dict:
    return {"schema_version": 1, "video": "coco:x", "samples": samples}


def _make_source_image(raw_dir: str) -> Path:
    """A source_root with train/a.jpg, matching the sam3 report fixtures' split/file_name."""
    source_root = Path(raw_dir) / "source"
    (source_root / "train").mkdir(parents=True)
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(source_root / "train" / "a.jpg"), canvas)
    return source_root


class ComputeAgreementTests(unittest.TestCase):
    def test_coco_sample_both_signals_agree(self):
        sample = {"index": 0, "source_kind": "coco", "predictions": {"qwen_context": "13"}}
        predictions = compute_agreement(sample, sam3_recovered_indices={0})

        self.assertTrue(predictions["sam3_recovered"])
        self.assertEqual(predictions["agreement"], "sam3+qwen agree")

    def test_coco_sample_sam3_recovered_qwen_abstains(self):
        sample = {"index": 0, "source_kind": "coco", "predictions": {"qwen_context": ""}}
        predictions = compute_agreement(sample, sam3_recovered_indices={0})

        self.assertEqual(predictions["agreement"], "sam3 only, qwen abstains")

    def test_coco_sample_sam3_missed_qwen_reads(self):
        sample = {"index": 0, "source_kind": "coco", "predictions": {"qwen_context": "7"}}
        predictions = compute_agreement(sample, sam3_recovered_indices=set())

        self.assertFalse(predictions["sam3_recovered"])
        self.assertEqual(predictions["agreement"], "qwen only, sam3 missed")

    def test_coco_sample_neither_signal(self):
        sample = {"index": 0, "source_kind": "coco", "predictions": {"qwen_context": ""}}
        predictions = compute_agreement(sample, sam3_recovered_indices=set())

        self.assertEqual(predictions["agreement"], "neither: sam3 missed, qwen abstains")

    def test_sam3_new_sample_has_no_sam3_recovered_key(self):
        # sam3_new samples are SAM3's own proposal -- "recovered against existing COCO"
        # is a coco-sample-only concept, so this key must not appear for them.
        sample = {"index": 5, "source_kind": "sam3_new", "predictions": {"qwen_context": "9"}}
        predictions = compute_agreement(sample, sam3_recovered_indices=set())

        self.assertNotIn("sam3_recovered", predictions)
        self.assertEqual(predictions["agreement"], "sam3+qwen agree")

    def test_sam3_new_sample_qwen_abstains(self):
        sample = {"index": 5, "source_kind": "sam3_new", "predictions": {"qwen_context": ""}}
        predictions = compute_agreement(sample, sam3_recovered_indices=set())

        self.assertEqual(predictions["agreement"], "sam3 only, qwen abstains")

    def test_missing_qwen_key_treated_as_abstain(self):
        # a sample whose Qwen request hasn't run yet has no "qwen_context" key at all
        sample = {"index": 0, "source_kind": "coco", "predictions": {}}
        predictions = compute_agreement(sample, sam3_recovered_indices=set())

        self.assertEqual(predictions["agreement"], "neither: sam3 missed, qwen abstains")


class AppendNewSam3CandidatesTests(unittest.TestCase):
    def _sam3_report(self, matched: bool) -> dict:
        return {"candidates": [{
            "split": "train",
            "file_name": "a.jpg",
            "box": [10.0, 10.0, 20.0, 20.0],
            "confidence": 0.42,
            "matched_annotation_id": 1 if matched else None,
        }]}

    def test_appends_only_unmatched_proposals(self):
        with TemporaryDirectory() as raw_dir:
            source_root = _make_source_image(raw_dir)
            output_dir = Path(raw_dir) / "out"
            dataset = _dataset([])

            dataset = append_new_sam3_candidates(
                dataset, self._sam3_report(matched=False), source_root, output_dir
            )

            self.assertEqual(len(dataset["samples"]), 1)
            added = dataset["samples"][0]
            self.assertEqual(added["source_kind"], "sam3_new")
            self.assertEqual(added["sam3_confidence"], 0.42)
            self.assertEqual(added["predictions"], {})
            self.assertTrue((output_dir / added["crop_path"]).is_file())

    def test_matched_proposals_are_not_appended(self):
        with TemporaryDirectory() as raw_dir:
            source_root = _make_source_image(raw_dir)
            output_dir = Path(raw_dir) / "out"
            dataset = _dataset([])

            dataset = append_new_sam3_candidates(
                dataset, self._sam3_report(matched=True), source_root, output_dir
            )

        self.assertEqual(dataset["samples"], [])

    def test_indices_continue_from_existing_max(self):
        with TemporaryDirectory() as raw_dir:
            source_root = _make_source_image(raw_dir)
            output_dir = Path(raw_dir) / "out"
            dataset = _dataset([
                {"index": 0, "source_kind": "coco", "predictions": {}},
                {"index": 7, "source_kind": "coco", "predictions": {}},
            ])

            dataset = append_new_sam3_candidates(
                dataset, self._sam3_report(matched=False), source_root, output_dir
            )

        self.assertEqual(dataset["samples"][-1]["index"], 8)

    def test_rerun_does_not_duplicate_already_appended_candidate(self):
        with TemporaryDirectory() as raw_dir:
            source_root = _make_source_image(raw_dir)
            output_dir = Path(raw_dir) / "out"
            dataset = _dataset([])
            report = self._sam3_report(matched=False)

            dataset = append_new_sam3_candidates(dataset, report, source_root, output_dir)
            dataset = append_new_sam3_candidates(dataset, report, source_root, output_dir)

        self.assertEqual(len(dataset["samples"]), 1)


if __name__ == "__main__":
    unittest.main()
