import json
from collections import Counter
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_jersey_audit_set import (
    HEIGHT_BANDS,
    HEIGHT_BANDS_1080P,
    band_label,
    height_band_label,
    stratified_sample,
)
from scripts.build_number_eval_set import (
    BAND_QUOTA_SCALE,
    clip_name,
    parse_clip_spec,
    records_from_cache,
    renumber_crops,
)


class ClipSpecTests(unittest.TestCase):
    def test_splits_video_and_cache_on_the_last_colon(self):
        video, cache = parse_clip_spec("data/raw/a.mp4:outputs/number_cache/.a_v1.npz")
        self.assertEqual(video, Path("data/raw/a.mp4"))
        self.assertEqual(cache, Path("outputs/number_cache/.a_v1.npz"))

    def test_a_spec_without_a_separator_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_clip_spec("data/raw/a.mp4")

    def test_clip_name_is_the_video_stem(self):
        self.assertEqual(clip_name(Path("data/raw/2026-06-04_Melsungen.mp4")),
                         "2026-06-04_Melsungen")


class BandTests(unittest.TestCase):
    """The 1080p bands exist because the 640 ones collapse production into one bucket."""

    def test_existing_640_banding_is_unchanged(self):
        self.assertEqual(
            [height_band_label(h) for h in (10, 17, 21, 30, 100)],
            ["<16", "16-19", "20-23", ">=24", ">=24"],
        )

    def test_1080p_bands_separate_the_production_range(self):
        # Measured production quantiles: p05 14, p25 20, p50 25, p75 30, p95 40.
        # Under HEIGHT_BANDS the last four of these all collapse to ">=24".
        heights = (14, 20, 25, 30, 40, 154)
        self.assertEqual(
            [band_label(h, HEIGHT_BANDS_1080P) for h in heights],
            ["<18", "18-21", "22-25", "26-30", "31-40", ">=41"],
        )
        self.assertEqual(
            len({height_band_label(h) for h in (25, 30, 40, 154)}), 1
        )

    def test_every_band_set_covers_zero_to_infinity(self):
        for bands in (HEIGHT_BANDS, HEIGHT_BANDS_1080P):
            self.assertEqual(bands[0][1], 0.0)
            self.assertEqual(bands[-1][2], float("inf"))
            for (_, _, high), (_, low, _) in zip(bands, bands[1:]):
                self.assertEqual(high, low, f"gap or overlap in {bands}")

    def test_stratified_sample_honours_the_band_set_it_is_given(self):
        # Heights 25/30/40 are three distinct 1080p bands but one 640 band.
        records = [
            {"box_height": h, "source_clip": "c", "annotation_id": f"{i}"}
            for i, h in enumerate([25, 30, 40] * 4)
        ]
        self.assertEqual(len(stratified_sample(records, 1, 0)), 1)
        self.assertEqual(
            len(stratified_sample(records, 1, 0, bands=HEIGHT_BANDS_1080P)), 3
        )


class BandQuotaTests(unittest.TestCase):
    """The >=41 band is ~5% of production and mostly detector false positives."""

    def _records(self, heights):
        return [
            {"box_height": h, "source_clip": "c", "annotation_id": f"{i}"}
            for i, h in enumerate(heights)
        ]

    def test_named_bands_take_their_override_and_others_keep_per_band(self):
        records = self._records([25] * 10 + [50] * 10)
        selected = stratified_sample(
            records, 6, 0, bands=HEIGHT_BANDS_1080P, band_quota={">=41": 2}
        )
        counts = Counter(band_label(r["box_height"], HEIGHT_BANDS_1080P) for r in selected)
        self.assertEqual(counts["22-25"], 6)
        self.assertEqual(counts[">=41"], 2)

    def test_no_override_reproduces_uniform_sampling(self):
        records = self._records([25] * 10 + [50] * 10)
        self.assertEqual(
            len(stratified_sample(records, 4, 0, bands=HEIGHT_BANDS_1080P)),
            len(stratified_sample(
                records, 4, 0, bands=HEIGHT_BANDS_1080P, band_quota={}
            )),
        )

    def test_quota_never_exceeds_available_candidates(self):
        records = self._records([50] * 3)
        selected = stratified_sample(
            records, 99, 0, bands=HEIGHT_BANDS_1080P, band_quota={">=41": 99}
        )
        self.assertEqual(len(selected), 3)

    def test_the_configured_scale_halves_the_low_yield_band(self):
        per_band = 20 * 7  # per-band-per-clip 20, seven clips
        quota = {
            label: max(1, round(per_band * scale))
            for label, scale in BAND_QUOTA_SCALE.items()
        }
        self.assertEqual(quota, {">=41": 70})


class RecordsFromCacheTests(unittest.TestCase):
    def _cache(self, directory: Path, offsets, boxes, confidence) -> Path:
        path = directory / "cache.npz"
        np.savez(
            path,
            offsets=np.asarray(offsets, dtype=np.int64),
            boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
            confidence=np.asarray(confidence, dtype=float),
        )
        return path

    def test_boxes_are_attributed_to_the_frame_their_offsets_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = self._cache(
                tmp,
                offsets=[0, 2, 2, 3],  # frame 0 has two boxes, frame 1 none, frame 2 one
                boxes=[[0, 0, 10, 20], [5, 5, 15, 30], [1, 1, 9, 25]],
                confidence=[0.9, 0.9, 0.9],
            )
            records = records_from_cache(Path("clip.mp4"), cache, 0.0)
        self.assertEqual([r["frame"] for r in records], [0, 0, 2])
        self.assertEqual([r["box_height"] for r in records], [20.0, 25.0, 24.0])
        self.assertEqual(records[0]["source_clip"], "clip")

    def test_low_confidence_and_degenerate_boxes_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = self._cache(
                tmp,
                offsets=[0, 3],
                boxes=[[0, 0, 10, 20], [0, 0, 10, 0], [0, 0, 10, 20]],
                confidence=[0.2, 0.9, 0.8],
            )
            records = records_from_cache(Path("clip.mp4"), cache, 0.5)
        # first fails confidence, second is zero-height, only the third survives
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["confidence"], 0.8)

    def test_a_stride_skipped_frame_simply_contributes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = self._cache(
                tmp, offsets=[0, 1, 1, 1, 2],
                boxes=[[0, 0, 10, 20], [0, 0, 10, 20]], confidence=[0.9, 0.9],
            )
            records = records_from_cache(Path("clip.mp4"), cache, 0.0)
        self.assertEqual([r["frame"] for r in records], [0, 3])


class RenumberCropsTests(unittest.TestCase):
    """Render gaps must not desync filename index from sample index.

    The reviewer and the Qwen harness both address samples by list position, so a
    frame that failed to decode must not leave the two numbering schemes drifting.
    """

    def test_gaps_from_failed_renders_are_closed_without_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            crops = Path(tmp) / "crops"
            contexts = Path(tmp) / "contexts"
            crops.mkdir()
            contexts.mkdir()
            # selected indices 0, 2 and 5 rendered; 1, 3, 4 failed to decode
            samples = []
            for final, original in enumerate((0, 2, 5)):
                for directory in (crops, contexts):
                    (directory / f"number_{original:04d}.jpg").write_bytes(
                        bytes([original])
                    )
                samples.append({
                    "index": final,
                    "crop_path": f"crops/number_{original:04d}.jpg",
                    "context_path": f"contexts/number_{original:04d}.jpg",
                })

            renumber_crops(samples, crops, contexts)

        self.assertEqual(
            [s["crop_path"] for s in samples],
            ["crops/number_0000.jpg", "crops/number_0001.jpg", "crops/number_0002.jpg"],
        )
        self.assertEqual(
            [s["context_path"] for s in samples],
            ["contexts/number_0000.jpg", "contexts/number_0001.jpg",
             "contexts/number_0002.jpg"],
        )


class TrialDatasetShapeTests(unittest.TestCase):
    """Guards the contract the reviewer and benchmark harness rely on."""

    def test_sample_schema_matches_what_the_harnesses_index_on(self):
        sample = {
            "index": 0, "frame": 100, "box": [1.0, 2.0, 3.0, 4.0], "predictions": {},
            "sample_id": "clip:100:0", "source_kind": "production_detection_1080p",
            "source_clip": "clip", "source_video": "clip.mp4",
            "detector_confidence": 0.8, "height_band": "22-25",
            "crop_path": "crops/number_0000.jpg",
            "context_path": "contexts/number_0000.jpg",
            "crop_width": 24, "crop_height": 24,
        }
        # benchmark_qwen_jersey_ocr resolves images by these two keys, and
        # label_jersey_numbers scores by list index.
        for key in ("crop_path", "context_path", "crop_width", "crop_height", "index"):
            self.assertIn(key, sample)
        self.assertTrue(sample["crop_path"].startswith("crops/"))
        self.assertTrue(sample["context_path"].startswith("contexts/"))
        json.dumps(sample)  # must be serialisable as written


if __name__ == "__main__":
    unittest.main()
