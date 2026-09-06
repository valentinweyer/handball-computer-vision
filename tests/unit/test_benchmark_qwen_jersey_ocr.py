import unittest

from scripts.benchmark_qwen_jersey_ocr import (
    parse_prediction,
    stratify_by_height,
    summarize_variant,
)


class PredictionParsingTests(unittest.TestCase):
    def test_parses_exact_number(self):
        self.assertEqual(parse_prediction("13"), ("13", "exact"))

    def test_parses_explicit_abstention(self):
        self.assertEqual(parse_prediction("NONE"), ("", "abstain"))

    def test_parses_json_response(self):
        self.assertEqual(
            parse_prediction('{"status":"readable","number":"24"}'),
            ("24", "json"),
        )

    def test_malformed_multi_number_response_abstains(self):
        self.assertEqual(parse_prediction("It might be 13 or 18"), ("", "malformed"))


class SummarizeVariantTests(unittest.TestCase):
    def test_matches_hand_computed_metrics(self):
        readable = {0: "13", 1: "24"}
        unreadable = {2, 3}
        raw = {
            "0": {"prediction": "13", "parse_status": "exact"},
            "1": {"prediction": "", "parse_status": "abstain"},
            "2": {"prediction": "", "parse_status": "abstain"},
            "3": {"prediction": "7", "parse_status": "exact"},
        }

        metrics = summarize_variant(readable, unreadable, raw)

        # readable: 0 correct, 1 abstained (wrong); unreadable: 2 correctly abstained,
        # 3 incorrectly guessed a number.
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["unreadable_abstention_rate"], 0.5)
        self.assertEqual(metrics["accepted_predictions"], 2)  # sample 0 + sample 3
        self.assertEqual(metrics["accepted_precision"], 0.5)  # only sample 0 is right
        self.assertEqual(metrics["overall_correct"], 2)  # sample 0 + sample 2
        self.assertEqual(metrics["overall_decision_accuracy"], 0.5)
        self.assertEqual(metrics["malformed_responses"], 0)
        self.assertEqual(metrics["completed_requests"], 4)

    def test_missing_raw_entries_do_not_crash_and_score_as_abstentions(self):
        # An index with no raw response at all reads the same as an explicit empty
        # prediction (score_model's `predictions.get(index, "")` default) - so it
        # counts as a correct abstention against unreadable truth, not as "unscored".
        metrics = summarize_variant({0: "13"}, {1}, {})

        self.assertEqual(metrics["completed_requests"], 0)
        self.assertEqual(metrics["coverage"], 0.0)
        self.assertEqual(metrics["unreadable_abstention_rate"], 1.0)
        self.assertEqual(metrics["overall_decision_accuracy"], 0.5)


class StratifyByHeightTests(unittest.TestCase):
    def test_partitions_readable_and_unreadable_by_sample_crop_height(self):
        samples = {
            0: {"crop_height": 10},   # "<16"
            1: {"crop_height": 18},   # "16-19"
            2: {"crop_height": 30},   # ">=24"
        }
        readable = {0: "1", 2: "9"}
        unreadable = {1}
        eligible = [0, 1, 2]

        bands = stratify_by_height(readable, unreadable, samples, eligible)

        self.assertEqual(bands["<16"], ({0: "1"}, set()))
        self.assertEqual(bands["16-19"], ({}, {1}))
        self.assertEqual(bands[">=24"], ({2: "9"}, set()))

    def test_every_eligible_index_lands_in_exactly_one_band(self):
        samples = {i: {"crop_height": h} for i, h in enumerate([5, 17, 22, 40, 12])}
        readable = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        bands = stratify_by_height(readable, set(), samples, list(samples))

        total = sum(len(band_readable) for band_readable, _ in bands.values())
        self.assertEqual(total, 5)


if __name__ == "__main__":
    unittest.main()
