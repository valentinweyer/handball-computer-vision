import unittest

from scripts.benchmark_qwen_jersey_ocr import parse_digit_sequence
from scripts.compare_read_modes import (
    DIGIT_VARIANT,
    WHOLE_VARIANT,
    classify_truncations,
    compare_h1,
    mcnemar_exact,
    pair_variants,
    score_arm,
)


def arm(whole, digit, whole_status="exact", digit_status="exact"):
    return {WHOLE_VARIANT: (whole, whole_status), DIGIT_VARIANT: (digit, digit_status)}


class McNemarTests(unittest.TestCase):
    """The gate is paired significance, not a difference of two accuracy figures."""

    def test_no_discordant_pairs_is_no_evidence(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_a_small_lopsided_split_is_not_significant(self):
        # 3 vs 0 cannot reach p<0.05 two-sided: the whole point of requiring the test.
        self.assertGreater(mcnemar_exact(3, 0), 0.05)

    def test_a_large_lopsided_split_is_significant(self):
        self.assertLess(mcnemar_exact(12, 1), 0.05)

    def test_an_even_split_is_maximally_unconvincing(self):
        self.assertEqual(mcnemar_exact(7, 7), 1.0)

    def test_the_test_is_symmetric(self):
        self.assertEqual(mcnemar_exact(9, 2), mcnemar_exact(2, 9))


class PairingTests(unittest.TestCase):
    def _report(self, whole, digit):
        return {"raw": {WHOLE_VARIANT: whole, DIGIT_VARIANT: digit}}

    def test_only_indices_present_in_both_arms_are_paired(self):
        report = self._report(
            {"1": {"prediction": "22", "parse_status": "exact"},
             "2": {"prediction": "7", "parse_status": "exact"}},
            {"1": {"prediction": "22", "parse_status": "exact"}},
        )
        self.assertEqual(sorted(pair_variants(report, [1, 2, 3])), [1])

    def test_truncated_responses_are_excluded_from_both_arms(self):
        # A truncation is a token-budget failure. Scoring it as an abstention would
        # let a harness misconfiguration masquerade as the model declining.
        report = self._report(
            {"1": {"prediction": "", "parse_status": "truncated"},
             "2": {"prediction": "22", "parse_status": "exact"}},
            {"1": {"prediction": "22", "parse_status": "exact"},
             "2": {"prediction": "", "parse_status": "truncated"}},
        )
        self.assertEqual(pair_variants(report, [1, 2]), {})


class ScoreArmTests(unittest.TestCase):
    def test_accuracy_coverage_and_abstention_are_scored_separately(self):
        paired = {
            1: arm("22", "22"),      # readable, both right
            2: arm("23", "22"),      # readable, whole wrong
            3: arm("", ""),          # readable, both abstain
            4: arm("", ""),          # unreadable, both abstain correctly
            5: arm("19", ""),        # unreadable, whole invents a number
        }
        readable = {1: "22", 2: "22", 3: "17"}
        unreadable = {4, 5}
        whole = score_arm(paired, WHOLE_VARIANT, readable, unreadable)
        self.assertEqual((whole["correct"], whole["wrong"]), (1, 1))
        self.assertAlmostEqual(whole["coverage"], 2 / 3)
        self.assertAlmostEqual(whole["selective_accuracy"], 0.5)
        self.assertAlmostEqual(whole["abstention_rate"], 0.5)
        self.assertEqual(whole["false_reads_on_unreadable"], 1)

        digit = score_arm(paired, DIGIT_VARIANT, readable, unreadable)
        self.assertEqual((digit["correct"], digit["wrong"]), (2, 0))
        self.assertAlmostEqual(digit["abstention_rate"], 1.0)


class H1Tests(unittest.TestCase):
    def test_discordant_pairs_drive_the_verdict_not_the_agreements(self):
        paired = {
            1: arm("22", "22"),   # both right -- contributes nothing
            2: arm("22", "22"),
            3: arm("99", "17"),   # digit only
            4: arm("99", "17"),
            5: arm("13", "99"),   # whole only
        }
        readable = {1: "22", 2: "22", 3: "17", 4: "17", 5: "13"}
        h1 = compare_h1(paired, readable)
        self.assertEqual(h1["both_correct"], 2)
        self.assertEqual(h1["digit_only"], 2)
        self.assertEqual(h1["whole_only"], 1)
        self.assertEqual(h1["p_value"], mcnemar_exact(1, 2))

    def test_unreadable_crops_are_not_part_of_the_accuracy_comparison(self):
        # Index 1 has a known number (whole right, digit abstained); index 2 has no
        # number at all and must not appear in the paired counts either way.
        paired = {1: arm("22", ""), 2: arm("", "")}
        scored = compare_h1(paired, {1: "22"})
        self.assertEqual(scored["whole_only"], 1)
        self.assertEqual(scored["both_correct"] + scored["neither"], 0)
        self.assertEqual(sum(compare_h1(paired, {}).get(k, 0)
                             for k in ("both_correct", "whole_only",
                                       "digit_only", "neither")), 0)


class H2Tests(unittest.TestCase):
    """The failure that forced suffix-folding: a half-seen 17 read as a bare 7."""

    def test_a_silent_truncation_rescued_by_an_explicit_partial_is_counted(self):
        paired = {
            1: arm("7", "", digit_status="partial_?7"),   # rescued
            2: arm("7", "17"),                            # digit read it fully
            3: arm("1", "", digit_status="abstain"),      # still silent-ish
            4: arm("17", "17"),                           # no truncation at all
        }
        readable = {1: "17", 2: "17", 3: "17", 4: "17"}
        h2 = classify_truncations(paired, readable)
        self.assertEqual(h2["silent_truncations_in_whole_mode"], 3)
        self.assertEqual(h2["digit_mode_returned_explicit_partial"], 1)
        self.assertEqual(h2["digit_mode_read_the_whole_number"], 1)

    def test_single_digit_truths_cannot_be_truncations(self):
        paired = {1: arm("7", "7")}
        self.assertEqual(
            classify_truncations(paired, {1: "7"})["silent_truncations_in_whole_mode"], 0
        )

    def test_a_wrong_digit_is_not_a_truncation(self):
        # "9" is not a digit of 17, so whole mode misread rather than truncated.
        paired = {1: arm("9", "")}
        self.assertEqual(
            classify_truncations(paired, {1: "17"})["silent_truncations_in_whole_mode"], 0
        )

    def test_the_partial_status_produced_by_the_parser_is_the_one_h2_looks_for(self):
        # Guards the contract between the two scripts.
        _value, status = parse_digit_sequence("? 7")
        self.assertTrue(status.startswith("partial_"))


if __name__ == "__main__":
    unittest.main()
