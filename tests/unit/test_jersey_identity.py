import unittest

import cv2
import numpy as np

from handball_cv.jersey.identity import NumberVoter, read_numbers


class _RecordingOCRModel:
    def __init__(self):
        self.crops = []
        self.kwargs = []

    def recognize(self, crop, **kwargs):
        self.crops.append(crop.copy())
        self.kwargs.append(kwargs)
        return [(None, "13", 0.9)]


class ReadNumbersTests(unittest.TestCase):
    def test_passes_tight_native_grayscale_crop_to_easyocr(self):
        frame_rgb = np.empty((20, 20, 3), dtype=np.uint8)
        frame_rgb[:] = (200, 100, 10)
        model = _RecordingOCRModel()

        result = read_numbers(model, frame_rgb, np.array([[0, 0, 20, 20]]))

        self.assertEqual(result, ["13"])
        self.assertEqual(model.crops[0].shape, (20, 20))
        expected = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        np.testing.assert_array_equal(model.crops[0], expected)
        self.assertEqual(model.kwargs[0]["allowlist"], "0123456789")
        self.assertEqual(model.kwargs[0]["horizontal_list"], [[0, 20, 0, 20]])

    def test_rejects_low_confidence_read(self):
        model = _RecordingOCRModel()
        model.recognize = lambda crop, **kwargs: [(None, "13", 0.49)]
        frame_rgb = np.zeros((20, 20, 3), dtype=np.uint8)

        self.assertEqual(
            read_numbers(model, frame_rgb, np.array([[0, 0, 20, 20]])), [""]
        )

    def test_voter_rejects_impossible_three_digit_output(self):
        voter = NumberVoter(min_votes=1, min_margin=0)
        voter.observe(1, "113")

        self.assertEqual(voter.best(1), (None, 0, 0.0))

    def test_default_margin_accepts_measured_dominant_number(self):
        # The three "3" reads are partial views of "13" (a crop catching only the
        # trailing digit), so they corroborate it rather than compete with it: 6+3=9
        # votes, and the runner-up drops to a stray single read. Before suffix folding
        # this same input scored ("13", 6, 0.25) -- same verdict, but the split evidence
        # left it much closer to the 0.2 margin floor than the observations warrant.
        #
        # The stray "0" is not a legal jersey number, so it is dropped after folding
        # and excluded from the denominator too: 8/11, not 8/12. It had no fold target
        # here, but would have merged into a "10"/"20"/"40" had one been present.
        voter = NumberVoter()
        for value in ["13"] * 6 + ["3"] * 3 + ["0", "1", "4"]:
            voter.observe(4, value)

        number, votes, margin = voter.best(4)
        self.assertEqual((number, votes), ("13", 9))
        self.assertAlmostEqual(margin, 8 / 11)

    def test_partial_trailing_digit_reads_resolve_to_the_full_number(self):
        # Measured on FelixClaar player 3 (jersey 17): reads split 17/7 evenly, and the
        # tie suppressed any verdict even though every read was consistent with 17.
        voter = NumberVoter()
        for value in ["17"] * 4 + ["7"] * 4 + ["10", "1", "11"]:
            voter.observe(3, value)

        self.assertEqual(voter.best(3)[0], "17")

    def test_a_few_long_reads_cannot_capture_a_dominant_short_number(self):
        # Measured on BHC-FAG player 27, visually confirmed as jersey 7: an absolute
        # min_promote_votes floor of 2 let two stray "77" reads absorb thirty-two "7"
        # reads and flip the verdict. Folding now also requires the longer reading to
        # hold min_promote_ratio of the support it is absorbing.
        voter = NumberVoter()
        for value in ["7"] * 32 + ["77"] * 2 + ["22"] * 14:
            voter.observe(27, value)

        self.assertEqual(voter.best(27)[0], "7")

    def test_comparably_supported_long_read_still_folds(self):
        # The guard must not become so strict that it blocks genuine partial views:
        # BHC-FAG player 23 reads {'7': 7, '77': 5}, where 77 has real support.
        voter = NumberVoter()
        for value in ["7"] * 7 + ["77"] * 5:
            voter.observe(23, value)

        self.assertEqual(voter.best(23)[0], "77")

    def test_a_single_long_read_cannot_capture_an_established_short_number(self):
        # A lone "17" misread must not steal a player genuinely wearing 7, so folding
        # requires the longer reading to clear min_promote_votes on its own.
        voter = NumberVoter()
        for value in ["7"] * 9 + ["17"]:
            voter.observe(5, value)

        self.assertEqual(voter.best(5)[0], "7")

    def test_ambiguous_equally_supported_extensions_do_not_fold(self):
        # "7" cannot arbitrate between 17 and 27 when both are equally supported;
        # folding either way would invent evidence, so the short read stays put.
        voter = NumberVoter(min_votes=1, min_margin=0.0)
        for value in ["17"] * 3 + ["27"] * 3 + ["7"] * 2:
            voter.observe(6, value)

        counts = voter._votes[6].resolved_counts(voter.min_promote_votes)
        self.assertEqual(counts.get("7"), 2)

    def test_leading_digit_reads_do_not_fold(self):
        # "1" is a prefix of "17", not a partial trailing view, and could equally be
        # its own number -- only trailing-digit reads are unambiguous partial views.
        voter = NumberVoter()
        for value in ["17"] * 3 + ["1"] * 4:
            voter.observe(7, value)

        counts = voter._votes[7].resolved_counts(voter.min_promote_votes)
        self.assertEqual(counts.get("1"), 4)

class ValidJerseyNumberTests(unittest.TestCase):
    """0 is not a legal handball number, but is a real partial view of 10/20/40."""

    def test_bare_zero_never_becomes_a_verdict(self):
        # Measured on BHC-FAG p3: the number detector fired on a shorts logo and the
        # reader called it "0" thirteen times, outvoting every real candidate.
        voter = NumberVoter()
        for value in ["0"] * 13 + ["53"] * 6:
            voter.observe(3, value)
        number, _votes, _margin = voter.best(3)
        self.assertNotEqual(number, "0")

    def test_zero_still_folds_into_a_number_ending_in_zero(self):
        # Measured on BHC-FAG p11 (jersey 40): a stray "0" is a trailing-digit view of
        # "40" and must strengthen it, not be discarded.
        voter = NumberVoter()
        for value in ["40"] * 5 + ["0"]:
            voter.observe(11, value)
        number, votes, _margin = voter.best(11)
        self.assertEqual((number, votes), ("40", 6))

    def test_three_digit_and_leading_zero_reads_are_rejected(self):
        voter = NumberVoter(min_votes=1, min_margin=0.0)
        for value in ["07"] * 5:
            voter.observe(1, value)
        self.assertIsNone(voter.best(1)[0])

if __name__ == "__main__":
    unittest.main()
