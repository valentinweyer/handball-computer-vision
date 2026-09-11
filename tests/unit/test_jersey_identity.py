import unittest

import cv2
import numpy as np

from handball_cv.jersey.identity import (
    NumberVoter, read_numbers, segment_breaks, segment_verdicts,
)


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


class VerdictHysteresisTests(unittest.TestCase):
    """A verdict is held once set, and the bar to set one is high enough to hold."""

    def test_resolved_number_is_not_retracted_by_contrary_noise(self):
        # Measured on BHC-FAG p13 (jersey 25): partial "2" reads dragged the margin
        # from 1.0 to 0.14 and the overlay label reverted to "P13" mid-clip.
        voter = NumberVoter()
        for value in ["25"] * 5:
            voter.observe(13, value)
        self.assertEqual(voter.best(13)[0], "25")
        for value in ["2", "2", "28"]:
            voter.observe(13, value)
        self.assertEqual(voter.best(13)[0], "25")

    def test_a_rival_that_clears_both_gates_still_replaces_the_held_value(self):
        # Hysteresis must not become ConsecutiveValueTracker's permanent lock.
        voter = NumberVoter()
        for value in ["25"] * 5:
            voter.observe(1, value)
        self.assertEqual(voter.best(1)[0], "25")
        for value in ["31"] * 12:
            voter.observe(1, value)
        self.assertEqual(voter.best(1)[0], "31")

    def test_unanimous_early_reads_qualify_but_a_split_does_not(self):
        # Margin, not count, separates a safe early commit from a premature one --
        # both of these are three-vote commits, and only one was right.
        #
        # BHC-FAG p3 is visually confirmed as jersey 53 and its first three reads
        # were 53/53/53 (margin 1.0). Requiring more votes suppressed a correct
        # answer, because the reads that followed were shorts-logo noise.
        good = NumberVoter()
        for value in ["53"] * 3:
            good.observe(3, value)
        self.assertEqual(good.best(3)[0], "53")

        # FelixClaar/BHC-FAG p2 is jersey 22, but EasyOCR's 2<->9 confusion put it
        # at {'92': 3, '22': 2} -- margin 0.2, contested from the very first read.
        split = NumberVoter()
        for value in ["92", "92", "22", "22", "92"]:
            split.observe(2, value)
        self.assertIsNone(split.best(2)[0])

    def test_merge_rederives_rather_than_inheriting_a_verdict(self):
        voter = NumberVoter()
        for value in ["25"] * 5:
            voter.observe(1, value)
        for value in ["31"] * 5:
            voter.observe(2, value)
        voter.merge(1, 2)
        # 25 and 31 now tie at 5 apiece: neither clears the margin, so no verdict.
        self.assertIsNone(voter.best(2)[0])


def reads(*rows):
    return [{"frame": f, "player_id": p, "value": v} for f, p, v in rows]


class SegmentBreakTests(unittest.TestCase):
    """Which run events cut an identity into separate people."""

    def test_reid_and_suspected_switch_break_but_label_events_do_not(self):
        events = [
            {"frame": 130, "type": "reid", "player_id": 5},
            {"frame": 530, "type": "suspected_id_switch", "player_id": 5},
            {"frame": 660, "type": "team_switch", "player_id": 5},
            {"frame": 700, "type": "link", "player_id": 5},
        ]
        self.assertEqual(segment_breaks(events), {5: [130, 530]})

    def test_repeated_events_on_one_frame_cut_once(self):
        events = [
            {"frame": 130, "type": "reid", "player_id": 5},
            {"frame": 130, "type": "suspected_id_switch", "player_id": 5},
        ]
        self.assertEqual(segment_breaks(events), {5: [130]})


class SegmentVerdictTests(unittest.TestCase):
    """A verdict earned late describes the start of the span that earned it.

    The measurement these stand for is on the 60s Melsungen clip: 7045 of 19097
    player-frames carry a number causally, and segment-local backfill adds 3273
    without a new read or a new model.
    """

    def test_a_late_verdict_backfills_its_own_earlier_frames(self):
        segments = segment_verdicts(
            reads((10, 1, "7"), (20, 1, "7"), (30, 1, "7")),
            events=[],
            live_frames={1: range(1, 51)},
        )
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.verdict, "7")
        self.assertEqual(segment.resolved_at, 30)
        self.assertEqual(segment.backfill, tuple(range(1, 30)))
        self.assertEqual(segment.labelled, tuple(range(30, 51)))

    def test_backfill_never_crosses_a_break(self):
        """p6 is 15 until frame 560 and somebody else after.

        Re-ID moves an identity onto whoever it believes has reappeared, so a
        whole-identity backfill would paint the second person's number over the
        first person's frames. Each side is replayed on its own reads alone.
        """
        segments = segment_verdicts(
            reads(
                (10, 6, "15"), (20, 6, "15"), (30, 6, "15"),
                (70, 6, "20"), (80, 6, "20"), (90, 6, "20"),
            ),
            events=[{"frame": 60, "type": "reid", "player_id": 6}],
            live_frames={6: range(1, 101)},
        )
        self.assertEqual([s.verdict for s in segments], ["15", "20"])
        first, second = segments
        self.assertEqual(first.backfill, tuple(range(1, 30)))
        self.assertNotIn(60, first.live)
        # The later verdict reaches back only to the break, never past it.
        self.assertEqual(min(second.backfill), 60)
        self.assertEqual(second.backfill, tuple(range(60, 90)))

    def test_a_verdict_that_changed_mid_segment_backfills_nothing(self):
        """p20 went 6 -> 15, and its early frames' own evidence said 6.

        Backfilling the final answer would overwrite what those frames showed.
        One segment in 14 on the Melsungen clip does this; withholding it costs
        284 frames and is the whole reason the guard exists.
        """
        segments = segment_verdicts(
            reads(
                (10, 20, "6"), (20, 20, "6"), (30, 20, "6"),
                *[(f, 20, "15") for f in range(40, 130, 10)],
            ),
            events=[],
            live_frames={20: range(1, 201)},
        )
        segment, = segments
        self.assertEqual(segment.verdict, "15")
        self.assertFalse(segment.stable)
        self.assertEqual(segment.backfill, ())
        # Coverage is unchanged, not reduced: what was drawn is still drawn.
        self.assertEqual(segment.labelled, tuple(range(30, 201)))

    def test_an_unresolved_segment_offers_nothing_but_is_still_reported(self):
        """A caller measuring coverage needs the denominator, not just the wins."""
        segments = segment_verdicts(
            reads((10, 4, "7"), (20, 4, "9")),
            events=[],
            live_frames={4: range(1, 51)},
        )
        segment, = segments
        self.assertIsNone(segment.verdict)
        self.assertIsNone(segment.resolved_at)
        self.assertEqual(segment.backfill, ())
        self.assertEqual(segment.labelled, ())
        self.assertEqual(len(segment.live), 50)

    def test_frames_the_identity_was_absent_are_never_backfilled(self):
        """Backfill labels frames that were drawn, and an absent player has none."""
        segments = segment_verdicts(
            reads((40, 3, "8"), (45, 3, "8"), (50, 3, "8")),
            events=[],
            live_frames={3: [1, 2, 3, 30, 40, 45, 50]},
        )
        segment, = segments
        self.assertEqual(segment.backfill, (1, 2, 3, 30, 40, 45))
        self.assertEqual(segment.labelled, (50,))

    def test_a_break_while_off_screen_does_not_plant_the_old_label(self):
        """p1 breaks at frame 540 and is not on screen again until 541.

        The carry-in that keeps a `suspected_id_switch` from losing a label it
        was still drawing must anchor on the segment's first *live* frame. Keyed
        to the break frame instead, it plants the previous person's verdict at
        the boundary -- which back-dates `resolved_at` to the segment start and
        erases the whole backfill. Cost 1178 frames on the 60s Melsungen clip.
        """
        segments = segment_verdicts(
            reads(
                (10, 1, "25"), (20, 1, "25"), (30, 1, "25"),
                (400, 1, "25"), (410, 1, "25"), (420, 1, "25"),
            ),
            events=[{"frame": 200, "type": "reid", "player_id": 1}],
            live_frames={1: [*range(1, 100), *range(201, 500)]},
        )
        first, second = segments
        self.assertEqual(first.verdict, "25")
        # The new segment re-earns the number on its own reads. `suspend` keeps
        # the tally, so the first agreeing read vouches for it -- frame 400, not
        # three reads later.
        self.assertEqual(second.start, 201)
        self.assertEqual(second.resolved_at, 400)
        self.assertEqual(second.backfill, tuple(range(201, 400)))
        self.assertEqual(second.value_at(250), "25")
        # The two segments stop and start at the live frames, not at the break.
        self.assertEqual((first.start, first.end), (1, 99))

    def test_breaks_outside_the_live_span_do_not_open_empty_segments(self):
        segments = segment_verdicts(
            reads((10, 2, "5"), (12, 2, "5"), (14, 2, "5")),
            events=[{"frame": 0, "type": "reid", "player_id": 2}, {"frame": 5, "type": "reid", "player_id": 2}, {"frame": 900, "type": "reid", "player_id": 2}],
            live_frames={2: range(5, 21)},
        )
        self.assertEqual([(s.start, s.end) for s in segments], [(5, 20)])

    def test_reads_land_in_the_segment_the_break_opens(self):
        """A break frame is the first frame of the new segment, not the last of the old."""
        segments = segment_verdicts(
            reads((60, 7, "9"), (70, 7, "9"), (80, 7, "9")),
            events=[{"frame": 60, "type": "reid", "player_id": 7}],
            live_frames={7: range(1, 101)},
        )
        before, after = segments
        self.assertIsNone(before.verdict)
        self.assertEqual(before.end, 59)
        self.assertEqual(after.start, 60)
        self.assertEqual(after.verdict, "9")


if __name__ == "__main__":
    unittest.main()
