"""A track the detector has stopped confirming must eventually end.

Rule 2 only retires a track whose *mask collapses*. A substitute who walks off
and sits on the bench stays fully visible, so SAM2 keeps a perfect mask and the
track survives to the end of the clip. Measured on Melsungen_Berlin_2min_1: 17.8
live tracks against 14.0 detections per frame, live > detections on 97% of
frames, and 34% of frames pinned at MAX_LIVE_OBJECTS -- at which point a
genuinely new player on court cannot be admitted at all.
"""
import unittest

from handball_cv.tracking.sam2_manager import (
    MIN_CONFIRM_CHECKPOINTS,
    UNMATCHED_CHECKPOINTS_MAX,
    Track,
)


class UndetectedRetirementTests(unittest.TestCase):
    """The counters that decide it, exercised the way the checkpoint loop does."""

    def test_a_healthy_but_undetected_track_survives_a_brief_dropout(self):
        # Measured over three 10-minute windows, a real on-court player vanishes
        # from the detector for 65 frames at p90 -- 6.5 checkpoints. Retiring
        # that fast would churn identities through re-ID at 0.55 rank-1.
        track = Track(obj_id=1)
        for _ in range(6):
            self.assertFalse(track.confirm("undetected", UNMATCHED_CHECKPOINTS_MAX))

    def test_it_is_retired_once_the_dropout_is_no_longer_plausible(self):
        track = Track(obj_id=1)
        fired = [track.confirm("undetected", UNMATCHED_CHECKPOINTS_MAX)
                 for _ in range(UNMATCHED_CHECKPOINTS_MAX)]
        self.assertEqual(fired[:-1], [False] * (UNMATCHED_CHECKPOINTS_MAX - 1))
        self.assertTrue(fired[-1])

    def test_one_detection_resets_the_run(self):
        # `clear_all_but` drops any condition not active this checkpoint, so a
        # single confirming detection restarts the count from zero.
        track = Track(obj_id=1)
        for _ in range(UNMATCHED_CHECKPOINTS_MAX - 1):
            track.confirm("undetected", UNMATCHED_CHECKPOINTS_MAX)
        track.clear_all_but(set())
        self.assertFalse(track.confirm("undetected", UNMATCHED_CHECKPOINTS_MAX))

    def test_a_collapsed_mask_still_ends_far_sooner(self):
        # The two rules are deliberately different speeds: a vanished mask is
        # unambiguous, an undetected-but-healthy one is not.
        self.assertLess(MIN_CONFIRM_CHECKPOINTS, UNMATCHED_CHECKPOINTS_MAX)
        track = Track(obj_id=1)
        fired = [track.confirm("gone") for _ in range(MIN_CONFIRM_CHECKPOINTS)]
        self.assertTrue(fired[-1])

    def test_the_two_rules_count_independently(self):
        track = Track(obj_id=1)
        for _ in range(5):
            track.confirm("undetected", UNMATCHED_CHECKPOINTS_MAX)
        self.assertFalse(track.confirm("gone"))     # first "gone" checkpoint
        self.assertTrue(track.confirm("gone"))      # second reaches the threshold


if __name__ == "__main__":
    unittest.main()
