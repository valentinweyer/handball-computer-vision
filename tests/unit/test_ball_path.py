"""The one-ball path has to prefer a coherent trajectory to a confident wrong one.

On BHC-FAG the greedy anchoring it replaces accepted consecutive segments 255 px
apart across 2-frame gaps -- objects no ball could have travelled between. These
pin the behaviour that rejects that, without pinning the exact costs, which are
an operating point rather than an invariant.
"""

import unittest

from handball_cv.ball.path import (
    Candidate,
    PathConfig,
    candidates_from_detections,
    select_path,
)


def straight(n, x0=100.0, step=15.0, score=0.8, start=0):
    """A ball crossing the court at a believable speed."""
    return {start + i: [Candidate(start + i, x0 + step * i, 200.0, score)]
            for i in range(n)}


class SelectPathTest(unittest.TestCase):
    config = PathConfig(max_speed=45.0, max_gap=15, absent_cost=0.55, speed_cost=0.012)

    def test_a_clean_trajectory_is_followed_whole(self):
        path = select_path(straight(10), 10, self.config)
        self.assertEqual([c.frame for c in path], list(range(10)))

    def test_an_unreachable_detection_is_rejected_however_confident(self):
        """The failure this module exists for: a 255px hop across 2 frames.

        The distractor scores higher than the real ball on its frame, so only
        the reachability constraint can reject it.
        """
        candidates = straight(6)
        candidates[3].append(Candidate(3, 900.0, 700.0, 0.99))
        path = select_path(candidates, 6, self.config)
        chosen = {c.frame: (c.x, c.y) for c in path}
        self.assertEqual(chosen[3], (145.0, 200.0), "took the confident outlier")

    def test_a_gap_is_stepped_over_when_the_ball_is_reachable(self):
        candidates = straight(4)
        candidates.update(straight(4, x0=100.0 + 15.0 * 7, start=7))
        path = select_path(candidates, 11, self.config)
        self.assertEqual([c.frame for c in path], [0, 1, 2, 3, 7, 8, 9, 10])

    def test_frames_it_skips_are_left_out_not_invented(self):
        candidates = straight(3)
        candidates.update(straight(3, x0=100.0 + 15.0 * 6, start=6))
        path = select_path(candidates, 9, self.config)
        self.assertNotIn(4, [c.frame for c in path])

    def test_a_lone_weak_detection_loses_to_a_gap(self):
        """A 0.25 detection costs 0.75 against 0.55 for skipping the frame."""
        candidates = straight(3)
        candidates[8] = [Candidate(8, 3000.0, 3000.0, 0.25)]
        path = select_path(candidates, 9, self.config)
        self.assertEqual([c.frame for c in path], [0, 1, 2])

    def test_a_gap_longer_than_max_gap_is_not_crossed(self):
        near = straight(2)
        far = straight(2, x0=130.0, start=40)
        path = select_path({**near, **far}, 42, self.config)
        self.assertEqual(len(path), 2, "crossed a 38-frame gap in one edge")

    def test_teleport_is_refused_even_with_nothing_else_on_offer(self):
        candidates = {0: [Candidate(0, 0.0, 0.0, 0.9)],
                      1: [Candidate(1, 1800.0, 900.0, 0.9)]}
        path = select_path(candidates, 2, self.config)
        self.assertEqual(len(path), 1)

    def test_no_candidates_is_an_empty_path_not_an_error(self):
        self.assertEqual(select_path({}, 10, self.config), [])


class CandidatesFromDetectionsTest(unittest.TestCase):
    def test_the_same_ball_seen_by_two_passes_counts_once(self):
        full = {0: [((90.0, 190.0, 113.0, 213.0), 0.6)]}
        crop = {0: [((92.0, 192.0, 115.0, 215.0), 0.8)]}
        merged = candidates_from_detections(full, crop, threshold=0.2)
        self.assertEqual(len(merged[0]), 1)
        self.assertAlmostEqual(merged[0][0].score, 0.8, msg="kept the weaker score")

    def test_two_genuinely_separate_boxes_are_kept_apart(self):
        source = {0: [((90.0, 190.0, 113.0, 213.0), 0.6),
                      ((900.0, 700.0, 923.0, 723.0), 0.7)]}
        self.assertEqual(len(candidates_from_detections(source)[0]), 2)

    def test_boxes_below_threshold_are_dropped(self):
        source = {0: [((90.0, 190.0, 113.0, 213.0), 0.1)]}
        self.assertEqual(candidates_from_detections(source, threshold=0.2), {})


if __name__ == "__main__":
    unittest.main()
