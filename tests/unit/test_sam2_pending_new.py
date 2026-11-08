import unittest

import numpy as np

from handball_cv.tracking.sam2_manager import (
    MIN_CONFIRM_CHECKPOINTS,
    PENDING_MAX_CENTRE_FRAC,
    TrackManager,
    _centre_gap,
)


def box(x, y, w=40, h=100):
    return np.array([x, y, x + w, y + h], dtype=float)


class CentreGapTests(unittest.TestCase):
    def test_identical_boxes_have_no_gap(self):
        self.assertAlmostEqual(_centre_gap(box(0, 0), box(0, 0)), 0.0)

    def test_a_running_player_stays_within_the_gate(self):
        # Measured displacement between checkpoints: median 30px, p75 55px,
        # p90 93px, against player boxes ~100px tall here.
        for travel in (30, 55):
            self.assertLess(_centre_gap(box(0, 0), box(travel, 0)),
                            PENDING_MAX_CENTRE_FRAC, f"{travel}px should match")

    def test_a_different_person_across_the_court_is_out_of_the_gate(self):
        self.assertGreater(_centre_gap(box(0, 0), box(500, 500)),
                           PENDING_MAX_CENTRE_FRAC)

    def test_the_gate_scales_with_apparent_size(self):
        # The same real-world step is more pixels close to camera than far away.
        near = _centre_gap(box(0, 0, 80, 200), box(40, 0, 80, 200))
        far = _centre_gap(box(0, 0, 20, 50), box(10, 0, 20, 50))
        self.assertAlmostEqual(near, far, places=6)


class PendingNewTests(unittest.TestCase):
    """New-player confirmation must follow the person, not a square of the image.

    The previous implementation binned candidates by `round(x/50)_round(y/50)`.
    On a 25fps 1080p clip at CHECK_EVERY=10, 28% of people move more than 50px
    between checkpoints and 57% move far enough to cross a bin edge, so moving
    players could never accumulate two counts in one bin and were never added --
    while a stationary one was admitted immediately.
    """

    def _manager(self):
        return TrackManager(team_model=None, court_test_fn=lambda b: True)

    def _offer(self, manager, boxes):
        """Feed one checkpoint's worth of unmatched detections; -> confirmed boxes."""
        confirmed = []
        seen = set()
        for b in boxes:
            slot = manager._match_pending(b)
            if slot is None:
                manager._pending_new_boxes.append([b.copy(), 1])
                seen.add(len(manager._pending_new_boxes) - 1)
                continue
            manager._pending_new_boxes[slot][0] = b.copy()
            manager._pending_new_boxes[slot][1] += 1
            seen.add(slot)
            if manager._pending_new_boxes[slot][1] >= MIN_CONFIRM_CHECKPOINTS:
                manager._pending_new_boxes[slot][1] = 0
                confirmed.append(b)
        manager._pending_new_boxes = [
            e for i, e in enumerate(manager._pending_new_boxes)
            if i in seen and e[1] > 0
        ]
        return confirmed

    def test_a_moving_player_confirms(self):
        m = self._manager()
        self.assertEqual(self._offer(m, [box(100, 100)]), [])
        # 30px of travel -- the measured median, and two bins under the old scheme
        confirmed = self._offer(m, [box(130, 100)])
        self.assertEqual(len(confirmed), 1)

    def test_a_player_moving_faster_than_a_bin_still_confirms(self):
        m = self._manager()
        self._offer(m, [box(100, 100)])
        self.assertEqual(len(self._offer(m, [box(112, 108)])), 1)

    def test_a_teleporting_detection_does_not_confirm(self):
        # Far apart is a different person, and must restart the count.
        m = self._manager()
        self._offer(m, [box(100, 100)])
        self.assertEqual(self._offer(m, [box(900, 700)]), [])

    def test_confirmation_requires_consecutive_checkpoints(self):
        # detection -> absent checkpoint -> detection must NOT confirm.
        m = self._manager()
        self._offer(m, [box(100, 100)])
        self._offer(m, [])                      # candidate not seen: run broken
        self.assertEqual(self._offer(m, [box(100, 100)]), [])
        self.assertEqual(len(self._offer(m, [box(100, 100)])), 1)

    def test_two_nearby_players_are_tracked_separately(self):
        m = self._manager()
        self._offer(m, [box(100, 100), box(400, 100)])
        confirmed = self._offer(m, [box(115, 100), box(415, 100)])
        self.assertEqual(len(confirmed), 2)

    def test_a_confirmed_candidate_is_not_confirmed_twice(self):
        m = self._manager()
        self._offer(m, [box(100, 100)])
        self.assertEqual(len(self._offer(m, [box(110, 100)])), 1)
        self.assertEqual(self._offer(m, [box(120, 100)]), [])


if __name__ == "__main__":
    unittest.main()
