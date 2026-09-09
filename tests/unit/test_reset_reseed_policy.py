"""The reset_reseed checkpoint policy must change memory, and nothing else.

It exists to price what EdgeTAM would force on this project: EdgeTAM's
predictor refuses to add an object once tracking has started, so McByte++
tears the session down and re-seeds everyone instead. That only measures what
it claims to measure if `TrackManager`'s decisions survive the rewrite intact
-- if the reseed quietly drops a correction, or re-seeds a body-swapped object
from the very mask the manager flagged as wrong, the experiment would blame
memory teardown for a bug in the harness.

These pin the decision-honouring half against a recording stub. The memory
teardown itself is the point of the policy and is not something to assert
away: `reset_state` is expected, exactly once, before any re-add.
"""
import unittest

import numpy as np

from handball_cv.tracking.sam2_driver import _apply_reset_reseed


class RecordingPredictor:
    """Records the calls `_apply_reset_reseed` makes, in order."""

    def __init__(self):
        self.calls = []

    def reset_state(self, state):
        self.calls.append(("reset_state",))

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.calls.append(("add_new_mask", obj_id, frame_idx, int(mask.sum())))

    def add_new_points_or_box(self, state, frame_idx, obj_id, box, **kw):
        self.calls.append(("add_new_box", obj_id, frame_idx, tuple(np.asarray(box))))

    # convenience views
    def seeded_by_mask(self):
        return {c[1] for c in self.calls if c[0] == "add_new_mask"}

    def seeded_by_box(self):
        return {c[1]: c[3] for c in self.calls if c[0] == "add_new_box"}


def mask_with(pixels: int, height: int = 40, width: int = 40) -> np.ndarray:
    m = np.zeros((height, width), dtype=bool)
    if pixels:
        m.flat[:pixels] = True
    return m


class ResetReseedPolicyTests(unittest.TestCase):

    def setUp(self):
        self.predictor = RecordingPredictor()
        self.masks = {1: mask_with(100), 2: mask_with(120), 3: mask_with(80)}

    def apply(self, actions, masks=None):
        _apply_reset_reseed(
            self.predictor, state={}, frame_idx=70,
            actions=actions, masks_by_id=self.masks if masks is None else masks,
        )
        return self.predictor

    def test_the_memory_bank_is_torn_down_once_before_any_reseed(self):
        p = self.apply([])
        kinds = [c[0] for c in p.calls]
        self.assertEqual(kinds.count("reset_state"), 1)
        self.assertEqual(kinds[0], "reset_state", "reset must precede every re-add")

    def test_an_untouched_survivor_is_reseeded_from_its_own_mask(self):
        # A box would lose the mask's shape, and shape is what propagation
        # carries forward -- so a healthy track keeps its mask.
        p = self.apply([])
        self.assertEqual(p.seeded_by_mask(), {1, 2, 3})
        self.assertEqual(p.seeded_by_box(), {})

    def test_a_removed_object_is_not_reseeded(self):
        p = self.apply([{"type": "remove", "obj_id": 2}])
        self.assertEqual(p.seeded_by_mask(), {1, 3})
        self.assertNotIn(2, p.seeded_by_box())

    def test_a_reprompted_object_is_reseeded_from_its_correction_box(self):
        box = [10.0, 20.0, 30.0, 40.0]
        p = self.apply([{"type": "reprompt", "obj_id": 1, "box": box}])
        self.assertEqual(p.seeded_by_box(), {1: (10.0, 20.0, 30.0, 40.0)})
        self.assertEqual(p.seeded_by_mask(), {2, 3}, "others keep their masks")

    def test_a_body_swapped_object_is_reseeded_from_the_box_not_its_mask(self):
        # This is the one that would silently ruin the experiment. A "reset"
        # means the mask is on the wrong player; re-seeding from it would carry
        # the swap through the teardown and look like a memory-loss failure.
        box = [1.0, 2.0, 3.0, 4.0]
        p = self.apply([{"type": "reset", "obj_id": 3, "box": box}])
        self.assertNotIn(3, p.seeded_by_mask())
        self.assertEqual(p.seeded_by_box()[3], (1.0, 2.0, 3.0, 4.0))

    def test_a_new_object_enters_from_its_detector_box(self):
        box = [5.0, 6.0, 7.0, 8.0]
        p = self.apply([{"type": "add", "obj_id": 9, "box": box}])
        self.assertEqual(p.seeded_by_box(), {9: (5.0, 6.0, 7.0, 8.0)})
        self.assertEqual(p.seeded_by_mask(), {1, 2, 3})

    def test_a_collapsed_mask_with_no_correction_is_dropped(self):
        # SAM2 would take an all-false mask as an empty object; there is no
        # usable seed, and the manager did not supply a box.
        p = self.apply([], masks={1: mask_with(100), 2: mask_with(0)})
        self.assertEqual(p.seeded_by_mask(), {1})
        self.assertNotIn(2, p.seeded_by_box())

    def test_every_decision_is_honoured_together(self):
        p = self.apply([
            {"type": "remove", "obj_id": 2},
            {"type": "reprompt", "obj_id": 1, "box": [0.0, 0.0, 5.0, 5.0]},
            {"type": "add", "obj_id": 7, "box": [9.0, 9.0, 12.0, 12.0]},
        ])
        self.assertEqual(p.seeded_by_mask(), {3})
        self.assertEqual(set(p.seeded_by_box()), {1, 7})
        self.assertEqual([c[0] for c in p.calls].count("reset_state"), 1)

    def test_every_reseed_lands_on_the_checkpoint_frame(self):
        p = self.apply([{"type": "add", "obj_id": 7, "box": [1.0, 1.0, 2.0, 2.0]}])
        frames = {c[2] for c in p.calls if c[0] != "reset_state"}
        self.assertEqual(frames, {70})


if __name__ == "__main__":
    unittest.main()
