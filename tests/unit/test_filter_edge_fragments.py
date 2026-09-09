"""`filter_edge_fragments` must stay a speed change, never a behaviour change.

It replaces `sv.filter_segments_by_distance(mode="edge")` in
`masks_from_logits`, which is on the propagation path for every player on every
frame. Its output becomes the boxes `TrackManager` reprompts from and the masks
the identity and jersey layers consume, so any drift here would show up as an
unexplained tracking result rather than as an obvious failure.

The three optimisations each have a way to be subtly wrong, and each has cases
below:

  * the bounding-box crop could lose a component or shift the distance
    threshold (which is relative to the *full* image diagonal, not the crop's);
  * `cv2.distanceTransform` could disagree with supervision's hand-rolled
    chamfer loop at the threshold boundary;
  * the single `np.unique` pass could keep the background label, or drop a
    component the per-label loop would have kept.

Supervision is the oracle: these assert byte-identical masks, not similar ones.
"""
import unittest

import numpy as np
import supervision as sv

from handball_cv.tracking.sam2_driver import filter_edge_fragments

RELATIVE_DISTANCE = 0.03  # the value masks_from_logits uses


def reference(mask: np.ndarray) -> np.ndarray:
    return sv.filter_segments_by_distance(
        mask, relative_distance=RELATIVE_DISTANCE, mode="edge"
    )


def blob(mask: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
    mask[y:y + h, x:x + w] = True
    return mask


class FilterEdgeFragmentsEquivalenceTests(unittest.TestCase):
    """Every case asserts equality with supervision's own result."""

    def assert_matches_supervision(self, mask: np.ndarray, message: str = "") -> None:
        expected = reference(mask)
        actual = filter_edge_fragments(mask, relative_distance=RELATIVE_DISTANCE)
        self.assertEqual(actual.dtype, np.bool_, message)
        self.assertEqual(actual.shape, mask.shape, message)
        differing = int(np.count_nonzero(expected != actual))
        self.assertEqual(differing, 0, f"{message}: {differing} pixels differ")

    def test_an_empty_mask_is_returned_unchanged(self):
        # SAM2 yields these for a fully occluded player; the early return must
        # not turn one into a crash or a non-bool array.
        self.assert_matches_supervision(np.zeros((240, 320), dtype=bool), "empty")

    def test_a_single_component_survives_untouched(self):
        mask = blob(np.zeros((480, 640), dtype=bool), 100, 200, 150, 60)
        self.assert_matches_supervision(mask, "single blob")

    def test_a_fragment_inside_the_threshold_is_kept(self):
        # A limb separated from the torso by a thin occluder: within
        # 0.03 * diagonal, so supervision keeps it and so must we.
        mask = blob(np.zeros((480, 640), dtype=bool), 100, 200, 150, 60)
        blob(mask, 100, 268, 40, 20)
        self.assert_matches_supervision(mask, "near fragment")

    def test_a_fragment_outside_the_threshold_is_dropped(self):
        mask = blob(np.zeros((480, 640), dtype=bool), 100, 200, 150, 60)
        blob(mask, 400, 600, 12, 12)
        self.assert_matches_supervision(mask, "far fragment")

    def test_components_at_the_image_borders_are_handled(self):
        # The crop is degenerate here: it spans the whole frame, so this is the
        # case where cropping must not change anything at all.
        mask = np.zeros((480, 640), dtype=bool)
        blob(mask, 0, 0, 30, 30)
        blob(mask, 450, 610, 30, 30)
        blob(mask, 200, 300, 100, 40)
        self.assert_matches_supervision(mask, "corners")

    def test_a_mask_far_from_the_origin_is_unaffected_by_the_crop(self):
        # The crop translates every component's coordinates; the equal-area
        # tie-break reads those coordinates, so it must be translation-safe.
        mask = blob(np.zeros((480, 640), dtype=bool), 330, 520, 120, 100)
        self.assert_matches_supervision(mask, "offset blob")

    def test_equal_area_components_break_the_tie_the_same_way(self):
        mask = np.zeros((480, 640), dtype=bool)
        blob(mask, 100, 100, 80, 40)
        blob(mask, 100, 400, 80, 40)  # identical area, further right
        self.assert_matches_supervision(mask, "equal-area tie")

    def test_a_fully_true_mask_is_one_component(self):
        self.assert_matches_supervision(np.ones((240, 320), dtype=bool), "all true")

    def test_a_single_pixel_is_one_component(self):
        mask = np.zeros((240, 320), dtype=bool)
        mask[120, 160] = True
        self.assert_matches_supervision(mask, "single pixel")

    def test_randomised_player_shaped_masks_at_full_resolution(self):
        # The real workload: a 1080p body blob, a few nearby fragments and some
        # distant speckle, which is the mix the filter exists to clean up.
        rng = np.random.default_rng(20260909)
        for case in range(12):
            mask = np.zeros((1080, 1920), dtype=bool)
            y, x = int(rng.integers(40, 700)), int(rng.integers(40, 1700))
            blob(mask, y, x, 300, 120)
            for _ in range(int(rng.integers(0, 4))):
                dy, dx = int(rng.integers(-90, 90)), int(rng.integers(-90, 90))
                blob(mask, np.clip(y + dy, 0, 1050), np.clip(x + dx, 0, 1890), 25, 20)
            for _ in range(int(rng.integers(0, 3))):
                blob(mask, int(rng.integers(0, 1060)), int(rng.integers(0, 1900)), 12, 12)
            with self.subTest(case=case):
                self.assert_matches_supervision(mask, f"random {case}")

    def test_a_non_boolean_mask_is_rejected(self):
        with self.assertRaises(TypeError):
            filter_edge_fragments(
                np.zeros((8, 8), dtype=np.uint8), relative_distance=RELATIVE_DISTANCE
            )


if __name__ == "__main__":
    unittest.main()
