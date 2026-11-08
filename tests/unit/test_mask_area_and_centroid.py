"""`mask_area_and_centroid` must return what np.nonzero + mean returned.

It replaces that spelling inside `TrackManager.update_from_propagation`, which
runs for every player on every propagated frame. Both of its outputs feed
lifecycle decisions rather than display: `Track.areas` drives the mask-collapse
test that triggers a reprompt or teardown, and `Track.centroids` drives the
centroid-jump guess that decides which of an overlapping pair gets removed. A
drift here would not fail loudly -- it would change which tracks get reset, and
surface much later as an unexplained identity result.

So the oracle is the previous implementation, and the assertions are for exact
equality, not closeness. That is a fair demand: `cv2.moments`' `m10/m00` and
`xs.mean()` are the same sum of integer coordinates divided by the same count,
both in float64.
"""
import unittest

import numpy as np

from handball_cv.tracking.sam2_manager import mask_area_and_centroid


def reference(mask: np.ndarray):
    """The spelling this replaces, straight out of update_from_propagation."""
    area = float(mask.sum())
    if area <= 0:
        return 0.0, None
    ys, xs = np.nonzero(mask)
    return area, (float(xs.mean()), float(ys.mean()))


class MaskAreaAndCentroidTests(unittest.TestCase):

    def assert_matches_reference(self, mask: np.ndarray, message: str = "") -> None:
        want_area, want_centroid = reference(mask)
        got_area, got_centroid = mask_area_and_centroid(mask)
        self.assertEqual(got_area, want_area, f"{message}: area")
        if want_centroid is None:
            self.assertIsNone(got_centroid, f"{message}: centroid")
        else:
            self.assertIsNotNone(got_centroid, f"{message}: centroid")
            self.assertEqual(got_centroid[0], want_centroid[0], f"{message}: centroid x")
            self.assertEqual(got_centroid[1], want_centroid[1], f"{message}: centroid y")

    def test_an_empty_mask_reports_no_area_and_no_centroid(self):
        # SAM2 yields these for a fully occluded player. The caller relies on
        # the None to skip appending a centroid and to leave last_seen alone.
        area, centroid = mask_area_and_centroid(np.zeros((240, 320), dtype=bool))
        self.assertEqual(area, 0.0)
        self.assertIsNone(centroid)

    def test_a_single_pixel_sits_at_its_own_coordinates(self):
        mask = np.zeros((240, 320), dtype=bool)
        mask[173, 41] = True
        self.assert_matches_reference(mask, "single pixel")
        self.assertEqual(mask_area_and_centroid(mask), (1.0, (41.0, 173.0)))

    def test_a_rectangle_sits_at_its_analytic_centre(self):
        mask = np.zeros((240, 320), dtype=bool)
        mask[100:110, 200:220] = True
        self.assert_matches_reference(mask, "rectangle")
        self.assertEqual(mask_area_and_centroid(mask), (200.0, (209.5, 104.5)))

    def test_a_mask_far_from_the_origin_is_offset_back_correctly(self):
        # The crop is the whole point of the change, so the x0/y0 offset added
        # back to the moment is the thing most likely to be wrong.
        mask = np.zeros((1080, 1920), dtype=bool)
        mask[880:1000, 1700:1850] = True
        self.assert_matches_reference(mask, "offset")

    def test_masks_touching_each_border_are_handled(self):
        for name, box in {
            "top-left": (0, 0, 40, 40),
            "bottom-right": (200, 280, 40, 40),
            "full-width": (100, 0, 20, 320),
            "full-height": (0, 100, 240, 20),
        }.items():
            mask = np.zeros((240, 320), dtype=bool)
            y, x, h, w = box
            mask[y:y + h, x:x + w] = True
            with self.subTest(border=name):
                self.assert_matches_reference(mask, name)

    def test_a_whole_true_mask_is_the_frame_centre(self):
        self.assert_matches_reference(np.ones((240, 320), dtype=bool), "all true")

    def test_disconnected_components_average_across_all_of_them(self):
        # After edge-fragment filtering a mask can still hold several pieces;
        # the centroid is over every true pixel, not just the largest piece.
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:140, 100:140] = True
        mask[300:320, 500:520] = True
        self.assert_matches_reference(mask, "two components")

    def test_randomised_player_shaped_masks_at_full_resolution(self):
        rng = np.random.default_rng(20260909)
        for case in range(20):
            mask = np.zeros((1080, 1920), dtype=bool)
            y, x = int(rng.integers(0, 780)), int(rng.integers(0, 1800))
            mask[y:y + int(rng.integers(80, 300)), x:x + int(rng.integers(40, 120))] = True
            for _ in range(int(rng.integers(0, 3))):
                fy, fx = int(rng.integers(0, 1060)), int(rng.integers(0, 1900))
                mask[fy:fy + 20, fx:fx + 20] = True
            with self.subTest(case=case):
                self.assert_matches_reference(mask, f"random {case}")


if __name__ == "__main__":
    unittest.main()
