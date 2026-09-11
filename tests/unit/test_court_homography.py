"""Temporal court homography: the weighted fit, and what the prior is for.

The failure these guard against is the one measured on the Melsungen clip: a
view holding only the centre circle offers landmarks that are collinear by
construction, the independent fit goes degenerate, and the court projects to a
line -- at a *better* reprojection error than a healthy frame, because a handful
of near-collinear points is trivial to fit perfectly. So the degeneracy test
below asserts on where the court actually lands, never on residual.
"""
import unittest

import numpy as np
from sports import MeasurementUnit
from sports.handball import CourtConfiguration, League

from handball_cv.court.camera_motion import estimate_camera_motion
from handball_cv.court.homography import (
    CourtTracker,
    fit_homography,
    robust_fit,
)
from handball_cv.court.keypoints import (
    KEYPOINT_FLIP_INDEX,
    KEYPOINT_TO_VERTEX,
    court_points,
)

IMAGE_SIZE = (1920, 1080)


def court_vertices():
    config = CourtConfiguration(
        league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS
    )
    return np.array(config.vertices, dtype=np.float64)


def project(matrix, points):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def a_camera():
    """A homography putting the whole court inside a 1920x1080 frame."""
    return np.array([
        [0.42, 0.06, 120.0],
        [0.015, 0.34, 240.0],
        [1.0e-5, 9.0e-5, 1.0],
    ])


class WeightedFitTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array(
            [[0.0, 0.0], [4000.0, 0.0], [4000.0, 2000.0], [0.0, 2000.0], [1500.0, 700.0]]
        )
        self.matrix = a_camera()

    def test_recovers_a_known_homography(self):
        recovered = fit_homography(self.source, project(self.matrix, self.source))
        np.testing.assert_allclose(
            recovered / recovered[2, 2], self.matrix / self.matrix[2, 2], atol=1e-6
        )

    def test_needs_four_correspondences(self):
        self.assertIsNone(fit_homography(self.source[:3], self.source[:3]))

    def test_mismatched_lengths_return_none(self):
        self.assertIsNone(fit_homography(self.source, self.source[:4]))

    def test_weights_decide_which_points_are_honoured(self):
        # One correspondence is moved a long way off. At a negligible weight the
        # fit should ignore it; at a dominant weight it should chase it. This is
        # the knob the temporal prior rides on, so it has to actually bite.
        target = project(self.matrix, self.source)
        corrupted = target.copy()
        corrupted[-1] += [300.0, 220.0]

        ignored = fit_homography(
            self.source, corrupted, np.array([1.0, 1.0, 1.0, 1.0, 1e-6])
        )
        chased = fit_homography(
            self.source, corrupted, np.array([1.0, 1.0, 1.0, 1.0, 500.0])
        )
        error = lambda m: np.linalg.norm(project(m, self.source)[-1] - corrupted[-1])
        self.assertLess(error(chased), error(ignored) / 10)

    def test_robust_fit_rejects_an_outlier(self):
        target = project(self.matrix, np.vstack([self.source, [[2000.0, 1000.0]]]))
        target[-1] += [400.0, 400.0]
        _, mask = robust_fit(np.vstack([self.source, [[2000.0, 1000.0]]]), target, 5.0)
        self.assertTrue(mask[:5].all())
        self.assertFalse(mask[-1])


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.vertices = court_vertices()
        self.camera = a_camera()
        self.centre_line_slots = [
            slot for slot in range(len(KEYPOINT_TO_VERTEX))
            if abs(self.vertices[KEYPOINT_TO_VERTEX[slot]][0] - 2000.0) < 1.0
        ]

    def keypoints(self, slots):
        image = project(self.camera, court_points(slots, self.vertices))
        return [(s, float(x), float(y), 0.9) for s, (x, y) in zip(slots, image)]

    def court_corners(self, fit):
        corners = np.array([[0.0, 0.0], [4000.0, 0.0], [4000.0, 2000.0], [0.0, 2000.0]])
        return project(fit.homography, corners)

    def test_a_well_covered_frame_is_solved_from_its_own_keypoints(self):
        tracker = CourtTracker(self.vertices, IMAGE_SIZE)
        fit = tracker.update(self.keypoints(list(range(len(KEYPOINT_TO_VERTEX)))))
        self.assertEqual(fit.source, "keypoints")
        truth = project(
            self.camera, np.array([[0.0, 0.0], [4000.0, 0.0], [4000.0, 2000.0], [0.0, 2000.0]])
        )
        np.testing.assert_allclose(self.court_corners(fit), truth, atol=1.0)

    def test_low_confidence_keypoints_are_ignored(self):
        tracker = CourtTracker(self.vertices, IMAGE_SIZE, min_confidence=0.95)
        self.assertFalse(tracker.update(self.keypoints(list(range(20)))).usable)

    def test_the_prior_rescues_a_centre_line_only_view(self):
        # Exactly the clip's failure: a frame offering only the collinear centre
        # line, after frames that saw the whole court.
        self.assertGreaterEqual(len(self.centre_line_slots), 4)
        truth = project(
            self.camera, np.array([[0.0, 0.0], [4000.0, 0.0], [4000.0, 2000.0], [0.0, 2000.0]])
        )

        tracker = CourtTracker(self.vertices, IMAGE_SIZE, prior_weight=0.15)
        for _ in range(3):
            tracker.update(self.keypoints(list(range(len(KEYPOINT_TO_VERTEX)))))
        carried = tracker.update(self.keypoints(self.centre_line_slots), motion=np.eye(3))

        alone, _ = robust_fit(
            court_points(self.centre_line_slots, self.vertices),
            np.array([[x, y] for _, x, y, _ in self.keypoints(self.centre_line_slots)]),
            20.0,
        )
        self.assertTrue(carried.usable)
        np.testing.assert_allclose(self.court_corners(carried), truth, atol=25.0)
        if alone is not None:
            # The independent fit is free to put the court anywhere; assert only
            # that it is dramatically worse, not where it lands.
            self.assertGreater(
                np.abs(project(alone, np.array([[0.0, 0.0]]))[0] - truth[0]).max(), 100.0
            )

    def test_propagates_when_a_frame_has_no_usable_keypoints(self):
        tracker = CourtTracker(self.vertices, IMAGE_SIZE)
        tracker.update(self.keypoints(list(range(len(KEYPOINT_TO_VERTEX)))))
        fit = tracker.update([], motion=np.eye(3))
        self.assertEqual(fit.source, "propagated")
        self.assertTrue(fit.usable)

    def test_abstains_rather_than_coast_forever(self):
        tracker = CourtTracker(self.vertices, IMAGE_SIZE, max_age=3)
        tracker.update(self.keypoints(list(range(len(KEYPOINT_TO_VERTEX)))))
        for _ in range(3):
            tracker.update([], motion=np.eye(3))
        self.assertFalse(tracker.update([], motion=np.eye(3)).usable)

    def test_nothing_at_all_yields_an_unusable_fit(self):
        fit = CourtTracker(self.vertices, IMAGE_SIZE).update([])
        self.assertFalse(fit.usable)
        self.assertEqual(fit.source, "none")
        self.assertEqual(len(fit.to_court(np.zeros((0, 2)))), 0)


class CameraMotionTests(unittest.TestCase):
    def test_identical_frames_give_the_identity(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (240, 320), dtype=np.uint8)
        motion = estimate_camera_motion(frame, frame)
        self.assertIsNotNone(motion)
        np.testing.assert_allclose(motion / motion[2, 2], np.eye(3), atol=1e-3)

    def test_recovers_a_translation(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, (240, 320), dtype=np.uint8)
        shifted = np.roll(frame, 7, axis=1)
        motion = estimate_camera_motion(shifted[:, 20:-20], frame[:, 20:-20])
        self.assertIsNotNone(motion)
        self.assertAlmostEqual(motion[0, 2] / motion[2, 2], -7.0, delta=1.0)

    def test_a_blank_frame_returns_none_rather_than_a_wrong_answer(self):
        blank = np.zeros((240, 320), dtype=np.uint8)
        self.assertIsNone(estimate_camera_motion(blank, blank))

    def test_colour_input_is_a_programming_error(self):
        colour = np.zeros((240, 320, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            estimate_camera_motion(colour, colour)


if __name__ == "__main__":
    unittest.main()


class IdentityGateTests(unittest.TestCase):
    """A view with no goal in it cannot say which end of the court it sees.

    The court is symmetric, so one goal area's arc is the other's, and the
    detector -- answering one frame at a time -- has to guess. What makes this
    dangerous is that the wrong answer is *coherent*: mis-labelling landmarks by
    the court's own mirror produces a set that agrees perfectly with itself, so
    RANSAC has two self-consistent stories and no reason to prefer the true one.
    Random label noise, by contrast, RANSAC removes without help -- which is why
    the corruption below is a mirror relabelling and not a shuffle.

    The previous frame is what knows which end was in view.
    """

    MISLABELLED = 25  # of 37: enough that the mirrored story wins the vote

    def setUp(self):
        self.vertices = court_vertices()
        self.camera = a_camera()
        self.all_slots = list(range(len(KEYPOINT_TO_VERTEX)))
        self.corners = np.array(
            [[0.0, 0.0], [4000.0, 0.0], [4000.0, 2000.0], [0.0, 2000.0]]
        )
        self.truth = project(self.camera, self.corners)

    def keypoints(self, mislabelled=0):
        """Landmarks as seen, with the first `mislabelled` given their mirror's name."""
        mirrored = set(self.all_slots[:mislabelled])
        labels = [
            KEYPOINT_FLIP_INDEX[s] if s in mirrored else s for s in self.all_slots
        ]
        image = project(self.camera, court_points(self.all_slots, self.vertices))
        return [(l, float(x), float(y), 0.9) for l, (x, y) in zip(labels, image)]

    def settled_tracker(self, gate):
        tracker = CourtTracker(self.vertices, IMAGE_SIZE, identity_gate_px=gate)
        for _ in range(3):
            tracker.update(self.keypoints())
        return tracker

    def corner_error(self, fit):
        return float(np.abs(project(fit.homography, self.corners) - self.truth).max())

    def test_without_the_gate_a_coherent_mislabelling_moves_the_court(self):
        # Guards the guard: if RANSAC alone coped, the gate would prove nothing.
        fit = self.settled_tracker(None).update(
            self.keypoints(self.MISLABELLED), motion=np.eye(3)
        )
        self.assertGreater(self.corner_error(fit), 500.0)

    def test_the_gate_holds_the_court_where_history_says_it_is(self):
        fit = self.settled_tracker(400.0).update(
            self.keypoints(self.MISLABELLED), motion=np.eye(3)
        )
        self.assertLess(self.corner_error(fit), 25.0)

    def test_a_clean_frame_is_unaffected_by_the_gate(self):
        fit = self.settled_tracker(400.0).update(self.keypoints(), motion=np.eye(3))
        self.assertLess(self.corner_error(fit), 1.0)

    def test_the_gate_never_starves_the_solve(self):
        # A wrong or stale prediction must not be able to reject everything:
        # below four survivors the tracker falls back to the ungated set.
        tracker = CourtTracker(self.vertices, IMAGE_SIZE, identity_gate_px=1.0)
        tracker.update(self.keypoints())
        fit = tracker.update(self.keypoints(), motion=np.eye(3))
        self.assertTrue(fit.usable)
        self.assertGreaterEqual(fit.inliers, 4)
