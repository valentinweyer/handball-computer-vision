import unittest

import cv2
import numpy as np

from handball_cv.ball.appearance import BallObservation, calibrate

COURT = (230, 159, 122)     # blue playing surface
JERSEY = (40, 40, 200)      # red kit
BALL = (30, 140, 230)       # orange ball


def ball_patch(radius=9, size=40):
    patch = np.full((size, size, 3), COURT, np.uint8)
    cv2.circle(patch, (size // 2, size // 2), radius, BALL, -1)
    return patch


def observations(count=14, *, person_height=180.0):
    return [
        BallObservation(patch=ball_patch(), radius=9.0, person_height=person_height)
        for _ in range(count)
    ]


def player_frame(*, with_ball: bool):
    """A player box on court, optionally carrying the ball at chest height."""
    frame = np.full((400, 300, 3), COURT, np.uint8)
    cv2.rectangle(frame, (100, 50), (180, 250), JERSEY, -1)
    if with_ball:
        cv2.circle(frame, (140, 110), 10, BALL, -1)
    return frame


class CalibrationTests(unittest.TestCase):
    def test_abstains_below_the_minimum_number_of_sightings(self):
        self.assertIsNone(calibrate(observations(5), min_samples=12))

    def test_builds_a_model_once_there_is_enough_evidence(self):
        model = calibrate(observations(14), min_samples=12)

        self.assertIsNotNone(model)
        self.assertEqual(model.samples, 14)

    def test_the_size_prior_is_measured_against_player_height(self):
        model = calibrate(observations(14, person_height=180.0))

        self.assertAlmostEqual(model.player_ratio, 9.0 / 180.0)
        # A player twice as tall in the image is twice as close, so is the ball.
        self.assertAlmostEqual(model.radius_for((0, 0, 50, 360)), 18.0)

    def test_falls_back_to_a_stated_ratio_when_no_player_was_nearby(self):
        model = calibrate(observations(14, person_height=None), default_ratio=0.11)

        self.assertAlmostEqual(model.player_ratio, 0.11)


class LocateTests(unittest.TestCase):
    def setUp(self):
        self.model = calibrate(observations(14))
        self.box = (100, 50, 180, 250)

    def test_finds_a_carried_ball_inside_the_person_box(self):
        found = self.model.locate(player_frame(with_ball=True), self.box)

        self.assertIsNotNone(found)
        x, y, _ = found
        self.assertAlmostEqual(x, 140, delta=12)
        self.assertAlmostEqual(y, 110, delta=12)

    def test_abstains_when_the_player_is_not_carrying_it(self):
        self.assertIsNone(self.model.locate(player_frame(with_ball=False), self.box))

    def test_a_ball_below_the_searched_region_is_not_claimed(self):
        frame = np.full((400, 300, 3), COURT, np.uint8)
        cv2.rectangle(frame, (100, 50), (180, 250), JERSEY, -1)
        cv2.circle(frame, (140, 235), 10, BALL, -1)     # at the player's feet

        self.assertIsNone(self.model.locate(frame, self.box))

    def test_a_degenerate_box_abstains_rather_than_raising(self):
        self.assertIsNone(self.model.locate(player_frame(with_ball=True), (10, 10, 11, 11)))


if __name__ == "__main__":
    unittest.main()
