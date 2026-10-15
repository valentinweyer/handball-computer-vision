import unittest

import cv2
import numpy as np

from handball_cv.ball.candidates import (
    ball_candidates, believable_motion, overlay_mask, play_volume_mask,
)

IDENTITY = np.eye(3)
COURT_BGR = (230, 159, 122)     # sampled from Melsungen frame 710


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)


def frame_with_disc(x: int, y: int, *, radius=8, background=100, value=220):
    canvas = np.full((200, 300), background, np.uint8)
    cv2.circle(canvas, (x, y), radius, value, -1)
    return canvas


class ThreeFrameDifferenceTests(unittest.TestCase):
    def test_finds_a_moving_disc_where_it_is_now(self):
        previous, current, following = (frame_with_disc(x, 100) for x in (50, 80, 110))

        frame = ball_candidates(current, previous, following, IDENTITY, IDENTITY)

        self.assertEqual(len(frame.candidates), 1)
        self.assertAlmostEqual(frame.candidates[0].x, 80, delta=2)
        self.assertAlmostEqual(frame.candidates[0].y, 100, delta=2)

    def test_the_ghost_at_the_old_position_is_suppressed(self):
        previous, current, following = (frame_with_disc(x, 100) for x in (50, 80, 110))

        three = ball_candidates(current, previous, following, IDENTITY, IDENTITY)
        # Differencing against the same neighbour twice is a two-frame
        # difference wearing a three-frame signature: it cannot tell the disc's
        # old position from its new one, and reports both.
        two = ball_candidates(current, previous, previous, IDENTITY, IDENTITY)

        self.assertEqual(len(three.candidates), 1)
        self.assertEqual(len(two.candidates), 2)
        self.assertAlmostEqual(min(c.x for c in two.candidates), 50, delta=2)


class RejectionTests(unittest.TestCase):
    @staticmethod
    def static_logo():
        """A bright mark painted on the floor, with the camera motion 4px wrong."""
        canvas = np.full((200, 300), 40, np.uint8)
        cv2.circle(canvas, (150, 100), 7, 230, -1)
        return canvas, translation(4, 0)

    def test_a_static_mark_under_imperfect_compensation_is_rejected(self):
        canvas, motion = self.static_logo()

        frame = ball_candidates(canvas, canvas, canvas, motion, motion)

        self.assertEqual(frame.candidates, ())
        self.assertGreaterEqual(frame.rejected["misregistered"], 1)

    def test_and_it_is_the_registration_check_doing_the_rejecting(self):
        canvas, motion = self.static_logo()

        frame = ball_candidates(
            canvas, canvas, canvas, motion, motion, registration_match=2.0,
        )

        self.assertTrue(frame.candidates)

    def test_a_blob_on_a_person_is_not_a_free_ball(self):
        previous, current, following = (frame_with_disc(x, 100) for x in (50, 80, 110))

        frame = ball_candidates(
            current, previous, following, IDENTITY, IDENTITY,
            person_boxes=[(60, 70, 110, 160)],
        )

        self.assertEqual(frame.candidates, ())
        self.assertGreaterEqual(frame.rejected["person"], 1)

    def test_a_blob_outside_the_play_volume_is_counted_not_silently_dropped(self):
        previous, current, following = (frame_with_disc(x, 100) for x in (50, 80, 110))
        volume = np.zeros((200, 300), np.uint8)

        frame = ball_candidates(
            current, previous, following, IDENTITY, IDENTITY, play_volume=volume,
        )

        self.assertEqual(frame.candidates, ())
        self.assertEqual(frame.rejected["outside_volume"], 1)

    def test_something_far_too_big_to_be_a_ball_is_rejected_on_size(self):
        previous = np.full((200, 300), 100, np.uint8)
        current = previous.copy()
        cv2.rectangle(current, (40, 30), (240, 170), 220, -1)

        frame = ball_candidates(current, previous, previous, IDENTITY, IDENTITY)

        self.assertEqual(frame.candidates, ())
        self.assertGreaterEqual(frame.rejected["area"] + frame.rejected["shape"], 1)


class OverlayMaskTests(unittest.TestCase):
    @staticmethod
    def sampled(*, moving_background: bool):
        """Frames with a fixed score bug over a background that may or may not move."""
        frames = []
        rng = np.random.default_rng(5)
        for index in range(8):
            canvas = (rng.integers(0, 255, (200, 300), dtype=np.uint8)
                      if moving_background
                      else np.full((200, 300), 90, np.uint8))
            cv2.rectangle(canvas, (200, 10), (280, 40), 240, -1)   # the score bug
            frames.append(canvas)
        return frames

    def test_a_fixed_graphic_over_a_changing_scene_is_masked(self):
        mask = overlay_mask(self.sampled(moving_background=True))

        self.assertIsNotNone(mask)
        self.assertTrue(mask[25, 240], "the score bug was not masked")

    def test_the_moving_scene_itself_is_not_masked(self):
        mask = overlay_mask(self.sampled(moving_background=True))

        self.assertFalse(mask[150, 60])

    def test_abstains_when_nothing_in_the_sample_moved(self):
        # A locked-off sample cannot distinguish an overlay from a wall, and
        # masking the whole picture would be worse than masking none of it.
        self.assertIsNone(overlay_mask(self.sampled(moving_background=False)))

    def test_abstains_on_too_few_frames_to_judge(self):
        self.assertIsNone(overlay_mask(self.sampled(moving_background=True)[:2]))


class ExclusionByOverlayTests(unittest.TestCase):
    def test_a_candidate_inside_the_overlay_is_counted_not_silently_dropped(self):
        previous, current, following = (frame_with_disc(x, 100) for x in (50, 80, 110))
        overlay = np.zeros((200, 300), np.uint8)
        overlay[60:140, 40:140] = 255

        frame = ball_candidates(
            current, previous, following, IDENTITY, IDENTITY, overlay=overlay,
        )

        self.assertEqual(frame.candidates, ())
        self.assertEqual(frame.rejected["overlay"], 1)


class BelievableMotionTests(unittest.TestCase):
    SHAPE = (1080, 1920)

    def test_a_locked_off_camera_is_believable(self):
        self.assertTrue(believable_motion(IDENTITY, self.SHAPE))

    def test_a_hard_pan_is_still_believable(self):
        self.assertTrue(believable_motion(translation(60, -12), self.SHAPE))

    def test_a_transform_that_teleports_the_picture_is_not(self):
        self.assertFalse(believable_motion(translation(900, 0), self.SHAPE))

    def test_a_collapsing_transform_is_not(self):
        shrink = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]], float)

        self.assertFalse(believable_motion(shrink, self.SHAPE))

    def test_no_estimate_at_all_is_not_believable(self):
        self.assertFalse(believable_motion(None, self.SHAPE))


class PlayVolumeTests(unittest.TestCase):
    @staticmethod
    def court_band():
        """Court across the middle of the picture, stands above, bench below."""
        canvas = np.full((400, 600, 3), 30, np.uint8)
        canvas[200:300, :] = COURT_BGR
        return canvas

    def test_the_court_itself_is_inside_the_volume(self):
        mask = play_volume_mask(self.court_band(), rise_px=50, spread_px=0, fill_px=5)

        self.assertTrue(mask[250, 300])

    def test_the_volume_reaches_above_the_court_for_a_ball_in_flight(self):
        mask = play_volume_mask(self.court_band(), rise_px=50, spread_px=0, fill_px=5)

        self.assertTrue(mask[170, 300], "a thrown ball above the court was excluded")

    def test_the_volume_does_not_reach_down_to_the_bench(self):
        mask = play_volume_mask(self.court_band(), rise_px=50, spread_px=0, fill_px=5)

        self.assertFalse(mask[380, 300])


if __name__ == "__main__":
    unittest.main()
