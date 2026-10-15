"""The gate is the whole reason this tracker is not the ungated one.

Measured on the BHC-FAG test clip, anchoring and propagating with no gate covers
96% of frames and roughly 40% of what it reports is a player's head or a
goalkeeper's jersey. These pin the rules that throw that away, and -- just as
importantly -- the rules that keep a real ball through the frames where nothing
independent can see it.
"""

import unittest

from handball_cv.ball.anchored import (
    AnchorConfig,
    Propagated,
    accepted_prefix,
    anchor_candidates,
    build_tracks,
    classify_frames,
)
from handball_cv.ball.tracking import BallState


BALL_AREA = 400.0


def run(n, area=BALL_AREA, start=0):
    return [Propagated(frame=start + i, x=100.0 + 10 * i, y=50.0, area=area)
            for i in range(n)]


class AcceptedPrefixTest(unittest.TestCase):
    config = AnchorConfig(max_unconfirmed=5, min_area=150.0, max_area=600.0)

    def test_confirmed_throughout_is_kept_whole(self):
        observations = run(10)
        self.assertEqual(
            accepted_prefix(observations, [True] * 10, self.config), 10
        )

    def test_gap_shorter_than_lapse_is_bridged(self):
        """A real ball outruns the detector for a few frames mid-flight.

        The frames are kept because a later confirmation vouches for them --
        the track came back to something independent, so it never left it.
        """
        confirmed = [True, True, False, False, False, True, True]
        self.assertEqual(accepted_prefix(run(7), confirmed, self.config), 7)

    def test_unconfirmed_tail_is_trimmed_not_kept(self):
        """Nothing ever agreed with the tail, so nothing vouches for it."""
        confirmed = [True, True, True, False, False]
        self.assertEqual(accepted_prefix(run(5), confirmed, self.config), 3)

    def test_track_lapses_after_max_unconfirmed(self):
        confirmed = [True] + [False] * 20
        self.assertEqual(accepted_prefix(run(21), confirmed, self.config), 1)

    def test_drift_onto_a_torso_ends_the_track_at_the_size_change(self):
        """The goalkeeper-jersey failure: the mask swallows a torso.

        Confirmed throughout -- a detection happened to sit nearby -- so only
        the area rule can catch this one.
        """
        observations = run(3) + [
            Propagated(frame=3, x=140.0, y=50.0, area=860.0),
            Propagated(frame=4, x=150.0, y=50.0, area=1310.0),
        ]
        self.assertEqual(accepted_prefix(observations, [True] * 5, self.config), 3)

    def test_area_break_still_trims_to_last_confirmation(self):
        confirmed = [True, False, False, True]
        observations = run(3) + [Propagated(frame=3, x=1.0, y=1.0, area=9000.0)]
        self.assertEqual(accepted_prefix(observations, confirmed, self.config), 1)

    def test_never_confirmed_is_rejected_entirely(self):
        self.assertEqual(accepted_prefix(run(3), [False] * 3, self.config), 0)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            accepted_prefix(run(3), [True, True], self.config)


class AnchorCandidatesTest(unittest.TestCase):
    def test_only_the_best_box_on_a_frame_can_seed(self):
        """Handball has one ball; two seeds on one frame would compete."""
        detections = {0: [((0, 0, 10, 10), 0.3), ((50, 50, 60, 60), 0.9)]}
        self.assertEqual(
            anchor_candidates(detections, threshold=0.25),
            [(0, (50.0, 50.0, 60.0, 60.0), 0.9)],
        )

    def test_frames_below_threshold_cannot_seed(self):
        detections = {0: [((0, 0, 10, 10), 0.2)], 1: [((0, 0, 10, 10), 0.4)]}
        self.assertEqual(
            [frame for frame, _b, _s in anchor_candidates(detections, threshold=0.25)],
            [1],
        )


class BuildTracksTest(unittest.TestCase):
    """`propagate` is injected, so the lifecycle is testable without SAM2."""

    def setUp(self):
        self.config = AnchorConfig(anchor_threshold=0.25, max_unconfirmed=5,
                                   min_area=150.0, max_area=600.0,
                                   support_radius=35.0)

    def test_an_accepted_track_suppresses_later_anchors_inside_it(self):
        detections = {f: [((100.0 + 10 * f, 45.0, 110.0 + 10 * f, 55.0), 0.9)]
                      for f in range(6)}

        calls = []

        def propagate(frame, box):
            calls.append(frame)
            return run(6 - frame, start=frame)

        tracks = build_tracks(detections, propagate, config=self.config, n_frames=6)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(calls, [0], "one ball should not be anchored six times")
        self.assertEqual((tracks[0].first_frame, tracks[0].last_frame), (0, 5))

    def test_a_rejected_propagation_yields_no_track(self):
        """An anchor whose propagation is never confirmed leaves nothing."""
        detections = {0: [((1000.0, 1000.0, 1010.0, 1010.0), 0.9)]}

        def propagate(frame, box):
            # far from the anchor box, so nothing confirms it
            return [Propagated(frame=f, x=10.0, y=10.0, area=BALL_AREA)
                    for f in range(10)]

        tracks = build_tracks(detections, propagate, config=self.config, n_frames=10)
        self.assertEqual(tracks, [])

    def test_appearance_can_confirm_where_the_detector_is_silent(self):
        """The held-ball case: the detector cannot see it, colour can."""
        detections = {0: [((95.0, 45.0, 105.0, 55.0), 0.9)]}

        def propagate(frame, box):
            return run(8)

        without = build_tracks(detections, propagate, config=self.config, n_frames=8)
        with_appearance = build_tracks(
            detections, propagate, config=self.config, n_frames=8,
            resembles=lambda frame, x, y: True,
        )
        self.assertEqual(len(without[0].points), 1)
        self.assertEqual(len(with_appearance[0].points), 8)

    def test_propagation_past_the_clip_end_is_dropped(self):
        detections = {0: [((95.0, 45.0, 105.0, 55.0), 0.9)]}
        tracks = build_tracks(
            detections, lambda f, b: run(50), config=self.config, n_frames=4,
            resembles=lambda frame, x, y: True,
        )
        self.assertEqual(tracks[0].last_frame, 3)


class ClassifyFramesTest(unittest.TestCase):
    def test_untracked_frames_are_absent_not_interpolated(self):
        tracks = build_tracks(
            {0: [((95.0, 45.0, 105.0, 55.0), 0.9)]},
            lambda f, b: run(2),
            config=AnchorConfig(max_unconfirmed=5),
            n_frames=2,
        )
        frames = classify_frames(tracks, [[] for _ in range(5)])
        self.assertEqual([f.state for f in frames[2:]], [BallState.ABSENT] * 3)
        self.assertTrue(all(f.x is None for f in frames[2:]))

    def test_a_ball_inside_a_person_box_is_held_by_that_person(self):
        tracks = build_tracks(
            {0: [((95.0, 45.0, 105.0, 55.0), 0.9)]},
            lambda f, b: [Propagated(frame=0, x=100.0, y=50.0, area=BALL_AREA)],
            config=AnchorConfig(max_unconfirmed=5),
            n_frames=1,
        )
        frames = classify_frames(tracks, [[(0.0, 0.0, 500.0, 500.0), (80.0, 20.0, 130.0, 200.0)]])
        self.assertEqual(frames[0].state, BallState.HELD)
        self.assertEqual(frames[0].holder, 1, "the tighter box is the nearer player")

    def test_a_ball_clear_of_every_player_is_flight(self):
        tracks = build_tracks(
            {0: [((95.0, 45.0, 105.0, 55.0), 0.9)]},
            lambda f, b: [Propagated(frame=0, x=100.0, y=50.0, area=BALL_AREA)],
            config=AnchorConfig(max_unconfirmed=5),
            n_frames=1,
        )
        frames = classify_frames(tracks, [[(400.0, 400.0, 450.0, 600.0)]])
        self.assertEqual(frames[0].state, BallState.FLIGHT)
        self.assertIsNone(frames[0].holder)


if __name__ == "__main__":
    unittest.main()
