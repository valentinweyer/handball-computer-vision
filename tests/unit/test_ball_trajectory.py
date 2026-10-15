import unittest

import numpy as np

from handball_cv.ball.tracking import (
    BallState,
    BallTrack,
    TrackPoint,
    chain_tracks,
    project_points,
    resolve_possession,
)


def point(frame: int, x: float, y: float) -> TrackPoint:
    return TrackPoint(frame=frame, x=x, y=y)


def still(count: int):
    """A locked-off camera: every frame maps onto the next unchanged."""
    return [np.eye(3)] * count


def panning(count: int, dx: float, dy: float = 0.0):
    """A camera panning steadily, as a per-frame homography."""
    return [np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)] * count


def jitter(frames: int, count: int, *, amplitude=2.0, seed=0):
    """Crowd: blobs that stay put and wobble, one per centre per frame.

    The centres are scattered rather than gridded on purpose. Evenly spaced
    centres are themselves a constant-velocity lattice -- a chain can stride
    from one to the next forever and look exactly like a ball -- which says
    nothing about real footage and everything about the fixture.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform([0, 0], [1900, 1000], size=(count, 2))
    return [
        [
            point(frame, cx + rng.uniform(-amplitude, amplitude),
                  cy + rng.uniform(-amplitude, amplitude))
            for cx, cy in centres
        ]
        for frame in range(frames)
    ]


class ChainTracksTests(unittest.TestCase):
    def test_recovers_a_ballistic_path_buried_in_crowd_noise(self):
        per_frame = jitter(12, 20, seed=7)
        expected = []
        x, y, vx, vy = 100.0, 500.0, 22.0, -18.0
        for frame in range(12):
            per_frame[frame].append(point(frame, x, y))
            expected.append((x, y))
            x, y = x + vx, y + vy
            vy += 2.4                       # gravity, so the path curves

        tracks = chain_tracks(per_frame, still(12), min_length=5)

        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose([(p.x, p.y) for p in tracks[0].points], expected)

    def test_a_jiggling_crowd_alone_produces_no_track(self):
        self.assertEqual(
            chain_tracks(jitter(14, 24, seed=3), still(14), min_length=5), ()
        )

    def test_speed_gate_applies_to_every_step_not_just_the_first(self):
        # Two ball-speed steps, then the path collapses into a near-stationary
        # cluster. A generous tolerance lets the cluster sit inside the predicted
        # window, so only the per-step speed check can reject it.
        slowing = [0.0, 20.0, 40.0, 43.0, 46.0, 49.0, 52.0, 55.0]
        per_frame = [[point(frame, x, 300.0)] for frame, x in enumerate(slowing)]

        tracks = chain_tracks(
            per_frame, still(8), min_speed=9.0, max_speed=70.0,
            tolerance=30.0, min_length=5,
        )

        self.assertEqual(tracks, ())

    def test_the_same_path_at_ball_speed_throughout_is_kept(self):
        per_frame = [[point(frame, 20.0 * frame, 300.0)] for frame in range(8)]

        tracks = chain_tracks(
            per_frame, still(8), min_speed=9.0, max_speed=70.0,
            tolerance=30.0, min_length=5,
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].points), 8)

    def test_a_frame_without_evidence_ends_the_track_rather_than_bridging_it(self):
        per_frame = [
            [] if frame == 5 else [point(frame, 20.0 * frame, 300.0)]
            for frame in range(11)
        ]

        tracks = chain_tracks(per_frame, still(11), min_length=5)

        self.assertTrue(tracks)
        for track in tracks:
            frames = [p.frame for p in track.points]
            self.assertNotIn(5, frames)
            self.assertFalse(
                4 in frames and 6 in frames,
                "a gap was bridged: the track spans the frame with no evidence",
            )

    def test_unknown_camera_motion_stops_a_trajectory_crossing_it(self):
        per_frame = [[point(frame, 20.0 * frame, 300.0)] for frame in range(11)]
        motions = still(11)
        motions[5] = None                   # frames 5 and 6 cannot be related

        for track in chain_tracks(per_frame, motions, min_length=5):
            frames = [p.frame for p in track.points]
            self.assertFalse(5 in frames and 6 in frames)

    def test_a_burned_in_graphic_is_rejected_however_ballistic_it_looks(self):
        # The score bug never goes anywhere in the picture, but compensating the
        # pan sends it gliding across the compensated frame in a dead straight
        # line. On the first full-clip run this was the longest trajectory
        # found, at 37 frames -- longer and straighter than any real ball. Its
        # centroid wobbles as the residual reshapes, so the wobble is included
        # here: a per-step threshold does not separate it, only the excursion.
        rng = np.random.default_rng(11)
        overlay = [
            [point(frame, 1700.0 + rng.uniform(-3, 3), 60.0 + rng.uniform(-3, 3))]
            for frame in range(10)
        ]

        self.assertEqual(chain_tracks(overlay, panning(10, -16.0), min_length=5), ())

    def test_a_ball_that_also_moves_in_the_picture_survives_that_check(self):
        moving = [[point(frame, 200.0 + 6.0 * frame, 400.0)] for frame in range(10)]

        tracks = chain_tracks(moving, panning(10, -10.0), min_length=5)

        self.assertEqual(len(tracks), 1)
        # 6 px of image motion against a 10 px pan is a 16 px compensated step.
        self.assertAlmostEqual(tracks[0].mean_speed, 16.0)

    def test_a_ball_cannot_cross_the_picture_between_two_frames(self):
        # A wrong homography can make a huge image jump look like a modest
        # compensated step. The image ceiling is what refuses it regardless.
        leaping = [[point(frame, 200.0 + 700.0 * frame, 300.0)] for frame in range(8)]
        forgiving = panning(8, -700.0 + 16.0)

        self.assertEqual(
            chain_tracks(leaping, forgiving, min_length=5, max_image_speed=120.0), (),
        )

    def test_compensation_is_pairwise_so_a_long_pan_does_not_drift(self):
        # Composing 200 homographies into one reference frame is what let
        # 698 px image steps pass a 70 px compensated gate on the real clip.
        held = [[point(frame, 900.0, 500.0)] for frame in range(200)]

        self.assertEqual(chain_tracks(held, panning(200, -12.0), min_length=5), ())


class ProjectPointsTests(unittest.TestCase):
    def test_applies_a_translation(self):
        matrix = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1]], float)

        np.testing.assert_allclose(project_points([(1, 1)], matrix), [[6, -2]])

    def test_divides_through_by_the_homogeneous_coordinate(self):
        matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], float)

        np.testing.assert_allclose(project_points([(4, 6)], matrix), [[4, 6]])

    def test_empty_input_stays_empty(self):
        self.assertEqual(len(project_points([], np.eye(3))), 0)


class ResolvePossessionTests(unittest.TestCase):
    @staticmethod
    def arriving_track(last_frame=3):
        return BallTrack(points=tuple(
            point(frame, 100.0 + 10 * frame, 200.0)
            for frame in range(last_frame + 1)
        ))

    def test_flight_frames_come_from_the_track(self):
        frames = resolve_possession([self.arriving_track()], [[] for _ in range(6)])

        for frame in range(4):
            self.assertIs(frames[frame].state, BallState.FLIGHT)
        self.assertEqual((frames[3].x, frames[3].y), (130.0, 200.0))

    def test_a_track_ending_beside_a_player_hands_them_the_ball(self):
        catcher = [120.0, 170.0, 160.0, 350.0]
        boxes = [[] for _ in range(4)] + [[catcher] for _ in range(4)]

        frames = resolve_possession([self.arriving_track()], boxes)

        for frame in range(4, 8):
            self.assertIs(frames[frame].state, BallState.HELD)
            self.assertEqual(frames[frame].holder, 0)
        # Upper-middle of the box, which is a stated fallback, not a measurement.
        self.assertEqual((frames[4].x, frames[4].y), (140.0, 233.0))

    def test_no_player_within_reach_abstains_rather_than_guessing(self):
        far_away = [1400.0, 900.0, 1480.0, 1050.0]
        boxes = [[] for _ in range(4)] + [[far_away] for _ in range(4)]

        frames = resolve_possession([self.arriving_track()], boxes)

        for frame in range(4, 8):
            self.assertIs(frames[frame].state, BallState.ABSENT)
            self.assertIsNone(frames[frame].holder)

    def test_possession_expires_when_nothing_ever_confirms_it(self):
        catcher = [120.0, 170.0, 160.0, 350.0]
        boxes = [[] for _ in range(4)] + [[catcher] for _ in range(20)]

        frames = resolve_possession([self.arriving_track()], boxes, max_unconfirmed=5)

        self.assertIs(frames[8].state, BallState.HELD)
        self.assertIs(frames[12].state, BallState.ABSENT)

    def test_an_appearance_confirmation_places_the_ball_and_renews_possession(self):
        catcher = [120.0, 170.0, 160.0, 350.0]
        boxes = [[] for _ in range(4)] + [[catcher] for _ in range(20)]

        frames = resolve_possession(
            [self.arriving_track()], boxes, max_unconfirmed=5,
            confirmations={frame: (133.0, 210.0) for frame in range(4, 20)},
        )

        self.assertIs(frames[18].state, BallState.HELD)
        self.assertEqual((frames[18].x, frames[18].y), (133.0, 210.0))


if __name__ == "__main__":
    unittest.main()
