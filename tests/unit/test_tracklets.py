import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from handball_cv.tracking.tracklets import (
    Tracklet,
    connect_tracklets,
    normalise,
    split_all,
    split_tracklet,
)


def make_tracklet(embeddings, start=0, tracklet_id=1, source=1):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    frames = np.arange(start, start + n)
    boxes = np.tile(np.array([0.0, 0.0, 10.0, 20.0]), (n, 1))
    return Tracklet(tracklet_id, source, frames, boxes, embeddings)


def identity_a(n, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return base + rng.normal(0, noise, (n, 4)).astype(np.float32)


def identity_b(n, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    base = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return base + rng.normal(0, noise, (n, 4)).astype(np.float32)


class SplitTests(unittest.TestCase):
    def test_tracklet_holding_two_identities_is_split_at_the_changepoint(self):
        # The P4 case: one tracker_id that ran on one player then another.
        tracklet = make_tracklet(np.vstack([identity_a(12), identity_b(12)]))
        pieces = split_tracklet(tracklet, next_id=99)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(len(pieces[0]) + len(pieces[1]), 24)
        # Cut lands on the true boundary, not somewhere arbitrary.
        self.assertEqual(pieces[0].end, 11)
        self.assertEqual(pieces[1].start, 12)
        self.assertEqual(pieces[1].tracklet_id, 99)

    def test_a_single_identity_is_never_split(self):
        pieces = split_tracklet(make_tracklet(identity_a(24)), next_id=99)
        self.assertEqual(len(pieces), 1)

    def test_pose_variation_interleaved_in_time_is_not_split(self):
        # Same guard, the case that matters: two visually distinct clusters
        # that alternate frame to frame are one player turning, not two people.
        a, b = identity_a(12), identity_b(12)
        interleaved = np.empty((24, 4), dtype=np.float32)
        interleaved[0::2], interleaved[1::2] = a, b
        pieces = split_tracklet(make_tracklet(interleaved), next_id=99)
        self.assertEqual(len(pieces), 1, "interleaved clusters must not split")

    def test_a_short_tail_of_another_identity_does_not_split(self):
        # Two frames of a neighbour is not enough evidence to cut an identity.
        tracklet = make_tracklet(np.vstack([identity_a(20), identity_b(2)]))
        self.assertEqual(len(split_tracklet(tracklet, next_id=99)), 1)

    def test_split_preserves_frames_boxes_and_provenance(self):
        tracklet = make_tracklet(np.vstack([identity_a(10), identity_b(10)]), start=100)
        pieces = split_tracklet(tracklet, next_id=7)
        rebuilt = np.concatenate([p.frames for p in pieces])
        np.testing.assert_array_equal(rebuilt, tracklet.frames)
        self.assertEqual(sum(len(p.boxes) for p in pieces), len(tracklet.boxes))
        self.assertIn(tracklet.tracklet_id, pieces[1].parts)

    def test_split_all_allocates_fresh_ids(self):
        tracklets = [
            make_tracklet(np.vstack([identity_a(10), identity_b(10)]), tracklet_id=1),
            make_tracklet(identity_a(12), tracklet_id=2, start=50),
        ]
        out = split_all(tracklets)
        self.assertEqual(len(out), 3)
        self.assertEqual(len({t.tracklet_id for t in out}), 3)


class ConnectTests(unittest.TestCase):
    def test_same_player_across_a_gap_is_connected(self):
        a = make_tracklet(identity_a(10, seed=0), start=0, tracklet_id=1)
        b = make_tracklet(identity_a(10, seed=5), start=40, tracklet_id=2)
        groups = connect_tracklets([a, b])
        self.assertEqual(groups[1], groups[2])

    def test_different_players_are_not_connected(self):
        a = make_tracklet(identity_a(10), start=0, tracklet_id=1)
        b = make_tracklet(identity_b(10), start=40, tracklet_id=2)
        groups = connect_tracklets([a, b])
        self.assertNotEqual(groups[1], groups[2])

    def test_temporally_overlapping_tracklets_are_never_connected(self):
        # Two identical-looking teammates on screen together are still two
        # people; co-visibility outranks appearance similarity.
        a = make_tracklet(identity_a(20, seed=0), start=0, tracklet_id=1)
        b = make_tracklet(identity_a(20, seed=5), start=10, tracklet_id=2)
        groups = connect_tracklets([a, b])
        self.assertNotEqual(groups[1], groups[2])

    def test_a_chain_of_fragments_lands_in_one_group(self):
        parts = [
            make_tracklet(identity_a(8, seed=s), start=s * 20, tracklet_id=s + 1)
            for s in range(3)
        ]
        groups = connect_tracklets(parts)
        self.assertEqual(len({groups[t.tracklet_id] for t in parts}), 1)


class HelperTests(unittest.TestCase):
    def test_normalise_produces_unit_rows_and_survives_empty(self):
        unit = normalise(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-6)
        self.assertEqual(len(normalise(np.zeros((0, 4), dtype=np.float32))), 0)

    def test_representative_is_a_unit_vector(self):
        tracklet = make_tracklet(identity_a(10))
        self.assertAlmostEqual(
            float(np.linalg.norm(tracklet.representative())), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
