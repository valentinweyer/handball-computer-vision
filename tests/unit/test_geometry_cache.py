"""The cache that lets a finished render be drawn again without tracking again.

Exactness is the whole requirement: a redraw has to reproduce the original
overlay pixel for pixel wherever the labels did not change, so anything lossy
here shows up as a silhouette that moved.
"""
import unittest

import numpy as np

from handball_cv.tracking.geometry_cache import GeometryCache


def mask(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class GeometryCacheTests(unittest.TestCase):
    def setUp(self):
        self.shape = (48, 64)
        self.tmp = __import__("tempfile").mkdtemp()
        self.path = __import__("pathlib").Path(self.tmp) / "geometry.npz"

    def roundtrip(self, cache):
        return GeometryCache.load(cache.save(self.path))

    def test_masks_survive_exactly_including_where_players_overlap(self):
        """MaskCache's label map cannot do this, which is why this cache exists.

        Two handball players in a duel share pixels. Flattened to one label map
        the second one written wins them outright, and the first player's mask
        comes back with a bite taken out of it.
        """
        a = mask(self.shape, 4, 30, 8, 28)
        b = mask(self.shape, 20, 44, 18, 50)
        self.assertTrue((a & b).any())

        cache = GeometryCache(64, 48, 10)
        cache.add(3, [1, 2], [[8, 4, 28, 30], [18, 20, 50, 44]], [a, b], [0, 1], [0, 1])
        got = self.roundtrip(cache).frame(3)

        np.testing.assert_array_equal(got.masks[0], a)
        np.testing.assert_array_equal(got.masks[1], b)

    def test_row_order_and_identity_columns_are_preserved(self):
        cache = GeometryCache(64, 48, 10)
        cache.add(
            1, [17, 4], [[0, 0, 10, 10], [5, 5, 20, 20]],
            [mask(self.shape, 0, 10, 0, 10), mask(self.shape, 5, 20, 5, 20)],
            [4, 2], [-1, 1],
        )
        got = self.roundtrip(cache).frame(1)
        np.testing.assert_array_equal(got.player_ids, [17, 4])
        np.testing.assert_array_equal(got.color_index, [4, 2])
        np.testing.assert_array_equal(got.team_id, [-1, 1])
        np.testing.assert_allclose(got.boxes[1], [5, 5, 20, 20])

    def test_an_empty_mask_survives_as_an_empty_mask(self):
        """A track can hold a box while its mask has collapsed to nothing."""
        cache = GeometryCache(64, 48, 10)
        cache.add(2, [9], [[0, 0, 1, 1]], [np.zeros(self.shape, bool)], [0], [0])
        got = self.roundtrip(cache).frame(2)
        self.assertEqual(got.masks.shape, (1, 48, 64))
        self.assertFalse(got.masks[0].any())

    def test_a_visited_frame_with_no_players_is_still_a_frame(self):
        """A redraw that skipped it would come out short and shift the rest."""
        cache = GeometryCache(64, 48, 10)
        cache.add(1, [1], [[0, 0, 5, 5]], [mask(self.shape, 0, 5, 0, 5)], [0], [0])
        cache.add(2, [], np.zeros((0, 4)), [], [], [])
        cache.add(3, [2], [[1, 1, 6, 6]], [mask(self.shape, 1, 6, 1, 6)], [1], [1])
        loaded = self.roundtrip(cache)
        self.assertEqual(loaded.frame_indices, [1, 2, 3])
        self.assertEqual(len(loaded.frame(2).player_ids), 0)

    def test_frames_outside_the_recorded_range_read_as_empty(self):
        cache = GeometryCache(64, 48, 10)
        cache.add(5, [1], [[0, 0, 5, 5]], [mask(self.shape, 0, 5, 0, 5)], [0], [0])
        loaded = self.roundtrip(cache)
        self.assertEqual(len(loaded.frame(0).player_ids), 0)
        self.assertEqual(len(loaded.frame(9).player_ids), 0)
        self.assertEqual(len(loaded.frame(99).player_ids), 0)

    def test_a_schema_change_refuses_rather_than_drawing_garbage(self):
        cache = GeometryCache(64, 48, 10)
        cache.add(1, [1], [[0, 0, 5, 5]], [mask(self.shape, 0, 5, 0, 5)], [0], [0])
        cache.save(self.path)
        with np.load(self.path) as data:
            store = {k: data[k] for k in data.files}
        store["schema"] = np.asarray(999)
        np.savez_compressed(self.path, **store)
        with self.assertRaisesRegex(ValueError, "schema"):
            GeometryCache.load(self.path)


if __name__ == "__main__":
    unittest.main()
