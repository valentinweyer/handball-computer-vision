"""The translation between keypoint slots and court template vertices.

The failure this guards against is silent. Pair each detected landmark with the
wrong court vertex and ``cv2.findHomography`` still returns a matrix, with no
error and no warning -- it simply solves a contradictory system as best it can.
Downstream, a player's foot maps to a confident, wrong court position. That was
the state of the code before ``KEYPOINT_TO_VERTEX`` existed, and only a
measurement catches it.

So the cheap structural checks here are backed by two that would actually
notice: the export's own mirror table, which needs nothing on disk, and the
homography residual over the labelled images when they are present.
"""
import unittest
from pathlib import Path

import numpy as np
from sports import MeasurementUnit
from sports.handball import CourtConfiguration, League

from handball_cv.court.keypoints import (
    KEYPOINT_COUNT,
    KEYPOINT_FLIP_INDEX,
    KEYPOINT_TO_VERTEX,
    SELF_MIRRORING_SLOTS,
    court_points,
    vertex_indices,
)

DATASET = Path(__file__).resolve().parents[2] / "notebooks/keypointv333-1"


def court_vertices():
    config = CourtConfiguration(
        league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS
    )
    return np.array(config.vertices, dtype=np.float64)


class MappingStructureTests(unittest.TestCase):
    def test_is_a_permutation(self):
        self.assertEqual(len(KEYPOINT_TO_VERTEX), KEYPOINT_COUNT)
        self.assertEqual(sorted(KEYPOINT_TO_VERTEX), list(range(KEYPOINT_COUNT)))

    def test_matches_the_template_it_indexes(self):
        self.assertEqual(len(court_vertices()), KEYPOINT_COUNT)

    def test_centre_circle_lands_on_the_centre_of_the_court(self):
        # Slot 6 is the middle of the five-point cross on the centre circle --
        # the one landmark identifiable by eye, and the seed the recovery used.
        centre = court_points([6], court_vertices())[0]
        np.testing.assert_allclose(centre, [2000.0, 1000.0])

    def test_rejects_out_of_range_slots(self):
        with self.assertRaises(ValueError):
            vertex_indices([KEYPOINT_COUNT])
        with self.assertRaises(ValueError):
            vertex_indices([-1])

    def test_rejects_a_template_of_the_wrong_size(self):
        with self.assertRaises(ValueError):
            court_points([0], court_vertices()[:10])


class MirrorTableTests(unittest.TestCase):
    """Independent evidence: the export's ``flip_idx`` was never used to derive
    the mapping, so its agreement is not circular."""

    def setUp(self):
        self.vertices = court_vertices()
        self.mirrored = self.vertices.copy()
        self.mirrored[:, 0] = self.vertices[:, 0].max() - self.mirrored[:, 0]

    def agreeing_slots(self, mapping):
        return {
            s for s in range(KEYPOINT_COUNT)
            if np.allclose(
                self.vertices[mapping[KEYPOINT_FLIP_INDEX[s]]],
                self.mirrored[mapping[s]],
                atol=1.0,
            )
        }

    def test_agrees_except_on_the_two_known_centre_line_slots(self):
        agreeing = self.agreeing_slots(KEYPOINT_TO_VERTEX)
        self.assertEqual(
            sorted(set(range(KEYPOINT_COUNT)) - agreeing), sorted(SELF_MIRRORING_SLOTS)
        )

    def test_the_two_exceptions_are_their_own_mirror_image(self):
        # Both sit on the centre line, so mirroring the court left to right maps
        # each to itself; the export swaps them instead. Harmless, but it has to
        # stay the *explained* disagreement rather than a new one.
        for slot in SELF_MIRRORING_SLOTS:
            point = self.vertices[KEYPOINT_TO_VERTEX[slot]]
            np.testing.assert_allclose(point, self.mirrored[KEYPOINT_TO_VERTEX[slot]])

    def test_the_naive_identity_mapping_would_fail_this(self):
        # Guards the guard: a test that passes for any mapping proves nothing.
        naive = tuple(range(KEYPOINT_COUNT))
        self.assertLess(len(self.agreeing_slots(naive)), 10)


@unittest.skipUnless(
    any(DATASET.glob("*/labels/*.txt")),
    "labelled keypoint export not present (gitignored)",
)
class HomographyResidualTests(unittest.TestCase):
    """A planar court seen through a near-pinhole camera must fit a homography
    to within annotation noise. A wrong correspondence cannot."""

    EXPORT_WH = 640.0

    @classmethod
    def setUpClass(cls):
        cls.vertices = court_vertices()
        cls.samples = []
        for path in sorted(DATASET.glob("*/labels/*.txt"))[:120]:
            fields = path.read_text().split()
            if len(fields) < 5 + 3 * KEYPOINT_COUNT:
                continue
            label = np.array(
                fields[5 : 5 + 3 * KEYPOINT_COUNT], dtype=float
            ).reshape(KEYPOINT_COUNT, 3)
            slots = list(np.where(label[:, 2] > 0)[0])
            if len(slots) < 6:
                continue
            cls.samples.append((
                slots,
                np.stack(
                    [label[slots, 0] * cls.EXPORT_WH, label[slots, 1] * cls.EXPORT_WH],
                    axis=1,
                ).astype(np.float32),
            ))

    def median_residual(self, mapping):
        import cv2

        residuals = []
        for slots, src in self.samples:
            dst = self.vertices[[mapping[s] for s in slots]].astype(np.float32)
            matrix, _ = cv2.findHomography(src, dst)
            if matrix is None:
                continue
            projected = cv2.perspectiveTransform(
                src.reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
            residuals.append(np.median(np.linalg.norm(projected - dst, axis=1)))
        return float(np.median(residuals))

    def test_fits_to_within_annotation_noise(self):
        # Measured at 33 cm over the full export; 100 cm leaves room for the
        # subsample here without admitting a scrambled correspondence.
        self.assertLess(self.median_residual(KEYPOINT_TO_VERTEX), 100.0)

    def test_the_naive_identity_mapping_is_off_by_metres(self):
        naive = tuple(range(KEYPOINT_COUNT))
        self.assertGreater(self.median_residual(naive), 500.0)


if __name__ == "__main__":
    unittest.main()
