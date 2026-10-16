"""Translate the court-keypoint model's output order into court template vertices.

The keypoint model (an RF-DETR keypoint model: our fine-tune or Roboflow's
``keypointv333-uwois-xprdi`` version 4; its retired version 3 was a YOLO11s-pose
network, with the same slot order) emits 37 landmarks in the slot order its training export happened to use. The ``sports``
library's :class:`~sports.handball.CourtConfiguration` also lists 37 vertices,
in a *different* order. Nothing previously translated between the two:
``scripts/run_court_mapping.py`` indexed ``config.vertices`` with the model's
own slot index, pairing each detected landmark with an unrelated court
position. Every homography built that way was fitted to a scrambled
correspondence.

``KEYPOINT_TO_VERTEX`` is that missing translation. It was recovered from the
892 labelled images of the training export (see
``scripts/recover_court_keypoint_mapping.py``) and is checked three ways:

- **Homography fit residual** over those images falls from 922 cm to 33 cm
  (p90 1420 -> 51 cm; 88% of images now under 50 cm). A handball court is
  planar and a broadcast camera is very nearly a pinhole, so a correct
  correspondence must fit to within annotation noise. ~33 cm on a 40 m court is
  that noise: the export is 640x640, so one click pixel is already ~6 cm near
  the centre line and considerably more at the far end.
- **Leave-one-out** error over the same images falls from 1702 cm to 48 cm,
  confirming the fit is not merely absorbing the error into spare parameters.
- **The export's own ``flip_idx``** (its left/right mirror table, which the
  recovery never consults) goes from agreeing on 4 of 37 slots to 35 of 37.
  The two exceptions are ``SELF_MIRRORING_SLOTS`` below.

The mapping is keyed by *slot index*, which a prediction reports as
``class_id`` -- **not** ``class_name``. Those are two different numberings:
``class_id`` is the slot, while ``class_name`` is the label the annotator typed
into Roboflow. Matching version 3's predictions against the labelled export,
``class_id`` is the slot on 28 of 31 landmarks (the three exceptions are
single-observation nearest-neighbour confusions between adjacent centre-circle
points), whereas ``int(class_name) - 1`` is the slot on only 6 of 31.

So the original defect had two halves, and fixing either alone leaves a wrong
homography: ``run_court_mapping.py`` read ``class_name``, then indexed
``config.vertices`` with it as though slot order were vertex order. End to end
on one Melsungen frame, the old path found 4 RANSAC inliers of 11 confident
keypoints -- the minimum a homography needs, so nothing actually agreed --
against 8 once both halves are corrected, with the reprojected 6 m and 9 m lines
landing on the painted arcs.

Orientation
-----------
A handball court is unchanged by a half turn, so neither the residual nor the
mirror table can tell a mapping from the same mapping turned through 180
degrees: both fit every image equally well. The first shipped mapping was the
turned one, and fitted perfectly while numbering every landmark from the wrong
end -- KP01 "TL-Corner" at the bottom-right of the frame.

The annotation template settles it. It is drawn like ``draw_court`` and like
the broadcast camera: far sideline (court y=2000) on top, KP01 top-left,
KP01-KP13 on the left goal. :func:`upright` measures that orientation from the
labels themselves; over the 892 labelled images the shipped mapping is upright
on 887 and the turned one on 0. The other 5 are mirrored under either mapping
-- annotation slips, not a third orientation.

Both versions share this slot order: Roboflow's ``keypoints_metadata`` gives
the annotators' KP name for every slot, and it matches this mapping on all 37
(pinned in ``tests/unit/test_court_rfdetr_keypoints.py``).

Version 4 is served as ``rfdetr-keypoint-preview``, which the pinned
``inference`` 0.62.0 has no implementation class for, so ``get_model`` raises
``KeyError``; it runs from its ONNX export through
:mod:`handball_cv.court.rfdetr_keypoints`. Roboflow's mAP (99.5 for version 3,
97.0 for version 4) favoured version 3, but on the export's held-out test images
version 4 is the better model, decisively on middle-court views -- see that
module for the numbers.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "KEYPOINT_COUNT",
    "KEYPOINT_TO_VERTEX",
    "SELF_MIRRORING_SLOTS",
    "KEYPOINT_FLIP_INDEX",
    "vertex_indices",
    "court_points",
    "court_vertices",
    "upright",
]

KEYPOINT_COUNT = 37

#: ``KEYPOINT_TO_VERTEX[slot]`` is the index into ``CourtConfiguration.vertices``
#: that the model's 0-based keypoint ``slot`` actually denotes. A permutation,
#: oriented so the labels are :func:`upright` (see "Orientation" above).
KEYPOINT_TO_VERTEX: tuple[int, ...] = (
     1,  4,  7,  0,  5, 13,
    16, 19, 32, 31, 28, 27,
    30, 29,  2,  3, 15, 17,
    18, 14,  6,  9, 26, 23,
    24, 11, 12, 10, 20, 21,
    22,  8, 25, 33, 34, 35,
    36,
)

#: The export's left/right mirror table, verbatim from its ``data.yaml``. Kept
#: as verification data: it is independent evidence for the mapping above, and
#: the test that uses it needs no dataset on disk.
KEYPOINT_FLIP_INDEX: tuple[int, ...] = (
    10, 9, 24, 11, 8, 5, 6, 7, 4, 1, 0, 3, 15, 14, 13, 12, 17, 16, 19,
    18, 23, 22, 21, 20, 2, 28, 29, 30, 25, 26, 27, 32, 31, 35, 36, 33, 34,
)

#: The two slots where ``KEYPOINT_FLIP_INDEX`` disagrees with a left/right
#: mirror of the recovered mapping. Both land on the centre line, at the two
#: goalpost offsets -- (2000, 1150) and (2000, 850) in centimetres. Mirroring
#: the court left to right maps each to *itself*; the export swaps them, which
#: is a vertical rather than horizontal flip. The disagreement is a quirk of
#: how the export was annotated and constrains nothing about this mapping.
SELF_MIRRORING_SLOTS: tuple[int, int] = (18, 19)


def vertex_indices(slots: Iterable[int]) -> np.ndarray:
    """Map model keypoint slots to ``CourtConfiguration.vertices`` indices."""
    out = np.asarray(list(slots), dtype=int)
    if out.size and (out.min() < 0 or out.max() >= KEYPOINT_COUNT):
        raise ValueError(
            f"keypoint slots must lie in [0, {KEYPOINT_COUNT}); got "
            f"{out.min()}..{out.max()}"
        )
    return np.asarray(KEYPOINT_TO_VERTEX, dtype=int)[out]


def court_vertices() -> np.ndarray:
    """The IHF court's 37 template vertices in centimetres, ``(37, 2)``.

    The coordinate system every court position in this project is reported in:
    x along the 4000 cm length, y across the 2000 cm width, origin at a corner.
    """
    from sports import MeasurementUnit
    from sports.handball import CourtConfiguration, League

    config = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)
    return np.asarray(config.vertices, dtype=np.float64)


def court_points(slots: Iterable[int], vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Court coordinates for the given model keypoint slots.

    Args:
        slots: 0-based keypoint slot indices as emitted by the keypoint model.
        vertices: ``CourtConfiguration.vertices``, in the unit the caller wants
            back (the configuration's ``measurement_unit`` decides that).

    Returns:
        ``(len(slots), 2)`` array of court coordinates, ready to be the target
        of a homography whose source is the matching image points.
    """
    table = np.asarray(vertices, dtype=np.float64)
    if table.shape[0] != KEYPOINT_COUNT:
        raise ValueError(
            f"expected {KEYPOINT_COUNT} court vertices, got {table.shape[0]}"
        )
    return table[vertex_indices(slots)]


def upright(image_xy: np.ndarray, court_xy: np.ndarray) -> bool:
    """True if labelled points see the court the way the annotation template does.

    That is court +x pointing right in the image and court +y pointing up (far
    sideline, y=2000, on top), judged through the points' homography at their
    centroid, where it is best conditioned. A half-turned mapping fails both
    tests; a mirrored labelling fails one.
    """
    court_xy = np.asarray(court_xy, dtype=np.float32)
    matrix, _ = cv2.findHomography(court_xy, np.asarray(image_xy, dtype=np.float32))
    if matrix is None:
        return False
    centre = court_xy.mean(axis=0)
    probe = np.array([[centre, centre + (100.0, 0.0), centre + (0.0, 100.0)]], np.float32)
    origin, along_x, along_y = cv2.perspectiveTransform(probe, matrix)[0]
    return bool(along_x[0] > origin[0] and along_y[1] < origin[1])
