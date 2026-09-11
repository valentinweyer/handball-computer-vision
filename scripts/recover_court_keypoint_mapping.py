"""Re-derive and validate ``KEYPOINT_TO_VERTEX`` from the labelled keypoint export.

The keypoint model numbers its 37 court landmarks in one order; the ``sports``
court template numbers its 37 vertices in another. ``run_court_mapping.py`` used
to assume the two agreed, which fitted every homography to a scrambled
correspondence. This script recovers the translation between them and scores it.

It never assumes the shipped answer. The recovery is seeded only from the centre
circle -- the one structure identifiable by eye, a five-point cross whose centre,
left/right and near/far arms are unambiguous -- and every orientation of that
seed is tried, so a wrong guess about which arm is which cannot bias the result.
From each seed it alternates projecting the labelled points onto the court and
re-solving the assignment (Hungarian, so the mapping stays one-to-one) until the
assignment stops moving. Slots absent from the anchor image are recovered by
repeating the fit across the whole export and pooling votes.

The score is the thing to read. A handball court is planar and a broadcast camera
is near enough a pinhole, so a correct correspondence has to fit to within
annotation noise; a wrong one cannot. ``--compare`` additionally diffs the result
against the constant the package ships.

Usage:
    python -m scripts.recover_court_keypoint_mapping
    python -m scripts.recover_court_keypoint_mapping --compare
"""

from __future__ import annotations

import argparse
import glob
import collections
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from sports import MeasurementUnit
from sports.handball import CourtConfiguration, League

from handball_cv.court.keypoints import (
    KEYPOINT_COUNT,
    KEYPOINT_FLIP_INDEX,
    KEYPOINT_TO_VERTEX,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "notebooks/keypointv333-1"
# The export is letterboxed to a square; the stretch is uniform, so any
# consistent scale recovers the same homography.
EXPORT_WH = 640.0
# Court landmarks are annotated by hand. 250 cm is loose enough to admit that
# noise and tight enough to reject a landmark assigned to the wrong vertex.
SNAP_TOLERANCE_CM = 250.0
MIN_VISIBLE = 6


def load_label(path: Path) -> np.ndarray | None:
    """Return the (37, 3) x/y/visibility block of a YOLO-pose label, or None."""
    fields = path.read_text().split()
    if len(fields) < 5 + 3 * KEYPOINT_COUNT:
        return None
    block = fields[5 : 5 + 3 * KEYPOINT_COUNT]
    return np.array(block, dtype=float).reshape(KEYPOINT_COUNT, 3)


def visible(label: np.ndarray) -> list[int]:
    return list(np.where(label[:, 2] > 0)[0])


def image_points(label: np.ndarray, slots: list[int]) -> np.ndarray:
    return np.stack(
        [label[slots, 0] * EXPORT_WH, label[slots, 1] * EXPORT_WH], axis=1
    ).astype(np.float32)


def refine(src: np.ndarray, slots: list[int], seed: dict[int, int],
           vertices: np.ndarray, iters: int = 40) -> tuple[dict[int, int], float]:
    """Alternate homography fit and one-to-one re-assignment until stable."""
    anchors = [(s, v) for s, v in seed.items() if s in slots]
    if len(anchors) < 4:
        return {}, float("inf")
    rows = [slots.index(s) for s, _ in anchors]
    matrix, _ = cv2.findHomography(
        src[rows], vertices[[v for _, v in anchors]].astype(np.float32)
    )
    if matrix is None:
        return {}, float("inf")

    assignment: dict[int, int] = {}
    previous: dict[int, int] | None = None
    for _ in range(iters):
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        cost = np.linalg.norm(projected[:, None, :] - vertices[None, :, :], axis=2)
        rows_, cols = linear_sum_assignment(cost)
        assignment = {slots[i]: int(j) for i, j in zip(rows_, cols)}
        if assignment == previous:
            break
        previous = assignment
        refit, _ = cv2.findHomography(
            src, vertices[cols].astype(np.float32), cv2.RANSAC, 200.0
        )
        if refit is None:
            break
        matrix = refit

    if not assignment:
        return {}, float("inf")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    target = vertices[[assignment[s] for s in slots]]
    return assignment, float(np.median(np.linalg.norm(projected - target, axis=1)))


def seed_hypotheses(vertices: np.ndarray) -> list[dict[int, int]]:
    """Every orientation of the centre-circle cross.

    Slots 6/16/17/19/18 are the cross's centre, left, right, upper and lower
    arms as they appear in the frame; slot 5 sits where the centre line meets a
    sideline. Which court direction each corresponds to depends on the camera's
    side and on the template's handedness, so all eight combinations are tried
    and the residual decides.
    """
    centre_slot, left_slot, right_slot, up_slot, down_slot, sideline_slot = 6, 16, 17, 19, 18, 5
    centre = int(np.argmin(np.linalg.norm(vertices - [2000.0, 1000.0], axis=1)))
    west = int(np.argmin(np.linalg.norm(vertices - [1820.0, 1000.0], axis=1)))
    east = int(np.argmin(np.linalg.norm(vertices - [2180.0, 1000.0], axis=1)))
    north = int(np.argmin(np.linalg.norm(vertices - [2000.0, 1150.0], axis=1)))
    south = int(np.argmin(np.linalg.norm(vertices - [2000.0, 850.0], axis=1)))
    far = int(np.argmin(np.linalg.norm(vertices - [2000.0, 2000.0], axis=1)))
    near = int(np.argmin(np.linalg.norm(vertices - [2000.0, 0.0], axis=1)))

    hypotheses = []
    for mirror_x in (False, True):
        for flip_y in (False, True):
            for sideline in (far, near):
                hypotheses.append({
                    centre_slot: centre,
                    left_slot: east if mirror_x else west,
                    right_slot: west if mirror_x else east,
                    up_slot: south if flip_y else north,
                    down_slot: north if flip_y else south,
                    sideline_slot: sideline,
                })
    return hypotheses


def score(files: list[Path], mapping, vertices: np.ndarray) -> tuple[float, float, float]:
    """Median homography fit residual, median leave-one-out error, %% under 50 cm."""
    fit, loo = [], []
    for path in files:
        label = load_label(path)
        if label is None:
            continue
        slots = visible(label)
        if len(slots) < MIN_VISIBLE:
            continue
        src = image_points(label, slots)
        dst = vertices[[mapping(s) for s in slots]].astype(np.float32)
        matrix, _ = cv2.findHomography(src, dst)
        if matrix is not None:
            projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            fit.append(np.median(np.linalg.norm(projected - dst, axis=1)))
        errors = []
        for k in range(len(slots)):
            keep = np.ones(len(slots), bool)
            keep[k] = False
            held, _ = cv2.findHomography(src[keep], dst[keep])
            if held is None:
                continue
            point = cv2.perspectiveTransform(src[k].reshape(1, 1, 2), held).ravel()
            errors.append(np.linalg.norm(point - dst[k]))
        if errors:
            loo.append(np.median(errors))
    fit_a, loo_a = np.array(fit), np.array(loo)
    return float(np.median(fit_a)), float(np.median(loo_a)), float(100 * (fit_a < 50).mean())


def flip_agreement(mapping, vertices: np.ndarray) -> int:
    """Slots where the export's own mirror table agrees with the mapping."""
    mirrored = vertices.copy()
    mirrored[:, 0] = vertices[:, 0].max() - mirrored[:, 0]
    return sum(
        np.allclose(vertices[mapping(KEYPOINT_FLIP_INDEX[s])], mirrored[mapping(s)], atol=1.0)
        for s in range(KEYPOINT_COUNT)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--compare", action="store_true",
                        help="diff the recovered mapping against the shipped constant")
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(str(args.dataset / "*/labels/*.txt")))
    if not files:
        raise SystemExit(
            f"no labels under {args.dataset}. The export is gitignored; re-download "
            f"it from Roboflow (see notebooks/handball_court_keypoint_training.ipynb)."
        )
    config = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)
    vertices = np.array(config.vertices, dtype=np.float64)
    print(f"{len(files)} labelled images under {args.dataset}")

    anchor_path, anchor_label, anchor_slots = None, None, []
    for path in files:
        label = load_label(path)
        if label is None:
            continue
        slots = visible(label)
        if len(slots) > len(anchor_slots):
            anchor_path, anchor_label, anchor_slots = path, label, slots
    print(f"anchor: {anchor_path.name} ({len(anchor_slots)} visible slots)\n")

    src = image_points(anchor_label, anchor_slots)
    best, best_residual = {}, float("inf")
    for i, seed in enumerate(seed_hypotheses(vertices)):
        assignment, residual = refine(src, anchor_slots, seed, vertices)
        print(f"  seed {i}: median residual {residual:9.1f} cm")
        if residual < best_residual:
            best, best_residual = assignment, residual
    print(f"\nbest seed: {best_residual:.1f} cm on the anchor image")

    mapping = dict(best)
    for _ in range(6):
        votes: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        for path in files:
            label = load_label(path)
            if label is None:
                continue
            slots = visible(label)
            known = [s for s in slots if s in mapping]
            if len(known) < MIN_VISIBLE:
                continue
            points = image_points(label, slots)
            rows = [slots.index(s) for s in known]
            matrix, _ = cv2.findHomography(
                points[rows], vertices[[mapping[s] for s in known]].astype(np.float32),
                cv2.RANSAC, 100.0,
            )
            if matrix is None:
                continue
            projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            cost = np.linalg.norm(projected[:, None, :] - vertices[None, :, :], axis=2)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < SNAP_TOLERANCE_CM:
                    votes[slots[i]][int(j)] += 1
        slots_seen = sorted(votes)
        tally = np.zeros((len(slots_seen), KEYPOINT_COUNT))
        for i, s in enumerate(slots_seen):
            for j, n in votes[s].items():
                tally[i, j] = n
        rows, cols = linear_sum_assignment(-tally)
        pooled = {slots_seen[i]: int(cols[i]) for i in rows if tally[i, cols[i]] > 0}
        if pooled == mapping:
            break
        mapping = pooled

    recovered = tuple(mapping[s] for s in range(KEYPOINT_COUNT))
    assert sorted(recovered) == list(range(KEYPOINT_COUNT)), "mapping is not a permutation"

    print(f"\nrecovered all {len(mapping)}/{KEYPOINT_COUNT} slots (bijection confirmed)\n")
    fit0, loo0, under0 = score(files, lambda s: s, vertices)
    fit1, loo1, under1 = score(files, lambda s: recovered[s], vertices)
    print("                                   BEFORE      AFTER")
    print(f"  homography fit residual        {fit0:7.1f} cm {fit1:8.1f} cm")
    print(f"  leave-one-out error            {loo0:7.1f} cm {loo1:8.1f} cm")
    print(f"  images under 50 cm             {under0:7.1f} %  {under1:8.1f} %")
    print(f"  flip_idx agreement (of 37)     {flip_agreement(lambda s: s, vertices):7d}    "
          f"{flip_agreement(lambda s: recovered[s], vertices):8d}")

    if args.compare:
        if recovered == tuple(KEYPOINT_TO_VERTEX):
            print("\n--compare: recovered mapping matches handball_cv.court.keypoints exactly")
        else:
            print("\n--compare: MISMATCH against handball_cv.court.keypoints")
            for s in range(KEYPOINT_COUNT):
                if recovered[s] != KEYPOINT_TO_VERTEX[s]:
                    print(f"    slot {s}: recovered {recovered[s]}, shipped {KEYPOINT_TO_VERTEX[s]}")
    print("\nKEYPOINT_TO_VERTEX = (")
    for i in range(0, KEYPOINT_COUNT, 6):
        print("    " + " ".join(f"{v:2d}," for v in recovered[i : i + 6]))
    print(")")


if __name__ == "__main__":
    main()
