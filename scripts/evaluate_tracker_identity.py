"""Score any tracker's identity quality against the SAM2 reference for Han-Ber.

The reference is 198 frames of per-pixel identity labels (ids 1-14) produced by
SAM2, independent of whatever tracker is being tested, and human-verified as
clean -- every id follows one person for its whole life. That independence is
what makes automated comparison possible: label once, score any number of
trackers, instead of hand-labelling each tracker's own partitioning.

Two of the reference ids are not field players (12 is a bench player standing
at the sideline, 14 is a referee), so metrics are reported both ways.

Metrics, all derived by matching tracker boxes to reference-mask boxes by IoU:

  * **mixed tracklets**   -- tracker_ids that cover more than one reference
      identity. The direct analogue of the hand labels used on FelixClaar.
  * **wrong-identity frames** -- matched frames where a tracker_id is on some
      identity other than its own dominant one. This is the ~25% figure.
  * **id switches**       -- transitions in the reference identity a tracker_id
      is matched to, walking its frames in order.
  * **fragments**         -- distinct tracker_ids covering one reference id.
  * **coverage**          -- fraction of reference detections matched at all;
      a tracker that simply drops players would otherwise look perfect.

Usage:
    python -m scripts.evaluate_tracker_identity --trackers mcbyte bytetrack sort
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from trackers import (
    BoTSORTTracker,
    ByteTrackTracker,
    CBIoUTracker,
    McByteMaskConfig,
    McByteTracker,
    OCSORTTracker,
    SORTTracker,
)

from scripts.render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)

ROOT = Path(__file__).resolve().parents[1]
MATCH_IOU = 0.5


def parse_labels(path: Path) -> dict:
    """Human verification file -> {identity: (valid_from, valid_to_or_None)}.

    A reference identity is only usable while it still follows one player.
    "clean until f90" means everything after frame 90 is untrustworthy, so the
    identity is scored over frames 0..90 and ignored afterwards; "X" drops it
    entirely. Scoring a tracker against a span the reference itself got wrong
    would punish the tracker for the reference's error.
    """
    import re

    valid = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, comment = line.partition("#")
        match = re.match(r"\s*(\d+)", head)
        if not match:
            continue
        identity = int(match.group(1))
        text = (head[match.end():] + " " + comment).strip().lower()
        if not text:
            continue
        if re.search(r"\bx\b|unclear", text):
            continue                      # excluded outright
        frames = [int(f) for f in re.findall(r"f(\d+)", text)]
        if frames and re.search(r"unti|until", text):
            valid[identity] = (0, frames[0])
        else:
            valid[identity] = (0, None)
    return valid


class Sam2ReplayTracker:
    """Replays a scripts.run_sam2_reprompt_tracker dump through the shared
    Tracker interface (.update(detections, frame) -> sv.Detections).

    That driver runs the real SAM2 video predictor once, offline, since it
    propagates in chunks rather than accepting one frame at a time -- this
    class exists only so run_tracker's per-frame loop (built around trackers
    that ARE naturally frame-incremental) can score its output identically to
    every box tracker. `detections`/`frame` are ignored: SAM2 already made its
    association decision when the dump was produced, using the same cached
    detections as its checkpoint reprompt source.
    """

    def __init__(self, dump_path: Path):
        if not dump_path.exists():
            raise FileNotFoundError(
                f"{dump_path} not found -- run scripts.run_sam2_reprompt_tracker "
                "for this clip first."
            )
        data = np.load(dump_path)
        frame_index, tracker_id, boxes = data["frame_index"], data["tracker_id"], data["boxes"]
        self._by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for f in np.unique(frame_index):
            m = frame_index == f
            self._by_frame[int(f)] = (boxes[m].astype(float), tracker_id[m].astype(int))
        self._cursor = 0

    def update(self, detections: sv.Detections, frame=None) -> sv.Detections:
        xyxy, tracker_id = self._by_frame.get(
            self._cursor, (np.empty((0, 4)), np.empty((0,), dtype=int))
        )
        self._cursor += 1
        return sv.Detections(xyxy=xyxy, tracker_id=tracker_id)


def build_trackers(fps: float, device: str, video: Path | None = None) -> dict:
    """Every tracker in the package, at library defaults unless noted."""
    trackers = {
        "mcbyte_masks_on": lambda: McByteTracker(
            frame_rate=fps, lost_track_buffer=30, track_activation_threshold=0.7,
            enable_mask_manager=True, mask_config=McByteMaskConfig(device=device),
            minimum_mask_average_confidence=0.6, minimum_mask_coverage=0.9,
            minimum_mask_fill_ratio=0.05,
        ),
        "mcbyte_masks_off": lambda: McByteTracker(
            frame_rate=fps, lost_track_buffer=30, track_activation_threshold=0.7,
            enable_mask_manager=False,
        ),
        "bytetrack": lambda: ByteTrackTracker(
            frame_rate=fps, lost_track_buffer=30, track_activation_threshold=0.7),
        "botsort": lambda: BoTSORTTracker(
            frame_rate=fps, lost_track_buffer=30, track_activation_threshold=0.7),
        "ocsort": lambda: OCSORTTracker(frame_rate=fps, lost_track_buffer=30),
        "sort": lambda: SORTTracker(
            frame_rate=fps, lost_track_buffer=30, track_activation_threshold=0.7),
        "cbiou": lambda: CBIoUTracker(
            frame_rate=fps, lost_track_buffer=30, track_activation_threshold=0.7),
    }
    if video is not None:
        dump = ROOT / "runs" / "sam2_reprompt" / f"{video.stem}.npz"
        if dump.exists():
            trackers["sam2_reprompt"] = lambda: Sam2ReplayTracker(dump)
        # Checkpoint-policy comparison. Each policy writes its own artifact
        # rather than overwriting the one above, so a policy run never
        # silently redefines the baseline a previous result was scored on.
        for policy in ("reprompt", "reset_reseed"):
            candidate = (
                ROOT / "runs" / "sam2_reprompt" / f"{video.stem}_policy_{policy}.npz"
            )
            if candidate.exists():
                trackers[f"sam2_policy_{policy}"] = (
                    lambda path=candidate: Sam2ReplayTracker(path)
                )
    return trackers


def load_reference(reference_dir: Path, offset: int, valid: dict) -> dict:
    """-> {frame_index: {reference_id: xyxy}}, restricted to verified spans."""
    reference = {}
    for path in sorted(glob.glob(str(reference_dir / "*.npz"))):
        if not Path(path).stem.isdigit():
            continue
        frame_index = int(Path(path).stem) + offset
        label = np.load(path)["label"]
        boxes = {}
        for value in np.unique(label):
            if value == 0:
                continue
            identity = int(value)
            span = valid.get(identity)
            if span is None:
                continue                      # unverified or excluded
            start, stop = span
            if frame_index < start or (stop is not None and frame_index > stop):
                continue                      # beyond where the reference is trusted
            ys, xs = np.where(label == value)
            if len(xs) < 50:
                continue
            boxes[identity] = np.array(
                [xs.min(), ys.min(), xs.max(), ys.max()], dtype=float
            )
        reference[frame_index] = boxes
    return reference


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def run_tracker(factory, reference: dict, video: Path, detections: Path, device: str) -> list:
    """-> [(frame_index, tracker_id, reference_id_or_None), ...]."""
    cache = load_detection_cache(detections)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = factory()
    matches = []
    for frame_index, frame_bgr in enumerate(sv.get_video_frames_generator(str(video))):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracked = tracker.update(frame_detections(cache, frame_index), frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        truth = reference.get(frame_index, {})
        if not len(tracked) or not truth:
            continue
        reference_ids = sorted(truth)
        scores = iou_matrix(
            np.asarray(tracked.xyxy, dtype=float),
            np.stack([truth[r] for r in reference_ids]),
        )
        # Greedy one-to-one: best pair first, so a box cannot claim two ids.
        pairs = sorted(
            ((scores[i, j], i, j) for i in range(scores.shape[0])
             for j in range(scores.shape[1]) if scores[i, j] >= MATCH_IOU),
            reverse=True,
        )
        used_rows, used_cols = set(), set()
        assigned = {}
        for score, row, column in pairs:
            if row in used_rows or column in used_cols:
                continue
            used_rows.add(row); used_cols.add(column)
            assigned[row] = reference_ids[column]
        for row, tracker_id in enumerate(tracked.tracker_id):
            matches.append((frame_index, int(tracker_id), assigned.get(row)))
    return matches


def score(matches: list, keep_ids: set | None) -> dict:
    """Identity metrics from (frame, tracker_id, reference_id) triples."""
    rows = [
        m for m in matches
        if m[2] is not None and (keep_ids is None or m[2] in keep_ids)
    ]
    total_tracked = len([m for m in matches if keep_ids is None or m[2] in (keep_ids or ())])
    per_track = defaultdict(list)
    for frame_index, tracker_id, reference_id in sorted(rows):
        per_track[tracker_id].append((frame_index, reference_id))

    mixed, wrong_frames, switches, matched = 0, 0, 0, 0
    for tracker_id, entries in per_track.items():
        ids = [reference_id for _, reference_id in entries]
        counts = Counter(ids)
        dominant = counts.most_common(1)[0][0]
        matched += len(ids)
        wrong_frames += len(ids) - counts[dominant]
        if len(counts) > 1:
            mixed += 1
        switches += sum(1 for a, b in zip(ids, ids[1:]) if a != b)

    fragments = defaultdict(set)
    for tracker_id, entries in per_track.items():
        for _, reference_id in entries:
            fragments[reference_id].add(tracker_id)
    return {
        "tracklets": len(per_track),
        "mixed_tracklets": mixed,
        "matched_detections": matched,
        "wrong_identity_frames": wrong_frames,
        "wrong_identity_pct": 100.0 * wrong_frames / max(matched, 1),
        "id_switches": switches,
        "reference_ids_covered": len(fragments),
        "mean_fragments_per_reference_id": (
            float(np.mean([len(v) for v in fragments.values()])) if fragments else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path,
                        help="directory of NNNNN.npz label maps")
    parser.add_argument("--labels", required=True, type=Path,
                        help="human verification answers.txt for that reference")
    parser.add_argument("--exclude-ids", type=int, nargs="*", default=(),
                        help="reference ids that are not field players")
    parser.add_argument("--trackers", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    valid = parse_labels(args.labels)
    reference = load_reference(args.reference, args.offset, valid)
    reference_detections = sum(len(v) for v in reference.values())
    info = sv.VideoInfo.from_video_path(str(args.video))
    factories = build_trackers(info.fps, args.device, args.video)
    names = args.trackers or list(factories)
    keep = {i for i in valid if i not in set(args.exclude_ids)}

    truncated = {i: v[1] for i, v in valid.items() if v[1] is not None}
    print(f"reference: {len(reference)} frames, {reference_detections} trusted detections")
    print(f"verified identities: {sorted(valid)}")
    if truncated:
        print(f"truncated at: {truncated}")
    if args.exclude_ids:
        print(f"excluded as non-players: {sorted(args.exclude_ids)}")
    print()

    results = {}
    for name in names:
        if name not in factories:
            print(f"  unknown tracker: {name}")
            continue
        matches = run_tracker(
            factories[name], reference, args.video, args.detections, args.device
        )
        only = score(matches, keep)
        results[name] = {"players_only": only, "all_ids": score(matches, None)}
        print(
            f"{name:>18}  tracklets={only['tracklets']:3d}  "
            f"mixed={only['mixed_tracklets']:2d}  "
            f"wrong={only['wrong_identity_pct']:5.1f}%  "
            f"switches={only['id_switches']:3d}  "
            f"frags/id={only['mean_fragments_per_reference_id']:.2f}  "
            f"matched={only['matched_detections']:4d}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"video": str(args.video), "valid_spans": {str(k): v for k, v in valid.items()},
             "results": results}, indent=2))


if __name__ == "__main__":
    main()
