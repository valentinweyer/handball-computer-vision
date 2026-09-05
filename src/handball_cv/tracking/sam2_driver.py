"""Reusable SAM2 predictor + TrackManager checkpoint driver.

Extracted from `scripts.run_sam2_reprompt_tracker`, which originally inlined
this whole loop for one purpose: dumping a tracking-only `(frame_index,
tracker_id, box)` replay for `evaluate_tracker_identity`. A second consumer
needs the same propagation + periodic reprompting -- reading jersey numbers
per player -- and duplicating a ~140-line predictor/state-machine loop for
that would leave two copies of the exact re-ID/lifecycle sequencing this
project has already spent real effort getting right. This module is that loop,
factored so both consumers drive it once.

`SAM2_UPSTREAM_DIR` must already be on `sys.path` before importing this module
(see the guard at the top of `scripts.run_sam2_reprompt_tracker` -- SAM2 has no
importable package name until its checkout is added to the path, so the guard
has to run in the entry-point script, before any import of this module).
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterator

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from handball_cv.tracking.sam2_manager import TrackManager

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECK_EVERY_DEFAULT = 10  # matches run_pipeline.py's detector-checkpoint cadence


def ffmpeg_exe() -> str:
    """Bare `ffmpeg` if on PATH, else the imageio-ffmpeg bundled static binary.

    This host has no system ffmpeg and no passwordless sudo to install one;
    imageio-ffmpeg ships a working static binary as a plain pip package.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def ensure_frame_cache(video: Path, frame_cache_dir: Path) -> list[Path]:
    """Extract `video` to `frame_cache_dir` as numbered JPGs if not already done."""
    if not frame_cache_dir.exists() or not any(frame_cache_dir.glob("*.jpg")):
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
             "-q:v", "2", "-start_number", "0", str(frame_cache_dir / "%05d.jpg")],
            check=True,
        )
    return sorted(frame_cache_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def masks_from_logits(mask_logits: torch.Tensor) -> np.ndarray:
    """(N, 1, H, W) logits -> (N, H, W) bool, edge-fragment filtered."""
    masks = (mask_logits > 0.0).squeeze(1).cpu().numpy().astype(bool)
    return np.array([
        sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
        for m in masks
    ])


@dataclass
class Sam2FrameResult:
    """One propagated frame, in the order live objects were returned.

    `player_id` is SAM2's own `obj_id` -- `TrackManager` uses it directly as
    the registry's stable id (see `sam2_manager`'s module docstring), so no
    separate tracker_id/player_id translation exists here the way McByte needs
    one. `read_frame` is lazy: the tracking-only consumer never needs pixel
    data outside checkpoint frames, so eagerly decoding every propagated
    frame's JPEG would cost real time for no benefit to that caller.
    """
    frame_idx: int
    player_ids: np.ndarray   # (N,) int
    masks: np.ndarray        # (N, H, W) bool
    boxes: np.ndarray        # (N, 4) xyxy, mask-derived
    read_frame: Callable[[], np.ndarray]  # -> RGB frame, decoded once and cached


def drive_sam2(
    video: Path,
    frame_detections_fn: Callable[[int], "sv.Detections"],
    team_model,
    checkpoint: str,
    check_every: int = CHECK_EVERY_DEFAULT,
    frame_cache_dir: Path | None = None,
    max_frames: int | None = None,
    goalkeeper_class_id: int = 1,
    desc: str = "SAM2 reprompt",
) -> tuple[TrackManager, dict[int, np.ndarray], Iterator[Sam2FrameResult]]:
    """Seed and propagate SAM2 with periodic detector-checkpoint reprompting.

    `frame_detections_fn(frame_index) -> sv.Detections` is the same cached
    per-frame detection lookup every tracker in this project's evaluation
    scripts already uses (`scripts.render_raw_team_classification.
    frame_detections`, bound to a loaded `.npz` cache) -- both the frame-0 seed
    and every checkpoint's reprompt boxes come from it, so SAM2 is not given a
    stronger detection input than a caller comparing it against other trackers
    would give them.

    Returns `(track_manager, seed_boxes, frames)`:

    - `track_manager` is usable immediately (its `.events`/`.registry` fill in
      as `frames` is consumed).
    - `seed_boxes` is `{obj_id: xyxy}` at frame 0, straight from the detector
      -- there is no mask yet at the seed frame, so it is kept separate from
      `frames` rather than forced into a `Sam2FrameResult` with an empty
      `masks` array that would not line up in length with `player_ids`.
    - `frames` yields every propagated frame from frame 1 onward, each with
      masks/boxes/player_ids of matching length.

    `court_test_fn` is left permissive (accepts any new detection), matching
    the box trackers in `evaluate_tracker_identity`, which also add every
    unmatched detection without a court-membership check.
    """
    from sam2.build_sam import build_sam2_video_predictor

    frame_cache_dir = frame_cache_dir or (
        video.resolve().parent / "_sam2_frame_cache" / video.stem
    )
    frame_files = ensure_frame_cache(video, frame_cache_dir)
    num_frames = len(frame_files)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    def read_frame(idx: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(frame_files[idx])), cv2.COLOR_BGR2RGB)

    predictor = build_sam2_video_predictor(SAM2_CONFIG, checkpoint)

    frame0 = read_frame(0)
    det0 = frame_detections_fn(0)
    if len(det0) == 0:
        raise RuntimeError("no cached detections on frame 0")

    track_manager = TrackManager(team_model, court_test_fn=lambda box: True)
    is_gk0 = det0.class_id == goalkeeper_class_id
    obj_ids0 = track_manager.seed(0, det0.xyxy, frame0, is_gk0)

    state = predictor.init_state(video_path=str(frame_cache_dir))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for oid, xyxy in zip(obj_ids0, det0.xyxy):
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=oid, box=np.asarray(xyxy, dtype=np.float32)
            )

    seed_boxes = dict(zip(obj_ids0, det0.xyxy))

    def frames() -> Iterator[Sam2FrameResult]:
        next_new_fid = 1
        last_masks_by_id: dict[int, np.ndarray] = {}
        last_frame = frame0

        pbar = tqdm(total=num_frames - 1, desc=desc)
        t = 0
        while t < num_frames - 1:
            chunk_len = min(check_every, num_frames - 1 - t)
            chunk_end = t

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for fid, obj_ids, mask_logits in predictor.propagate_in_video(
                        state, start_frame_idx=t, max_frame_num_to_track=chunk_len):
                    if fid < next_new_fid or fid >= num_frames:
                        continue  # repeated boundary frame, or past our truncation

                    masks = masks_from_logits(mask_logits)
                    track_manager.update_from_propagation(fid, obj_ids, masks)

                    boxes_this = sv.mask_to_xyxy(masks=masks)
                    yield Sam2FrameResult(
                        frame_idx=fid,
                        player_ids=np.asarray(obj_ids, dtype=int),
                        masks=masks,
                        boxes=boxes_this,
                        read_frame=functools.lru_cache(maxsize=1)(lambda idx=fid: read_frame(idx)),
                    )

                    chunk_end = fid
                    if fid == t + chunk_len:
                        last_masks_by_id = dict(zip(obj_ids, masks))
                        last_frame = read_frame(fid)
                    pbar.update(1)

            next_new_fid = chunk_end + 1
            t = chunk_end
            if t >= num_frames - 1:
                break

            # ── detector checkpoint: decide add / remove / reprompt ──
            live_obj_ids = list(last_masks_by_id.keys())
            live_masks = (
                np.array([last_masks_by_id[oid] for oid in live_obj_ids])
                if live_obj_ids else np.empty((0, 0, 0), dtype=bool)
            )
            det = frame_detections_fn(chunk_end)
            det_is_goalkeeper = det.class_id == goalkeeper_class_id

            actions = track_manager.checkpoint(
                chunk_end, last_frame, live_obj_ids, live_masks, det.xyxy, det_is_goalkeeper
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for action in actions:
                    if action["type"] == "remove":
                        predictor.remove_object(state, obj_id=action["obj_id"])
                    elif action["type"] == "reprompt":
                        predictor.add_new_points_or_box(
                            state, frame_idx=chunk_end, obj_id=action["obj_id"],
                            box=np.asarray(action["box"], dtype=np.float32),
                            clear_old_points=True,
                        )
                    elif action["type"] == "add":
                        predictor.add_new_points_or_box(
                            state, frame_idx=chunk_end, obj_id=action["obj_id"],
                            box=np.asarray(action["box"], dtype=np.float32),
                        )
                    elif action["type"] == "reset":
                        # Body-swap, not drift: memory is contaminated with the
                        # wrong player's appearance, so reprompting in place
                        # would seed the "correction" from that wrong mask.
                        # Tear the object down and re-add it fresh under the
                        # same obj_id.
                        if len(state["obj_id_to_idx"]) > 1:
                            predictor.remove_object(state, obj_id=action["obj_id"])
                            predictor.add_new_points_or_box(
                                state, frame_idx=chunk_end, obj_id=action["obj_id"],
                                box=np.asarray(action["box"], dtype=np.float32),
                            )
                        else:
                            # Removing the only live object resets the whole
                            # session (see SAM2VideoPredictor.remove_object) --
                            # fall back to an in-place reprompt instead.
                            predictor.add_new_points_or_box(
                                state, frame_idx=chunk_end, obj_id=action["obj_id"],
                                box=np.asarray(action["box"], dtype=np.float32),
                                clear_old_points=True,
                            )
        pbar.close()

    return track_manager, seed_boxes, frames()
