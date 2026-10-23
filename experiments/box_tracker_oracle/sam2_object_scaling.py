"""How does SAM2's per-frame cost scale with the number of tracked players?

A contact-triggered hybrid would run SAM2 only on the players near a contact
(35-47% of player-frames in full games, `contact_coverage.py`). That only
saves time if cost grows with the object count rather than being dominated by
the shared image encoder. Same predictor construction and output-identical
accelerations as `sam2_driver.drive_sam2`, bfloat16, frames read lazily from
the video, every propagated frame's masks converted with `masks_from_logits`
as the tracking loop does. Fresh predictor state per object count; the first
`--warmup` frames are excluded.

    python -m experiments.box_tracker_oracle.sam2_object_scaling --counts 1 2 4 8 14
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from experiments.box_tracker_oracle.common import MAIN, OUT, Clip

sys.path.insert(0, str(MAIN / "sam2-upstream"))
CHECKPOINT = MAIN / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="han")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 14])
    parser.add_argument("--frames", type=int, default=70)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    from handball_cv.tracking.sam2_driver import (
        SAM2_CONFIG, init_state_lazily, install_rope_repeat_cache, masks_from_logits)
    from handball_cv.tracking.sam2_acceleration import (
        install_fused_rope, install_small_module_graphs)
    from handball_cv.video import VideoFrames

    clip = Clip(args.clip)
    start = min(clip.oracle_boxes)
    boxes0 = clip.oracle_boxes[start]
    ids = sorted(boxes0, key=lambda i: -np.prod(boxes0[i][2:] - boxes0[i][:2]))
    predictor = build_sam2_video_predictor(SAM2_CONFIG, str(CHECKPOINT))
    predictor.fill_hole_area = 0
    install_rope_repeat_cache()
    install_fused_rope()
    install_small_module_graphs(predictor)

    results = {}
    for count in args.counts:
        chosen = ids[:count]
        state = init_state_lazily(predictor, VideoFrames(clip.video, count=start + args.frames))
        times = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for oid in chosen:
                predictor.add_new_points_or_box(state, frame_idx=start, obj_id=int(oid),
                                                box=np.asarray(boxes0[oid], np.float32))
            torch.cuda.synchronize()
            last = time.perf_counter()
            for _, _, logits in predictor.propagate_in_video(state, start_frame_idx=start,
                                                              max_frame_num_to_track=args.frames):
                masks_from_logits(logits)
                torch.cuda.synchronize()
                now = time.perf_counter()
                times.append(now - last)
                last = now
        predictor.reset_state(state)
        steady = np.array(times[args.warmup:]) * 1000
        results[count] = {"objects": len(chosen), "ms_median": round(float(np.median(steady)), 1),
                          "ms_mean": round(float(steady.mean()), 1), "frames": len(steady)}
        print(count, results[count], flush=True)
    counts = np.array([r["objects"] for r in results.values()], float)
    ms = np.array([r["ms_median"] for r in results.values()])
    slope, intercept = np.polyfit(counts, ms, 1)
    fit = {"fixed_ms": round(float(intercept), 1), "per_object_ms": round(float(slope), 1)}
    print("linear fit", fit)
    (OUT / "sam2_object_scaling.json").write_text(
        json.dumps({"clip": args.clip, "results": results, "fit": fit}, indent=1))


if __name__ == "__main__":
    main()
