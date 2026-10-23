"""Run an RF-DETR TorchScript export on every frame of the study clips (GPU).

Same detector class, input path, threshold and cache schema as
`scripts.process_video`'s scan stage, but at stride 1 (box trackers need
every frame) and written under the study's artifact directory, never over a
stage output. Arm name = the export directory, e.g. `detector-expanded-v1`.

    python -m experiments.box_tracker_oracle.detect_every_frame detector-expanded-v1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import supervision as sv

from experiments.box_tracker_oracle.common import CLIPS, MAIN, OUT
FINETUNE_ROOT = MAIN.parent / "rfdetr-handball-finetune"   # the worktree has no sibling checkout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", help="directory under rfdetr-handball-finetune/exports")
    parser.add_argument("--clips", nargs="+", default=list(CLIPS))
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    import torch

    from handball_cv.detection.rfdetr_torchscript import TorchScriptDetector
    from handball_cv.pipeline.scan import scan_video
    from handball_cv.video import VideoFrames

    weights = FINETUNE_ROOT / "exports" / args.export / "rfdetr-large-handball.ts"
    detector = TorchScriptDetector(weights, device="cuda")
    out_dir = OUT / "detections" / args.export
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name in args.clips:
        video = MAIN / "data/raw" / f"{CLIPS[name][0]}.mp4"
        count = sv.VideoInfo.from_video_path(str(video)).total_frames
        torch.cuda.synchronize()
        started = time.perf_counter()
        scanned = scan_video(VideoFrames(video, count=count), detector=detector, motion=False,
                             detect_stride=1, threshold=args.threshold,
                             detector_id=str(weights), desc=name)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        np.savez_compressed(out_dir / f"{name}.npz", **scanned["detections"])
        timings[name] = {"frames": count, "seconds": round(seconds, 2),
                         "ms_per_frame": round(1000 * seconds / count, 1)}
        print(name, timings[name], flush=True)
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
