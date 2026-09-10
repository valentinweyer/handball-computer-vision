"""Run one trained SAM2/EfficientTAM arm through the shared tracking driver.

Use a fresh process per arm. Cached RF-DETR, team-model re-ID, BF16, interval 10,
1024 input, eager predictors and existing post-processing are held fixed.
The output directory must be new. Timing excludes result-file compression and
scores; the loop includes all checkpoint decisions, prompts and CPU mask work.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "felix": ("FelixClaar", "outputs/team_comparison/.FelixClaar_detections_v1.npz",
              "outputs/team_comparison/.FelixClaar_team.pkl"),
    "han": ("Han-Ber4_cached", "outputs/team_dataset/.Han-Ber4_detections_v1.npz",
            "outputs/team_correction_mcbyte/.Han-Ber4_team.pkl"),
    "bhc": ("BHC-FAG_window_cached", "outputs/team_dataset/.BHC-FAG_window_detections_v1.npz",
            "outputs/team_confidence_v2/.BHC-FAG_team.pkl"),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sam2", "efficienttam"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-masks", action="store_true",
                        help="diagnostic run: retain packed masks every frame; adds loop overhead")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("a trained checkpoint file is required")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.upstream.resolve()))
    # Import exactly one model family: both initialize Hydra globally.
    if args.backend == "sam2":
        from sam2.build_sam import build_sam2_video_predictor as builder
        config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    else:
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor as builder
        config = "configs/efficienttam/efficienttam_s.yaml"
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking.sam2_driver import drive_sam2, ensure_frame_cache
    from scripts.render_raw_team_classification import load_detection_cache, person_detections

    stem, detection_path, team_path = CASES[args.case]
    video = ROOT / "data/raw" / f"{stem}.mp4"
    frame_cache = ROOT / "data/cache/frames" / stem
    files = ensure_frame_cache(video, frame_cache)
    cache = load_detection_cache(ROOT / detection_path)
    # These three benchmark caches already contain only people. Do not alter the
    # historical workload silently if a future cache adds other classes.
    assert set(np.unique(cache["class_id"])) <= {1, 2}
    team = TeamModel.load(ROOT / team_path, device="cuda")
    manifest = {
        "backend": args.backend, "case": args.case, "config": config,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "upstream_revision": subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "project_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "driver_sha256": sha256(ROOT / "src/handball_cv/tracking/sam2_driver.py"),
        "runner_sha256": sha256(__file__),
        "video": str(video), "video_sha256": sha256(video),
        "detections_sha256": sha256(ROOT / detection_path),
        "team_model_sha256": sha256(ROOT / team_path),
        "jpeg_count": len(files), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "precision": "BF16 autocast",
        "compile_image_encoder": False, "vos_optimized": False,
        "check_every": 10, "checkpoint_policy": "reprompt",
        "reid": "team-model features (historical tracking-only configuration)",
        "ocr": False, "live_detection": False, "capture_masks": args.capture_masks,
    }
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")
        torch.cuda.synchronize()
        before = perf_counter()
        predictor = builder(config, str(args.checkpoint), vos_optimized=False,
                            hydra_overrides_extra=["++model.compile_image_encoder=false"])
        torch.cuda.synchronize()
        manifest["model_build_seconds"] = perf_counter() - before
        assert predictor.image_size == 1024
        factory = lambda checkpoint: predictor
        lookup = lambda fid: person_detections(cache, fid)
        # Physically bounded warmup state, independent of measured manager/history.
        with tempfile.TemporaryDirectory(prefix="efficienttam-warmup-") as tmp:
            for path in files[:21]:
                shutil.copyfile(path, Path(tmp) / path.name)
            manager, seeds, frames = drive_sam2(
                video, lookup, team, str(args.checkpoint), frame_cache_dir=Path(tmp),
                predictor_factory=factory, desc="warmup",
            )
            for result in frames:
                pass
            del manager, seeds, frames, result
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = perf_counter()
        manager, seeds, frames = drive_sam2(
            video, lookup, team, str(args.checkpoint), frame_cache_dir=frame_cache,
            predictor_factory=factory, desc=f"{args.backend} {args.case}",
        )
        torch.cuda.synchronize()
        manifest["fresh_state_seed_seconds"] = perf_counter() - before
        rows = [(0, int(oid), box.copy()) for oid, box in seeds.items()]
        frame_stats, packed = [], []
        before = perf_counter()
        resumed = before
        for result in frames:
            now = perf_counter()
            # No per-frame CUDA synchronize: masks already arrive on CPU. Whole
            # interval synchronization is the throughput authority.
            frame_stats.append({"frame": result.frame_idx, "objects": len(result.player_ids),
                                "resume_seconds": now-resumed,
                                "after_checkpoint": result.frame_idx > 1 and result.frame_idx % 10 == 1,
                                "mask_pixels": int(result.masks.sum()) if args.capture_masks else None})
            rows.extend((result.frame_idx, int(oid), box.copy())
                        for oid, box in zip(result.player_ids, result.boxes))
            if args.capture_masks:
                packed.append((result.frame_idx, result.player_ids.copy(),
                               np.packbits(result.masks, axis=-1), result.masks.shape[-1]))
            resumed = perf_counter()
        torch.cuda.synchronize()
        elapsed = perf_counter() - before
        assert len(frame_stats) == len(files)-1
        times = [r["resume_seconds"] for r in frame_stats]
        manifest.update({
            "loop_seconds": elapsed, "unique_propagated_frames": len(frame_stats),
            "loop_fps": len(frame_stats)/elapsed,
            "resume_latency_ms_median": float(np.median(times)*1000),
            "resume_latency_ms_p95": float(np.percentile(times,95)*1000),
            "mean_active_objects": float(np.mean([r["objects"] for r in frame_stats])),
            "peak_active_objects": max(r["objects"] for r in frame_stats),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "warnings": sorted({str(w.message) for w in captured}),
        })
    np.savez_compressed(args.output / "replay.npz",
                        frame_index=np.array([r[0] for r in rows], dtype=int),
                        tracker_id=np.array([r[1] for r in rows], dtype=int),
                        boxes=np.array([r[2] for r in rows], dtype=float).reshape(-1,4),
                        source=str(video))
    (args.output / "events.json").write_text(json.dumps(manager.events, indent=2, default=json_default))
    (args.output / "frames.json").write_text(json.dumps(frame_stats, default=json_default))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default))
    if packed:
        masks_dir = args.output / "masks"
        masks_dir.mkdir()
        for fid, ids, masks, width in packed:
            np.savez_compressed(masks_dir / f"{fid:05d}.npz", ids=ids, masks=masks, width=width)
    print(json.dumps({k: manifest[k] for k in ("backend", "case", "loop_fps", "loop_seconds",
                                             "mean_active_objects", "peak_active_objects")}, indent=2))


if __name__ == "__main__":
    main()
