"""Run a bounded production-manager comparison on GPU; one backend per process."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import torch

from experiments.efficienttam.run_comparison import CASES, ROOT, sha256, json_default
from experiments.sam3_multiplex.adapter import CHECKPOINT, MultiplexAdapter, build_tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sam3", "sam2"), default="sam3")
    parser.add_argument("--case", choices=CASES, default="felix")
    parser.add_argument("--max-frames", type=int, default=141)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--capture-masks", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.max_frames >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    stem, detection_file, team_file = CASES[args.case]
    files = sorted((ROOT / "data/cache/frames" / stem).glob("*.jpg"), key=lambda p: int(p.stem))[:args.max_frames]
    video = ROOT / "data/raw" / f"{stem}.mp4"
    checkpoint = CHECKPOINT if args.backend == "sam3" else ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
    upstream = ROOT / ("sam3" if args.backend == "sam3" else "sam2-upstream")
    sys.path.insert(0, str(upstream))
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking.sam2_driver import drive_sam2
    from scripts.render_raw_team_classification import load_detection_cache, person_detections

    manifest = {
        "backend": args.backend, "case": args.case,
        "protocol": "shared_production_manager_periodic_RFDETR_boxes",
        "frames": len(files), "check_every": 10, "checkpoint_policy": "reprompt",
        "precision": "BF16 autocast", "compile": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "upstream_revision": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha256(__file__),
        "adapter_sha256": sha256(Path(__file__).with_name("adapter.py")),
        "driver_sha256": sha256(ROOT / "src/handball_cv/tracking/sam2_driver.py"),
        "detections_sha256": sha256(ROOT / detection_file),
        "team_model_sha256": sha256(ROOT / team_file),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "capture_masks": args.capture_masks,
        "warmup_frames": args.warmup_frames,
        "scope": "tracking-only; cached RF-DETR and historical team-model reID; no OCR/render/live detection",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = perf_counter()
    torch.manual_seed(0)
    if args.backend == "sam3":
        model, loading = build_tracker()
        predictor = MultiplexAdapter(model)
        manifest["strict_loading"] = loading
        manifest["prompt_conversion"] = "trained SAM3 interactive box head -> batched binary mask seeds/reconditioning"
    else:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path=None,
            hydra_overrides_extra=["++model.compile_image_encoder=false"],
            device="cuda", vos_optimized=False,
        )
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        predictor.load_state_dict(weights, strict=True)
        del weights
    torch.cuda.synchronize()
    manifest["model_build_seconds"] = perf_counter() - started
    cache = load_detection_cache(ROOT / detection_file)
    team = TeamModel.load(ROOT / team_file, device="cuda")
    lookup = lambda fid: person_detections(cache, fid)
    factory = lambda _: predictor

    def drive(directory):
        return drive_sam2(video, lookup, team, str(checkpoint),
                          frame_cache_dir=directory, predictor_factory=factory,
                          desc=f"{args.backend} {args.case}")

    if args.warmup_frames:
        with tempfile.TemporaryDirectory(prefix="sam31-warm-") as temporary:
            for path in files[:args.warmup_frames]:
                (Path(temporary) / path.name).symlink_to(path.resolve())
            manager, seeds, frames = drive(Path(temporary))
            for result in frames:
                pass
            del manager, seeds, frames, result
        if args.backend == "sam3":
            del predictor.last_state
        gc.collect()
        torch.cuda.empty_cache()

    if args.capture_masks:
        (args.output / "masks").mkdir()
    rows, stats = [], []
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="sam31-quality-") as temporary:
        for path in files:
            (Path(temporary) / path.name).symlink_to(path.resolve())
        started = perf_counter()
        manager, seeds, frames = drive(Path(temporary))
        torch.cuda.synchronize()
        manifest["fresh_state_seed_seconds"] = perf_counter() - started
        rows.extend((0, int(oid), box.copy()) for oid, box in seeds.items())
        started = previous = perf_counter()
        for result in frames:
            now = perf_counter()
            row = {"frame": result.frame_idx, "objects": len(result.player_ids),
                   "resume_seconds": now - previous,
                   "nonempty_targets": int(result.masks.any(axis=(1, 2)).sum())}
            if args.backend == "sam3":
                row["buckets"] = predictor.bucket_count(predictor.last_state)
                row["groups"] = len(predictor.last_state["groups"])
            stats.append(row)
            rows.extend((result.frame_idx, int(oid), box.copy())
                        for oid, box in zip(result.player_ids, result.boxes))
            if args.capture_masks and 90 <= result.frame_idx <= 140:
                np.savez_compressed(args.output / "masks" / f"{result.frame_idx:05d}.npz",
                                    ids=result.player_ids, masks=np.packbits(result.masks, axis=-1),
                                    width=result.masks.shape[-1])
            previous = perf_counter()
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
    manifest.update({
        "loop_seconds": elapsed, "loop_fps": len(stats) / elapsed,
        "unique_propagated_frames": len(stats),
        "resume_latency_ms_median": float(np.median([s["resume_seconds"] for s in stats]) * 1000),
        "resume_latency_ms_p95": float(np.percentile([s["resume_seconds"] for s in stats], 95) * 1000),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "events": len(manager.events),
        "max_buckets": max((s.get("buckets", 0) for s in stats), default=0),
    })
    np.savez_compressed(args.output / "replay.npz",
                        frame_index=np.array([r[0] for r in rows], dtype=int),
                        tracker_id=np.array([r[1] for r in rows], dtype=int),
                        boxes=np.array([r[2] for r in rows], dtype=float).reshape(-1, 4), source=str(video))
    (args.output / "events.json").write_text(json.dumps(manager.events, indent=2, default=json_default))
    (args.output / "frames.json").write_text(json.dumps(stats, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
