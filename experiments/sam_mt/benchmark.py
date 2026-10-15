"""Compare native SAM-MT and SAM2 propagation with identical detector-derived points.

No periodic corrections, entrants, retirement or re-ID: this isolates the models.
Run each arm in a fresh process because both source trees import as `sam2`.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import warnings

import numpy as np
import torch

from experiments.efficienttam.run_comparison import CASES, ROOT, sha256
from experiments.sam_mt.check_setup import CHECKPOINT, CHECKPOINT_SHA256, CONFIG, REVISION


def gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or result.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sam_mt", "sam2"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--capture-masks", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    upstream = ROOT / ("SAM-MT" if args.backend == "sam_mt" else "sam2-upstream")
    config = CONFIG if args.backend == "sam_mt" else "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = CHECKPOINT if args.backend == "sam_mt" else ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if args.backend == "sam_mt":
        assert revision == REVISION and sha256(checkpoint) == CHECKPOINT_SHA256
    sys.path.insert(0, str(upstream))
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
    from handball_cv.tracking.sam2_driver import masks_from_logits
    import supervision as sv

    assert Path(sam2.__file__).resolve().is_relative_to(upstream.resolve())
    assert torch.cuda.is_available(), "GPU required; no CPU fallback"
    stem, detection_file, _ = CASES[args.case]
    files = sorted((ROOT / "data/cache/frames" / stem).glob("*.jpg"), key=lambda p: int(p.stem))
    if args.max_frames is not None:
        files = files[:args.max_frames]
    assert len(files) > 21
    with np.load(ROOT / detection_file, allow_pickle=False) as cache:
        start, end = map(int, cache["offsets"][:2])
        boxes = cache["boxes"][start:end]
        assert set(cache["class_id"][start:end]) <= {1, 2}
    # Two positive clicks per target, on the vertical centreline of each box.
    points = np.stack([
        np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                         boxes[:, 1] + fraction * (boxes[:, 3] - boxes[:, 1])))
        for fraction in (0.30, 0.50)
    ], axis=1).astype(np.float32)
    ids = np.arange(1, len(boxes) + 1)
    points_per_object = [2] * len(boxes)
    torch.manual_seed(0)
    manifest = {
        "backend": args.backend, "case": args.case, "protocol": "native_seed_once_two_points",
        "upstream_revision": revision, "runner_sha256": sha256(__file__),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "config": config, "detections_sha256": sha256(ROOT / detection_file),
        "frames": len(files), "seed_targets": len(ids), "points": points.tolist(),
        "precision": "BF16 autocast", "compile_image_encoder": False,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "postprocessing": "upstream default plus shared edge-fragment filter at 0.03",
        "capture_masks": args.capture_masks, "gpu_processes_before": gpu_processes(),
        "scope": "native propagation, original-resolution CPU masks and boxes; no checkpoint manager or live detection",
    }
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")
        torch.cuda.synchronize()
        started = perf_counter()
        predictor = build_sam2_video_predictor(
            config, ckpt_path=None, device="cuda", vos_optimized=False,
            hydra_overrides_extra=["++model.compile_image_encoder=false"],
        )
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        predictor.load_state_dict(weights, strict=True)
        del weights
        torch.cuda.synchronize()
        manifest["model_build_seconds"] = perf_counter() - started
        assert all(p.device.type == "cuda" for p in predictor.parameters())

        def run(frame_dir, collect):
            rows, stats, retained = [], [], []
            torch.cuda.synchronize()
            started = perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(str(frame_dir), offload_video_to_cpu=False, offload_state_to_cpu=False)
                assert state["images"].is_cuda and state["storage_device"].type == "cuda"
                if args.backend == "sam_mt":
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=1,
                        points=torch.as_tensor(points.reshape(1, -1, 2), device="cuda"),
                        labels=torch.ones((1, 2 * len(ids)), dtype=torch.int32, device="cuda"),
                        points_per_object=points_per_object,
                    )
                    iterator = predictor.propagate_in_video(state, points_per_object=points_per_object)
                else:
                    for oid, clicks in zip(ids, points):
                        predictor.add_new_points_or_box(
                            state, frame_idx=0, obj_id=int(oid), points=clicks,
                            labels=np.ones(2, dtype=np.int32),
                        )
                    iterator = predictor.propagate_in_video(state)

                def consume(output):
                    fid = int(output[0])
                    if args.backend == "sam_mt":
                        logits = output[3][0].unsqueeze(1)
                        output_ids = ids
                    else:
                        logits = output[2]
                        output_ids = np.asarray(output[1], dtype=int)
                    assert logits.is_cuda and len(logits) == len(ids)
                    masks = masks_from_logits(logits)
                    xyxy = sv.mask_to_xyxy(masks=masks)
                    if collect:
                        rows.extend((fid, int(oid), box) for oid, box in zip(output_ids, xyxy))
                        stats.append({"frame": fid, "nonempty_targets": int(np.count_nonzero(masks.any(axis=(1, 2))))})
                        if args.capture_masks and 95 <= fid <= 130:
                            retained.append((fid, output_ids.copy(), masks))

                first = next(iterator)
                assert first[0] == 0
                consume(first)
                torch.cuda.synchronize()
                setup_seconds = perf_counter() - started
                started = perf_counter()
                count = 0
                for output in iterator:
                    count += 1
                    consume(output)
                torch.cuda.synchronize()
                loop_seconds = perf_counter() - started
            return rows, stats, retained, setup_seconds, loop_seconds, count

        with tempfile.TemporaryDirectory(prefix="sam-mt-warm-") as warm:
            for path in files[:21]:
                (Path(warm) / path.name).symlink_to(path.resolve())
            run(Path(warm), False)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix="sam-mt-measured-") as measured:
            for path in files:
                (Path(measured) / path.name).symlink_to(path.resolve())
            rows, stats, retained, setup, elapsed, count = run(Path(measured), True)
        assert count == len(files) - 1
        manifest.update({
            "fresh_state_seed_seconds": setup, "loop_seconds": elapsed,
            "unique_propagated_frames": count, "loop_fps": count / elapsed,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "warnings": sorted({str(w.message) for w in captured}),
            "gpu_processes_after": gpu_processes(),
        })
    np.savez_compressed(args.output / "replay.npz",
                        frame_index=np.array([r[0] for r in rows], dtype=int),
                        tracker_id=np.array([r[1] for r in rows], dtype=int),
                        boxes=np.array([r[2] for r in rows], dtype=float).reshape(-1, 4),
                        source=str(ROOT / "data/raw" / f"{stem}.mp4"))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "events.json").write_text("[]\n")
    (args.output / "frames.json").write_text(json.dumps(stats))
    if retained:
        (args.output / "masks").mkdir()
        for fid, output_ids, masks in retained:
            np.savez_compressed(args.output / "masks" / f"{fid:05d}.npz",
                                ids=output_ids, masks=np.packbits(masks, axis=-1), width=masks.shape[-1])
    print(json.dumps({k: manifest[k] for k in ("backend", "case", "seed_targets", "frames", "loop_fps", "loop_seconds")}, indent=2))


if __name__ == "__main__":
    main()
