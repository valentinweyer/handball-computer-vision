"""CPU-only ARC-SORT accuracy screen using stride-1 production detections.

Run with the vendor and isolated dependencies on PYTHONPATH; no installation
into the pipeline environment, no production imports changed, no GPU access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import cv2
import numpy as np
import supervision as sv
import torch
from scipy.optimize import linear_sum_assignment
from loguru import logger
from trackers import SORTTracker

from arcsort import ARCTracker
from arcsort.tracker import load_config
from arcsort.interop import from_supervision, to_supervision
from experiments.box_tracker_oracle.common import Clip, MAIN, OUT, colour_feature
from experiments.box_tracker_oracle.appearance_sort import AppearanceSORT
from experiments.box_tracker_oracle.run import wrong_runs
from scripts.evaluate_tracker_identity import run_tracker, score, Sam2ReplayTracker
from scripts.render_raw_team_classification import load_detection_cache, person_detections


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def frames(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def prepare_model(vendor, weights):
    sys.path.insert(0, str(vendor / "third_party/osnet"))
    from osnet import osnet_x1_0

    # Official Deep-EIoU release. Loading remains on CPU; optimizer is unused.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items() if torch.is_tensor(v)}
    n_classes = state["classifier.weight"].shape[0]
    net = osnet_x1_0(num_classes=n_classes, pretrained=False, loss="softmax").eval()
    net.load_state_dict(state, strict=True)
    return net


def encode(net, frame, boxes):
    batch = []
    height, width = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = np.round(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        patch = frame[y1:y2, x1:x2]
        if not patch.size:
            patch = np.zeros((256, 128, 3), np.uint8)
        else:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = cv2.resize(patch, (128, 256), interpolation=cv2.INTER_LINEAR)
        batch.append(patch.transpose(2, 0, 1).astype(np.float32))
    if not batch:
        return np.empty((0, 512), np.float32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    result = []
    with torch.inference_mode():
        for start in range(0, len(batch), 16):
            x = torch.from_numpy(np.stack(batch[start:start + 16])) / 255.0
            result.append(torch.nn.functional.normalize(net((x - mean) / std), dim=1).numpy())
    return np.concatenate(result)


def cache_features(clip, cache, output, net, fingerprints):
    path = output / f"{clip.name}_osnet.npz"
    meta = path.with_suffix(".json")
    if path.exists():
        if json.loads(meta.read_text()) != fingerprints:
            raise RuntimeError("refusing to reuse feature cache with changed inputs")
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    result = {}
    n = 0
    for f, frame in enumerate(frames(clip.video)):
        det = person_detections(cache, f)
        result[f"boxes_{f + 1}"] = det.xyxy.astype(np.float64)
        result[f"embs_{f + 1}"] = encode(net, frame, det.xyxy)
        n += 1
        if n % 50 == 0:
            print(f"{clip.name}: CPU features {n}/{len(cache['offsets']) - 1}", flush=True)
    assert n == len(cache["offsets"]) - 1
    np.savez_compressed(path, **result)
    meta.write_text(json.dumps(fingerprints, indent=2))
    return result


class ArcAdapter:
    def __init__(self, config, embeddings=None):
        self.tracker = ARCTracker(config)
        self.rows = []
        if embeddings is not None:
            comp = self.tracker.embedding_computer
            comp._emb_dim = 512
            comp._cache = {int(k.split("_")[1]): (v, embeddings[k.replace("boxes_", "embs_")])
                           for k, v in embeddings.items() if k.startswith("boxes_")}
            # Refuse missing-cache fallback: upstream hardcodes CUDA for live ReID.
            def missing(*args, **kwargs):
                raise RuntimeError("CPU feature cache miss; GPU fallback prohibited")
            comp._compute_from_frame = missing

    def update(self, detections, frame=None):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame is not None else None
        tracked = self.tracker.update(from_supervision(detections), bgr)
        f = self.tracker.frame_count
        for x1, y1, x2, y2, tid in tracked:
            self.rows.append(f"{f},{int(tid)},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},1,-1,-1,-1")
        return to_supervision(tracked)


class MotReplay:
    def __init__(self, lines):
        self.by_frame = {}
        for line in lines:
            row = [float(x) for x in line.split(",")]
            f, tid, x, y, w, h = row[:6]
            self.by_frame.setdefault(int(f) - 1, []).append([x, y, x + w, y + h, tid])
        self.cursor = 0

    def update(self, detections, frame=None):
        rows = np.asarray(self.by_frame.get(self.cursor, []), dtype=float).reshape(-1, 5)
        self.cursor += 1
        return to_supervision(rows)


def metrics(matches, clip):
    m = score(matches, clip.keep)
    denominator = sum(sum(i in clip.keep for i in truth) for truth in clip.reference.values())
    m["reference_observations"] = denominator
    m["recall_pct"] = 100 * m["matched_detections"] / denominator
    m["correct_pct"] = 100 * (m["matched_detections"] - m["wrong_identity_frames"]) / denominator
    m["wrong_runs"] = wrong_runs(matches, clip.keep)
    # Additional fragmentation-sensitive count. This is NOT standard IDF1:
    # unlabelled people/spans prevent a complete false-positive denominator.
    kept = [(f, t, r) for f, t, r in matches if r in clip.keep]
    tids = sorted({t for _, t, _ in kept})
    rids = sorted(clip.keep)
    counts = Counter((t, r) for _, t, r in kept)
    matrix = np.array([[counts[t, r] for r in rids] for t in tids])
    one_to_one = int(matrix[linear_sum_assignment(-matrix)].sum()) if matrix.size else 0
    m["one_to_one_correct_pct"] = 100 * one_to_one / denominator
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=["felix", "han", "bhc"])
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    vendor = root / "vendor/arcsort"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert not torch.cuda.is_available()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    config = load_config(vendor / "arcsort/configs/sportsmot.yaml")
    weight_sha = sha(root / "sports_model.pth.tar-60")
    # Lock before any evaluation. Defaults, not fitted on the test clips.
    lock = {"upstream_commit": "2a580c592c838bce99b89a6a2dbd7ad6f50c84be",
            "device": "cpu", "precision": "float32", "detector_stride": 1,
            "detector": "detector-expanded-v1", "detector_cache_floor": 0.3,
            "person_classes": [1, 2], "config": config,
            "weight_sha256": weight_sha, "clips": args.clips,
            "offline": {"max_gap": 300, "merge_thresh": 0.85, "motion_tol": "inf",
                        "split": True, "split_thresh": 0.45, "split_persist": 5},
            "interpolation_max_gap": 10, "baselines_activation": 0.5,
            "secondary_local_confidence": {"track_high_thresh": 0.5, "new_track_thresh": 0.5},
            "colour_gate": 0.28,
            "note": "Existing cache floor 0.3 truncates BYTE's 0.1-0.3 low-score range. No tuning."}
    lock_path = root / "protocol.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError("protocol already locked differently")
    lock_path.write_text(json.dumps(lock, indent=2))
    net = prepare_model(vendor, root / "sports_model.pth.tar-60")
    interp = module("arc_interpolation", vendor / "tools/interpolate_tracks.py")
    merge = module("arc_merge", vendor / "tools/merge_tracklets.py")
    results = {}
    manifest = {}
    for name in args.clips:
        clip = Clip(name)
        path = OUT / "detections/detector-expanded-v1" / f"{name}.npz"
        cache = load_detection_cache(path)
        n_frames = int(sv.VideoInfo.from_video_path(str(clip.video)).total_frames)
        offsets = cache["offsets"]
        # A sparse scan cache does not qualify even if interpolated later.
        assert len(offsets) == n_frames + 1, (name, len(offsets), n_frames)
        assert np.all(np.diff(offsets) > 0), f"check unscanned/empty frame slots in {name}"
        fp = {"video_sha256": sha(clip.video), "detections_sha256": sha(path),
              "weights_sha256": weight_sha, "device": "cpu", "dtype": "float32"}
        manifest[name] = {**fp, "video": str(clip.video), "cache": str(path),
                          "frames": n_frames, "frames_with_detections": int(sum(np.diff(offsets) > 0)),
                          "minimum_confidence": float(cache["confidence"].min()),
                          "person_detections": sum(len(person_detections(cache, f)) for f in range(n_frames)),
                          "reference_sha256": hashlib.sha256("".join(
                              sha(p) for p in sorted(clip.reference_dir.glob("*.npz"))).encode()).hexdigest()}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        emb = cache_features(clip, cache, root, net, fp)
        det_for = lambda f: person_detections(cache, f)
        colours = []
        for f, frame in enumerate(frames(clip.video)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = det_for(f).xyxy
            colours.append(np.stack([colour_feature(rgb, b) for b in boxes]) if len(boxes)
                           else np.empty((0, 128), np.float32))
        variants = {
            "sort": SORTTracker(frame_rate=clip.fps, lost_track_buffer=30, track_activation_threshold=0.5),
            "sort_strict": SORTTracker(frame_rate=clip.fps, lost_track_buffer=10,
                                       minimum_iou_threshold=0.5, track_activation_threshold=0.5),
            "colour_gate": AppearanceSORT(colours, "gate", 0.28, frame_rate=clip.fps,
                                          lost_track_buffer=30, track_activation_threshold=0.5),
            "arc_motion": ArcAdapter({**config, "use_reid": False, "use_gated_reid": False}),
            "arc_osnet": ArcAdapter(config, emb),
            "arc_osnet_localconf": ArcAdapter({**config, "track_high_thresh": 0.5,
                                               "new_track_thresh": 0.5}, emb),
            "sam2_stored": Sam2ReplayTracker(MAIN / "runs/sam2_reprompt" /
                                             f"{clip.video.stem}_policy_reprompt.npz"),
        }
        for variant, tracker in variants.items():
            matches = run_tracker(lambda: tracker, clip.reference, clip.video, det_for, "cpu")
            m = metrics(matches, clip)
            results[f"{name}/{variant}"] = m
            (root / f"{name}_{variant}_matches.json").write_text(json.dumps(matches))
            print(f"{name}/{variant}: correct={m['correct_pct']:.2f}% "
                  f"wrong={m['wrong_identity_pct']:.2f}% frags={m['mean_fragments_per_reference_id']:.2f}", flush=True)
            if variant.startswith("arc_"):
                out = root / variant
                out.mkdir(exist_ok=True)
                (out / f"{name}.txt").write_text("\n".join(tracker.rows) + "\n")
                (out / f"{name}_arbitration.json").write_text(json.dumps(tracker.tracker.arbitration_swaps, indent=2))
            if variant.startswith("arc_osnet"):
                lines = interp.interpolate_sequence(tracker.rows, max_gap=10)
                interp_stage = "arc_interp" if variant == "arc_osnet" else "arc_interp_localconf"
                final_stage = "arc_final" if variant == "arc_osnet" else "arc_final_localconf"
                p = root / interp_stage / f"{name}.txt"
                p.parent.mkdir(exist_ok=True)
                p.write_text("\n".join(lines) + "\n")
                final = root / final_stage / f"{name}.txt"
                ma = argparse.Namespace(split=True, split_thresh=0.45, split_persist=5,
                                        merge_thresh=0.85, max_gap=300, motion_tol=float("inf"))
                merge.process_sequence(p, root / f"{name}_osnet.npz", final, ma)
                for stage, stage_lines in [(interp_stage, lines), (final_stage, final.read_text().splitlines())]:
                    replay = MotReplay(stage_lines)
                    sm = run_tracker(lambda: replay, clip.reference, clip.video, det_for, "cpu")
                    results[f"{name}/{stage}"] = metrics(sm, clip)
                    (root / f"{name}_{stage}_matches.json").write_text(json.dumps(sm))
                    print(f"{name}/{stage}: {json.dumps(results[f'{name}/{stage}'])}", flush=True)
            (root / "results.json").write_text(json.dumps(results, indent=2))
        # Input cache is read-only throughout; verify content has not changed.
        assert sha(path) == fp["detections_sha256"]
    assert not torch.cuda.is_initialized()
    (root / "complete.json").write_text(json.dumps({"complete": True, "cuda_initialized": False,
                                                     "cases": len(results), "pid": os.getpid()}, indent=2))


if __name__ == "__main__":
    main()
