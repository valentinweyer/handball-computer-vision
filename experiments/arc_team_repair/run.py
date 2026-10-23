"""CPU accuracy test of optional retrospective ARC team-overlap repair.

python -m experiments.arc_team_repair.run --team-overlap-repair --root RUN_DIR
Without the flag the exact original MOT files are copied, with no team model
loaded. The native appearance-only offline merger is deliberately not used.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch

from experiments.arc_sort_cpu.run import frames, sha, MotReplay, metrics, module
from experiments.arc_team_repair.repair import repair_tracks, RepairConfig, parse_lines
from experiments.box_tracker_oracle.common import MAIN, OUT, Clip
from scripts.evaluate_tracker_identity import iou_matrix, run_tracker
from scripts.render_raw_team_classification import load_detection_cache, person_detections
from handball_cv.teams.model import (
    TeamModel, crop_quality, jersey_color_features, torso_contamination,
    MIN_TEAM_VOTE_CONFIDENCE,
)

PREVIOUS = MAIN / "runs/arc-sort-cpu-2026-10-21"
MODELS = {"felix": "score_FelixClaar", "han": "score_Han-Ber4_cached",
          "bhc": "score_BHC-FAG_window_cached"}
READ_POLICY = {"confidence": float(MIN_TEAM_VOTE_CONFIDENCE), "quality": 0.30,
               "contamination_max": 0.05, "other_box_iou_max": 0.10,
               "match_iou_min": 0.50, "match_margin_min": 0.10,
               "field_player_class": 2,
               "backend": "existing per-video TeamModel high-confidence colour branch"}


def observations(clip, source_lines, output):
    model_path = MAIN / "runs/bundles" / MODELS[clip.name] / "team_model.pkl"
    cache_path = OUT / "detections/detector-expanded-v1" / f"{clip.name}.npz"
    cache = load_detection_cache(cache_path)
    key = {"model_sha256": sha(model_path), "detections_sha256": sha(cache_path),
           "video_sha256": sha(clip.video), "policy": READ_POLICY,
           "source_sha256": sha(PREVIOUS / "arc_osnet_localconf" / f"{clip.name}.txt")}
    dest = output / f"{clip.name}_team_reads.json"
    if dest.exists():
        saved = json.loads(dest.read_text())
        assert saved["inputs"] == key, "team read cache provenance changed"
        return {(r["frame"], r["track"]): r["team"] for r in saved["reads"]}, saved
    # Reuse the production colour reducers/prototypes, without constructing
    # a SigLIP network. Ambiguous colour receives no qualified observation.
    model = TeamModel.load(model_path, classifier=SimpleNamespace(), device="cpu")
    paths = parse_lines(source_lines)
    reads = []
    counts = Counter()
    decoded = 0
    for f, bgr in enumerate(frames(clip.video)):
        decoded += 1
        det = person_detections(cache, f)
        boxes = det.xyxy
        if not len(boxes):
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crops = model._crop_boxes(rgb, boxes)
        teams, confidences = model._color_prediction(jersey_color_features(crops))
        quality = [crop_quality(c) for c in crops]
        contamination = torso_contamination(boxes)
        overlap = iou_matrix(boxes, boxes)
        np.fill_diagonal(overlap, 0.0)
        clean = np.array([
            int(det.class_id[j]) == 2 and confidences[j] >= READ_POLICY["confidence"]
            and quality[j].accepted and quality[j].score >= READ_POLICY["quality"]
            and contamination[j] <= READ_POLICY["contamination_max"]
            and overlap[j].max() < READ_POLICY["other_box_iou_max"]
            for j in range(len(boxes))
        ])
        tids = sorted(paths.get(f, {}))
        if not tids:
            continue
        pairs = iou_matrix(np.stack([paths[f][t] for t in tids]), boxes)
        for row, tid in enumerate(tids):
            counts["track_rows"] += 1
            col = int(pairs[row].argmax())
            if not clean[col]:
                counts["unclear_detection"] += 1
                continue
            horizontal = np.delete(pairs[row], col)
            vertical = np.delete(pairs[:, col], row)
            rival = max(horizontal.max(initial=0), vertical.max(initial=0))
            if pairs[row, col] < READ_POLICY["match_iou_min"] or pairs[row, col] - rival < READ_POLICY["match_margin_min"]:
                counts["ambiguous_box_match"] += 1
                continue
            reads.append({"frame": f, "track": tid, "team": int(teams[col]),
                          "confidence": float(confidences[col]), "quality": quality[col].score,
                          "detector_index": col})
            counts["trusted"] += 1
        if decoded % 100 == 0:
            print(f"{clip.name}: frame-local CPU team observations {decoded}", flush=True)
    assert decoded == len(cache["offsets"]) - 1
    saved = {"inputs": key, "counts": dict(counts), "reads": reads}
    dest.write_text(json.dumps(saved, indent=2))
    return {(r["frame"], r["track"]): r["team"] for r in reads}, saved


def correctness(matches, keep):
    count = defaultdict(Counter)
    for f, tid, rid in matches:
        if rid in keep:
            count[tid][rid] += 1
    dominant = {t: c.most_common(1)[0][0] for t, c in count.items()}
    return {(f, rid) for f, tid, rid in matches if rid in keep and dominant[tid] == rid}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--team-overlap-repair", action="store_true", default=False)
    ap.add_argument("--clips", nargs="+", default=["felix", "han", "bhc"])
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert not torch.cuda.is_available()
    cfg = RepairConfig()
    protocol = {"repair": asdict(cfg), "read_policy": READ_POLICY,
                "source_sha256": {p: sha(Path(__file__).parent / p)
                                  for p in ("run.py", "repair.py")},
                "baseline": "def6081 / arc_osnet_localconf",
                "enabled": args.team_overlap_repair, "clips": args.clips,
                "detector_stride": 1, "detector_cache_floor": 0.3,
                "native_offline_merge": False, "device": "cpu"}
    lock = root / "protocol.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == protocol, "refuse changed protocol"
    else:
        lock.write_text(json.dumps(protocol, indent=2))
    old_results = json.loads((PREVIOUS / "results.json").read_text())
    interp = module("repair_interpolation", PREVIOUS / "vendor/arcsort/tools/interpolate_tracks.py")
    results = {}
    for name in args.clips:
        clip = Clip(name)
        source = PREVIOUS / "arc_osnet_localconf" / f"{name}.txt"
        source_bytes = source.read_bytes()
        lines = source_bytes.decode().splitlines()
        reads, read_audit = observations(clip, lines, root) if args.team_overlap_repair else ({}, {})
        off, _ = repair_tracks(lines, reads, enabled=False)
        assert off == lines
        assert ("\n".join(off) + "\n").encode() == source_bytes, "off mode not byte-identical"
        frozen_reads = dict(reads)
        selected, audit = repair_tracks(lines, reads, enabled=args.team_overlap_repair, config=cfg)
        # No evidence is changed or learned from repaired identities.
        output = root / ("enabled" if args.team_overlap_repair else "disabled")
        output.mkdir(exist_ok=True)
        (output / f"{name}.txt").write_text("\n".join(selected) + "\n")
        (root / f"{name}_repair_audit.json").write_text(json.dumps(audit, indent=2))
        before_fields = [ln.split(",") for ln in lines]
        after_fields = [ln.split(",") for ln in selected]
        assert all(a[:1] + a[2:] == b[:1] + b[2:] for a, b in zip(before_fields, after_fields))
        variants = {"disabled": off, "selected": selected,
                    "disabled_interp": interp.interpolate_sequence(off, max_gap=10),
                    "selected_interp": interp.interpolate_sequence(selected, max_gap=10)}
        matches = {}
        for variant, rows in variants.items():
            replay = MotReplay(rows)
            m = run_tracker(lambda: replay, clip.reference, clip.video, lambda f: None, "cpu")
            matches[variant] = m
            results[f"{name}/{variant}"] = metrics(m, clip)
            (root / f"{name}_{variant}_matches.json").write_text(json.dumps(m))
        # Existing output must reproduce its original measurements unchanged.
        for stage, previous_key in [("disabled", "arc_osnet_localconf"),
                                    ("disabled_interp", "arc_interp_localconf")]:
            prior = old_results[f"{name}/{previous_key}"]
            for field in ("correct_pct", "wrong_identity_frames", "matched_detections", "id_switches"):
                assert results[f"{name}/{stage}"][field] == prior[field], (name, stage, field)
        paired = {}
        for before, after in [("disabled", "selected"), ("disabled_interp", "selected_interp")]:
            a = correctness(matches[before], clip.keep)
            b = correctness(matches[after], clip.keep)
            paired[after] = {"gained_correct": len(b - a), "lost_correct": len(a - b)}
        assert reads == frozen_reads
        assert source.read_bytes() == source_bytes
        results[f"{name}/repair"] = {"corrections": len(audit["corrections"]),
                                      "changed_rows": audit.get("changed_rows", 0),
                                      "decisions": dict(Counter(e.get("decision") for e in audit["encounters"])),
                                      "handover_decisions": dict(Counter(e.get("decision") for e in audit.get("handovers", []))),
                                      "paired": paired}
        (root / "results.json").write_text(json.dumps(results, indent=2))
        print(name, results[f"{name}/repair"], flush=True)
    assert not torch.cuda.is_initialized()
    (root / "complete.json").write_text(json.dumps({"complete": True, "cuda_initialized": False,
                                                    "default_enabled": False, "clips": args.clips}, indent=2))


if __name__ == "__main__":
    main()
