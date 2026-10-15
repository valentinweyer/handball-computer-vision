"""Score only processed frames using the established identity matcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.efficienttam.run_comparison import CASES, ROOT, sha256
from experiments.efficienttam.score_comparison import REFERENCES, continuity, handoffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from scripts.evaluate_tracker_identity import (
        Sam2ReplayTracker, load_reference, parse_labels, run_tracker, score,
    )

    manifest = json.loads((args.run / "manifest.json").read_text())
    case = manifest["case"]
    frame_count = int(manifest.get("frames", manifest.get("jpeg_count")))
    ref_dir, label_dir, exclude = REFERENCES[case]
    label_path = ROOT / "runs/tracklet_labels" / label_dir / "answers.txt"
    valid = parse_labels(label_path)
    keep = set(valid) - set(exclude)
    reference = load_reference(ROOT / "source" / ref_dir, offset=0, valid=valid)
    reference = {fid: truth for fid, truth in reference.items() if fid < frame_count}
    stem, detections, _ = CASES[case]
    matches = run_tracker(lambda: Sam2ReplayTracker(args.run / "replay.npz"),
                          reference, ROOT / "data/raw" / f"{stem}.mp4",
                          ROOT / detections, device="cuda")
    sections = {}
    players = {}
    windows = [("processed_prefix", 0, frame_count - 1)]
    if case == "felix":
        windows.append(("overlap", 95, min(130, frame_count - 1)))
    for name, lower, upper in windows:
        if upper < lower:
            continue
        truth = {fid: row for fid, row in reference.items() if lower <= fid <= upper}
        rows = [row for row in matches if lower <= row[0] <= upper]
        metrics = score(rows, keep)
        denominator = sum(sum(rid in keep for rid in row) for row in truth.values())
        metrics["reference_detections"] = denominator
        metrics["correct_pct"] = 100 * (metrics["matched_detections"] - metrics["wrong_identity_frames"]) / max(1, denominator)
        metrics["recall_pct"] = 100 * metrics["matched_detections"] / max(1, denominator)
        timelines = continuity(truth, rows, keep)
        metrics["missing_frames"] = sum(row["missing_frames"] for row in timelines.values())
        metrics["predicted_id_transitions"] = sum(len(row["predicted_id_transitions"]) for row in timelines.values())
        metrics["range"] = [lower, upper]
        sections[name] = metrics
        players[name] = timelines
    report = {"backend": manifest["backend"], "case": case, "sections": sections,
              "players": players, "handoffs": handoffs(matches, keep, {}),
              "labels_sha256": sha256(label_path),
              "caveat": "Box-reference matching; visual mask inspection required. Window dominant IDs are local to that window."}
    (args.run / "score.json").write_text(json.dumps(report, indent=2))
    (args.run / "matches.json").write_text(json.dumps(matches))
    print(json.dumps(sections, indent=2))


if __name__ == "__main__":
    main()
