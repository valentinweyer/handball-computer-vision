"""Score explicit experiment replays with the unchanged historical matcher.

Additional timelines expose missing runs and ID fragmentation, including gaps
under a reused ID. These box/reference diagnostics do not prove identity during
complete occlusion or replace visual review of same-team overlap events.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from experiments.efficienttam.run_comparison import CASES, ROOT, sha256

REFERENCES = {
    "felix": (".FelixClaar_ref_masks", "FelixClaar_reference", []),
    "han": (".Han-Ber4_sam2_masks", "HanBer4_sam2_reference", [12, 14]),
    "bhc": (".BHC-FAG_window_ref_masks", "BHC-FAG_window_reference", []),
}


def continuity(reference, matches, keep):
    """Constant-ID match runs, not an assertion that every such run is correct."""
    matched = {(frame, rid): tid for frame, tid, rid in matches if rid in keep}
    result = {}
    for rid in sorted(keep):
        timeline = [[frame, matched.get((frame, rid))]
                    for frame, truth in sorted(reference.items()) if rid in truth]
        runs = []
        transitions = []
        previous_matched = None
        previous_frame = None
        for frame, tid in timeline:
            if previous_frame is not None and frame != previous_frame+1:
                previous_matched = None  # do not bridge untrusted reference gaps
            if tid is not None:
                if previous_matched is not None and previous_matched[1] != tid:
                    transitions.append({"frame": frame, "previous_frame": previous_matched[0],
                                        "from": previous_matched[1], "to": tid})
                previous_matched = (frame, tid)
            if runs and runs[-1]["id"] == tid and runs[-1]["end"] == frame-1:
                runs[-1]["end"] = frame
            else:
                runs.append({"start": frame, "end": frame, "id": tid})
            previous_frame = frame
        missing = [r for r in runs if r["id"] is None]
        present = [r for r in runs if r["id"] is not None]
        result[rid] = {
            "reference_frames": len(timeline),
            "missing_frames": sum(r["end"]-r["start"]+1 for r in missing),
            "missing_runs": missing,
            "longest_constant_id_match_run": max((r["end"]-r["start"]+1 for r in present), default=0),
            "constant_id_match_runs": len(present),
            "predicted_id_transitions": transitions,
            "timeline": timeline,
        }
    return result


def handoffs(matches, keep, teams):
    by_track = defaultdict(list)
    for frame, tid, rid in sorted(matches):
        if rid in keep:
            by_track[tid].append((frame, rid))
    result = []
    for tid, entries in by_track.items():
        for before, after in zip(entries, entries[1:]):
            if before[1] == after[1]:
                continue
            a, b = teams.get(str(before[1])), teams.get(str(after[1]))
            result.append({"tracker_id": tid, "previous_frame": before[0], "frame": after[0],
                           "from_reference": before[1], "to_reference": after[1],
                           "same_team": a == b if a in ("A", "B") and b in ("A", "B") else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from scripts.evaluate_tracker_identity import (
        Sam2ReplayTracker, load_reference, parse_labels, run_tracker, score,
    )
    manifest = json.loads((args.run / "manifest.json").read_text())
    case = manifest["case"]
    ref_dir, label_dir, exclude = REFERENCES[case]
    label_path = ROOT / "runs/tracklet_labels" / label_dir / "answers.txt"
    valid = parse_labels(label_path)
    keep = set(valid) - set(exclude)
    reference = load_reference(ROOT / "source" / ref_dir, offset=0, valid=valid)
    stem, detections, _ = CASES[case]
    matches = run_tracker(lambda: Sam2ReplayTracker(args.run / "replay.npz"),
                          reference, ROOT / "data/raw" / f"{stem}.mp4",
                          ROOT / detections, device="cuda")
    metrics = score(matches, keep)
    denominator = sum(sum(rid in keep for rid in truth) for truth in reference.values())
    all_reference = sum(len(truth) for truth in reference.values())
    metrics.update({"reference_detections": denominator,
                    "all_reference_detections": all_reference,
                    "historical_correct_pct": 100*(metrics["matched_detections"]-metrics["wrong_identity_frames"])/all_reference,
                    "recall_pct": 100*metrics["matched_detections"]/denominator,
                    "correct_pct": 100*(metrics["matched_detections"]-metrics["wrong_identity_frames"])/denominator})
    teams = {}
    if case == "han":
        team_labels = json.loads((ROOT / "data/annotations/team/Han-Ber4-tracklets.json").read_text())
        teams = {rid: row["code"] for rid, row in team_labels["tracks"].items()}
    timelines = continuity(reference, matches, keep)
    metrics["missing_frames"] = sum(row["missing_frames"] for row in timelines.values())
    metrics["missing_runs"] = sum(len(row["missing_runs"]) for row in timelines.values())
    metrics["predicted_id_transitions"] = sum(len(row["predicted_id_transitions"]) for row in timelines.values())
    report = {"case": case, "backend": manifest["backend"], "metrics": metrics,
              "reference": str(ROOT / "source" / ref_dir), "labels_sha256": sha256(label_path),
              "valid_spans": valid, "excluded": exclude,
              "handoffs": handoffs(matches, keep, teams), "players": timelines,
              "events": json.loads((args.run / "events.json").read_text())}
    (args.run / "score.json").write_text(json.dumps(report, indent=2))
    (args.run / "matches.json").write_text(json.dumps(matches))
    print(json.dumps({"case": case, "backend": manifest["backend"], **metrics}, indent=2))


if __name__ == "__main__":
    main()
