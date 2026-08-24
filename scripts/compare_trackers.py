"""Score tracker runs on fragmentation *and* identity confusion.

Both pipelines already log fragmentation (a track dies, a new one is born, the
re-ID gallery maybe stitches them). Neither logs the failure that actually
matters in a scrum: a track that stays alive but slides onto a different player.
That is invisible in the event log and nearly invisible by eye, because the
overlay hides the jersey.

This scores it directly. For each track column, crops are re-embedded with the
same SigLIP model the pipelines use for team classification, sampled every
`--stride` frames. A track that stays on one player keeps a high cosine
similarity between consecutive samples; a sudden collapse means the box changed
occupant. That is a proxy, not ground truth -- a player turning away or being
occluded also drops similarity -- so treat the count as comparative between
runs on the same clip, not as an absolute swap count.

Usage:
    python compare_trackers.py sam2 mcbyte
    python compare_trackers.py sam2 mcbyte --stride 3 --swap-threshold 0.80
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))

import cv2
import numpy as np

import supervision as sv
from sports import TeamClassifier

from handball_cv.teams.model import MIN_TEAM_VOTE_CONFIDENCE, TeamModel, torso_boxes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path(os.getenv("HANDBALL_CV_RUN_DIR", PROJECT_ROOT / "runs"))


def load_run(name: str) -> dict:
    with np.load(RUN_DIR / f"{name}.npz") as d:
        return {k: d[k] for k in d.files}


def team_cache_path(source_video: str) -> Path:
    p = Path(source_video)
    return p.parent / f".{p.stem}_team.pkl"


def labels_path(source_video: str) -> Path:
    p = Path(source_video)
    return p.parent / f".{p.stem}_team_labels.json"


def load_labels(source_video: str):
    """(reference_run_name, {(frame_idx, col): "HAN"|"BER"|"GK"}) or (None, {})
    if no label file exists yet. Ground truth from team_labels.py; SKIP entries
    are dropped."""
    path = labels_path(source_video)
    if not path.exists():
        return None, {}
    data = json.loads(path.read_text())
    labels = {}
    for key, value in data["labels"].items():
        if value == "SKIP":
            continue
        frame_str, col_str = key.split("_")
        labels[(int(frame_str), int(col_str))] = value
    return data["reference_run"], labels


def sample_embeddings(run: dict, classifier, stride: int, team_model=None):
    """(series, team_series) sampled every `stride` frames for every valid box.

    series: {column: [(frame_idx, embedding), ...]} -- for the swap-detection
    cosine metric, same as before.
    team_series: {column: [(frame_idx, predicted_team, confidence), ...]} --
    only populated when `team_model` is given; used for the team-purity metric.
    """
    boxes = run["boxes"]                       # (T, P, 4); row t == source frame t+1
    source = str(run["source"])
    wanted = set(range(0, len(boxes), stride))

    series: dict = {}
    team_series: dict = {}

    for t, frame in enumerate(sv.get_video_frames_generator(source)):
        idx = t - 1                            # source frame 0 is not in `boxes`
        if idx < 0 or idx not in wanted or idx >= len(boxes):
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        row = boxes[idx]
        valid = np.isfinite(row).all(axis=1)
        cols = np.nonzero(valid)[0]
        if len(cols) == 0:
            continue

        torso_xyxy = torso_boxes(row[cols])  # same crop the pipeline itself embeds/classifies on
        crops = [sv.crop_image(frame_rgb, b) for b in torso_xyxy]
        keep = [c.size > 0 for c in crops]
        cols_ok = cols[keep]
        crops_ok = [c for c, ok in zip(crops, keep) if ok]
        if not crops_ok:
            continue

        raw_embeddings = classifier.extract_features(crops_ok)  # one SigLIP pass
        norm_embeddings = raw_embeddings / (np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-8)
        for col, emb in zip(cols_ok, norm_embeddings):
            series.setdefault(int(col), []).append((idx, emb))

        if team_model is not None:
            # reuses raw_embeddings instead of team_model re-cropping/re-extracting
            teams, conf = team_model.predict_crops(crops_ok, embeddings=raw_embeddings)
            for col, team, c in zip(cols_ok, teams, conf):
                team_series.setdefault(int(col), []).append((idx, int(team), float(c)))

    return series, team_series


def team_purity(run: dict, team_series: dict) -> float:
    """% of confident per-frame team predictions agreeing with the run's final
    (dumped) team label for that column. Low-confidence predictions are
    excluded -- they're not what the pipeline would have gated on either.
    """
    if not team_series:
        return float("nan")
    teams_final = run["teams"]
    agree, total = 0, 0
    for col, samples in team_series.items():
        final = int(teams_final[col])
        for _idx, team, conf in samples:
            if conf < MIN_TEAM_VOTE_CONFIDENCE:
                continue
            total += 1
            agree += int(team == final)
    return 100.0 * agree / total if total else float("nan")


def team_accuracy(run: dict, run_name: str, team_model, labels_ref: tuple) -> dict:
    """Best-mapping accuracy of `team_model`'s predictions against ground-truth
    labels, scored at the exact (frame_idx, col) pairs the labels were made on.

    `team_model` predicts arbitrary cluster ids (0/1) with no inherent name --
    unlike `team_purity`, which only checks self-agreement, this compares
    against real HAN/BER labels, so the cluster->name mapping that maximises
    agreement is picked first (standard "best-map" scoring for unsupervised
    clusters against ground truth) and reused to report a single number.
    GK-labelled points are scored separately (predicted cluster + confidence,
    no accuracy penalty) since GK isn't one of the model's two classes.

    `labels_ref` is `(reference_run_name, labels_dict)` from `load_labels`.
    Labels are keyed by (frame_idx, col) from ONE specific reference run's
    column ordering -- scoring them against a *different* run's `boxes` would
    silently compare each label to whatever different player happens to share
    that column index. Refuses rather than doing that quietly.
    """
    reference_run_name, labels = labels_ref
    if not labels:
        return {"team_accuracy_%": float("nan"), "labels_scored": 0}
    if run_name != reference_run_name:
        print(f"note: labels were made on run {reference_run_name!r}, not {run_name!r} "
              "-- skipping team_accuracy_% (column indices would not line up)")
        return {"team_accuracy_%": float("nan"), "labels_scored": 0}

    source = str(run["source"])
    boxes = run["boxes"]

    by_frame: dict = {}
    for (fid, col), gt in labels.items():
        by_frame.setdefault(fid, []).append((col, gt))

    predictions = []  # (gt_label, predicted_cluster, confidence)
    for t, frame_bgr in enumerate(sv.get_video_frames_generator(source)):
        idx = t - 1
        if idx not in by_frame:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pairs = by_frame[idx]
        cols = np.array([c for c, _ in pairs])
        gts = [g for _, g in pairs]
        row_boxes = boxes[idx][cols]
        finite = np.isfinite(row_boxes).all(axis=1)
        if not finite.all():
            row_boxes, gts = row_boxes[finite], [g for g, f in zip(gts, finite) if f]
        if len(row_boxes) == 0:
            continue
        teams, conf = team_model.predict(frame_rgb, row_boxes)
        predictions.extend(zip(gts, teams.tolist(), conf.tolist()))

    non_gk = [(gt, team, c) for gt, team, c in predictions if gt != "GK"]
    gk_conf = [c for gt, _, c in predictions if gt == "GK"]

    if not non_gk:
        return {"team_accuracy_%": float("nan"), "labels_scored": len(predictions)}

    best_acc, best_map = -1.0, None
    for mapping in ({0: "HAN", 1: "BER"}, {0: "BER", 1: "HAN"}):
        acc = sum(1 for gt, team, _ in non_gk if mapping[team] == gt) / len(non_gk)
        if acc > best_acc:
            best_acc, best_map = acc, mapping

    return {
        "team_accuracy_%": 100.0 * best_acc,
        "labels_scored": len(non_gk),
        "_cluster_map": best_map,
        "_gk_mean_confidence": float(np.mean(gk_conf)) if gk_conf else float("nan"),
    }


def score(
    run: dict, run_name: str, series: dict, stride: int, threshold: float,
    team_series: dict = None, team_model=None, labels_ref: tuple = None,
) -> dict:
    boxes = run["boxes"]
    track_ids = run["track_ids"]
    T, P = boxes.shape[0], boxes.shape[1]

    presence = np.isfinite(boxes).all(axis=2)  # (T, P)

    holes = 0
    for col in range(P):
        seen = np.nonzero(presence[:, col])[0]
        if len(seen) > 1:
            # frames missing strictly between a column's first and last sighting
            holes += (seen[-1] - seen[0] + 1) - len(seen)

    sims, drops = [], []
    for col, samples in series.items():
        for (fa, ea), (fb, eb) in zip(samples, samples[1:]):
            if fb - fa > stride * 2:           # a real gap, not a swap
                continue
            s = float(np.dot(ea, eb))
            sims.append(s)
            if s < threshold:
                drops.append((fb + 1, int(track_ids[col]), s))

    drops.sort(key=lambda d: d[2])
    result = {
        "tracks": P,
        "mean_presence_%": 100.0 * presence.sum() / (T * P) if P else 0.0,
        "gap_frames": holes,
        "samples_compared": len(sims),
        "mean_cosine": float(np.mean(sims)) if sims else float("nan"),
        "p05_cosine": float(np.percentile(sims, 5)) if sims else float("nan"),
        f"suspected_swaps(<{threshold})": len(drops),
    }
    if team_series is not None:
        result["team_purity_%"] = team_purity(run, team_series)
    if team_model is not None and labels_ref is not None:
        result.update(team_accuracy(run, run_name, team_model, labels_ref))
    result["_worst"] = drops[:8]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names dumped under source/.runs")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--swap-threshold", type=float, default=0.85)
    args = ap.parse_args()

    classifier = TeamClassifier(device="cuda")  # extract_features only; no fit needed

    first_run = load_run(args.runs[0])
    source = str(first_run["source"])
    team_cache = team_cache_path(source)
    team_model = TeamModel.load(team_cache, device="cuda") if team_cache.exists() else None
    if team_model is None:
        print(f"note: no cached TeamModel at {team_cache} -- skipping team_purity_%/team_accuracy_%")

    labels_ref = load_labels(source)
    if not labels_ref[1]:
        print(f"note: no labels at {labels_path(source)} -- skipping team_accuracy_% "
              "(run: python team_labels.py sample / save)")

    results = {}
    for name in args.runs:
        run = load_run(name)
        series, team_series = sample_embeddings(run, classifier, args.stride, team_model)
        results[name] = score(run, name, series, args.stride, args.swap_threshold,
                               team_series, team_model, labels_ref)

    keys = [k for k in next(iter(results.values())) if not k.startswith("_")]
    width = max(len(k) for k in keys)
    print(f"\n{'metric'.ljust(width)}  " + "  ".join(n.rjust(12) for n in args.runs))
    print("-" * (width + 2 + 14 * len(args.runs)))
    for k in keys:
        cells = []
        for n in args.runs:
            v = results[n][k]
            cells.append(f"{v:12.2f}" if isinstance(v, float) else f"{v:>12}")
        print(f"{k.ljust(width)}  " + "  ".join(cells))

    for name in args.runs:
        worst = results[name]["_worst"]
        if worst:
            print(f"\nlowest-similarity transitions in {name} (frame, track_id, cosine):")
            for frame, tid, s in worst:
                print(f"  frame {frame:>4}  track {tid:>3}  cos={s:.3f}")
        cluster_map = results[name].get("_cluster_map")
        if cluster_map is not None:
            gk_conf = results[name]["_gk_mean_confidence"]
            print(f"\n{name}: cluster->name map {cluster_map}, "
                  f"GK mean confidence {gk_conf:.3f} (excluded from accuracy, not the model's class)")


if __name__ == "__main__":
    main()
