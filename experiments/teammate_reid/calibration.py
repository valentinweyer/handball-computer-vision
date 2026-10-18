"""Leave-one-match-out check of per-broadcast score calibration (development data only).

Stage 1 failed because an absolute cosine floor set on Kiel did not transfer:
the score level moves from broadcast to broadcast. This module checks, before
any new GPU time is spent, whether a rule set on two matches holds on the
third. All three matches (Kiel-Lemgo, Eisenach-Hamburg, Gummersbach-Loewen)
are development data from here on; a rule that survives every rotation is
then locked and tested on fresh windows.

Calibration is label-free and per video. Each query's non-top candidate
scores are mostly teammates who are not the query (impostors). Their median
and MAD over that video's real queries (never the synthetic duplicates, whose
scores are a subset) give the broadcast's teammate-vs-teammate baseline.
Scores become (score - median) / spread and margins margin / spread. This is a
completed-video, transductive statistic: it reads only similarity scores, never
an identity label.

Threshold choice on the training matches: the (score, margin) gate with the
highest coverage that keeps precision >= target and pooled unknown
false-match rate <= 5%, at targets 0.90 (the criterion) and 0.95 (a safety
margin). Both are reported; neither is tuned on a test match.

    python -m experiments.teammate_reid.calibration
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.teammate_reid.evaluation import l2n, operating_point
from experiments.teammate_reid.transfer import CRITERION, point_with_parts, verdict

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
SOURCES = {  # results dir -> matches it contributes
    REPO / "runs/smart-reid-prototype-2026-10-18/results": ("Kiel-Lemgo",),
    REPO / "runs/smart-reid-shadow/stage1": ("Eisenach-Hamburg", "Gummersbach-Loewen"),
}
OUT = REPO / "runs/smart-reid-shadow/calibration"
CONFIGS = [  # (feature, calibrated)
    ("digits/frozen_global", False),       # the Stage 1 primary, for reference
    ("raw/frozen_stripes", False),
    ("raw/frozen_stripes", True),
    ("strict/frozen_stripes", False),
    ("strict/frozen_stripes", True),
]
TARGETS = (0.90, 0.95)
GRID = 60


def embeddings(feats: dict) -> dict:
    out = {}
    for mode in ("raw", "digits", "strict"):
        g = l2n(feats[f"{mode}_global"])
        s = l2n(feats[f"{mode}_stripes"].astype(np.float32))
        out[f"{mode}/frozen_global"] = g
        out[f"{mode}/frozen_stripes"] = l2n(np.concatenate([s.reshape(len(s), -1), g], 1))
    return out


def scored_rows(results: Path, matches: tuple, feature: str) -> list[dict]:
    """Per-query rows with every candidate score kept, for one feature."""
    from experiments.teammate_reid.benchmark import fold_queries, load_reviewed
    from experiments.teammate_reid.evaluation import decide, score_query
    crops, fragments, _ = load_reviewed(results)
    data = np.load(results / "features.npz")
    emb = embeddings({k: data[k] for k in data.files})[feature]
    rows = []
    for video in sorted({f["video"] for f in fragments if f["match"] in matches}):
        match = next(f["match"] for f in fragments if f["video"] == video)
        for q in fold_queries(fragments, crops, {video}):
            scores = score_query(q, emb, "gallery_agg")
            best, s1, margin = decide(scores)
            rows.append({"video": video, "match": match, "fragment": q.fragment, "identity": q.identity,
                         "known": q.known, "synthetic_unknown": q.synthetic_unknown, "reentry": q.reentry,
                         "best": best, "score": s1, "margin": margin,
                         "correct": bool(q.known and best == q.identity), "candidates": len(scores),
                         "duration_s": q.meta["duration_s"],
                         "others": sorted(scores.values(), reverse=True)[1:]})
    return rows


def video_stats(rows: list[dict]) -> dict:
    """video -> (median, spread) of non-top candidate scores over its real queries."""
    stats = {}
    for video in sorted({r["video"] for r in rows}):
        pool = np.array([s for r in rows if r["video"] == video and not r["synthetic_unknown"]
                         for s in r["others"]])
        med = float(np.median(pool))
        spread = float(1.4826 * np.median(np.abs(pool - med)))
        stats[video] = (med, max(spread, 1e-6))
    return stats


def calibrate(rows: list[dict], stats: dict) -> list[dict]:
    out = []
    for r in rows:
        med, spread = stats[r["video"]]
        out.append({**r, "score": (r["score"] - med) / spread,
                    "margin": r["margin"] / spread if np.isfinite(r["margin"]) else r["margin"]})
    return out


def select_point(rows: list[dict], target: float, fmr_max: float = CRITERION["unknown_false_match_rate_max"],
                 grid: int = GRID) -> dict | None:
    """Highest-coverage gate on these (training) rows meeting the precision and false-match bounds."""
    scores = np.array([r["score"] for r in rows if np.isfinite(r["score"])])
    margins = np.array([r["margin"] for r in rows if np.isfinite(r["margin"])])
    best = None
    for tau in np.unique(np.quantile(scores, np.linspace(0, 1, grid))):
        for gate in np.unique(np.quantile(margins, np.linspace(0, 1, grid))):
            p = operating_point(rows, float(tau), float(gate))
            if not p["accepted"] or p["precision"] < target or p["unknown_false_match_rate"] > fmr_max:
                continue
            if best is None or (p["coverage"], p["tau"]) > (best["coverage"], best["tau"]):
                best = p
    return best


def lomo() -> dict:
    report = {"criterion": CRITERION, "folds": {}, "stats": {}}
    for feature, calibrated in CONFIGS:
        tag = f"{feature}|{'calibrated' if calibrated else 'absolute'}"
        rows = [r for results, matches in SOURCES.items() for r in scored_rows(results, matches, feature)]
        if calibrated:
            stats = video_stats(rows)
            report["stats"][feature] = stats
            rows = calibrate(rows, stats)
        matches = sorted({r["match"] for r in rows})
        for target in TARGETS:
            key = f"{tag}|train_p{target:.2f}"
            report["folds"][key] = {}
            for test in matches:
                train = [r for r in rows if r["match"] != test]
                held = [r for r in rows if r["match"] == test]
                chosen = select_point(train, target)
                if chosen is None:
                    report["folds"][key][test] = {"chosen": None}
                    continue
                result = point_with_parts(held, chosen["tau"], chosen["margin"])
                report["folds"][key][test] = {
                    "chosen": {k: chosen[k] for k in ("tau", "margin", "coverage", "precision")},
                    "test": result, "verdict": verdict(result, CRITERION)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lomo.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main() -> None:
    report = lomo()
    print(f"{'config':52s} {'test match':20s} {'cov':>5s} {'prec':>5s} {'acc':>4s} {'fmr':>5s} {'natf':>5s}  pass")
    for key, folds in report["folds"].items():
        for test, f in folds.items():
            if f["chosen"] is None:
                print(f"{key:52s} {test:20s}   no training point meets the bounds")
                continue
            t = f["test"]
            print(f"{key:52s} {test:20s} {t['coverage']:5.2f} {t['precision']:5.2f} {t['accepted']:4d} "
                  f"{t['unknown_false_match_rate']:5.3f} {t['unknown_false_match_rate_natural']:5.3f}  "
                  f"{'PASS' if f['verdict']['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
