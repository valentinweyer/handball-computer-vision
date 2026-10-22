"""Label-free per-video score calibration for the split_v2 rule (development, tuning matches only).

The shared gate is the weak point: one (tau, m) for every video, while similarity levels differ
between videos and kits (finetune_v1 B diagnosis). Teammates on court at the same time are
different people, so their scores give a per-(video, team) impostor level without labels or
numbers. Variants, fixed before any result was seen:

- `raw`: split_v2 as locked (baseline);
- `shift`: score - median impostor score; margin unchanged;
- `z`: (score - median) / (1.4826 * MAD); margin / (1.4826 * MAD).

Evaluated with split_v2's own leave-one-match-out over its 8 tuning matches: the gate is chosen
on 7 matches and applied unchanged to the 8th, which is exactly the shared-gate case.

    python -m experiments.teammate_reid.calibrate
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from experiments.teammate_reid.benchmark import load_reviewed
from experiments.teammate_reid.candidate_calibration import aggregate
from experiments.teammate_reid.openset import DEV_SOURCES, RUNS
from experiments.teammate_reid.split_test import TEST, TUNING
from experiments.teammate_reid.symmetric import CHOICE, _pick, lomo, rows_for

OUT = RUNS / "calibrate"
MIN_OVERLAP = 25            # frames two fragments must share to count as co-present
MIN_PAIRS = 10              # fewer impostor pairs: fall back to the video's pooled level
VARIANTS = ("raw", "shift", "z")


def impostor_scores(crops: list, fragments: list, emb: np.ndarray) -> dict:
    """(video, team) -> scores of co-present teammate fragments on different tracks."""
    by_fragment: dict = {}
    for c in sorted(crops, key=lambda c: c["frame"]):
        by_fragment.setdefault(c["fragment"], []).append(c["crop_id"])
    out: dict = {}
    frags = [f for f in fragments if by_fragment.get(f["key"])]
    for a, b in itertools.combinations(frags, 2):
        if a["video"] != b["video"] or a["team"] != b["team"] or a["track_id"] == b["track_id"]:
            continue
        if min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1 < MIN_OVERLAP:
            continue
        ca, cb = by_fragment[a["key"]], by_fragment[b["key"]]
        s = 0.5 * (aggregate(emb[_pick(ca, "spread8")] @ emb[cb].T) + aggregate(emb[_pick(cb, "spread8")] @ emb[ca].T))
        out.setdefault((a["video"], a["team"]), []).append(float(s))
    return out


def levels() -> tuple[dict, dict]:
    """fragment key -> (median, scale) of its (video, team), and a summary per (video, team)."""
    per_fragment, summary = {}, {}
    for results, matches in DEV_SOURCES.items():
        if not any(m in TUNING for m in matches):
            continue
        crops, fragments, _ = load_reviewed(results)
        fragments = [f for f in fragments if f["match"] in TUNING]
        assert not {f["match"] for f in fragments} & set(TEST)
        emb = np.load(results / "features_sportsreid_vitb16.npz")["raw"]
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        imp = impostor_scores(crops, fragments, emb)
        for video in {f["video"] for f in fragments}:
            pooled = [s for (v, _), ss in imp.items() if v == video for s in ss]
            for team in {f["team"] for f in fragments if f["video"] == video}:
                ss = imp.get((video, team), [])
                use = ss if len(ss) >= MIN_PAIRS else pooled
                med = float(np.median(use))
                scale = float(1.4826 * np.median(np.abs(np.array(use) - med)))
                summary[f"{video}|{team}"] = {"pairs": len(ss), "fallback": len(ss) < MIN_PAIRS,
                                              "median": med, "scale": scale}
                for f in fragments:
                    if f["video"] == video and f["team"] == team:
                        per_fragment[f["key"]] = (med, scale)
    return per_fragment, summary


def calibrated(rows: list, per_fragment: dict, variant: str) -> list:
    if variant == "raw":
        return rows
    out = []
    for r in rows:
        med, scale = per_fragment[r["fragment"]]
        score, margin = r["score"] - med, r["margin"]
        if variant == "z":
            score, margin = score / scale, margin / scale
        out.append({**r, "score": score, "margin": margin})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, universe = rows_for(CHOICE["gallery"], CHOICE["query_crops"])
    per_fragment, summary = levels()
    report = {"choice": CHOICE, "levels": summary, "variants": {}}
    print(f"{'variant':16s} {'folds':>5s} {'prec':>6s} {'right':>6s} {'wrong':>5s} {'unkFMR':>7s} "
          f"{'cov(univ)':>9s} {'s right':>8s} {'s wrong':>8s}")
    for variant, target in itertools.product(VARIANTS, ("tight", "at_bounds")):
        r = lomo(calibrated(rows, per_fragment, variant), True, target, universe)
        report["variants"][f"{variant}|{target}"] = r
        print(f"{variant + '|' + target:16s} {r['folds_passed']:>2d}/{r['folds']:<2d} {r['precision']:6.3f} "
              f"{r['correct']:6d} {r['wrong_known']:5d} {r['unknown_fmr']:7.3f} {r['coverage_universe']:9.3f} "
              f"{r['correct_seconds']:8.0f} {r['wrong_seconds']:8.0f}", flush=True)
    (OUT / "lomo.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
