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

    python -m experiments.teammate_reid.calibrate            # tuning LOMO (exploratory)
    python -m experiments.teammate_reid.calibrate lock       # write locked/calibrate_v1.json once
    python -m experiments.teammate_reid.calibrate confirm    # score the lock once on CONFIRM
"""
from __future__ import annotations

import datetime
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.teammate_reid.benchmark import load_reviewed
from experiments.teammate_reid.candidate_calibration import aggregate
from experiments.teammate_reid.openset import DEV_SOURCES, REPO, RUNS, TARGETS
from experiments.teammate_reid.split_test import TEST, TUNING
from experiments.teammate_reid.symmetric import CHOICE, Arrays, _pick, lomo, results_rows, rows_for

OUT = RUNS / "calibrate"
MIN_OVERLAP = 25            # frames two fragments must share to count as co-present
MIN_PAIRS = 10              # fewer impostor pairs: fall back to the video's pooled level
VARIANTS = ("raw", "shift", "z")
LOCK = Path(__file__).parent / "locked/calibrate_v1.json"
SPLIT_V2 = Path(__file__).parent / "locked/split_v2.json"
# development matches that played no part in split_v2 or in this calibration: every mined window
CONFIRM = [(REPO / "runs/smart-reid-prototype-2026-10-18/results", "Kiel-Lemgo"),
           (RUNS / "stage1", "Eisenach-Hamburg"), (RUNS / "stage1", "Gummersbach-Loewen")] + [
          (RUNS / "final" / w, m) for w, m in (("Saarlouis_Muenster_66m00_10min", "Saarlouis-Muenster"),
                                                ("Bielefeld_Aurich_86m00_10min", "Bielefeld-Aurich"),
                                                ("Wuerzburg_Regensburg_66m00_10min", "Wuerzburg-Regensburg"),
                                                ("BalingenII_Erlangen_66m00_10min", "BalingenII-Erlangen"))] + [
          (RUNS / "extra_dev" / f"{w}_{t}", m)
          for w, m, ts in (("Kiel_Lemgo", "Kiel-Lemgo", ("56m00_10min", "76m00_10min")),
                           ("Eisenach_Hamburg", "Eisenach-Hamburg", ("57m00_9min", "76m00_10min")),
                           ("VfL_GM_Loewen", "Gummersbach-Loewen", ("56m00_10min", "76m00_10min")),
                           ("Saarlouis_Muenster", "Saarlouis-Muenster", ("56m00_10min", "76m00_10min")),
                           ("Bielefeld_Aurich", "Bielefeld-Aurich", ("62m00_10min", "96m00_10min")),
                           ("Wuerzburg_Regensburg", "Wuerzburg-Regensburg", ("56m00_10min", "76m00_10min")),
                           ("BalingenII_Erlangen", "BalingenII-Erlangen", ("56m00_10min", "76m00_10min")))
          for t in ts]
CONFIRM_MATCHES = sorted({m for _, m in CONFIRM})


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


def levels_for(results: Path, matches: set) -> tuple[dict, dict]:
    """fragment key -> (median, scale) of its (video, team), and a summary per (video, team)."""
    per_fragment, summary = {}, {}
    crops, fragments, _ = load_reviewed(results)
    fragments = [f for f in fragments if f["match"] in matches]
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


def levels() -> tuple[dict, dict]:
    """Impostor levels of the tuning matches."""
    per_fragment, summary = {}, {}
    for results, matches in DEV_SOURCES.items():
        if any(m in TUNING for m in matches):
            p, s = levels_for(results, set(TUNING))
            per_fragment.update(p)
            summary.update(s)
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


def _stats(rows: list, gate: list) -> dict:
    a = Arrays(rows, halves=True)
    st = a.stats(np.ones(len(rows), bool), np.array([gate]))
    one = {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    one["unknown_false_accepts"] = round(one["fmr"] * one["unknown_queries"])
    one["wrong_known"] = int(one["accepted_known"] - one["correct"])
    return one


def lock() -> dict:
    """Shift variant, gate chosen on all tuning matches as split_v2's lock does. Never rewritten."""
    if LOCK.is_file():
        raise SystemExit(f"{LOCK} exists; a lock is never rewritten")
    rows, _ = rows_for(CHOICE["gallery"], CHOICE["query_crops"])
    per_fragment, _ = levels()
    shifted = calibrated(rows, per_fragment, "shift")
    gate = Arrays(shifted, False).choose(np.ones(len(shifted), bool), *TARGETS[CHOICE["target"]])
    locked = {
        "locked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 cwd=Path(__file__).parent).stdout.strip(),
        "base_rule": "locked/split_v2.json (gallery, query crops, aggregation, halves guard)",
        "variant": "shift: score - median score of co-present same-team fragments on other tracks, per (video, team); "
                   "margin unchanged",
        "min_overlap_frames": MIN_OVERLAP, "min_pairs": MIN_PAIRS, "fallback": "video's pooled impostor scores",
        "target": CHOICE["target"], "gate_shifted": [float(g) for g in gate],
        "gate_raw": json.loads(SPLIT_V2.read_text())["gate"],
        "chosen_on": list(TUNING),
        "confirm": {"windows": [str(r.relative_to(REPO)) + "|" + m for r, m in CONFIRM],
                    "matches": CONFIRM_MATCHES,
                    "note": "development matches outside split_v2 tuning; spent for calibration by this run (user, 2026-10-23)"},
        "bars": {"C1": "pooled unknown FMR (shift) <= 0.02 (split_v2's tight target)",
                 "C2": "correct known names (shift) >= 0.95 * correct (raw split_v2)",
                 "C3": "known precision (shift) >= 0.97",
                 "verdict": "CONFIRMED if C1-C3 hold, else NOT CONFIRMED; reported per match with n, no change after"},
    }
    LOCK.write_text(json.dumps(locked, indent=1))
    return locked


def confirm() -> dict:
    lk = json.loads(LOCK.read_text())
    out_path = OUT / "confirm.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: the confirmation runs once")
    allowed = set(CONFIRM_MATCHES)
    assert not allowed & set(TUNING)
    raw, shift, per_window = [], [], {}
    for results, match in CONFIRM:
        r, _ = results_rows(results, (match,), CHOICE["gallery"], CHOICE["query_crops"], allowed=lambda m: m in allowed)
        per_fragment, summary = levels_for(results, {match})
        s = calibrated(r, per_fragment, "shift")
        for row, srow in zip(r, s):
            row["window"] = srow["window"] = results.name
        raw += r
        shift += s
        per_window[results.name] = {"match": match, "queries": len(r), "levels": summary}
    res = {"lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
           "raw": _stats(raw, lk["gate_raw"]), "shift": _stats(shift, lk["gate_shifted"]), "per_match": {}}
    for m in CONFIRM_MATCHES:
        res["per_match"][m] = {"raw": _stats([x for x in raw if x["match"] == m], lk["gate_raw"]),
                               "shift": _stats([x for x in shift if x["match"] == m], lk["gate_shifted"])}
    a, b = res["raw"], res["shift"]
    res["bars"] = {"C1": bool(b["fmr"] <= 0.02), "C2": bool(b["correct"] >= 0.95 * a["correct"]),
                   "C3": bool(b["precision"] >= 0.97)}
    res["verdict"] = "CONFIRMED" if all(res["bars"].values()) else "NOT CONFIRMED"
    res["per_window"] = per_window
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    return res


def main() -> None:
    if sys.argv[1:] == ["lock"]:
        print(json.dumps(lock(), indent=1))
        return
    if sys.argv[1:] == ["confirm"]:
        r = confirm()
        keys = ("precision", "correct", "wrong_known", "unknown_false_accepts", "unknown_queries", "fmr",
                "coverage", "correct_seconds", "wrong_seconds")
        for c in ("raw", "shift"):
            print(c, {k: round(r[c][k], 3) for k in keys})
        for m, v in r["per_match"].items():
            print(f"{m:22s}", {c: (v[c]["correct"], v[c]["wrong_known"], v[c]["unknown_false_accepts"],
                                   v[c]["unknown_queries"]) for c in ("raw", "shift")})
        print(r["bars"], r["verdict"])
        return
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
