"""Lever 1 + half-split check: more names at the same precision? (tuning matches only)

Variants, all with the split-v1 encoder (SportsReID ViT-B/16, raw crops) and the
absolute two-gate rule (score >= tau, margin over the runner-up >= m):

- galleries `before` (the split-v1 protocol), `before_notrack` (same, without the
  query's own track), `both_notrack` (also fragments starting >= 2 s after the query,
  without the query's own track; `evaluation.build_queries`);
- query crops `first8` (as locked) or `spread8` (8 crops evenly over the fragment);
- `halves`: additionally abstain when the fragment's first and second half (each
  >= 2 crops) pick different players -- the guard against a box that switches person.
  Tuning queries are number-vouched fragments, so this measures what the check costs;
  what it catches only shows on real unresolved segments.

Each variant's gate is chosen on 7 tuning matches and applied to the 8th (criterion v2
bounds at the split-v1 lock's `tight` target, and at the bounds). Every vouched fragment
with >= 3 crops is one slot in a fixed universe, so correct names and correct seconds
compare across variants whose galleries differ. The split-v1 test matches are never
loaded (asserted).

    python -m experiments.teammate_reid.symmetric          # compare variants (tuning LOMO)
    python -m experiments.teammate_reid.symmetric lock     # write locked/split_v2.json once
"""
from __future__ import annotations

import datetime
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from experiments.teammate_reid.benchmark import load_reviewed
from experiments.teammate_reid.candidate_calibration import aggregate
from experiments.teammate_reid.evaluation import build_queries
from experiments.teammate_reid.openset import CRITERION, DEV_SOURCES, GRID, RUNS, TARGETS
from experiments.teammate_reid.split_test import TEST, TUNING

GALLERIES = {"before": (False, False), "before_notrack": (False, True), "both_notrack": (True, True)}
QUERY_MODES = ("first8", "spread8")
M = 8
OUT = RUNS / "symmetric"
LOCK = Path(__file__).parent / "locked/split_v2.json"
# chosen 2026-10-19 from the tuning LOMO (`main`): most correct names at the split-v1
# precision; the half-split check is applied as a guard on top of the gate, not tuned
CHOICE = {"gallery": "both_notrack", "query_crops": "spread8", "rule": "absolute",
          "target": "tight", "halves_guard": True}


def _pick(crops: list, mode: str) -> list:
    if mode == "first8" or len(crops) <= M:
        return crops[:M]
    return [crops[i] for i in np.linspace(0, len(crops) - 1, M).round().astype(int)]


def _best(sim: np.ndarray, qc: list, galleries: dict) -> tuple:
    raw = sorted(((aggregate(sim[np.ix_(qc, g)]), i) for i, g in galleries.items()), reverse=True)
    margin = raw[0][0] - raw[1][0] if len(raw) > 1 else -np.inf
    return raw[0][1], raw[0][0], margin


def rows_for(gallery: str, query_mode: str) -> tuple[list, list]:
    """(query rows, universe) over the tuning matches only."""
    rows, universe = [], []
    for results, matches in DEV_SOURCES.items():
        matches = tuple(m for m in matches if m in TUNING)
        if matches:
            r, u = results_rows(results, matches, gallery, query_mode, allowed=lambda m: m in TUNING)
            rows += r
            universe += u
    return rows, universe


def results_rows(results: Path, matches: tuple, gallery: str, query_mode: str, allowed) -> tuple[list, list]:
    """(query rows, universe) of one results dir: universe = every vouched fragment with >= 3 crops.

    `allowed(match)` is asserted for every match loaded, so a split cannot leak."""
    after, exclude_track = GALLERIES[gallery]
    rows, universe = [], []
    crops, fragments, _ = load_reviewed(results)
    emb = np.load(results / "features_sportsreid_vitb16.npz")["raw"]
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    by_fragment: dict = {}
    for c in sorted(crops, key=lambda c: c["frame"]):
        by_fragment.setdefault(c["fragment"], []).append(c["crop_id"])
    for video in sorted({f["video"] for f in fragments if f["match"] in matches}):
        frags = [f for f in fragments if f["video"] == video and by_fragment.get(f["key"])]
        match = frags[0]["match"]
        assert allowed(match) and match not in TEST
        ids = sorted({c for f in frags for c in by_fragment[f["key"]]})
        index = {c: i for i, c in enumerate(ids)}
        sim = emb[ids] @ emb[ids].T
        local = {k: [index[c] for c in v] for k, v in by_fragment.items() if k in {f["key"] for f in frags}}
        universe += [{"match": match, "fragment": f["key"], "duration_s": (f["end"] - f["start"] + 1) / 25.0}
                     for f in frags if len(local[f["key"]]) >= 3]
        for q in build_queries(frags, local, after=after, exclude_track=exclude_track):
            qc = _pick(q.crops, query_mode)
            best, score, margin = _best(sim, qc, q.galleries)
            h = len(q.crops) // 2
            halves = None
            if h >= 2 and len(q.galleries) > 1:
                first, second = (_best(sim, _pick(part, "spread8"), q.galleries)[0]
                                 for part in (q.crops[:h], q.crops[h:]))
                halves = first == second
            rows.append({"match": match, "fragment": q.fragment, "identity": q.identity, "known": q.known,
                         "synthetic_unknown": q.synthetic_unknown, "duration_s": q.meta["duration_s"],
                         "best": best, "score": score, "margin": margin, "halves_agree": halves,
                         "candidates": len(q.galleries)})
    return rows, universe


class Arrays:
    """Vectorised scoring of one row set over a grid of (tau, m) gates."""

    def __init__(self, rows: list, halves: bool):
        self.rows = rows
        score = np.array([r["score"] for r in rows], float)
        if halves:
            score = np.where([r["halves_agree"] is False for r in rows], -np.inf, score)
        self.score, self.margin = score, np.array([r["margin"] for r in rows], float)
        self.known = np.array([r["known"] for r in rows])
        self.right = np.array([r["best"] == r["identity"] for r in rows]) & self.known
        self.dur = np.array([r["duration_s"] for r in rows], float)
        self.match = np.array([r["match"] for r in rows])

    def grid(self, sel: np.ndarray) -> list:
        q = lambda v: np.unique(np.quantile(v[sel & np.isfinite(v)], np.linspace(0, 1, GRID)))
        return list(itertools.product(q(self.score), q(self.margin)))

    def stats(self, sel: np.ndarray, gates: np.ndarray) -> dict:
        acc = (self.score[None, sel] >= gates[:, :1]) & (self.margin[None, sel] >= gates[:, 1:])
        known, right, dur = self.known[sel], self.right[sel], self.dur[sel]
        acc_known, correct = (acc & known).sum(1), (acc & right).sum(1)
        n_unknown = max(1, (~known).sum())
        return {"accepted_known": acc_known, "correct": correct,
                "precision": np.where(acc_known > 0, correct / np.maximum(acc_known, 1), np.nan),
                "coverage": correct / max(1, known.sum()), "fmr": (acc & ~known).sum(1) / n_unknown,
                "correct_seconds": (acc * (right * dur)).sum(1),
                "wrong_seconds": (acc * ((~right) * dur)).sum(1), "known_queries": int(known.sum()),
                "unknown_queries": int((~known).sum())}

    def choose(self, sel: np.ndarray, kp: float, fmr: float):
        gates = np.array(self.grid(sel))
        s = self.stats(sel, gates)
        ok = (s["accepted_known"] > 0) & (s["precision"] >= kp) & (s["fmr"] <= fmr)
        if not ok.any():
            return None
        return gates[np.flatnonzero(ok)[np.argmax(s["coverage"][ok])]]


def lomo(rows: list, halves: bool, target: str, universe: list) -> dict:
    a = Arrays(rows, halves)
    kp, fmr = TARGETS[target]
    tot = {k: 0.0 for k in ("accepted_known", "correct", "unknown_fm", "known", "unknown",
                            "correct_seconds", "wrong_seconds")}
    folds = {}
    for m in TUNING:
        gate = a.choose(a.match != m, kp, fmr)
        test = a.match == m
        if gate is None:
            s = a.stats(test, np.array([[np.inf, np.inf]]))
        else:
            s = a.stats(test, np.array([gate]))
        one = {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        one["gate"] = None if gate is None else [float(g) for g in gate]
        one["passes"] = bool(np.isfinite(one["precision"]) and one["precision"] >= CRITERION["known_precision_min"]
                             and one["fmr"] <= CRITERION["unknown_false_match_rate_max"]
                             and one["coverage"] >= CRITERION["coverage_min"])
        folds[m] = one
        tot["accepted_known"] += one["accepted_known"]
        tot["correct"] += one["correct"]
        tot["unknown_fm"] += one["fmr"] * one["unknown_queries"]
        tot["known"] += one["known_queries"]
        tot["unknown"] += one["unknown_queries"]
        tot["correct_seconds"] += one["correct_seconds"]
        tot["wrong_seconds"] += one["wrong_seconds"]
    uni_n, uni_s = len(universe), sum(u["duration_s"] for u in universe)
    return {"folds_passed": sum(f["passes"] for f in folds.values()), "folds": len(folds),
            "precision": tot["correct"] / max(1, tot["accepted_known"]), "correct": int(tot["correct"]),
            "wrong_known": int(tot["accepted_known"] - tot["correct"]),
            "unknown_fmr": tot["unknown_fm"] / max(1, tot["unknown"]), "unknown_queries": int(tot["unknown"]),
            "coverage_known": tot["correct"] / max(1, tot["known"]), "known_queries": int(tot["known"]),
            "coverage_universe": tot["correct"] / uni_n, "universe": uni_n,
            "correct_seconds": tot["correct_seconds"], "correct_seconds_share": tot["correct_seconds"] / uni_s,
            "wrong_seconds": tot["wrong_seconds"], "per_fold": folds}


def lock() -> dict:
    """Write the v2 choice with its gate chosen on all tuning matches. Refuses to overwrite."""
    if LOCK.is_file():
        raise SystemExit(f"{LOCK} exists; a lock is never rewritten")
    rows, universe = rows_for(CHOICE["gallery"], CHOICE["query_crops"])
    a = Arrays(rows, False)
    everything = np.ones(len(rows), bool)
    gate = a.choose(everything, *TARGETS[CHOICE["target"]])
    in_sample = {k: (float(v[0]) if isinstance(v, np.ndarray) else v)
                 for k, v in a.stats(everything, np.array([gate])).items()}
    locked = {
        "locked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "encoder": "sportsreid_vitb16", "mask": "raw", "aggregation": "top-3 mean per query crop, mean over 8",
        **CHOICE, "gate": [float(g) for g in gate], "min_sep_frames": 50, "tuning": TUNING,
        "never_used": {"split_v1_test": TEST, "note": "new footage is scored once against this file"},
        "criterion": CRITERION,
        "tuning_lomo": {k: v for k, v in lomo(rows, False, CHOICE["target"], universe).items() if k != "per_fold"},
        "tuning_in_sample": in_sample,
    }
    LOCK.write_text(json.dumps(locked, indent=2, default=float))
    return locked


def main() -> None:
    if sys.argv[1:] == ["lock"]:
        r = lock()
        print(json.dumps({k: r[k] for k in ("gate", "tuning_lomo")}, indent=1, default=float))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"criterion": CRITERION, "tuning": TUNING, "variants": {}}
    print(f"{'variant':44s} {'folds':>5s} {'prec':>6s} {'right':>6s} {'wrong':>5s} {'unkFMR':>7s} "
          f"{'cov(univ)':>9s} {'s right':>8s} {'s wrong':>8s}")
    for gallery, query_mode in itertools.product(GALLERIES, QUERY_MODES):
        rows, universe = rows_for(gallery, query_mode)
        for halves, target in itertools.product((False, True), ("tight", "at_bounds")):
            key = f"{gallery}|{query_mode}|{'halves' if halves else 'plain'}|{target}"
            r = lomo(rows, halves, target, universe)
            report["variants"][key] = r
            print(f"{key:44s} {r['folds_passed']:>2d}/{r['folds']:<2d} {r['precision']:6.3f} {r['correct']:6d} "
                  f"{r['wrong_known']:5d} {r['unknown_fmr']:7.3f} {r['coverage_universe']:9.3f} "
                  f"{r['correct_seconds']:8.0f} {r['wrong_seconds']:8.0f}", flush=True)
    (OUT / "lomo.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
