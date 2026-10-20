"""Score the final windows once against `locked/split_v2.json`.

The four lower-league windows (`heldout.FINAL_WINDOWS`) were cut after the v2 lock was
committed and were never used for tuning or selection. This reads only the lock: the
gallery, query-crop and half-split settings and the gate come from it, nothing is chosen
here. Fragments the assistant review excluded (`<window>/review.json`) are dropped by
`load_reviewed`, as on every earlier bank.

Reported per match and pooled, with 95% intervals from a bootstrap over identities
within each match: known precision, coverage of known queries, unknown false-match
rate, the criterion v2 verdict, and correct/wrong names and seconds over the universe
of vouched fragments (>= 3 crops). Refuses to run twice.

    python -m experiments.teammate_reid.final_test
"""
from __future__ import annotations

import datetime
import hashlib
import json

import numpy as np

from experiments.teammate_reid.heldout import FINAL_WINDOWS
from experiments.teammate_reid.openset import CRITERION, RUNS
from experiments.teammate_reid.split_test import TEST, TUNING
from experiments.teammate_reid.symmetric import LOCK, Arrays, results_rows

FINAL = RUNS / "final"
OUT = FINAL / "test.json"
KP, FMR, COV = (CRITERION[k] for k in ("known_precision_min", "unknown_false_match_rate_max", "coverage_min"))


def summary(rows: list, halves: bool, gate: list) -> dict:
    s = Arrays(rows, halves).stats(np.ones(len(rows), bool), np.array([gate]))
    s = {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    s["wrong_known"] = int(s["accepted_known"] - s["correct"])
    s["unknown_false_matches"] = int(round(s["fmr"] * s["unknown_queries"])) if rows else 0
    s["passes"] = bool(np.isfinite(s["precision"]) and s["precision"] >= KP and s["fmr"] <= FMR
                       and s["coverage"] >= COV)
    return s


def clustered_ci(rows: list, halves: bool, gate: list, reps: int = 2000, seed: int = 0) -> dict:
    clusters: dict = {}
    for r in rows:
        clusters.setdefault(r["match"], {}).setdefault(r["identity"], []).append(r)
    groups = [list(by_id.values()) for by_id in clusters.values()]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ("precision", "coverage", "fmr")}
    for _ in range(reps):
        sample = [r for g in groups for i in rng.integers(0, len(g), len(g)) for r in g[i]]
        s = summary(sample, halves, gate)
        for k in draws:
            if np.isfinite(s[k]):
                draws[k].append(s[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
            for k, v in draws.items()}


def main() -> None:
    if OUT.is_file():
        raise SystemExit(f"{OUT} exists; the final windows are scored once")
    lock = json.loads(LOCK.read_text())
    gate, halves = lock["gate"], lock["halves_guard"]
    final = lambda m: m not in TUNING and m not in TEST
    rows, universe, inputs = [], [], {}
    for name, match, _source, _exposure in FINAL_WINDOWS:
        assert final(match), match
        results = FINAL / name
        r, u = results_rows(results, (match,), lock["gallery"], lock["query_crops"], allowed=final)
        rows += r
        universe += u
        inputs[name] = {f: hashlib.sha256((results / f).read_bytes()).hexdigest()
                        for f in ("fragments.json", "crops.json", "features_sportsreid_vitb16.npz", "review.json")}
    per_match = {}
    for _name, match, *_ in FINAL_WINDOWS:
        sub = [r for r in rows if r["match"] == match]
        uni = [u for u in universe if u["match"] == match]
        per_match[match] = {**summary(sub, halves, gate), "universe": len(uni),
                            "universe_seconds": sum(u["duration_s"] for u in uni),
                            "identities": len({r["identity"] for r in sub if r["known"]})}
    pooled = summary(rows, halves, gate)
    uni_s = sum(u["duration_s"] for u in universe)
    report = {
        "scored_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(), "lock": lock,
        "criterion": {"known_precision_min": KP, "unknown_false_match_rate_max": FMR, "coverage_min": COV},
        "pass": pooled["passes"],
        "pooled": {**pooled, "universe": len(universe), "universe_seconds": uni_s,
                   "coverage_universe": pooled["correct"] / max(1, len(universe)),
                   "correct_seconds_share": pooled["correct_seconds"] / max(uni_s, 1e-9)},
        "pooled_ci95": clustered_ci(rows, halves, gate),
        "per_match": per_match,
        "inputs_sha256": inputs,
        "labels": "queries are number-vouched fragments labelled by their own OCR reads; "
                  "mixed fragments removed by assistant visual review, not by a human annotator",
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("pass", "pooled_ci95")}, indent=1))
    keys = ("known_queries", "accepted_known", "correct", "wrong_known", "precision", "coverage",
            "unknown_queries", "unknown_false_matches", "fmr", "correct_seconds", "wrong_seconds", "passes")
    for m, s in {**per_match, "POOLED": report["pooled"]}.items():
        print(f"{m:22s} " + " ".join(f"{k}={s[k]:.3f}" if isinstance(s[k], float) else f"{k}={s[k]}" for k in keys))


if __name__ == "__main__":
    main()
