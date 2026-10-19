"""Nested leave-one-match-out: tune everything on some matches, validate on an isolated one.

The encoder comparison chose its winner after seeing every development fold, which
flatters it. Here each match takes one turn as the isolated validation match and
influences nothing about its own decision:

- outer loop: hold out match M (validation);
- on the other matches only: score every candidate (encoder x rule x training target)
  by an inner leave-one-match-out, pick the candidate passing the most inner folds
  (ties: higher pooled inner coverage, then fewer unknown false matches), and choose
  its gate on all of them;
- apply that one locked choice to M, once.

The outer results estimate the whole procedure -- model selection included -- on
matches it never saw. Rows come from the cached `runs/smart-reid-shadow/<tag>/rows.json`
written by `openset.main`, so nothing is re-encoded.

    python -m experiments.teammate_reid.nested
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.teammate_reid.openset import CRITERION, TARGETS, choose, passes, score

RUNS = Path("/home/valentinweyer/projects/handball-computer-vision/runs/smart-reid-shadow")
FEATURE_TAGS = {
    "prtreid_stripes": "openset_11",
    "personvit": "openset_11_personvit",
    "personvit_strict": "openset_11_personvit_strict",
    "sportsreid": "openset_11_sportsreid_vitb16",
    "sportsreid_strict": "openset_11_sportsreid_vitb16_strict",
    "fused": "openset_11_fused_pvit_sreid",
    "fused_strict": "openset_11_fused_pvit_sreid_strict",
}
RULES = ("absolute", "min_crops")
KP, FMR = CRITERION["known_precision_min"], CRITERION["unknown_false_match_rate_max"]


def load_rows() -> dict:
    out = {}
    for name, tag in FEATURE_TAGS.items():
        rows = json.loads((RUNS / tag / "rows.json").read_text())
        for r in rows:
            for k in ("score", "margin", "z_raw", "z_gap"):
                r[k] = float(r[k])
        out[name] = rows
    return out


def candidates() -> list:
    return [(f, rule, t) for f in FEATURE_TAGS for rule in RULES for t in TARGETS]


def inner_score(rows: list, rule: str, target: str) -> tuple:
    """(folds passed, pooled coverage, -pooled unknown FMR) over an inner leave-one-match-out."""
    kp, fmr = TARGETS[target]
    passed, correct, known, fm, unknown = 0, 0, 0, 0, 0
    for test in sorted({r["match"] for r in rows}):
        chosen = choose([r for r in rows if r["match"] != test], rule, kp, fmr)
        if chosen is None:
            continue
        s = score([r for r in rows if r["match"] == test], rule, chosen[0])
        passed += passes(s, KP, FMR)
        correct += s["correct"]
        known += s["known_queries"]
        fm += round(s["unknown_false_match_rate"] * s["unknown_queries"])
        unknown += s["unknown_queries"]
    return passed, correct / max(1, known), -fm / max(1, unknown)


def outer_fold(args: tuple) -> dict:
    validation, all_rows = args
    ranked = []
    for f, rule, target in candidates():
        train = [r for r in all_rows[f] if r["match"] != validation]
        ranked.append((inner_score(train, rule, target), (f, rule, target)))
    ranked.sort(key=lambda x: x[0], reverse=True)
    (passed, cov, nfmr), (f, rule, target) = ranked[0]
    kp, fmr = TARGETS[target]
    train = [r for r in all_rows[f] if r["match"] != validation]
    gate, _ = choose(train, rule, kp, fmr)
    s = score([r for r in all_rows[f] if r["match"] == validation], rule, gate)
    return {"validation": validation, "chosen": {"feature": f, "rule": rule, "target": target,
                                                  "gate": [float(g) for g in gate],
                                                  "inner": {"folds_passed": passed, "coverage": cov,
                                                            "unknown_fmr": -nfmr}},
            "validation_result": s, "pass": bool(passes(s, KP, FMR)),
            "runner_up": [{"candidate": c, "inner": list(sc)} for sc, c in ranked[1:4]]}


def main() -> None:
    rows = load_rows()
    matches = sorted({r["match"] for r in rows["sportsreid"]})
    with ProcessPoolExecutor(max_workers=len(matches)) as pool:
        folds = list(pool.map(outer_fold, [(m, rows) for m in matches]))
    known = sum(f["validation_result"]["known_queries"] for f in folds)
    acc = sum(f["validation_result"]["accepted_known"] for f in folds)
    correct = sum(f["validation_result"]["correct"] for f in folds)
    unknown = sum(f["validation_result"]["unknown_queries"] for f in folds)
    fm = sum(round(f["validation_result"]["unknown_false_match_rate"] * f["validation_result"]["unknown_queries"])
             for f in folds)
    pooled = {"known_precision": correct / max(1, acc), "coverage": correct / max(1, known),
              "unknown_false_match_rate": fm / max(1, unknown),
              "folds_passed": sum(f["pass"] for f in folds), "folds": len(folds),
              "wrong_seconds": sum(f["validation_result"]["wrong_seconds"] for f in folds),
              "correct_seconds": sum(f["validation_result"]["correct_seconds"] for f in folds)}
    out = RUNS / "nested"
    out.mkdir(exist_ok=True)
    (out / "nested.json").write_text(json.dumps({"criterion": CRITERION, "pooled": pooled, "folds": folds},
                                                indent=2, default=float))
    for f in folds:
        c, v = f["chosen"], f["validation_result"]
        print(f"{f['validation']:22s} chose {c['feature']:18s} {c['rule']:9s} {c['target']:9s} "
              f"(inner {c['inner']['folds_passed']}/10) -> kp {v['known_precision']:.3f} cov {v['coverage']:.2f} "
              f"fmr {v['unknown_false_match_rate']:.3f} nat {v['natural_false_matches']}/{v['natural_unknowns']} "
              f"{'PASS' if f['pass'] else 'fail'}")
    print("pooled:", {k: round(v, 3) if isinstance(v, float) else v for k, v in pooled.items()})


if __name__ == "__main__":
    main()
