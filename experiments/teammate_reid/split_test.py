"""Fixed tuning/test split over the processed matches: select on tuning, test once.

The split follows the source folders of the detector project: `train/` matches tune,
`eval/` matches test. It is fixed here, in code, before any selection is run on it.

- `select`: on the TUNING matches only (test rows are dropped first, and asserted
  absent), every candidate (encoder x rule x training target) is scored by
  leave-one-match-out over the tuning matches; the one passing the most folds wins
  (ties: higher pooled coverage, then fewer unknown false matches), its gate is chosen
  on all tuning matches, and the choice is written to `locked/split_v1.json`. Refuses
  to overwrite.
- `test`: reads only that lock and scores the TEST matches once (criterion v2), per
  match and pooled, with identity-clustered 95% intervals.

Caveat recorded with the result: the three test matches appeared in the earlier
11-match encoder comparison, which shaped the candidate list (not this selection).

    python -m experiments.teammate_reid.split_test select
    python -m experiments.teammate_reid.split_test test
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.teammate_reid.nested import FEATURE_TAGS, RULES, inner_score, load_rows
from experiments.teammate_reid.openset import CRITERION, TARGETS, choose, passes, score

TUNING = ("Ludwigshafen-Wetzlar", "RNL-SGF", "Balingen-Kiel", "Berlin-BHC", "Lemgo-Magdeburg",
          "Goeppingen-Leipzig", "Altenholz-Luebeck", "Pforzheim-Bietigheim")
TEST = ("Kiel-Lemgo", "Eisenach-Hamburg", "Gummersbach-Loewen")
LOCK = Path(__file__).parent / "locked/split_v1.json"
OUT = Path("/home/valentinweyer/projects/handball-computer-vision/runs/smart-reid-shadow/split_v1")
KP, FMR = CRITERION["known_precision_min"], CRITERION["unknown_false_match_rate_max"]


def tuning_rows(rows: dict) -> dict:
    out = {f: [r for r in rs if r["match"] in TUNING] for f, rs in rows.items()}
    assert not any(r["match"] in TEST for rs in out.values() for r in rs)
    return out


def _rank(args: tuple) -> tuple:
    (feature, rule, target), rows = args
    return inner_score(rows, rule, target), (feature, rule, target)


def select() -> dict:
    if LOCK.is_file():
        raise SystemExit(f"{LOCK} exists; the split's choice is locked and never rewritten")
    rows = tuning_rows(load_rows())
    jobs = [((f, rule, t), rows[f]) for f in FEATURE_TAGS for rule in RULES for t in TARGETS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        ranked = sorted(pool.map(_rank, jobs), key=lambda x: x[0], reverse=True)
    (passed, cov, nfmr), (feature, rule, target) = ranked[0]
    gate, s = choose(rows[feature], rule, *TARGETS[target])
    locked = {
        "locked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "tuning": TUNING, "test": TEST, "criterion": CRITERION,
        "feature": feature, "rule": rule, "target": target, "gate": [float(g) for g in gate],
        "tuning_lomo": {"folds_passed": passed, "folds": len(TUNING), "coverage": cov, "unknown_fmr": -nfmr},
        "tuning_in_sample": {k: (float(v) if isinstance(v, float) else v) for k, v in s.items()},
        "ranking": [{"candidate": list(c), "folds_passed": sc[0], "coverage": sc[1], "unknown_fmr": -sc[2]}
                    for sc, c in ranked],
    }
    LOCK.write_text(json.dumps(locked, indent=2, default=float))
    return locked


def clustered_ci(rows: list, rule: str, gate: tuple, reps: int = 2000, seed: int = 0) -> dict:
    clusters: dict = {}
    for r in rows:
        clusters.setdefault(r["match"], {}).setdefault(r["identity"], []).append(r)
    rng = np.random.default_rng(seed)
    keys = ("known_precision", "coverage", "unknown_false_match_rate")
    draws = {k: [] for k in keys}
    for _ in range(reps):
        sample = [r for by_id in clusters.values() for i in rng.integers(0, len(by_id), len(by_id))
                  for r in list(by_id.values())[i]]
        s = score(sample, rule, gate)
        for k in keys:
            if np.isfinite(s[k]):
                draws[k].append(s[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def test() -> dict:
    lock = json.loads(LOCK.read_text())
    rows = [r for r in load_rows()[lock["feature"]] if r["match"] in TEST]
    gate = tuple(lock["gate"])
    pooled = score(rows, lock["rule"], gate)
    report = {
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(), "lock": lock,
        "pooled": pooled, "pooled_ci95": clustered_ci(rows, lock["rule"], gate),
        "pass": bool(passes(pooled, KP, FMR)),
        "per_match": {m: score([r for r in rows if r["match"] == m], lock["rule"], gate) for m in TEST},
        "caveat": "test matches appeared in the earlier 11-match encoder comparison (candidate list); "
                  "clubs Kiel, Lemgo and Loewen also play in tuning matches (frozen encoders, per-match galleries)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("select", "test"))
    args = parser.parse_args()
    if args.command == "select":
        r = select()
        print(json.dumps({k: r[k] for k in ("feature", "rule", "target", "gate", "tuning_lomo")}, indent=1))
        for c in r["ranking"][:6]:
            print(c)
    else:
        r = test()
        print(json.dumps({k: r[k] for k in ("pass", "pooled", "pooled_ci95", "per_match")}, indent=1, default=float))


if __name__ == "__main__":
    main()
