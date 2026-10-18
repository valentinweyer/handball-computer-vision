"""Stage 1: does the matcher keep its Kiel precision on held-out matches?

The operating points are copied from the prototype's fold-A benchmark (all
chosen on Kiel rows) into `thresholds.json` **before** any held-out window is
mined. The evaluation then reads that file and never re-derives a threshold.
The primary method and the go/no-go criterion are fixed here too, so neither
can be chosen after seeing held-out results.

    python -m experiments.teammate_reid.transfer lock
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
PROTOTYPE = REPO / "runs/smart-reid-prototype-2026-10-18/results"
OUT = REPO / "runs/smart-reid-shadow"
THRESHOLDS = OUT / "thresholds.json"

PRIMARY = "digits/frozen_global|gallery_agg"
SECONDARY = ("digits/frozen_stripes|gallery_agg", "strict/frozen_global|gallery_agg",
             "strict/frozen_stripes|gallery_agg")
CRITERION = {"precision_min": 0.90, "unknown_false_match_rate_max": 0.05, "coverage_min": 0.20,
             "scope": "pooled over held-out matches, primary method; every bound must hold"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock() -> dict:
    if THRESHOLDS.is_file():
        raise SystemExit(f"{THRESHOLDS} already exists; thresholds are locked and never rewritten")
    benchmark = PROTOTYPE / "benchmark.json"
    summaries = json.loads(benchmark.read_text())["folds"]["A_cross_match"]["summaries"]
    points = {}
    for config in (PRIMARY, *SECONDARY):
        point = summaries[config]["two_gate_p90_point"]
        points[config] = {"tau": point["tau"], "margin": point["margin"],
                          "kiel_fold_a": {k: point[k] for k in ("coverage", "precision", "accepted",
                                                                 "unknown_false_match_rate")}}
    locked = {
        "locked_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": str(benchmark), "source_sha256": sha256(benchmark),
        "selection": "two-gate P90 point of fold A (Kiel-Lemgo test rows): an in-sample oracle on Kiel, "
                     "frozen here before any held-out window was mined",
        "primary": PRIMARY, "points": points, "criterion": CRITERION,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    THRESHOLDS.write_text(json.dumps(locked, indent=2))
    return locked


def load_thresholds() -> dict:
    """The locked file, with its hash: evaluation reads this and never recomputes a threshold."""
    data = json.loads(THRESHOLDS.read_text())
    data["file_sha256"] = sha256(THRESHOLDS)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("lock",))
    args = parser.parse_args()
    if args.command == "lock":
        print(json.dumps(lock(), indent=2))


if __name__ == "__main__":
    main()
