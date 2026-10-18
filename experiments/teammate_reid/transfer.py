"""Stage 1: does the matcher keep its Kiel precision on held-out matches?

The operating points are copied from the prototype's fold-A benchmark (all
chosen on Kiel rows) into `thresholds.json` **before** any held-out window is
mined. The evaluation then reads that file and never re-derives a threshold.
The primary method and the go/no-go criterion are fixed here too, so neither
can be chosen after seeing held-out results.

    python -m experiments.teammate_reid.transfer lock
    python -m experiments.teammate_reid.run --set heldout --out runs/smart-reid-shadow/stage1 mine
    python -m experiments.teammate_reid.run --set heldout --out runs/smart-reid-shadow/stage1 extract
    python -m experiments.teammate_reid.run --out runs/smart-reid-shadow/stage1 sheets
    # review.json in stage1/ is the assistant's contact-sheet review (mixed fragments)
    python -m experiments.teammate_reid.transfer evaluate

Interpretation fixed with the criterion, before any held-out data was mined:
- The false-match rate in the criterion is the one the locked point was
  measured with on Kiel: natural and synthetic unknowns pooled
  (`operating_point`). Natural and synthetic rates are also reported
  separately, and a natural rate above the bound is called out even when the
  pooled rate passes.
- The query protocol is the prototype's (`build_queries`: gallery fragments
  end >= 2 s before the query, same team, within one video), because that is
  the protocol the thresholds were measured under.
- Confidence intervals come from a bootstrap over (match, identity)
  clusters, resampled within each match: crops and fragments of one player
  are never treated as independent.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
PROTOTYPE = REPO / "runs/smart-reid-prototype-2026-10-18/results"
OUT = REPO / "runs/smart-reid-shadow"
THRESHOLDS = OUT / "thresholds.json"
STAGE1 = OUT / "stage1"

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


def _rate(rows: list, unknown_kind: str) -> float:
    pool = [r for r in rows if not r["known"] and r["synthetic_unknown"] == (unknown_kind == "synthetic")]
    return sum(r["accepted"] for r in pool) / len(pool) if pool else float("nan")


def point_with_parts(rows: list, tau: float, margin: float) -> dict:
    """`operating_point` plus the natural and synthetic unknown rates apart."""
    from experiments.teammate_reid.evaluation import accept, operating_point
    point = operating_point(rows, tau, margin)
    marked = [{**r, "accepted": accept(r, tau, margin)} for r in rows]
    point["unknown_false_match_rate_natural"] = _rate(marked, "natural")
    point["unknown_false_match_rate_synthetic"] = _rate(marked, "synthetic")
    point["unknown_natural_queries"] = sum(not r["known"] and not r["synthetic_unknown"] for r in rows)
    point["unknown_synthetic_queries"] = sum(r["synthetic_unknown"] for r in rows)
    point["known_queries"] = sum(r["known"] for r in rows)
    return point


def clustered_ci(rows: list, tau: float, margin: float, reps: int = 2000, seed: int = 0) -> dict:
    """95% intervals from resampling (match, identity) clusters within each match."""
    clusters: dict = {}
    for r in rows:
        clusters.setdefault(r["match"], {}).setdefault(r["identity"], []).append(r)
    rng = np.random.default_rng(seed)
    keys = ("precision", "coverage", "unknown_false_match_rate", "unknown_false_match_rate_natural")
    draws = {k: [] for k in keys}
    for _ in range(reps):
        sample = []
        for by_identity in clusters.values():
            ids = list(by_identity)
            for i in rng.integers(0, len(ids), len(ids)):
                sample += by_identity[ids[i]]
        point = point_with_parts(sample, tau, margin)
        for k in keys:
            if np.isfinite(point[k]):
                draws[k].append(point[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
            for k, v in draws.items()}


def verdict(point: dict, criterion: dict) -> dict:
    checks = {
        "precision": bool(np.isfinite(point["precision"]) and point["precision"] >= criterion["precision_min"]),
        "unknown_false_match_rate": point["unknown_false_match_rate"] <= criterion["unknown_false_match_rate_max"],
        "coverage": point["coverage"] >= criterion["coverage_min"],
    }
    natural = point["unknown_false_match_rate_natural"]
    return {"checks": checks, "pass": all(checks.values()),
            "natural_unknown_rate_above_bound": bool(np.isfinite(natural)
                                                     and natural > criterion["unknown_false_match_rate_max"])}


def evaluate(out: Path = STAGE1) -> dict:
    from experiments.teammate_reid.benchmark import fold_queries, frozen_embeddings, load_reviewed
    from experiments.teammate_reid.evaluation import outcomes, summarize

    locked = load_thresholds()
    crops, fragments, review = load_reviewed(out)
    prototype_matches = {"Kiel-Lemgo", "BHC-FAG", "Melsungen-Berlin", "FelixClaar", "Han-Ber4"}
    leaked = {f["match"] for f in fragments} & prototype_matches
    if leaked:
        raise SystemExit(f"held-out evaluation would include development matches: {sorted(leaked)}")
    data = np.load(out / "features.npz")
    embeddings = frozen_embeddings({k: data[k] for k in data.files})
    match_of = {f["video"]: f["match"] for f in fragments}

    report = {"thresholds_sha256": locked["file_sha256"], "primary": locked["primary"],
              "criterion": locked["criterion"], "review": review.get("reviewer"),
              "videos": {}, "configs": {}}
    all_rows = {}
    for config, point in locked["points"].items():
        name, method = config.split("|")
        rows = []
        for video in sorted(match_of):
            queries = fold_queries(fragments, crops, {video})
            video_rows = outcomes(queries, embeddings[name], method)
            for q, r in zip(queries, video_rows):
                r.update(video=video, match=match_of[video], team=q.team)
            rows += video_rows
        all_rows[config] = rows
        tau, margin = point["tau"], point["margin"]
        per_match = {m: point_with_parts([r for r in rows if r["match"] == m], tau, margin)
                     for m in sorted(set(match_of.values()))}
        pooled = point_with_parts(rows, tau, margin)
        accepted_wrong = sorted(
            ({k: r[k] for k in ("fragment", "identity", "best", "score", "margin", "duration_s", "known",
                                "synthetic_unknown")}
             for r in rows if r["score"] >= tau and r["margin"] >= margin and not r["correct"]),
            key=lambda r: -r["duration_s"])
        report["configs"][config] = {
            "locked": {"tau": tau, "margin": margin, "kiel_fold_a": point["kiel_fold_a"]},
            "pooled": pooled, "pooled_ci95": clustered_ci(rows, tau, margin),
            "per_match": per_match, "accepted_wrong": accepted_wrong,
            # descriptive only: never used to choose anything, the oracle is on these rows
            "descriptive_unlocked_summary": summarize(rows),
        }
    primary = report["configs"][locked["primary"]]
    report["verdict"] = verdict(primary["pooled"], locked["criterion"])
    report["videos"] = {v: {"match": m, "fragments": sum(f["video"] == v for f in fragments),
                            "identities": len({f["identity"] for f in fragments if f["video"] == v})}
                        for v, m in match_of.items()}
    (out / "transfer.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "rows_heldout.json").write_text(json.dumps(all_rows, default=float))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("lock", "evaluate"))
    parser.add_argument("--out", type=Path, default=STAGE1)
    args = parser.parse_args()
    if args.command == "lock":
        print(json.dumps(lock(), indent=2))
    else:
        report = evaluate(args.out)
        primary = report["configs"][report["primary"]]
        print(json.dumps({"verdict": report["verdict"], "pooled": primary["pooled"],
                          "ci95": primary["pooled_ci95"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
