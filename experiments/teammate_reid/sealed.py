"""Evaluate the locked open-set rule on the sealed matches, once (criterion v2).

Reads only `locked/thresholds_sealed.json`; nothing here chooses a threshold.
Known-player precision, the unknown false-match rate and coverage are bounded
separately, pooled over the sealed matches; per-match results and clustered
95% intervals (identities resampled within each match) are reported beside them.

    python -m experiments.teammate_reid.sealed --out runs/smart-reid-shadow/sealed_preview --tag preview
    python -m experiments.teammate_reid.sealed --out runs/smart-reid-shadow/sealed --tag verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.teammate_reid.openset import CRITERION, decide, passes, query_rows, score

LOCK = Path(__file__).parent / "locked/thresholds_sealed.json"


def is_error(row: dict, gate: tuple) -> bool:
    """An accepted proposal that is wrong: a known player linked to a teammate, or any link for an unknown."""
    proposed = decide(row, "absolute", gate)
    return proposed is not None and (not row["known"] or proposed != row["identity"])


def clustered_ci(rows: list, gate: tuple, reps: int = 2000, seed: int = 0) -> dict:
    clusters: dict = {}
    for r in rows:
        clusters.setdefault(r["match"], {}).setdefault(r["identity"], []).append(r)
    rng = np.random.default_rng(seed)
    keys = ("known_precision", "coverage", "unknown_false_match_rate")
    draws = {k: [] for k in keys}
    for _ in range(reps):
        sample = [r for by_id in clusters.values() for i in rng.integers(0, len(by_id), len(by_id))
                  for r in list(by_id.values())[i]]
        s = score(sample, "absolute", gate)
        for k in keys:
            if np.isfinite(s[k]):
                draws[k].append(s[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
            for k, v in draws.items()}


def evaluate(out: Path, tag: str) -> dict:
    lock = json.loads(LOCK.read_text())
    from experiments.teammate_reid.benchmark import load_reviewed
    _, fragments, review = load_reviewed(out)
    matches = tuple(sorted({f["match"] for f in fragments}))
    leaked = set(matches) & set(lock["development_matches"])
    if leaked:
        raise SystemExit(f"sealed evaluation would include development matches: {sorted(leaked)}")
    rows = query_rows(out, matches)
    gate = (lock["tau"], lock["margin"], 0)
    pooled = score(rows, "absolute", gate)
    report = {
        "tag": tag, "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "gate": {"tau": lock["tau"], "margin": lock["margin"]}, "criterion": CRITERION,
        "review": review.get("reviewer"), "pooled": pooled, "pooled_ci95": clustered_ci(rows, gate),
        "pass": bool(passes(pooled, CRITERION["known_precision_min"], CRITERION["unknown_false_match_rate_max"])),
        "per_match": {m: score([r for r in rows if r["match"] == m], "absolute", gate) for m in matches},
        "accepted_errors": sorted(
            ({k: r[k] for k in ("match", "fragment", "identity", "best", "score", "margin", "known",
                                "synthetic_unknown", "duration_s")} for r in rows if is_error(r, gate)),
            key=lambda r: -r["duration_s"]),
    }
    (out / f"sealed_{tag}.json").write_text(json.dumps(report, indent=2, default=float))
    (out / f"rows_{tag}.json").write_text(json.dumps(rows, default=float))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    r = evaluate(args.out, args.tag)
    print(json.dumps({k: r[k] for k in ("tag", "gate", "pass", "pooled", "pooled_ci95", "per_match")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
