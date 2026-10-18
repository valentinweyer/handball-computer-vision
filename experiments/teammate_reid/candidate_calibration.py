"""Candidate-relative score calibration from the gallery (development data only).

The per-broadcast normalisation in `calibration.py` divided every score in a
video by one baseline drawn from the queries' own scores, and it made transfer
worse. This is the per-candidate version from the 2026-10-19 research note
(gallery-distance profiles / identity-conditioned score normalisation). It asks
whether a score is unusual *for this player among these teammates*.

For a query and each candidate teammate c:

- **Impostor profile of c.** Every other gallery player's fragment that exists
  at the query's time (same team, same video, ending >= 2 s before the query) is
  scored against c's gallery exactly as a query would be: its first 8 crops,
  top-3 mean. These are known non-matches by their own number reads. The
  query's own labels never enter.
- **Shrinkage.** The mean and variance are shrunk toward the profile pooled over
  all of this query's candidates with weight `PRIOR_N`, so a candidate with
  few impostor fragments cannot get an extreme z from a tiny sample.
- **Decision.** z_c = (s_c - mu_c) / sigma_c. Take the best candidate by z, then
  gate on z_best and on its gap to the runner-up. With one candidate there is no
  impostor and no runner-up, so the matcher abstains.

Gallery labels are number-vouched automatic labels, so this is not label-free.
It is gallery-only, though: no test-query label or test-query score is used.

    python -m experiments.teammate_reid.candidate_calibration
"""
from __future__ import annotations

import json

import numpy as np

from experiments.teammate_reid.calibration import OUT, SOURCES, TARGETS, embeddings, select_point
from experiments.teammate_reid.transfer import CRITERION, point_with_parts, verdict

M_CROPS = 8
TOP = 3
MIN_SEP = 50
PRIOR_N = 5.0
FEATURES = ("raw/frozen_stripes", "strict/frozen_stripes")


def aggregate(sims: np.ndarray) -> float:
    """Top-`TOP` mean per query crop, then the mean over query crops (as `score_query`)."""
    k = min(TOP, sims.shape[1])
    return float(np.sort(sims, axis=1)[:, -k:].mean(1).mean())


def shrunk(values: list, pooled_mu: float, pooled_var: float) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return pooled_mu, np.sqrt(pooled_var)
    mu = (n * np.mean(values) + PRIOR_N * pooled_mu) / (n + PRIOR_N)
    var = (n * np.var(values) + PRIOR_N * pooled_var) / (n + PRIOR_N)
    return float(mu), float(np.sqrt(max(var, 1e-8)))


def query_row(q, fragments_by_key: dict, crops_by_fragment: dict, sim: np.ndarray) -> dict:
    """Candidate-relative decision for one query (`sim` is crop x crop cosine for its video)."""
    gallery_frags = [fragments_by_key[k] for k in fragments_by_key
                     if fragments_by_key[k]["team"] == q.team and fragments_by_key[k]["identity"] in q.galleries
                     and k != q.fragment and fragments_by_key[k]["end"] + MIN_SEP <= q.start]
    qc = q.crops[:M_CROPS]
    raw, profile = {}, {}
    for identity, ids in q.galleries.items():
        raw[identity] = aggregate(sim[np.ix_(qc, ids)])
        profile[identity] = [aggregate(sim[np.ix_(crops_by_fragment[g["key"]][:M_CROPS], ids)])
                             for g in gallery_frags if g["identity"] != identity]
    pooled = [s for v in profile.values() for s in v]
    if len(raw) < 2 or len(pooled) < 2:
        best = max(raw, key=raw.get) if raw else None
        return {"best": best, "score": -np.inf, "margin": -np.inf, "candidates": len(raw)}
    pmu, pvar = float(np.mean(pooled)), float(np.var(pooled))
    z = {}
    for identity, s in raw.items():
        mu, sd = shrunk(profile[identity], pmu, pvar)
        z[identity] = (s - mu) / sd
    ranked = sorted(z.items(), key=lambda kv: -kv[1])
    return {"best": ranked[0][0], "score": ranked[0][1], "margin": ranked[0][1] - ranked[1][1],
            "candidates": len(raw)}


def candidate_rows(results, matches: tuple, feature: str) -> list[dict]:
    from experiments.teammate_reid.benchmark import fold_queries, load_reviewed
    crops, fragments, _ = load_reviewed(results)
    data = np.load(results / "features.npz")
    emb = embeddings({k: data[k] for k in data.files})[feature]
    by_fragment: dict = {}
    for c in sorted(crops, key=lambda c: c["frame"]):
        by_fragment.setdefault(c["fragment"], []).append(c["crop_id"])
    rows = []
    for video in sorted({f["video"] for f in fragments if f["match"] in matches}):
        frags = {f["key"]: f for f in fragments if f["video"] == video and by_fragment.get(f["key"])}
        ids = sorted({i for k in frags for i in by_fragment[k]})
        index = {c: i for i, c in enumerate(ids)}
        local = emb[ids]
        sim = local @ local.T
        local_crops = {k: [index[c] for c in by_fragment[k]] for k in frags}
        match = next(iter(frags.values()))["match"]
        for q in fold_queries(list(frags.values()), crops, {video}):
            # re-index the query and its galleries into this video's similarity matrix
            q.crops = [index[c] for c in q.crops]
            q.galleries = {i: [index[c] for c in v] for i, v in q.galleries.items()}
            decision = query_row(q, frags, local_crops, sim)
            rows.append({"video": video, "match": match, "fragment": q.fragment, "identity": q.identity,
                         "known": q.known, "synthetic_unknown": q.synthetic_unknown, "reentry": q.reentry,
                         **decision, "correct": bool(q.known and decision["best"] == q.identity),
                         "duration_s": q.meta["duration_s"]})
    return rows


def lomo() -> dict:
    report = {"criterion": CRITERION, "prior_n": PRIOR_N, "folds": {}}
    for feature in FEATURES:
        rows = [r for results, matches in SOURCES.items() for r in candidate_rows(results, matches, feature)]
        known = [r for r in rows if r["known"]]
        report.setdefault("rank1", {})[feature] = float(np.mean([r["correct"] for r in known]))
        for target in TARGETS:
            key = f"{feature}|candidate_z|train_p{target:.2f}"
            report["folds"][key] = {}
            for test in sorted({r["match"] for r in rows}):
                chosen = select_point([r for r in rows if r["match"] != test], target)
                if chosen is None:
                    report["folds"][key][test] = {"chosen": None}
                    continue
                result = point_with_parts([r for r in rows if r["match"] == test], chosen["tau"], chosen["margin"])
                report["folds"][key][test] = {
                    "chosen": {k: chosen[k] for k in ("tau", "margin", "coverage", "precision")},
                    "test": result, "verdict": verdict(result, CRITERION)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lomo_candidate.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main() -> None:
    report = lomo()
    print("rank-1 by candidate z:", {k: round(v, 3) for k, v in report["rank1"].items()})
    print(f"{'config':48s} {'test match':20s} {'z/gap':>11s} {'cov':>5s} {'prec':>5s} {'acc':>4s} {'fmr':>5s} {'natf':>5s}  pass")
    for key, folds in report["folds"].items():
        for test, f in folds.items():
            if f["chosen"] is None:
                print(f"{key:48s} {test:20s}   no training point meets the bounds")
                continue
            t, c = f["test"], f["chosen"]
            print(f"{key:48s} {test:20s} {c['tau']:5.2f}/{c['margin']:5.2f} {t['coverage']:5.2f} {t['precision']:5.2f} "
                  f"{t['accepted']:4d} {t['unknown_false_match_rate']:5.3f} {t['unknown_false_match_rate_natural']:5.3f}  "
                  f"{'PASS' if f['verdict']['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
