"""Open-set rejection rules, leave-one-match-out over the development bank.

The locked stripe rule failed its fresh test only on players with no gallery:
every accepted error was a gallery-absent player matched to a teammate, and no
known player was ever linked to the wrong teammate. This cycle compares cheap
rejection rules on all development matches. Every rule uses full-crop stripes
and gallery + 8-crop aggregation; only the acceptance decision changes.

- R0 `absolute`: best raw score >= tau and margin over the runner-up >= m.
- R1 `support`: R0, plus the chosen player's gallery holds >= k trusted fragments.
- R2 `agreement`: R0, plus at least a share `a` of the query's crops pick the same
  player on their own (an unknown player tends to scatter over teammates).
- R3 `z_rank`: pick by candidate-relative z (`candidate_calibration`), gate on that
  player's raw score and on the z gap to the runner-up.

Scored against criterion v2 (`locked/criterion_v2.json`): known-player precision,
the unknown false-match rate and coverage, each bounded separately. Gates are
chosen on the training matches of each fold at the bounds and at a tighter
margin, and applied unchanged to the held-out match.

    python -m experiments.teammate_reid.openset
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from experiments.teammate_reid.calibration import embeddings
from experiments.teammate_reid.candidate_calibration import aggregate, query_row as z_decision

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
RUNS = REPO / "runs/smart-reid-shadow"
DEV_SOURCES = {
    REPO / "runs/smart-reid-prototype-2026-10-18/results": ("Kiel-Lemgo",),
    RUNS / "stage1": ("Eisenach-Hamburg", "Gummersbach-Loewen"),
    RUNS / "fresh": ("Ludwigshafen-Wetzlar", "RNL-SGF"),
    RUNS / "dev": ("Balingen-Kiel", "Berlin-BHC"),
    RUNS / "dev_b": ("Lemgo-Magdeburg", "Goeppingen-Leipzig"),
}
CRITERION = json.loads((Path(__file__).parent / "locked/criterion_v2.json").read_text())
TARGETS = {"at_bounds": (CRITERION["known_precision_min"], CRITERION["unknown_false_match_rate_max"]),
           "tight": (0.97, 0.02)}
FEATURE = "raw/frozen_stripes"
M_CROPS = 8
GRID = 30


def query_rows(results: Path, matches: tuple, feature: str = FEATURE) -> list[dict]:
    """One row per query with every quantity any rule gates on."""
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
        sim = emb[ids] @ emb[ids].T
        local_crops = {k: [index[c] for c in by_fragment[k]] for k in frags}
        match = next(iter(frags.values()))["match"]
        for q in fold_queries(list(frags.values()), crops, {video}):
            q.crops = [index[c] for c in q.crops]
            q.galleries = {i: [index[c] for c in v] for i, v in q.galleries.items()}
            qc = q.crops[:M_CROPS]
            raw = {i: aggregate(sim[np.ix_(qc, g)]) for i, g in q.galleries.items()}
            ranked = sorted(raw.items(), key=lambda kv: -kv[1])
            best, s1 = ranked[0]
            margin = s1 - ranked[1][1] if len(ranked) > 1 else -np.inf
            per_crop = [max(q.galleries, key=lambda i: np.sort(sim[c, q.galleries[i]])[-3:].mean())
                        for c in qc]
            support = sum(1 for f in frags.values() if f["identity"] == best and f["team"] == q.team
                          and f["key"] != q.fragment and f["end"] + 50 <= q.start)
            z = z_decision(q, frags, local_crops, sim)
            rows.append({
                "video": video, "match": match, "fragment": q.fragment, "identity": q.identity,
                "known": q.known, "synthetic_unknown": q.synthetic_unknown, "duration_s": q.meta["duration_s"],
                "best": best, "score": s1, "margin": margin, "candidates": len(raw),
                "agreement": float(np.mean([p == best for p in per_crop])),
                "support": support, "query_crops": len(qc),
                "z_best": z["best"], "z_gap": z["margin"], "z_raw": raw.get(z["best"], -np.inf),
            })
    return rows


def decide(row: dict, rule: str, gate: tuple) -> str | None:
    """The proposed identity, or None to abstain."""
    if rule == "z_rank":
        tau, gap = gate
        ok = row["z_raw"] >= tau and row["z_gap"] >= gap
        return row["z_best"] if ok else None
    tau, margin, extra = gate
    ok = row["score"] >= tau and row["margin"] >= margin
    if rule == "support":
        ok = ok and row["support"] >= extra
    elif rule == "agreement":
        ok = ok and row["agreement"] >= extra
    return row["best"] if ok else None


def score(rows: list, rule: str, gate: tuple) -> dict:
    known = [r for r in rows if r["known"]]
    unknown = [r for r in rows if not r["known"]]
    kp = [(r, decide(r, rule, gate)) for r in known]
    acc_known = [(r, p) for r, p in kp if p is not None]
    correct = [r for r, p in acc_known if p == r["identity"]]
    fm = [r for r in unknown if decide(r, rule, gate) is not None]
    natural = [r for r in unknown if not r["synthetic_unknown"]]
    return {
        "known_queries": len(known), "unknown_queries": len(unknown),
        "accepted_known": len(acc_known), "correct": len(correct),
        "known_precision": len(correct) / len(acc_known) if acc_known else float("nan"),
        "coverage": len(correct) / max(1, len(known)),
        "unknown_false_match_rate": len(fm) / max(1, len(unknown)),
        "natural_unknowns": len(natural),
        "natural_false_matches": sum(decide(r, rule, gate) is not None for r in natural),
        "wrong_seconds": float(sum(r["duration_s"] for r, p in acc_known if p != r["identity"])
                               + sum(r["duration_s"] for r in fm)),
        "correct_seconds": float(sum(r["duration_s"] for r in correct)),
    }


def passes(s: dict, kp_min: float, fmr_max: float, cov_min: float = CRITERION["coverage_min"]) -> bool:
    return (np.isfinite(s["known_precision"]) and s["known_precision"] >= kp_min
            and s["unknown_false_match_rate"] <= fmr_max and s["coverage"] >= cov_min)


def gates(rows: list, rule: str) -> list:
    q = lambda key: np.unique(np.quantile([r[key] for r in rows if np.isfinite(r[key])], np.linspace(0, 1, GRID)))
    if rule == "z_rank":
        return list(itertools.product(q("z_raw"), q("z_gap")))
    extra = {"absolute": [0], "support": [1, 2, 3], "agreement": [0.5, 0.625, 0.75, 0.875, 1.0]}[rule]
    return list(itertools.product(q("score"), q("margin"), extra))


def choose(rows: list, rule: str, kp_min: float, fmr_max: float) -> tuple | None:
    """Highest-coverage gate on these (training) rows meeting the precision and false-match bounds."""
    best = None
    for gate in gates(rows, rule):
        s = score(rows, rule, gate)
        if s["accepted_known"] and s["known_precision"] >= kp_min and s["unknown_false_match_rate"] <= fmr_max:
            if best is None or s["coverage"] > best[1]["coverage"]:
                best = (gate, s)
    return best


def lomo(rows: list) -> dict:
    report = {"criterion": CRITERION, "targets": TARGETS, "rules": {}}
    matches = sorted({r["match"] for r in rows})
    for rule in ("absolute", "support", "agreement", "z_rank"):
        for tname, (kp, fmr) in TARGETS.items():
            folds, held = {}, []
            for test in matches:
                chosen = choose([r for r in rows if r["match"] != test], rule, kp, fmr)
                test_rows = [r for r in rows if r["match"] == test]
                if chosen is None:
                    folds[test] = {"gate": None}
                    continue
                s = score(test_rows, rule, chosen[0])
                folds[test] = {"gate": [float(g) for g in chosen[0]], "test": s,
                               "pass": passes(s, CRITERION["known_precision_min"],
                                              CRITERION["unknown_false_match_rate_max"])}
                held += [dict(r, _gate=chosen[0]) for r in test_rows]
            # pooled held-out: each match scored at its own fold's gate
            pooled = {k: 0 for k in ("known", "acc", "correct", "unknown", "fm")}
            for r in held:
                p = decide(r, rule, r["_gate"])
                if r["known"]:
                    pooled["known"] += 1
                    pooled["acc"] += p is not None
                    pooled["correct"] += p == r["identity"]
                else:
                    pooled["unknown"] += 1
                    pooled["fm"] += p is not None
            summary = {"known_precision": pooled["correct"] / max(1, pooled["acc"]),
                       "coverage": pooled["correct"] / max(1, pooled["known"]),
                       "unknown_false_match_rate": pooled["fm"] / max(1, pooled["unknown"]),
                       "folds_passed": sum(f.get("pass", False) for f in folds.values()), "folds": len(folds)}
            report["rules"][f"{rule}|{tname}"] = {"folds": folds, "pooled_lomo": summary}
    return report


def main() -> None:
    rows = [r for results, matches in DEV_SOURCES.items() for r in query_rows(results, matches)]
    out = RUNS / "openset"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, default=float))
    report = lomo(rows)
    (out / "lomo.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"{len(rows)} rows from {len({r['match'] for r in rows})} matches")
    print(f"{'rule|target':24s} {'folds':>6s} {'known prec':>10s} {'coverage':>9s} {'unk FMR':>8s}")
    for key, v in report["rules"].items():
        s = v["pooled_lomo"]
        print(f"{key:24s} {s['folds_passed']:>2d}/{s['folds']:<3d} {s['known_precision']:10.3f} "
              f"{s['coverage']:9.3f} {s['unknown_false_match_rate']:8.3f}")


if __name__ == "__main__":
    main()
