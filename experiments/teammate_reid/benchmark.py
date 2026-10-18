"""Gallery-memory vs learned-appearance ablation on mined, reviewed fragments.

Folds (the split unit is the underlying match, never a video):
- ``A_cross_match``: train on every non-Kiel match, test on Kiel-Lemgo.
- ``B_same_match_other_team``: test one Kiel team, train on the non-Kiel matches
  plus the other Kiel team -- same broadcast, disjoint people. Weaker
  separation than A, reported separately and never pooled with it.
- ``C_reverse``: train on Kiel-Lemgo, test on BHC-FAG and Melsungen-Berlin.

Frozen embeddings need no training data and are scored on exactly the same
queries as each fold's learned heads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import time

import numpy as np

from experiments.teammate_reid.evaluation import (
    METHODS, build_queries, chance_rank1, cluster_bootstrap, l2n, outcomes, runtime_point, summarize)
from experiments.teammate_reid.features import STRIPE_NAMES
from experiments.teammate_reid.train import embed, train_head

SEEDS = (0, 1, 2)
MASK_MODES = ("digits", "strict")
TEST_MATCH = "Kiel-Lemgo"


def load_reviewed(out: Path) -> tuple[list, list, dict]:
    crops = json.loads((out / "crops.json").read_text())
    fragments = [f for f in json.loads((out / "fragments.json").read_text()) if f["status"] == "accepted"]
    review_path = out / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {"excluded": {}}
    excluded = set(review.get("excluded", {}))
    excluded_ids = set(review.get("excluded_identities", {}))
    keep = [f for f in fragments if f["key"] not in excluded and f["identity"] not in excluded_ids]
    keys = {f["key"] for f in keep}
    crops = [c for c in crops if c["fragment"] in keys]
    return crops, keep, review


def frozen_embeddings(feats: dict) -> dict:
    out = {"raw/frozen_global": l2n(feats["raw_global"])}
    for mode in MASK_MODES:
        g = l2n(feats[f"{mode}_global"])
        s = l2n(feats[f"{mode}_stripes"].astype(np.float32))
        out[f"{mode}/frozen_global"] = g
        out[f"{mode}/frozen_stripes"] = l2n(np.concatenate([s.reshape(len(s), -1), g], 1))
    return out


def fold_queries(fragments: list, crops: list, videos: set, team: int | None = None) -> list:
    by_fragment: dict = {}
    for c in sorted(crops, key=lambda c: c["frame"]):
        by_fragment.setdefault(c["fragment"], []).append(c["crop_id"])
    queries = []
    for video in sorted(videos):
        frags = [f for f in fragments if f["video"] == video and (team is None or f["team"] == team)]
        queries += build_queries(frags, by_fragment)
    return queries


def evaluate_all(queries: list, embeddings: dict, configs: list) -> dict:
    results = {}
    for name, method, anchor_only in configs:
        rows = outcomes(queries, embeddings[name], method, anchor_only=anchor_only)
        tag = f"{name}|{method}{'|anchor' if anchor_only else ''}"
        summary = summarize(rows)
        if name.endswith("frozen_global") and method == "ema_single" and not anchor_only:
            summary["runtime_gates"] = runtime_point(rows)
        results[tag] = {"summary": summary, "rows": rows}
    return results


def run_benchmark(out: Path, device: str = "cuda") -> dict:
    out = Path(out)
    crops, fragments, review = load_reviewed(out)
    data = np.load(out / "features.npz")
    feats = {k: data[k] for k in data.files}
    row_of = {c["crop_id"]: c for c in crops}
    frozen = frozen_embeddings(feats)

    kiel_teams = sorted({f["team"] for f in fragments if f["match"] == TEST_MATCH})
    folds = {
        "A_cross_match": [({c["video"] for c in crops if c["match"] == TEST_MATCH}, None,
                           lambda c: c["match"] != TEST_MATCH)],
        "B_same_match_other_team": [
            ({c["video"] for c in crops if c["match"] == TEST_MATCH}, t,
             (lambda t: lambda c: c["match"] != TEST_MATCH or c["team"] != t)(t)) for t in kiel_teams],
        "C_reverse": [({c["video"] for c in crops if c["match"] in ("BHC-FAG", "Melsungen-Berlin")},
                       None, lambda c: c["match"] not in ("BHC-FAG", "Melsungen-Berlin"))],
    }
    frozen_configs = [(name, m, False) for name in frozen for m in METHODS] + \
        [(name, "gallery_agg", True) for name in frozen]

    report = {"folds": {}, "review": {k: v for k, v in review.items() if k != "notes"},
              "data": {"crops": len(crops), "fragments": len(fragments)}}
    for fold, parts in folds.items():
        fold_rows: dict = {}
        train_infos, timing = [], {}
        for test_videos, team, is_train in parts:
            queries = fold_queries(fragments, crops, test_videos, team)
            train_crops = [c for c in crops if is_train(c)]
            # guard: no test match fragment may train a head, except fold B's other team
            assert not any(c["video"] in test_videos and (team is None or c["team"] == team)
                           for c in train_crops)
            train_ids = np.array([c["crop_id"] for c in train_crops])
            labels = [(c["video"], c["team"], c["identity"]) for c in train_crops]
            embeddings = dict(frozen)
            learned_configs = []
            for mode in MASK_MODES:
                S, G = feats[f"{mode}_stripes"], feats[f"{mode}_global"]
                for kind in ("proj_global", "stripe_head"):
                    for seed in SEEDS:
                        t0 = time.time()
                        head, info = train_head(kind, S[train_ids], G[train_ids], labels,
                                                seed=seed, device=device)
                        timing.setdefault(f"{mode}/{kind}/train_s", []).append(time.time() - t0)
                        t0 = time.time()
                        name = f"{mode}/{kind}/s{seed}"
                        embeddings[name] = embed(head, S, G, device)
                        timing.setdefault(f"{mode}/{kind}/embed_ms_per_crop", []).append(
                            1000 * (time.time() - t0) / len(G))
                        train_infos.append({**info, "mode": mode, "team_tested": team})
                        learned_configs += [(name, m, False) for m in METHODS]
                        if kind == "stripe_head" and mode == "digits":
                            for drop, region in enumerate((*STRIPE_NAMES, "global")):
                                ab = f"{name}/drop_{region}"
                                embeddings[ab] = embed(head, S, G, device, drop=(drop,))
                                learned_configs.append((ab, "gallery_agg", False))
            res = evaluate_all(queries, embeddings, frozen_configs + learned_configs)
            for tag, r in res.items():
                fold_rows.setdefault(tag, []).extend(r["rows"])
            fold_rows.setdefault("_queries", []).extend(
                [{"fragment": q.fragment, "identity": q.identity, "known": q.known,
                  "synthetic_unknown": q.synthetic_unknown, "reentry": q.reentry,
                  "candidates": len(q.galleries), "chance": 1 / len(q.galleries)} for q in queries])
            fold_rows.setdefault("_chance", []).append(chance_rank1(queries))

        query_list = fold_rows.pop("_queries")
        chance = float(np.mean([q["chance"] for q in query_list if q["known"]])) if query_list else None
        fold_rows.pop("_chance")
        summaries = {tag: summarize(rows) for tag, rows in fold_rows.items()}
        for tag in summaries:
            if tag.endswith("frozen_global|ema_single"):
                summaries[tag]["runtime_gates"] = runtime_point(fold_rows[tag])
        # learned: mean and spread across seeds
        learned = {}
        for tag, s in summaries.items():
            parts_ = tag.split("|")
            name = parts_[0]
            if not re.search(r"/s\d+", name):
                continue
            key = re.sub(r"/s\d+", "", name) + "|" + "|".join(parts_[1:])
            learned.setdefault(key, []).append(s)
        seed_mean = {k: {m: [float(np.nanmean([x[m] for x in v])), float(np.nanstd([x[m] for x in v]))]
                         for m in ("rank1", "rank1_reentry", "coverage_at_p95", "coverage_at_p98",
                                   "precision_at_cov50")} for k, v in learned.items()}
        # paired, identity-clustered bootstrap against the runtime-shaped frozen baseline
        boot = {}
        for mode in MASK_MODES:
            base_tag = f"{mode}/frozen_global|ema_single"
            for other in (f"{mode}/frozen_global|gallery_agg", f"{mode}/frozen_stripes|gallery_agg",
                          f"{mode}/stripe_head/s0|gallery_agg", f"{mode}/proj_global/s0|gallery_agg",
                          f"{mode}/stripe_head/s0|ema_single"):
                a, b = fold_rows.get(base_tag), fold_rows.get(other)
                if not a or not b:
                    continue
                boot[f"{other} - {base_tag}"] = {
                    "rank1": cluster_bootstrap(
                        a, b, lambda rs: np.mean([r["correct"] for r in rs if r["known"]])
                        if any(r["known"] for r in rs) else np.nan),
                    "coverage_at_p95": cluster_bootstrap(
                        a, b, lambda rs: summarize(rs)["coverage_at_p95"], reps=300),
                }
        report["folds"][fold] = {
            "queries": {"known": sum(q["known"] for q in query_list),
                        "unknown_natural": sum((not q["known"]) and not q["synthetic_unknown"]
                                               for q in query_list),
                        "unknown_synthetic": sum(q["synthetic_unknown"] for q in query_list),
                        "reentry": sum(q["reentry"] and q["known"] for q in query_list),
                        "identities": len({q["identity"] for q in query_list if q["known"]})},
            "chance_rank1": chance,
            "summaries": summaries, "learned_seed_mean_sd": seed_mean, "bootstrap": boot,
            "train": [{k: v for k, v in i.items() if k != "log"} for i in train_infos],
            "train_logs": [i["log"] for i in train_infos[:4]],
            "timing": {k: [float(np.mean(v)), float(np.std(v))] for k, v in timing.items()},
        }
        (out / f"rows_{fold}.json").write_text(json.dumps(
            {k: v for k, v in fold_rows.items() if "/drop_" not in k and ("/s1" not in k and "/s2" not in k)},
            default=float))
        print(fold, json.dumps(report["folds"][fold]["queries"]), "chance", chance)
    (out / "benchmark.json").write_text(json.dumps(report, indent=1, default=float))
    return report
