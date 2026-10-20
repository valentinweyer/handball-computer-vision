"""One-shot fresh test of `locked/coverage_v1.json` on the higher-league bank.

Steps (each refuses to redo what is sealed):

    python -m experiments.teammate_reid.coverage2_fresh decide   # v2 + candidate, sealed together
    python -m experiments.teammate_reid.coverage2_fresh queue    # blind strips, shuffled codes
    python -m experiments.teammate_reid.coverage2_fresh score    # after labels are frozen

`decide` needs the bank's population (`stage2 --bank higher population`) and evidence
store (`coverage2 extract --banks higher`), both built after the gallery review.
Labels: `<OUT>/assistant_labels_v1.json` (+ `.sha256`), optional user adjudication
`<OUT>/user_adjudication_v1.json` (+ `.sha256`), reported apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.teammate_reid import stage2
from experiments.teammate_reid.coverage2_rules import Bank, Rule, decide

LOCK = Path(__file__).parent / "locked/coverage_v1.json"
OUT = stage2.RUNS / "coverage_fresh"
RANDOM_EXTRA = 40
SEED = 20261021


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rules() -> dict:
    lock = json.loads(LOCK.read_text())
    return {"v2": Rule(**lock["baseline"]["rule"]), "candidate": Rule(**lock["candidate"]["rule"])}


def cmd_decide() -> dict:
    path = OUT / "decisions_SEALED.json"
    if path.exists():
        raise SystemExit(f"{path} exists; decisions are sealed once")
    bank = Bank("higher")
    out = {name: {rid: decide(bank, rid, rule) for rid in bank.segments} for name, rule in rules().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lock_sha256": _sha(LOCK), "decisions": out}, default=float))
    (OUT / "decisions_SEALED.sha256").write_text(_sha(path) + "\n")
    return {k: sum(d["name"] is not None for d in v.values()) for k, v in out.items()}


def cmd_queue() -> dict:
    """Every segment either rule names, plus a seeded random sample of the rest; codes shuffled."""
    import cv2
    import pandas as pd
    path = OUT / "queue_SEALED.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    dec = json.loads((OUT / "decisions_SEALED.json").read_text())["decisions"]
    named = sorted({rid for v in dec.values() for rid, d in v.items() if d["name"] is not None})
    rest = sorted(set(dec["v2"]) - set(named))
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(rest, size=min(RANDOM_EXTRA, len(rest)), replace=False).tolist())
    items = [{"review_id": r, "group": "named"} for r in named] + [{"review_id": r, "group": "random"} for r in sample]
    order = rng.permutation(len(items))
    items = [{**items[i], "code": f"H{k:03d}"} for k, i in enumerate(order)]
    path.write_text(json.dumps({"items": items, "seed": SEED}, indent=1))
    stage2.configure("higher")
    pop = {p["review_id"]: p for p in json.loads((stage2.OUT / "population.json").read_text())["population"]}
    pages = OUT / "pages"
    pages.mkdir(exist_ok=True)
    strips = {}
    for match, spec in stage2.SPECS.items():
        its = [it for it in items if pop[it["review_id"]]["match"] == match]
        t = pd.read_parquet(spec.bundle / "tracks.parquet", columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        cap = cv2.VideoCapture(str(spec.bundle / json.loads((spec.bundle / "manifest.json").read_text())["files"]["video"]))
        for it in its:
            p = pop[it["review_id"]]
            tt = t[(t.track_id == p["track_id"]) & (t.frame >= p["start"]) & (t.frame <= p["end"])].set_index("frame")
            frames = list(tt.index[np.linspace(0, len(tt) - 1, min(12, len(tt))).round().astype(int)])
            tiles = []
            for f in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, im = cap.read()
                r = tt.loc[f]
                z = stage2._zoom(im, (r.x1, r.y1, r.x2, r.y2))
                cv2.putText(z, str(f), (6, 530), 0, 0.6, (255, 255, 255), 2)
                tiles.append(z)
            tiles += [np.zeros_like(tiles[0])] * (12 - len(tiles))
            head = np.full((40, 6 * stage2.ZOOM_W, 3), 25, np.uint8)
            cv2.putText(head, f"{it['code']}   {match}   ({p['duration_s']} s)", (8, 28), 0, 0.9, (0, 255, 255), 2)
            strips[it["code"]] = np.vstack([head, np.hstack(tiles[:6]), np.hstack(tiles[6:])])
    codes = sorted(strips)
    for k in range(0, len(codes), 2):
        cv2.imwrite(str(pages / f"page_{k // 2:02d}.jpg"), np.vstack([strips[c] for c in codes[k:k + 2]]),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    return {"named_by_either": len(named), "random": len(sample), "pages": (len(codes) + 1) // 2}


def _frozen(path: Path) -> dict:
    sha = path.with_suffix(".sha256")
    if not sha.exists() or sha.read_text().split()[0] != _sha(path):
        raise SystemExit(f"{path.name} is not frozen")
    return json.loads(path.read_text())["labels"]


def _cluster_ci(rows: list, reps: int = 2000, seed: int = 0) -> list:
    """Precision of `rows` (dicts with ok: bool, cluster) bootstrapped over clusters."""
    if not rows:
        return [None, None]
    by: dict = {}
    for r in rows:
        by.setdefault(r["cluster"], []).append(r["ok"])
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        sample = [x for k in rng.choice(len(keys), len(keys)) for x in by[keys[k]]]
        draws.append(np.mean(sample))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def cmd_score() -> dict:
    out_path = OUT / "result.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; the fresh bank is scored once")
    lock = json.loads(LOCK.read_text())
    sealed = OUT / "decisions_SEALED.json"
    assert (OUT / "decisions_SEALED.sha256").read_text().split()[0] == _sha(sealed)
    dec = json.loads(sealed.read_text())["decisions"]
    assert json.loads(sealed.read_text())["lock_sha256"] == _sha(LOCK)
    queue = {it["review_id"]: it for it in json.loads((OUT / "queue_SEALED.json").read_text())["items"]}
    labels = _frozen(OUT / "assistant_labels_v1.json")
    adj_path = OUT / "user_adjudication_v1.json"
    adj = _frozen(adj_path) if adj_path.exists() else {}
    bank = Bank("higher")

    def ref(rid):
        code = queue[rid]["code"] if rid in queue else None
        if code in adj:
            return {**adj[code], "source": "user_adjudicated"}
        if code in labels:
            return {**labels[code], "source": "assistant"}
        return None

    def outcome(rid, name):
        if name is None:
            return "abstained"
        r = ref(rid)
        if r is None:
            return "unlabelled"
        if r.get("flag") == "mixed":
            return "mixed"
        if r.get("flag") in ("not_a_player", "bench"):
            return "bench_or_nonplayer"
        if r.get("number") is None:
            return "unreadable"
        truth = (int(r["team"]), str(r["number"]))
        if tuple(name) == truth:
            seg = bank.segments[rid]
            return "correct" if (seg.get("on_court_frac") is None or seg["on_court_frac"] >= 0.5) else "correct_off_court"
        known = truth in {(t, n) for t, n, _ in bank.candidates(bank.segments[rid]).values()}
        return "wrong" if known else "unknown_named"

    per = {}
    for rid in bank.segments:
        a, b = dec["v2"][rid]["name"], dec["candidate"][rid]["name"]
        per[rid] = {"v2": outcome(rid, a), "candidate": outcome(rid, b), "v2_name": a, "candidate_name": b,
                    "changed": (tuple(a) if a else None) != (tuple(b) if b else None)}
    errs = ("wrong", "unknown_named", "mixed", "bench_or_nonplayer")
    added = [rid for rid, p in per.items() if p["changed"] and p["candidate_name"] is not None]
    lost = [rid for rid, p in per.items() if p["changed"] and p["v2_name"] is not None and p["candidate_name"] is None]
    added_out = Counter(per[r]["candidate"] for r in added)
    unverified = added_out["unreadable"] + added_out["unlabelled"]
    # synthetic unknowns on labelled segments whose player has a gallery
    from experiments.teammate_reid.coverage2_stress import synthetic_unknowns
    refs = {("higher", rid): [{"source": "human", "truth": (int(r["team"]), str(r["number"])), "flag": None}]
            for rid in bank.segments for r in [ref(rid)] if r and r.get("number") is not None and not r.get("flag")}
    simple = lambda refs_, key: (refs_.get(key) or [None])[0]
    syn = {name: dict(synthetic_unknowns(bank, rule, refs, simple)) for name, rule in rules().items()}
    pop_s = sum(s["duration_s"] * (1 - s["baseline_shown"]) for s in bank.segments.values())

    def summary(name):
        c = Counter(p[name] for p in per.values())
        new_s = sum(bank.segments[r]["duration_s"] * (1 - bank.segments[r]["baseline_shown"])
                    for r, p in per.items() if p[name] == "correct")
        zk = sum(1 for r, p in per.items() if p[name] == "correct" and bank.segments[r]["baseline_shown"] == 0)
        rows = [{"ok": p[name] == "correct", "cluster": f"{bank.segments[r]['match']}|{p[name + '_name']}"}
                for r, p in per.items() if p[name] in ("correct",) + errs]
        return {"counts": dict(c), "new_correct_seconds": new_s, "coverage_seconds": new_s / pop_s,
                "coverage_segments": c["correct"] / len(per), "zero_known_correct": zk,
                "precision_verified": c["correct"] / max(1, c["correct"] + sum(c[e] for e in errs)),
                "precision_ci95_clustered": _cluster_ci(rows)}

    s_v2, s_c = summary("v2"), summary("candidate")
    added_errors = sum(added_out[e] for e in errs) + sum(per[r]["candidate"] in errs for r in lost)
    verified_added = added_out["correct"]
    if added_errors > 0 or syn["candidate"].get("false_match", 0) > syn["v2"].get("false_match", 0):
        verdict = "FAIL"
    elif verified_added < 5 or (added and unverified > len(added) / 2):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS (no gross failure; not proof of non-inferiority)"
    gain = (s_c["counts"].get("correct", 0) - s_v2["counts"].get("correct", 0)) / max(1, s_v2["counts"].get("correct", 0))
    n = verified_added
    result = {"lock_sha256": _sha(LOCK), "labels_sha256": _sha(OUT / "assistant_labels_v1.json"),
              "adjudication_sha256": _sha(adj_path) if adj else None, "verdict": verdict,
              "coverage_target_met": gain >= 0.10, "relative_gain_correct": gain,
              "paired": {"added": dict(added_out), "lost": [(r, per[r]["v2"]) for r in lost],
                         "both_named_same": sum(1 for p in per.values() if not p["changed"] and p["v2_name"] is not None)},
              "added_precision_one_sided_lower95_if_zero_errors": (0.05 ** (1 / n)) if n and added_errors == 0 else None,
              "synthetic_unknowns": syn, "v2": s_v2, "candidate": s_c, "population": len(per),
              "zero_known_population": sum(1 for s in bank.segments.values() if s["baseline_shown"] == 0),
              "unnamed_player_seconds": pop_s, "quality_rule": lock["fresh_test"]["quality_rule"],
              "per_segment": per}
    out_path.write_text(json.dumps(result, indent=1, default=str))
    return {k: v for k, v in result.items() if k != "per_segment"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("decide", "queue", "score"))
    a = parser.parse_args()
    print(json.dumps({"decide": cmd_decide, "queue": cmd_queue, "score": cmd_score}[a.command](), indent=1, default=str))


if __name__ == "__main__":
    main()
