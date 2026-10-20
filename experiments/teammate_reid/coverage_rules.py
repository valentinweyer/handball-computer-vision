"""Rule variants for smart re-ID on baseline-unresolved segments, scored on the dev banks' labels.

Reads the evidence cache of `coverage.extract` and the human labels of both spent
Stage 2 banks (their `score_v3.json` items: blind labels, blind re-check, and the
post-unsealing adjudication). A rule is a pure function of one segment's cached
evidence, so every variant is evaluated without the GPU.

The labelled segments are a stratified sample: in each bank every segment the old
rule accepted was labelled, plus 40 drawn at random from the rest. Population
estimates weight each random item by (non-accepted segments / 40). Raw counts are
reported beside them.

Outcomes per labelled segment and rule: `correct` (named the labelled player),
`wrong` (named someone else, or a non-player), `unknown_named` (wrong, and the
labelled player had no gallery at all: a real unknown named), `abstained`, or
`unscored` (label unclear/mixed).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from experiments.teammate_reid import stage2
from experiments.teammate_reid.coverage import BANKS, OUT
from experiments.teammate_reid.symmetric import _pick

FPS = 25.0


@dataclass(frozen=True)
class Rule:
    tau: float = 0.7839630554462301          # split-v2 lock
    margin: float = 0.02255046162112006
    single_tau: float | None = None           # accept a lone candidate at this score; None = never
    co_min_frames: int = 1                    # veto at >= this many co-present frames (old: 1)
    co_ignore_duplicates: bool = False        # do not count frames where the other box is this box (IoU >= 0.5)
    read_min: int = 2                         # reads needed for a number to contradict (old: 2)
    vmax_cm_s: float | None = None            # veto a candidate whose nearest gallery fragment implies a faster move
    halves: bool = True
    unsettled: str = "allow"                  # "allow" (both teams' galleries) | "majority" (its majority team) | "abstain"
    min_team_gallery: int = 0                 # abstain when the segment's team has fewer gallery identities


V2 = Rule()


def _supported(reads: dict, min_count: int) -> set:
    full = {v for v, c in reads.items() if c >= min_count and len(v) > 1}
    return full | {v for v, c in reads.items() if c >= min_count and len(v) == 1 and not any(f.endswith(v) for f in full)}


def _on_court(pos) -> bool:
    """Court positions spike far off court when feet are hidden or the fit is poor; ignore those."""
    return pos is not None and -300 <= pos[0] <= 4300 and -300 <= pos[1] <= 2300


def _too_fast(seg: dict, cand: dict, vmax: float) -> bool:
    for side, here, gap in (("before", seg["start_pos"], lambda c: seg["start"] - c["frame"]),
                            ("after", seg["end_pos"], lambda c: c["frame"] - seg["end"])):
        c = cand[side]
        if c is None or not _on_court(c["pos"]) or not _on_court(here):
            continue
        dt = max(gap(c), 1) / FPS
        dist = float(np.hypot(c["pos"][0] - here[0], c["pos"][1] - here[1]))
        if dt <= 10.0 and dist / dt > vmax:
            return True
    return False


def decide(seg: dict, rule: Rule, scores=None, vetoed: frozenset = frozenset()) -> dict:
    """The rule's decision for one segment: name (team, number) or None, with the reason.

    `scores(candidate)` overrides the per-crop scores (a gallery grown by chaining);
    `vetoed` holds extra (team, number) identities ruled out for this segment."""
    if rule.unsettled == "abstain" and not seg["settled"]:
        return {"name": None, "reason": "team_unsettled"}
    if seg["team_gallery_size"] < rule.min_team_gallery:
        return {"name": None, "reason": "thin_gallery"}
    supported = _supported(seg["own_reads"], rule.read_min)
    n = len(seg["crop_frames"])
    spread = _pick(list(range(n)), "spread8")
    h = n // 2
    alive = []
    for c in seg["candidates"]:
        if not c["crop_scores"]:
            continue
        if rule.unsettled == "majority" and not seg["settled"] and c["team"] != seg["team"]:
            continue
        co = c["co_frames"] - (c["dup_frames"] if rule.co_ignore_duplicates else 0)
        if co >= rule.co_min_frames or c["frag_overlap_frames"] > 0:
            continue
        if supported and c["number"] not in supported:
            continue
        if rule.vmax_cm_s is not None and _too_fast(seg, c, rule.vmax_cm_s):
            continue
        if (c["team"], c["number"]) in vetoed:
            continue
        s = np.asarray(c["crop_scores"]) if scores is None else scores(c)
        alive.append((float(s[spread].mean()), c, s))
    alive.sort(key=lambda x: -x[0])
    if not alive:
        return {"name": None, "reason": "no_candidate"}
    best_score, best, _ = alive[0]
    name = (best["team"], best["number"])
    if len(alive) == 1:
        ok = rule.single_tau is not None and best_score >= rule.single_tau
        return {"name": name if ok else None, "reason": "single" if ok else "single_candidate", "score": best_score}
    margin = best_score - alive[1][0]
    if best_score < rule.tau or margin < rule.margin:
        return {"name": None, "reason": "below_gate", "score": best_score, "margin": margin}
    if rule.halves and h >= 2:
        halves = []
        for part in (list(range(h)), list(range(h, n))):
            idx = _pick(part, "spread8")
            halves.append(max(alive, key=lambda a: a[2][idx].mean())[1]["identity"])
        if halves[0] != halves[1]:
            return {"name": None, "reason": "halves_disagree"}
    return {"name": name, "reason": "accepted", "score": best_score, "margin": margin}


def chained(bank: dict, first: Rule, second: Rule, min_sep: int = 50) -> dict:
    """Two passes. Names from `first` join their player's gallery for the second pass
    (only from segments >= `min_sep` frames away and on another track, as galleries
    are built), and veto that player on every segment overlapping them in time.
    Returns review_id -> decision; pass-1 names are kept as they were."""
    emb = np.load(OUT / f"{bank['bank']}_query_embeddings.npz")
    emb = {k: emb[k].astype(np.float32) for k in emb.files}
    for k, v in emb.items():
        emb[k] = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)
    first_pass = {rid: decide(seg, first) for rid, seg in bank["segments"].items()}
    named: dict = {}
    for rid, d in first_pass.items():
        if d["name"] is not None:
            seg = bank["segments"][rid]
            named.setdefault((seg["match"], tuple(d["name"])), []).append(seg)
    out = {}
    for rid, seg in bank["segments"].items():
        if first_pass[rid]["name"] is not None:
            out[rid] = {**first_pass[rid], "pass": 1}
            continue
        vetoed = frozenset(name for (match, name), segs in named.items() if match == seg["match"]
                           and any(o["start"] <= seg["end"] and o["end"] >= seg["start"] for o in segs))
        q = emb[rid]

        def scores(c, seg=seg, q=q):
            top = np.asarray(c["crop_top3"], dtype=np.float32)
            extra = [emb[o["review_id"]] for o in named.get((seg["match"], (c["team"], c["number"])), [])
                     if o["track_id"] != seg["track_id"]
                     and (o["end"] + min_sep <= seg["start"] or o["start"] >= seg["end"] + min_sep)]
            if not extra:
                return top.mean(1)
            sims = q @ np.concatenate(extra).T
            return np.sort(np.concatenate([top, sims], axis=1), axis=1)[:, -3:].mean(1)

        out[rid] = {**decide(seg, second, scores=scores, vetoed=vetoed), "pass": 2}
    return out


def evaluate_decisions(banks: list, decisions: list) -> dict:
    """`evaluate` for precomputed decisions (one dict per bank)."""
    tot = {"correct": 0, "wrong": 0, "unknown_named": 0, "est_correct": 0.0, "est_wrong": 0.0,
           "est_correct_s": 0.0, "population": 0, "seconds": 0.0, "named_unlabelled": 0}
    for b, dec in zip(banks, decisions):
        tot["population"] += b["population"]
        tot["seconds"] += b["seconds"]
        for rid, seg in b["segments"].items():
            name = dec[rid]["name"]
            lab = b["labels"].get(rid)
            if lab is None:
                tot["named_unlabelled"] += name is not None
                continue
            o = outcome(seg, lab, name)
            w = b["weights"][rid]
            if o == "correct":
                tot["correct"] += 1
                tot["est_correct"] += w
                tot["est_correct_s"] += w * seg["duration_s"]
            elif o in ("wrong", "unknown_named"):
                tot[o] += 1
                tot["est_wrong"] += w
    errs = tot["wrong"] + tot["unknown_named"]
    tot["precision_labelled"] = tot["correct"] / max(1, tot["correct"] + errs)
    tot["est_precision"] = tot["est_correct"] / max(1e-9, tot["est_correct"] + tot["est_wrong"])
    tot["est_coverage"] = tot["est_correct"] / tot["population"]
    tot["est_coverage_s"] = tot["est_correct_s"] / tot["seconds"]
    return tot


def load_bank(bank: str) -> dict:
    """Evidence segments joined with their labels and strata."""
    stage2.configure(bank)
    ev = {s["review_id"]: s for s in json.loads((OUT / f"{bank}_evidence.json").read_text())["segments"]}
    sc = json.loads((stage2.OUT / "score_v3.json").read_text())
    labels = {}
    for it in sc["items"]:
        lab, truth = it["label"], it["truth"]
        if truth is not None:
            truth = (int(truth[0]), str(truth[1]).replace("ß", "8"))   # one frozen label typo: "1ß" for 18
        labels[it["review_id"]] = {"truth": truth, "flag": lab.get("flag"), "group": it["group"]}
    sizes: dict = {}
    for seg in ev.values():
        for c in seg["candidates"]:
            sizes.setdefault((seg["match"], c["team"]), set()).add(c["number"])
    for seg in ev.values():   # gallery identities of the segment's own team in this match
        seg["team_gallery_size"] = len(sizes.get((seg["match"], seg["team"]), ()))
    n_old = sum(1 for it in sc["items"] if it["group"] == "accepted")
    n_rest = len(ev) - n_old
    weights = {rid: (1.0 if l["group"] == "accepted" else n_rest / sum(x["group"] == "random" for x in labels.values()))
               for rid, l in labels.items()}
    return {"bank": bank, "segments": ev, "labels": labels, "weights": weights,
            "population": len(ev), "seconds": sum(s["duration_s"] for s in ev.values())}


def outcome(seg: dict, lab: dict, name) -> str:
    if name is None:
        return "abstained"
    if lab["flag"] == "not_a_player":
        return "wrong"
    if lab["truth"] is None:
        return "unscored"
    if tuple(name) == lab["truth"]:
        return "correct"
    known = any((c["team"], c["number"]) == lab["truth"] for c in seg["candidates"])
    return "wrong" if known else "unknown_named"


def evaluate(banks: list, rule: Rule) -> dict:
    return evaluate_decisions(banks, [{rid: decide(seg, rule) for rid, seg in b["segments"].items()} for b in banks])


def check_reproduces_final_bank(bank: dict) -> int:
    """V2 must reproduce the final bank's sealed decisions (same protocol)."""
    stage2.configure("final")
    sealed = {r["review_id"]: r for r in json.loads((stage2.OUT / "proposals_SEALED.json").read_text())["rows"]}
    diff = 0
    for rid, seg in bank["segments"].items():
        mine = decide(seg, V2)["name"]
        theirs = sealed[rid]["proposal"]
        theirs = None if theirs is None else (int(theirs.split("|team")[1].split("|")[0]), theirs.split("#")[-1])
        diff += (mine != theirs)
    return diff


def fmt(r: dict) -> str:
    return (f"labelled: {r['correct']:3d} right {r['wrong']:2d} wrong {r['unknown_named']:2d} unknown-named "
            f"(P {r['precision_labelled']:.3f}) | est: P {r['est_precision']:.3f} cov {r['est_coverage']:.3f} "
            f"({r['est_coverage_s']:.3f} of seconds) | unlabelled names {r['named_unlabelled']}")


def main() -> None:
    banks = [load_bank(b) for b in BANKS]
    print("V2 vs sealed final-bank decisions, segments differing:", check_reproduces_final_bank(banks[1]))
    results = {}
    base = evaluate(banks, V2)
    print(f"{'V2 (locked)':58s} {fmt(base)}")
    results["v2"] = {"rule": asdict(V2), **base}
    variants = {
        "single 0.80": replace(V2, single_tau=0.80),
        "single 0.75": replace(V2, single_tau=0.75),
        "co-presence ignores duplicate boxes": replace(V2, co_ignore_duplicates=True),
        "co-presence needs 5 frames": replace(V2, co_min_frames=5),
        "co-presence needs 13 frames (0.5 s)": replace(V2, co_min_frames=13),
        "1 read contradicts": replace(V2, read_min=1),
        "speed veto 900 cm/s": replace(V2, vmax_cm_s=900.0),
        "speed veto 700 cm/s": replace(V2, vmax_cm_s=700.0),
    }
    for name, rule in variants.items():
        r = evaluate(banks, rule)
        results[name] = {"rule": asdict(rule), **r}
        print(f"{name:58s} {fmt(r)}")
    print("\ngate grid (tau x margin), guards: 1 read contradicts + speed 900 + single 0.80:")
    guarded = replace(V2, read_min=1, vmax_cm_s=900.0, single_tau=0.80)
    for tau, m in itertools.product((0.784, 0.76, 0.74, 0.72, 0.70), (0.0226, 0.03, 0.045, 0.06)):
        r = evaluate(banks, replace(guarded, tau=tau, margin=m))
        results[f"guarded tau {tau} m {m}"] = {"rule": asdict(replace(guarded, tau=tau, margin=m)), **r}
        print(f"  tau {tau:.3f} m {m:.4f}   {fmt(r)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rules.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
