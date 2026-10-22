"""How much of the unnamed player-time could a matcher name at all? (development banks only)

The Stage 2 queues hold every segment split_v2 named plus a seeded random sample of the
rest, all blind human-labelled. Named segments count in full; the rest is estimated from
the random sample, weighted by unnamed seconds, with a bootstrap CI. A segment is
reachable when its labelled player has a gallery identity in the window (or only in the
extra windows). For reachable random segments it also reports where the true player
ranks and why v2 abstains, and the top-1 scores of right and wrong candidates.

    python -m experiments.teammate_reid.ceiling
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.teammate_reid import stage2
from experiments.teammate_reid.coverage2_rules import _outliers, _pick, _supported, decide, is_duplicate
from experiments.teammate_reid.extra_galleries import ExtendedBank, _rules, references_with_verify

BANKS = ("test_v1", "final")
CATS = ("window_gallery", "extra_only", "no_gallery", "unreadable", "mixed", "not_player")


def _unnamed_s(seg: dict) -> float:
    return seg["duration_s"] * (1 - seg["baseline_shown"])


def _identities(bank, match: str) -> set:
    return {(int(g["team"]), str(g["number"])) for g in bank.galleries[match]}


def _truth(r) -> tuple | str:
    if r.get("flag") == "mixed":
        return "mixed"
    if r.get("flag") == "not_a_player":
        return "not_player"
    if not r["truth"]:
        return "unreadable"
    return int(r["truth"][0]), str(r["truth"][1])


def alive(bank, rid: str, rule) -> list:
    """Candidates that survive v2's vetoes, best first: (score, team, number). Mirrors `decide`."""
    seg = bank.segments[rid]
    scores = bank.crop_scores(rid, rule.emb, rule.clean_gallery)
    use = np.ones(len(seg["crop_frames"]), bool)
    if rule.clean_query:
        q = bank.q[rule.emb][rid]
        use &= ~_outliers(q @ q.T)
    idx = np.flatnonzero(use)
    sel = idx[_pick(list(range(len(idx))), "spread8")] if rule.query == "spread8" else idx
    supported = _supported(seg["own_reads"])
    out = []
    for ident, (team, number, s) in scores.items():
        cp = seg["copresence"].get(ident, {"n": 0, "frag_overlap_frames": 0})
        if cp["frag_overlap_frames"] > 0 or (cp.get("n", 0) > 0 and not is_duplicate(cp)):
            continue
        if supported and number not in supported:
            continue
        out.append((float(s[sel].mean()), int(team), str(number)))
    return sorted(out, reverse=True)


def main(reps: int = 2000) -> dict:
    _, ref, _ = references_with_verify()
    rule = _rules()["v2"]
    totals, total_s, strata = defaultdict(float), 0.0, []
    ranks, reasons, top1 = Counter(), Counter(), {"right": [], "wrong": []}
    always = Counter()
    for name in BANKS:
        base, ext = ExtendedBank(extra=False, bank=name), ExtendedBank(extra=True, bank=name)
        stage2.configure(name)
        mem = json.loads((stage2.OUT / "queue_membership_SEALED.json").read_text())
        accepted = set(mem["accepted"])

        def cat(rid):
            t = _truth(ref((name, rid)))
            if isinstance(t, str):
                return t
            m = base.segments[rid]["match"]
            return ("window_gallery" if t in _identities(base, m)
                    else "extra_only" if t in _identities(ext, m) else "no_gallery")

        total_s += sum(_unnamed_s(s) for s in base.segments.values())
        rest_s = sum(_unnamed_s(s) for r, s in base.segments.items() if r not in accepted)
        acc = [(cat(r), _unnamed_s(base.segments[r])) for r in mem["accepted"]]
        smp = [(cat(r), _unnamed_s(base.segments[r])) for r in mem["random_sample"]]
        strata.append((acc, smp, rest_s))
        for c, s in acc:
            totals[c] += s
        for c, s in smp:
            totals[c] += s / sum(x for _, x in smp) * rest_s

        for rid in mem["random_sample"]:
            t, cands = _truth(ref((name, rid))), alive(base, rid, rule)
            if not isinstance(t, str) and t in _identities(base, base.segments[rid]["match"]):
                pos = next((i for i, c in enumerate(cands) if c[1:] == t), None)
                ranks["vetoed" if pos is None else "1" if pos == 0 else "2-3" if pos < 3 else "4+"] += 1
                if pos == 0:
                    reasons[decide(base, rid, rule).get("reason") or "named"] += 1
            if not cands:
                always["no candidate"] += 1
            elif isinstance(t, str):
                always[t] += 1
            else:
                right = cands[0][1:] == t
                always["right" if right else "wrong_or_unknown"] += 1
                top1["right" if right else "wrong"].append(round(cands[0][0], 3))

    rng, boot = np.random.default_rng(0), []
    for _ in range(reps):
        t = defaultdict(float)
        for acc, smp, rest_s in strata:
            draw = [smp[i] for i in rng.integers(0, len(smp), len(smp))]
            for c, s in acc:
                t[c] += s
            for c, s in draw:
                t[c] += s / sum(x for _, x in draw) * rest_s
        boot.append([t[c] / total_s for c in CATS])
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    report = {
        "unnamed_seconds": round(total_s),
        "shares": {c: {"share": round(totals[c] / total_s, 3), "ci95": [round(lo[i], 3), round(hi[i], 3)]}
                   for i, c in enumerate(CATS)},
        "reachable_random_truth_rank": dict(ranks),
        "abstain_reason_when_top1_right": dict(reasons),
        "random_sample_if_top1_always_named": dict(always),
        "top1_scores": {k: sorted(v) for k, v in top1.items()},
        "tau": rule.tau,
    }
    for c, v in report["shares"].items():
        print(f"{c:15s} {v['share']:6.1%}  CI {v['ci95'][0]:.1%}-{v['ci95'][1]:.1%}")
    for k in ("reachable_random_truth_rank", "abstain_reason_when_top1_right", "random_sample_if_top1_always_named"):
        print(k, report[k])
    out = stage2.RUNS / "ceiling"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
