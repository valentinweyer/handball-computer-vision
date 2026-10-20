"""Score round-2 rules on the development banks against every available reference.

References, kept apart by source and never merged into one "truth":
- `human`: the user's blind Stage 2 labels (both banks);
- `human_adjudicated`: the user's post-unsealing corrections (not blind);
- `assistant_v1` / `assistant_v2`: the assistant's blind full-resolution labels of
  segments a candidate rule named outside the human queue. Development evidence only.

Per rule and segment the outcome is: correct, wrong (another known player, or a
non-player), unknown_named (the labelled player has no gallery: a real unknown got
a name), mixed (the box holds two people), unreadable (labelled, number not
confirmable), unlabelled (named but no reference yet) or abstained.

Coverage gains count only correct names on segments that are on court: a segment
with fewer than half its court positions inside the 40 x 20 m court (+1 m) is
counted as off court (bench, warm-up) and reported separately. Newly correct
player-time is the segment's duration times the share of its rows the baseline
left unnamed. The zero-known stratum is the segments with no named row at all.

    python -m experiments.teammate_reid.coverage2_eval
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from experiments.teammate_reid import stage2
from experiments.teammate_reid.coverage2 import OUT

ANN = Path(__file__).parent / "annotations"
REGRESSION = {("test_v1", "S000"): "Eisenach #99 named; assistant reads #55 on the shorts (unconfirmed)",
              ("test_v1", "S011"): "Eisenach #99 named; assistant reads #55 on the shorts (unconfirmed)",
              ("final", "S241"): "Bielefeld: human #7, assistant reads #6 (disputed)",
              ("final", "S158"): "Balingen II: true #7, v2 assigns #10",
              ("test_v1", "S259"): "Kiel: mixed #4 / #23"}


def references() -> dict:
    """(bank, review_id) -> list of {source, truth (team, number) | None, flag}."""
    refs: dict = {}
    for bank in ("test_v1", "final"):
        stage2.configure(bank)
        adj = json.loads(stage2.ADJUDICATION.read_text())["labels"] if stage2.ADJUDICATION.exists() else {}
        blind = json.loads(stage2.LABELS.read_text())["labels"]
        sc = json.loads((stage2.OUT / "score_v3.json").read_text())
        for it in sc["items"]:
            rid = it["review_id"]
            lab = blind.get(rid, {})
            truth = it["truth"]
            if rid in adj:   # the blind label as it was, then the adjudication
                refs.setdefault((bank, rid), []).append(_entry("human", lab))
                refs[(bank, rid)].append({"source": "human_adjudicated",
                                          "truth": tuple(truth) if truth else None, "flag": adj[rid].get("flag")})
            else:
                refs.setdefault((bank, rid), []).append(
                    {"source": "human", "truth": (int(truth[0]), str(truth[1]).replace("ß", "8")) if truth else None,
                     "flag": lab.get("flag")})
    for version in ("v1", "v2"):
        path = ANN / f"coverage_dev_assistant_labels_{version}.json"
        queue = ANN / f"coverage_dev_queue_{version}.json"
        if not path.exists():
            continue
        labels = json.loads(path.read_text())["labels"]
        for it in json.loads(queue.read_text())["items"]:
            if it.get("control"):
                continue
            a = labels[it["code"]]
            refs.setdefault((it["bank"], it["review_id"]), []).append(
                {"source": f"assistant_{version}", "truth": (a["team"], a["number"]) if a.get("number") else None,
                 "flag": a.get("flag"), "note": a.get("note", "")})
    return refs


def _entry(source: str, lab: dict) -> dict:
    n = (lab.get("number") or "").strip()
    t = lab.get("team")
    return {"source": source, "truth": (int(t), n.replace("ß", "8").lstrip("0") or "0") if n and t is not None else None,
            "flag": lab.get("flag")}


def reference(refs: dict, key) -> dict | None:
    """The reference used for scoring: adjudicated > human > assistant. Two assistant
    reads that disagree are not settled by either: the segment counts as disputed
    (unverified), never as the read that matches a prediction."""
    rs = refs.get(key, [])
    assistant = [r for r in rs if r["source"].startswith("assistant")]
    if not any(r["source"].startswith("human") for r in rs) and len(assistant) > 1:
        reads = {(r["truth"], r["flag"]) for r in assistant}
        if len(reads) > 1:
            return {"source": "assistant_disputed", "truth": None, "flag": None,
                    "note": "; ".join(f"{r['source']}: {r['truth'] or r['flag']}" for r in assistant)}
    for src in ("human_adjudicated", "human", "assistant_v2", "assistant_v1"):
        for r in rs:
            if r["source"] == src:
                return r
    return None


def outcome(bank, rid: str, name, ref: dict | None) -> str:
    if name is None:
        return "abstained"
    if ref is None:
        return "unlabelled"
    if ref["source"] == "assistant_disputed":
        return "disputed"
    if ref["flag"] == "mixed":
        return "mixed"
    if ref["flag"] == "not_a_player":
        return "wrong"
    if ref["truth"] is None:
        return "unreadable"
    if tuple(name) == tuple(ref["truth"]):
        return "correct"
    known = ref["truth"] in {(t, n) for t, n, _ in bank.candidates(bank.segments[rid]).values()}
    return "wrong" if known else "unknown_named"


def on_court(seg: dict) -> bool:
    f = seg.get("on_court_frac")
    return f is None or f >= 0.5


def score(banks: list, decisions: dict, refs: dict) -> dict:
    """decisions: (bank, rid) -> decision. Census over every named segment."""
    tot = Counter()
    secs = Counter()
    per = {}
    for b in banks:
        for rid, seg in b.segments.items():
            d = decisions[(b.name, rid)]
            ref = reference(refs, (b.name, rid))
            o = outcome(b, rid, d["name"], ref)
            if o == "correct" and not on_court(seg):
                o = "correct_off_court"
            per[(b.name, rid)] = {"outcome": o, "name": d["name"], "reason": d["reason"],
                                  "source": ref["source"] if ref else None}
            tot[o] += 1
            new_s = seg["duration_s"] * (1 - seg["baseline_shown"])
            secs[o] += new_s
            if seg["baseline_shown"] == 0:
                tot[f"zero_known|{o}"] += 1
                secs[f"zero_known|{o}"] += new_s
            if o in ("correct", "wrong", "unknown_named", "mixed"):
                digits = bool(b.q_digit[rid].any())
                tot[f"{'digit_box' if digits else 'no_digit_box'}|{o}"] += 1
                tot[f"src:{ref['source']}|{o}"] += 1
    pop = sum(b.population for b in banks)
    zero = sum(1 for b in banks for s in b.segments.values() if s["baseline_shown"] == 0)
    pop_s = sum(s["duration_s"] * (1 - s["baseline_shown"]) for b in banks for s in b.segments.values())
    errors = tot["wrong"] + tot["unknown_named"] + tot["mixed"]
    return {"counts": dict(tot), "population": pop, "zero_known_population": zero,
            "unnamed_player_seconds": pop_s, "new_correct_seconds": secs["correct"],
            "new_correct_seconds_zero_known": secs["zero_known|correct"],
            "coverage_segments": tot["correct"] / pop, "coverage_seconds": secs["correct"] / pop_s,
            "precision_strict": tot["correct"] / max(1, tot["correct"] + errors),
            "unverified": tot["unreadable"] + tot["unlabelled"] + tot["disputed"], "per_segment": per}
