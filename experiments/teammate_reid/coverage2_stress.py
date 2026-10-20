"""Two checks for a coverage rule on banks with labelled unresolved segments.

`synthetic_unknowns`: for every segment whose labelled player has a gallery, remove
that player's gallery and decide again. The segment is now an unknown by
construction; any name is a false match. This is the case the lone-candidate change
could make worse (removing the truth can leave one candidate standing).

`digit_split`: correct names split by whether any query crop had a detected number
box, and whether the named player also wins with digits and a torso-and-shorts band
hidden on both sides (`strict`). A name that survives `strict` did not depend on a
printed number; raw-crop agreement alone is not evidence of number independence.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from experiments.teammate_reid.coverage2_rules import Bank, decide
from experiments.teammate_reid.symmetric import _pick


def synthetic_unknowns(bank: Bank, rule, refs: dict, reference) -> Counter:
    out = Counter()
    for rid, seg in bank.segments.items():
        ref = reference(refs, (bank.name, rid))
        if not ref or not ref["truth"] or ref["flag"] or ref["source"].startswith("assistant"):
            continue
        truth = tuple(ref["truth"])
        if truth not in {(t, n) for t, n, _ in bank.candidates(seg).values()}:
            continue
        key = (rid, rule.emb, rule.clean_gallery)
        full = bank.crop_scores(rid, rule.emb, rule.clean_gallery)
        bank._cache[key] = {i: v for i, v in full.items() if (v[0], v[1]) != truth}
        d = decide(bank, rid, rule)
        bank._cache[key] = full
        out["trials"] += 1
        out["false_match" if d["name"] is not None else "abstained"] += 1
    return out


def digit_split(bank: Bank, rule, refs: dict, reference) -> Counter:
    out = Counter()
    for rid, seg in bank.segments.items():
        d = decide(bank, rid, rule)
        ref = reference(refs, (bank.name, rid))
        if d["name"] is None or not ref or not ref["truth"] or tuple(d["name"]) != tuple(ref["truth"]):
            continue
        digits = "digit_box_seen" if bank.q_digit[rid].any() else "no_digit_box"
        n = len(seg["crop_frames"])
        sel = np.asarray(range(n))[_pick(list(range(n)), "spread8")]
        st = bank.crop_scores(rid, "strict", False)
        ranked = sorted(((float(v[2][sel].mean()), (v[0], v[1])) for v in st.values()), reverse=True)
        strict_ok = bool(ranked) and ranked[0][1] == tuple(d["name"])
        out[f"{digits}|strict_{'agrees' if strict_ok else 'differs'}"] += 1
    return out
