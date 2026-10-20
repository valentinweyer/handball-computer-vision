"""Rule variants over the round-2 evidence store (`coverage2.py`), scored per segment.

Every rule sees exactly the same population, candidates and crops; only the decision
differs. The baseline (`V2`) reproduces the split-v2 lock: same-team galleries from
vouched fragments ending >= 2 s before or starting >= 2 s after the segment, never its
own track; vetoes for co-presence and number contradiction; 8 query crops spread over
the segment, top-3 mean per crop, mean over crops; score >= tau and margin >= m; the
half-split guard.

Options, each motivated by the abstention analysis in the round-2 report:

- `single_tau`: name a lone surviving candidate at this score (v2 always abstains);
- `dup`: how co-presence treats a box showing the candidate's number that overlaps
  the segment's own box. "none" = every co-present frame vetoes (v2); "iou50" = frames
  at IoU >= 0.5 do not count (round-1 candidate; distinct players can pass it);
  "sustained" = the veto is lifted only when the other track overlaps on >= 90% of at
  least 10 co-present frames with median IoU >= 0.7 (a duplicate track of one person);
- `query`: "spread8" (v2) or "all" (every sampled crop);
- `clean_query`: drop query crops whose mean similarity to the segment's other crops
  is an outlier (median - 3 MAD), so a crop of another person cannot vote;
- `clean_gallery`: drop gallery crops far from their own identity's centroid
  (below median - 3 MAD of that identity), so a contaminated anchor cannot match;
- `support`: minimum share of query crops whose own best candidate is the named one
  (stable evidence across independent crops), applied on top of the gate;
- `mixed_guard`: abstain when the query crops split into two appearance groups that
  each hold >= 25% of the crops and are farther apart than the identity's own spread
  (a box that switches person; catches switches the half-split check misses);
- `emb`: "raw" or "strict" (digits and a torso-and-shorts band hidden on both sides);
- `tier2` = (tau2, m2, strict_tau, strict_m): a segment below the raw gate is still
  named when raw score >= tau2 and margin >= m2 AND the same candidate also wins with
  digits and the torso-and-shorts band hidden on both sides, at strict score >=
  strict_tau and strict margin >= strict_m. Motivation (abstention analysis): a
  look-alike unknown teammate scores high on kit, so its similarity collapses under
  the strict mask; the same player keeps head, limbs and shoes;
- `lone_strict`: a lone candidate must also reach this strict score.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from experiments.teammate_reid import stage2
from experiments.teammate_reid.coverage2 import OUT
from experiments.teammate_reid.symmetric import _pick

LOCK_GATE = (0.7839630554462301, 0.02255046162112006)
MIN_SEP = 50


@dataclass(frozen=True)
class Rule:
    tau: float = LOCK_GATE[0]
    margin: float = LOCK_GATE[1]
    single_tau: float | None = None
    dup: str = "none"
    query: str = "spread8"
    clean_query: bool = False
    clean_gallery: bool = False
    support: float | None = None
    mixed_guard: bool = False
    halves: bool = True
    emb: str = "raw"
    tier2: tuple | None = None
    lone_strict: float | None = None
    crowd_max: float | None = None    # tier-2 names only when <= this share of frames has another box over half of it


V2 = Rule()


def _norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def _supported(reads: dict) -> set:
    full = {v for v, c in reads.items() if c >= 2 and len(v) > 1}
    return full | {v for v, c in reads.items() if c >= 2 and len(v) == 1 and not any(f.endswith(v) for f in full)}


def is_duplicate(c: dict) -> bool:
    return c.get("n", 0) >= 10 and c["frac50"] >= 0.9 and c["median"] >= 0.7


def _outliers(sim: np.ndarray) -> np.ndarray:
    """Rows whose mean similarity to the others is below median - 3 MAD."""
    n = len(sim)
    if n < 5:
        return np.zeros(n, bool)
    m = (sim.sum(1) - np.diag(sim)) / (n - 1)
    med, mad = np.median(m), np.median(np.abs(m - np.median(m)))
    return m < med - 3 * max(mad, 1e-3)


class Bank:
    """One bank's store, with embeddings normalised and per-match gallery indices."""

    def __init__(self, bank: str):
        self.name = bank
        meta = json.loads((OUT / f"{bank}_store.json").read_text())
        arr = np.load(OUT / f"{bank}_store.npz")
        self.segments = {s["review_id"]: s for s in meta["segments"]}
        self.galleries = meta["galleries"]
        self.negatives = meta["negatives"]
        self.q = {e: {rid: _norm(arr[f"q_{e}|{rid}"]) for rid in self.segments} for e in ("raw", "strict")}
        self.q_digit = {rid: arr[f"q_digit|{rid}"] for rid in self.segments}
        self.g, self.g_index, self.g_keep = {}, {}, {}
        for match, frags in self.galleries.items():
            ids = arr[f"gal_ids|{match}"].tolist()
            self.g_index[match] = {c: i for i, c in enumerate(ids)}
            self.g[match] = {e: _norm(arr[f"gal_{e}|{match}"]) for e in ("raw", "strict")}
            self.g_keep[match] = {}
            for e in ("raw", "strict"):   # gallery cleaning: per identity, drop crops far from its centroid
                keep = np.ones(len(ids), bool)
                by_ident: dict = {}
                for f in frags:
                    by_ident.setdefault(f["identity"], []).extend(self.g_index[match][c] for c in f["crop_ids"])
                for idx in by_ident.values():
                    if len(idx) < 5:
                        continue
                    x = self.g[match][e][idx]
                    centre = _norm(x.mean(0, keepdims=True))[0]
                    s = x @ centre
                    med, mad = np.median(s), np.median(np.abs(s - np.median(s)))
                    keep[np.asarray(idx)[s < med - 3 * max(mad, 1e-3)]] = False
                self.g_keep[match][e] = keep
        stage2.configure(bank)
        self.population = len(self.segments)
        self.seconds = sum(s["duration_s"] for s in self.segments.values())
        self._cache: dict = {}
        geo = OUT / f"{bank}_geometry.json"
        self.geometry = json.loads(geo.read_text()) if geo.exists() else {}

    def candidates(self, seg: dict) -> dict:
        """identity -> (team, number, gallery row indices) under the v2 gallery protocol."""
        out: dict = {}
        for f in self.galleries[seg["match"]]:
            if seg["settled"] and f["team"] != seg["team"]:
                continue
            before = f["end"] + MIN_SEP <= seg["start"]
            after = f["start"] >= seg["end"] + MIN_SEP
            if not (before or after) or f["track_id"] == seg["track_id"]:
                continue
            c = out.setdefault(f["identity"], (int(f["team"]), f["number"], []))
            c[2].extend(self.g_index[seg["match"]][x] for x in f["crop_ids"])
        return out

    def crop_scores(self, rid: str, emb: str, clean_gallery: bool) -> dict:
        """identity -> per-query-crop top-3 mean similarity (all crops)."""
        key = (rid, emb, clean_gallery)
        if key not in self._cache:
            seg = self.segments[rid]
            q = self.q[emb][rid]
            G = self.g[seg["match"]][emb]
            keep = self.g_keep[seg["match"]][emb]
            out = {}
            for ident, (team, number, idx) in self.candidates(seg).items():
                idx = [i for i in idx if keep[i]] if clean_gallery else idx
                if not idx:
                    continue
                sims = q @ G[idx].T
                out[ident] = (team, number, np.sort(sims, axis=1)[:, -min(3, sims.shape[1]):].mean(1))
            self._cache[key] = out
        return self._cache[key]


def decide(bank: Bank, rid: str, rule: Rule) -> dict:
    seg = bank.segments[rid]
    scores = bank.crop_scores(rid, rule.emb, rule.clean_gallery)
    n = len(seg["crop_frames"])
    use = np.ones(n, bool)
    if rule.clean_query:
        q = bank.q[rule.emb][rid]
        use &= ~_outliers(q @ q.T)
    idx_all = np.flatnonzero(use)
    sel = idx_all[_pick(list(range(len(idx_all))), "spread8")] if rule.query == "spread8" else idx_all
    supported = _supported(seg["own_reads"])
    alive = []
    for ident, (team, number, s) in scores.items():
        cp = seg["copresence"].get(ident, {"n": 0, "frag_overlap_frames": 0})
        if cp["frag_overlap_frames"] > 0:
            continue
        if cp.get("n", 0) > 0:
            if rule.dup == "none":
                continue
            if rule.dup == "iou50" and cp["frac50"] < 1.0:
                continue
            if rule.dup == "sustained" and not is_duplicate(cp):
                continue
        if supported and number not in supported:
            continue
        alive.append((float(s[sel].mean()), ident, team, number, s))
    alive.sort(key=lambda a: -a[0])
    if not alive:
        return {"name": None, "reason": "no_candidate"}
    best = alive[0]
    name = (best[2], best[3])

    def strict_view():
        """Score and margin of the raw winner when digits and the kit band are hidden."""
        ss = bank.crop_scores(rid, "strict", rule.clean_gallery)
        st = sorted(((float(ss[a[1]][2][sel].mean()), a[1]) for a in alive if a[1] in ss), reverse=True)
        if not st or st[0][1] != best[1]:
            return None, None
        return st[0][0], (st[0][0] - st[1][0]) if len(st) > 1 else float("inf")

    crowded = (rule.crowd_max is not None
               and bank.geometry.get(rid, {}).get("crowd50", 1.0) > rule.crowd_max)
    if len(alive) == 1:
        ok = rule.single_tau is not None and best[0] >= rule.single_tau
        if ok and rule.lone_strict is not None:
            s_score, _ = strict_view()
            ok = s_score is not None and s_score >= rule.lone_strict
        return {"name": name if ok else None, "reason": "single" if ok else "single_candidate", "score": best[0]}
    margin = best[0] - alive[1][0]
    tier = 1
    if best[0] < rule.tau or margin < rule.margin:
        ok = False
        if rule.tier2 is not None and not crowded and best[0] >= rule.tier2[0] and margin >= rule.tier2[1]:
            s_score, s_margin = strict_view()
            ok = s_score is not None and s_score >= rule.tier2[2] and s_margin >= rule.tier2[3]
        if not ok:
            return {"name": None, "reason": "below_gate", "score": best[0], "margin": margin, "best": name}
        tier = 2
    if rule.halves and n // 2 >= 2:
        picks = []
        for part in (list(range(n // 2)), list(range(n // 2, n))):
            p = np.asarray(part)[_pick(list(range(len(part))), "spread8")]
            picks.append(max(alive, key=lambda a: a[4][p].mean())[1])
        if picks[0] != picks[1]:
            return {"name": None, "reason": "halves_disagree"}
    if rule.support is not None:
        stack = np.stack([a[4][idx_all] for a in alive])
        share = float((stack.argmax(0) == 0).mean())
        if share < rule.support:
            return {"name": None, "reason": "low_support", "support": share}
    if rule.mixed_guard and len(idx_all) >= 8 and _two_people(bank.q[rule.emb][rid][idx_all]):
        return {"name": None, "reason": "mixed_guard"}
    return {"name": name, "reason": "accepted" if tier == 1 else "accepted_tier2", "score": best[0], "margin": margin}


def _two_people(q: np.ndarray) -> bool:
    """Two-means split of the crops; two persons if both groups hold >= 25% and the
    groups are less similar to each other than either is internally."""
    sim = q @ q.T
    a, b = np.unravel_index(np.argmin(sim), sim.shape)
    for _ in range(10):
        ga = sim[:, a] >= sim[:, b]
        if ga.all() or (~ga).all():
            return False
        ca, cb = _norm(q[ga].mean(0, keepdims=True))[0], _norm(q[~ga].mean(0, keepdims=True))[0]
        a2, b2 = int(np.argmax(q @ ca)), int(np.argmax(q @ cb))
        if (a2, b2) == (a, b):
            break
        a, b = a2, b2
    share = min(ga.mean(), 1 - ga.mean())
    within = min(sim[np.ix_(ga, ga)].mean(), sim[np.ix_(~ga, ~ga)].mean())
    between = sim[np.ix_(ga, ~ga)].mean()
    return share >= 0.25 and between < within - 0.12


CANDIDATES = {
    "v2": V2,
    "lone_only": replace(V2, single_tau=0.80),
    "dup_iou50_only": replace(V2, dup="iou50"),
    "dup_sustained_only": replace(V2, dup="sustained"),
    "narrow (lone + iou50)": replace(V2, single_tau=0.80, dup="iou50"),
}


def describe(rule: Rule) -> dict:
    return asdict(rule)
