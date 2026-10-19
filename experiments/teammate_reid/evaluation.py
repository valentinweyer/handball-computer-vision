"""Within-team identity proposals from appearance, scored with abstention.

Query construction (`build_queries`): within one test video, a query is an
accepted fragment whose start lies at least `min_sep` frames after the end of
every gallery fragment it is compared with. Galleries hold only number-vouched
fragments (mined labels, reviewed), never appearance-derived ones, so nothing
here can confirm itself. A query whose own identity has no gallery yet is a
natural *unknown player* query; every known-player query is also scored a
second time with its true identity removed (synthetic unknown), where any
accepted proposal is a false match.

Matching (`score_query`):
- ``ema_single``: the runtime shape -- one EMA vector per identity
  (alpha 0.3 over its gallery crops in time order, `sam2_manager`), compared with
  the query fragment's first crop.
- ``ema_agg``: the same EMA gallery, the query as the mean of its first `m` crops.
- ``gallery_single`` / ``gallery_agg``: every gallery crop kept as a view; an
  identity scores the mean of its `top` best cosines, per query crop, averaged
  over the first `m` query crops for ``_agg``.

A proposal is accepted when the best score clears `tau` and beats the runner-up
by `margin`; otherwise the query abstains. Confidence for the threshold-free
curves is the margin (best minus runner-up), so the curves compare how well a
method ranks its own proposals, not a particular threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMA_ALPHA = 0.3
METHODS = ("ema_single", "ema_agg", "gallery_single", "gallery_agg")


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def ema(vectors: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    value = vectors[0]
    for v in vectors[1:]:
        value = alpha * v + (1 - alpha) * value
    return value


@dataclass
class Query:
    video: str
    team: int
    identity: str
    fragment: str
    start: int
    end: int
    crops: list                      # crop ids, time order
    galleries: dict                  # identity -> crop ids, time order
    anchors: dict                    # identity -> crop ids of its first fragment only
    known: bool                      # own identity is among the candidates
    reentry: bool                    # no own gallery fragment on the same tracker id
    synthetic_unknown: bool = False
    meta: dict = field(default_factory=dict)


def build_queries(fragments: list, crops_by_fragment: dict, fps: float = 25.0,
                  min_sep: int = 50, min_query_crops: int = 3, after: bool = False,
                  exclude_track: bool = False) -> list[Query]:
    """Queries for one video's accepted fragments (dicts with key/identity/team/start/end/track_id).

    Gallery fragments end >= `min_sep` frames before the query; with `after`, fragments
    starting >= `min_sep` frames after it count too (a completed-video pass). With
    `exclude_track`, no fragment of the query's own track is gallery: a production
    track that is vouched anywhere is already named, so unresolved segments never have
    their own track's continuation to match against."""
    frags = sorted((f for f in fragments if crops_by_fragment.get(f["key"])), key=lambda f: f["start"])
    first = {}
    for f in frags:
        first.setdefault(f["identity"], f["key"])
    out = []
    for q in frags:
        qcrops = crops_by_fragment[q["key"]]
        if len(qcrops) < min_query_crops:
            continue
        galleries, owners = {}, {}
        for g in frags:
            before, later = g["end"] + min_sep <= q["start"], after and g["start"] >= q["end"] + min_sep
            if g["key"] == q["key"] or g["team"] != q["team"] or not (before or later):
                continue
            if exclude_track and g["track_id"] == q["track_id"]:
                continue
            galleries.setdefault(g["identity"], []).extend(crops_by_fragment[g["key"]])
            owners.setdefault(g["identity"], []).append(g)
        if not galleries:
            continue
        anchors = {i: crops_by_fragment[first[i]] for i in galleries
                   if any(g["key"] == first[i] for g in owners[i])}
        known = q["identity"] in galleries
        reentry = known and all(g["track_id"] != q["track_id"] for g in owners[q["identity"]])
        base = Query(q["video"], q["team"], q["identity"], q["key"], q["start"], q["end"],
                     list(qcrops), galleries, anchors, known, reentry,
                     meta={"duration_s": (q["end"] - q["start"] + 1) / fps,
                           "opened_by": q.get("opened_by")})
        out.append(base)
        if known and len(galleries) > 1:
            out.append(Query(**{**base.__dict__,
                                "galleries": {i: c for i, c in galleries.items() if i != q["identity"]},
                                "anchors": {i: c for i, c in anchors.items() if i != q["identity"]},
                                "known": False, "synthetic_unknown": True}))
    return out


def score_query(query: Query, emb: np.ndarray, method: str, m: int = 8, top: int = 3,
                anchor_only: bool = False) -> dict:
    """identity -> score for one query; `emb` rows are L2-normalised crop embeddings."""
    galleries = query.anchors if anchor_only else query.galleries
    q = emb[query.crops[:m] if method.endswith("_agg") else query.crops[:1]]
    scores = {}
    for identity, ids in galleries.items():
        g = emb[ids]
        if method.startswith("ema"):
            proto = l2n(ema(g))
            qv = l2n(q.mean(0))
            scores[identity] = float(qv @ proto)
        else:
            sims = q @ g.T                                   # (mq, ng)
            k = min(top, sims.shape[1])
            scores[identity] = float(np.sort(sims, axis=1)[:, -k:].mean(1).mean())
    return scores


def decide(scores: dict) -> tuple[str | None, float, float]:
    """(best identity, best score, margin over runner-up).

    With a single candidate there is nothing to beat, so the margin is -inf
    and the margin gate abstains: appearance cannot be verified against an
    alternative. (`runtime_point` reproduces the runtime, which skips the
    margin check in that case.)"""
    if not scores:
        return None, -np.inf, -np.inf
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = ranked[0]
    margin = s1 - ranked[1][1] if len(ranked) > 1 else -np.inf
    return best, s1, margin


def outcomes(queries: list, emb: np.ndarray, method: str, **kw) -> list[dict]:
    rows = []
    for q in queries:
        scores = score_query(q, emb, method, **kw)
        best, s1, margin = decide(scores)
        rows.append({"fragment": q.fragment, "identity": q.identity, "known": q.known,
                     "synthetic_unknown": q.synthetic_unknown, "reentry": q.reentry,
                     "best": best, "score": s1, "margin": margin,
                     "correct": bool(q.known and best == q.identity),
                     "candidates": len(scores), "duration_s": q.meta["duration_s"]})
    return rows


def accept(row: dict, tau: float, margin: float) -> bool:
    return row["score"] >= tau and row["margin"] >= margin


def operating_point(rows: list, tau: float, margin: float) -> dict:
    """Coverage of known players, precision over everything accepted, and the
    damage: wrong links and the seconds of footage they would mislabel."""
    known = [r for r in rows if r["known"]]
    unknown = [r for r in rows if not r["known"]]
    acc_known = [r for r in known if accept(r, tau, margin)]
    acc_unknown = [r for r in unknown if accept(r, tau, margin)]
    correct = [r for r in acc_known if r["correct"]]
    wrong = [r for r in acc_known if not r["correct"]] + acc_unknown
    accepted = len(acc_known) + len(acc_unknown)
    return {
        "tau": tau, "margin": margin,
        "coverage": len(correct) / max(1, len(known)),
        "precision": len(correct) / accepted if accepted else float("nan"),
        "accepted": accepted, "correct_links": len(correct),
        "wrong_links_known": len(acc_known) - len(correct),
        "false_matches_unknown": len(acc_unknown),
        "unknown_false_match_rate": len(acc_unknown) / max(1, len(unknown)),
        "wrong_link_seconds": float(sum(r["duration_s"] for r in wrong)),
        "correct_link_seconds": float(sum(r["duration_s"] for r in correct)),
    }


def runtime_point(rows: list, tau: float = 0.7, margin: float = 0.02) -> dict:
    """`PlayerRegistry.reid_match` gates: cosine floor, margin only with a runner-up."""
    patched = [{**r, "margin": np.inf if r["candidates"] == 1 else r["margin"]} for r in rows]
    return operating_point(patched, tau, margin)


def margin_curve(rows: list) -> list[dict]:
    """Precision/coverage as the margin gate sweeps, with no absolute floor.

    Unknown queries are included, so precision counts false matches on
    players the gallery does not hold. Ties are cut at distinct margins."""
    margins = sorted({r["margin"] for r in rows if np.isfinite(r["margin"])}, reverse=True)
    return [operating_point(rows, -np.inf, m) for m in margins]


def summarize(rows: list) -> dict:
    known = [r for r in rows if r["known"]]
    curve = margin_curve(rows)

    def coverage_at(p):
        ok = [c["coverage"] for c in curve if c["accepted"] and c["precision"] >= p]
        return max(ok, default=0.0)

    def precision_at(cov):
        ok = [c["precision"] for c in curve if c["coverage"] >= cov]
        return max(ok, default=float("nan"))

    p95 = max((c for c in curve if c["accepted"] and c["precision"] >= 0.95),
              key=lambda c: c["coverage"], default=None)
    closed = margin_curve(known)
    closed_cov95 = max((c["coverage"] for c in closed if c["accepted"] and c["precision"] >= 0.95),
                       default=0.0)
    pareto = two_gate_pareto(rows)
    two_gate_cov95 = max((c["coverage"] for c in pareto if c["precision"] >= 0.95), default=0.0)
    two_gate_cov90 = max((c["coverage"] for c in pareto if c["precision"] >= 0.90), default=0.0)
    return {
        "known_queries": len(known),
        "unknown_queries": len(rows) - len(known),
        "rank1": float(np.mean([r["correct"] for r in known])) if known else float("nan"),
        "rank1_reentry": float(np.mean([r["correct"] for r in known if r["reentry"]]))
        if any(r["reentry"] for r in known) else float("nan"),
        "reentry_queries": sum(r["reentry"] for r in known),
        "coverage_at_p95": coverage_at(0.95),
        "coverage_at_p98": coverage_at(0.98),
        "precision_at_cov25": precision_at(0.25),
        "precision_at_cov50": precision_at(0.50),
        "oracle_p95_point": p95,
        "closed_set_coverage_at_p95": closed_cov95,
        "two_gate_coverage_at_p95": two_gate_cov95,
        "two_gate_coverage_at_p90": two_gate_cov90,
        "two_gate_p90_point": max((c for c in pareto if c["precision"] >= 0.90),
                                  key=lambda c: c["coverage"], default=None),
    }


def two_gate_pareto(rows: list, grid: int = 25) -> list[dict]:
    """Operating points over a (score floor, margin) grid, both swept over
    their observed quantiles. Chosen on the evaluated rows themselves, so it is
    an oracle bound shared equally by every method, not a deployable threshold."""
    scores = np.array([r["score"] for r in rows if np.isfinite(r["score"])])
    margins = np.array([r["margin"] for r in rows if np.isfinite(r["margin"])])
    if not len(scores) or not len(margins):
        return []
    taus = np.unique(np.quantile(scores, np.linspace(0, 1, grid)))
    gates = np.unique(np.quantile(margins, np.linspace(0, 1, grid)))
    points = [operating_point(rows, float(t), float(m)) for t in taus for m in gates]
    return [p for p in points if p["accepted"]]


def chance_rank1(queries: list) -> float:
    known = [q for q in queries if q.known]
    return float(np.mean([1 / len(q.galleries) for q in known])) if known else float("nan")


def cluster_bootstrap(rows_a: list, rows_b: list, stat, reps: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over identities of stat(b) - stat(a); rows align by index."""
    identities = sorted({r["identity"] for r in rows_a})
    index = {i: [k for k, r in enumerate(rows_a) if r["identity"] == i] for i in identities}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        pick = rng.choice(identities, size=len(identities), replace=True)
        ks = [k for i in pick for k in index[i]]
        deltas.append(stat([rows_b[k] for k in ks]) - stat([rows_a[k] for k in ks]))
    deltas = np.asarray([d for d in deltas if np.isfinite(d)])
    point = stat(rows_b) - stat(rows_a)
    return {"delta": float(point), "ci95": [float(np.percentile(deltas, 2.5)),
                                            float(np.percentile(deltas, 97.5))]
            if len(deltas) else [float("nan")] * 2}
