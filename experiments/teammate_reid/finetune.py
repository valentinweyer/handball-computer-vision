"""Appearance adapter on frozen SportsReID, trained within matches, tested for seen/unseen parity.

Everything follows `locked/finetune_v1.json` (plan: docs/smart-reid-finetune-plan-2026-10-22.md):

- split by team along the video folders: A trains (train/ videos), B is inner validation,
  C holds seen teams in a new match and D unseen teams (both eval/ videos, never trained);
- strict-masked crops, frozen SportsReID ViT-B/16, residual low-rank adapter (rank 16);
- supervised contrastive loss within one match (teammates full weight, opponents 0.3) plus
  an anchor to the base embedding; no identity head, no identity across matches;
- A windows: fragments ending before frame 10000 train, fragments starting at or after it
  are the seen same-match block, straddlers are dropped;
- parity: coverage at known precision >= 0.95 and unknown FMR <= 0.03 (split_v2 benchmark),
  adapter minus base, unseen (D) against seen in a new match (C), team-clustered bootstrap.

    python -m experiments.teammate_reid.finetune encode   # strict embeddings, plus 2 views on A (GPU)
    python -m experiments.teammate_reid.finetune pilot    # train on A, report B and A's blocks only
    python -m experiments.teammate_reid.finetune train    # the frozen run -> adapter.pt
    python -m experiments.teammate_reid.finetune parity   # C and D once -> parity.json (refuses twice)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments.teammate_reid.benchmark import load_reviewed
from experiments.teammate_reid.evaluation import build_queries
from experiments.teammate_reid.openset import RUNS
from experiments.teammate_reid.symmetric import Arrays, _best, _pick

LOCK = Path(__file__).parent / "locked/finetune_v1.json"
OUT = RUNS / "finetune"
PROTO = RUNS.parent / "smart-reid-prototype-2026-10-18/results"
SPLIT_FRAME = 10000          # 2/3 of a 10-minute window at 25 fps
MIN_SEP = 50
KP, FMR = 0.95, 0.03

# (group, match, results dir, windows): windows None = every window of that match in the dir
SOURCES = [
    ("A", "Balingen-Kiel", RUNS / "dev", None), ("A", "Berlin-BHC", RUNS / "dev", None),
    ("A", "Lemgo-Magdeburg", RUNS / "dev_b", None), ("A", "Goeppingen-Leipzig", RUNS / "dev_b", None),
    ("A", "Ludwigshafen-Wetzlar", RUNS / "fresh", None), ("A", "RNL-SGF", RUNS / "fresh", None),
    ("A", "Dresden-Melsungen", RUNS / "finetune_train/DRE_MTM_66m00_10min", None),
    ("A", "Vinnhorst-Potsdam", RUNS / "finetune_train/Vinnhorst_Potsdam_66m00_10min", None),
    ("B", "Altenholz-Luebeck", RUNS / "sealed", None), ("B", "Pforzheim-Bietigheim", RUNS / "sealed", None),
    ("C", "Kiel-Lemgo", PROTO, None),
    ("C", "Kiel-Lemgo", RUNS / "extra_dev/Kiel_Lemgo_56m00_10min", None),
    ("C", "Kiel-Lemgo", RUNS / "extra_dev/Kiel_Lemgo_76m00_10min", None),
    ("C", "Magdeburg-Wetzlar", RUNS / "higher/Magdeburg_Wetzlar_66m00_10min", None),
    ("C", "Melsungen-Loewen", RUNS / "higher/Melsungen_Loewen_66m00_10min", None),
    ("C", "Gummersbach-Loewen", RUNS / "stage1", None),
    ("C", "Gummersbach-Loewen", RUNS / "extra_dev/VfL_GM_Loewen_56m00_10min", None),
    ("C", "Gummersbach-Loewen", RUNS / "extra_dev/VfL_GM_Loewen_76m00_10min", None),
    ("D", "Eisenach-Hamburg", RUNS / "stage1", None),
    ("D", "Eisenach-Hamburg", RUNS / "extra_dev/Eisenach_Hamburg_57m00_9min", None),
    ("D", "Eisenach-Hamburg", RUNS / "extra_dev/Eisenach_Hamburg_76m00_10min", None),
    ("D", "Stuttgart-Erlangen", RUNS / "higher/Stuttgart_Erlangen_66m00_10min", None),
    ("D", "Saarlouis-Muenster", RUNS / "final/Saarlouis_Muenster_66m00_10min", None),
    ("D", "Saarlouis-Muenster", RUNS / "extra_dev/Saarlouis_Muenster_56m00_10min", None),
    ("D", "Saarlouis-Muenster", RUNS / "extra_dev/Saarlouis_Muenster_76m00_10min", None),
    ("D", "Bielefeld-Aurich", RUNS / "final/Bielefeld_Aurich_86m00_10min", None),
    ("D", "Bielefeld-Aurich", RUNS / "extra_dev/Bielefeld_Aurich_62m00_10min", None),
    ("D", "Bielefeld-Aurich", RUNS / "extra_dev/Bielefeld_Aurich_96m00_10min", None),
    ("D", "Wuerzburg-Regensburg", RUNS / "final/Wuerzburg_Regensburg_66m00_10min", None),
    ("D", "Wuerzburg-Regensburg", RUNS / "extra_dev/Wuerzburg_Regensburg_56m00_10min", None),
    ("D", "Wuerzburg-Regensburg", RUNS / "extra_dev/Wuerzburg_Regensburg_76m00_10min", None),
]
# Teams seen in training, as the MAIN window's anonymous cluster of the match. Gummersbach-Loewen:
# cluster 0 is Loewen in yellow (the same kit and players as RNL-SGF team 0), cluster 1 is
# Gummersbach in blue (checked on gallery crops 2026-10-22). Every other C/D team is unseen.
SEEN_CLUSTERS = {"Kiel-Lemgo": {0, 1}, "Magdeburg-Wetzlar": {0, 1}, "Melsungen-Loewen": {0, 1},
                 "Gummersbach-Loewen": {0}}


# ---------------------------------------------------------------- data

def _tag(results: Path) -> str:
    return str(results.relative_to(RUNS.parent)).replace("/", "__")


def _alignment() -> dict:
    """window -> {extra cluster: main cluster}, from the extra-galleries torso-colour mapping."""
    path = RUNS / "extra_galleries/variants.json"
    return {w.lower(): {int(k): int(v) for k, v in a["map"].items()}
            for w, a in json.loads(path.read_text())["alignment"].items()}


def _to_main(align: dict, results: Path, video: str) -> dict:
    """Cluster mapping onto the match's main window: identity for a main window, the torso-colour
    alignment for an extra window (which must exist: a silent identity would swap teams)."""
    if "extra_dev" in results.parts:
        return align[video.lower()]
    return {0: 0, 1: 1}


def load_window_sets(groups: tuple = ("A", "B", "C", "D")) -> list[dict]:
    """One record per (match, window): crops, accepted fragments, main-window team mapping."""
    align, out = _alignment(), []
    for group, match, results, _ in SOURCES:
        if group not in groups:
            continue
        crops, frags, _ = load_reviewed(results)
        for video in sorted({f["video"] for f in frags if f["match"] == match}):
            fv = [f for f in frags if f["video"] == video]
            keys = {f["key"] for f in fv}
            out.append({"group": group, "match": match, "results": results, "video": video,
                        "fragments": fv, "crops": sorted((c for c in crops if c["fragment"] in keys),
                                                         key=lambda c: (c["frame"], c["crop_id"])),
                        "to_main": _to_main(align, results, video)})
    return out


def block(fragment: dict) -> str | None:
    """'train' (ends before SPLIT_FRAME), 'heldout' (starts at or after it) or None (straddles)."""
    if fragment["end"] < SPLIT_FRAME:
        return "train"
    if fragment["start"] >= SPLIT_FRAME:
        return "heldout"
    return None


def team_key(w: dict, team: int) -> str:
    return f"{w['match']}|{w['to_main'][int(team)]}"


def seen_team(w: dict, team: int) -> bool:
    return w["to_main"][int(team)] in SEEN_CLUSTERS.get(w["match"], set())


# ---------------------------------------------------------------- encoding

def _views(img: np.ndarray, rng: np.random.Generator) -> list:
    """original, horizontal mirror, crop jitter 8% + brightness/contrast/saturation jitter 0.15."""
    import cv2
    h, w = img.shape[:2]
    dx, dy = (int(rng.uniform(0, 0.08) * s) for s in (w, h))
    x0, y0 = int(rng.integers(0, dx + 1)), int(rng.integers(0, dy + 1))
    jit = cv2.resize(img[y0:y0 + h - dy, x0:x0 + w - dx], (w, h)).astype(np.float32)
    b, c, s = (1 + rng.uniform(-0.15, 0.15) for _ in range(3))
    grey = jit.mean(2, keepdims=True)
    jit = np.clip(((grey + (jit - grey) * s) - 128) * c + 128 * b, 0, 255).astype(np.uint8)
    return [img, np.ascontiguousarray(img[:, ::-1]), jit]


def cmd_encode() -> None:
    """Strict-masked SportsReID embeddings of every crop (A also gets 2 augmented views)."""
    import cv2
    from experiments.teammate_reid.encoders import SportsReIDViT, local_digit_boxes
    from experiments.teammate_reid.masking import apply_mask
    enc, emb_dir = SportsReIDViT(), OUT / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    dirs: dict = {}
    for group, _, results, _ in SOURCES:
        dirs[results] = dirs.get(results, False) or group == "A"
    for results, need_views in dirs.items():
        path = emb_dir / f"{_tag(results)}.npz"
        if path.exists() and (not need_views or "v1" in np.load(path).files):
            continue
        if not (results / "crops.json").exists():
            print(f"missing (not mined yet): {results}", flush=True)
            continue
        crops = json.loads((results / "crops.json").read_text())
        crops = sorted(crops, key=lambda c: c["crop_id"])
        rng = np.random.default_rng(0)
        out = {k: [] for k in (("v0", "v1", "v2") if need_views else ("v0",))}
        for i in range(0, len(crops), 128):
            batch = []
            for c in crops[i:i + 128]:
                img = cv2.cvtColor(cv2.imread(str(results / "crops" / f"{c['crop_id']}.jpg")), cv2.COLOR_BGR2RGB)
                masked = apply_mask(img, local_digit_boxes(c, img.shape), "strict", seed_key=str(c["crop_id"]))[0]
                batch.append(_views(masked, rng) if need_views else [masked])
            for v, key in enumerate(out):
                out[key].append(enc.encode([b[v] for b in batch]).astype(np.float16))
        np.savez_compressed(path, crop_ids=np.array([c["crop_id"] for c in crops]),
                            **{k: np.concatenate(v) for k, v in out.items()})
        print(f"{path.name}: {len(crops)} crops, views {list(out)}", flush=True)


def embeddings(results: Path) -> dict:
    """crop_id -> row, and the view arrays (float32, L2-normalised)."""
    z = np.load(OUT / "emb" / f"{_tag(results)}.npz")
    views = {k: z[k].astype(np.float32) for k in z.files if k.startswith("v")}
    views = {k: v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None) for k, v in views.items()}
    return {"row": {int(c): i for i, c in enumerate(z["crop_ids"])}, **views}


# ---------------------------------------------------------------- adapter and training

def make_adapter(dim: int = 512, rank: int = 16, seed: int = 0):
    import torch

    class Adapter(torch.nn.Module):
        """y = l2n(x + U V^T x); U starts at zero, so the untrained adapter is the identity."""

        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.V = torch.nn.Parameter(torch.randn(dim, rank, generator=g) / dim ** 0.5)
            self.U = torch.nn.Parameter(torch.zeros(dim, rank))

        def forward(self, x):
            return torch.nn.functional.normalize(x + (x @ self.V) @ self.U.T, dim=-1)

    return Adapter()


def contrastive_loss(y, ident, frag, frame, team, temperature: float, opp_weight: float):
    """Supervised contrastive loss within one match. Positives: same identity, other fragment,
    >= MIN_SEP frames apart. Same-identity crops that are not positives leave the denominator.
    Teammates weigh 1, opponents `opp_weight`."""
    import torch
    s = y @ y.T / temperature
    same = ident[:, None] == ident[None, :]
    eye = torch.eye(len(y), dtype=torch.bool)
    pos = same & (frag[:, None] != frag[None, :]) & ((frame[:, None] - frame[None, :]).abs() >= MIN_SEP)
    keep = ~eye & (pos | ~same)
    w = torch.where(team[:, None] == team[None, :], 1.0, opp_weight) * keep
    denom = torch.logsumexp(s + torch.log(w.clamp_min(1e-12)) + torch.where(keep, 0.0, -1e9), dim=1)
    n_pos = pos.sum(1)
    has = n_pos > 0
    if not has.any():
        return s.sum() * 0
    pos_mean = (s * pos).sum(1)[has] / n_pos[has]
    return (denom[has] - pos_mean).mean()


def training_pools(windows: list, cap: int, seed: int) -> list[dict]:
    """Per A match: capped training crops (train block only) with identity/fragment/frame/team."""
    rng, pools = np.random.default_rng(seed), []
    for w in windows:
        if w["group"] != "A":
            continue
        e = embeddings(w["results"])
        frags = {f["key"]: f for f in w["fragments"] if block(f) == "train"}
        by_id: dict = {}
        for c in w["crops"]:
            if c["fragment"] in frags:
                by_id.setdefault(frags[c["fragment"]]["identity"], []).append(c)
        rows = []
        for cs in by_id.values():
            if len(cs) > cap:
                cs = [cs[i] for i in sorted(rng.choice(len(cs), cap, replace=False))]
            rows += cs
        ids = {i: n for n, i in enumerate(sorted(by_id))}
        fidx = {k: n for n, k in enumerate(sorted(frags))}
        pools.append({"match": w["match"], "X": np.stack([np.stack([e[v][e["row"][c["crop_id"]]] for c in rows])
                                                          for v in ("v0", "v1", "v2")]),
                      "ident": np.array([ids[frags[c["fragment"]]["identity"]] for c in rows]),
                      "frag": np.array([fidx[c["fragment"]] for c in rows]),
                      "frame": np.array([c["frame"] for c in rows]), "team": np.array([c["team"] for c in rows])})
    return pools


def train(lock: dict) -> tuple:
    import torch
    t = lock["training"]
    seed, cap = t["seed"], t["crop_cap_per_identity"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = training_pools(load_window_sets(("A",)), cap, seed)
    model = make_adapter(rank=16, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(t["epochs"]):
        losses = []
        for p in pools:
            idents = np.unique(p["ident"])
            for _ in range(max(1, len(p["ident"]) // 64)):
                pick = rng.choice(idents, min(8, len(idents)), replace=False)
                rows = np.concatenate([rng.choice(np.flatnonzero(p["ident"] == i), 8) for i in pick])
                view = rng.integers(0, 3, len(rows))
                x = torch.tensor(p["X"][view, rows])
                y = model(x)
                loss = contrastive_loss(y, *(torch.tensor(p[k][rows]) for k in ("ident", "frag", "frame", "team")),
                                        temperature=0.1, opp_weight=0.3)
                loss = loss + 1.0 * ((y - x) ** 2).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        history.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1}: loss {history[-1]:.4f}", flush=True)
    return model, {"loss": history, "pools": {p["match"]: len(p["ident"]) for p in pools}}


# ---------------------------------------------------------------- benchmark

def rows_for(w: dict, adapter=None, which: str | None = None) -> list[dict]:
    """split_v2 benchmark rows (both_notrack, spread8, top-3, halves) on one window's strict
    embeddings, base or adapted. `which` restricts fragments to an A time block."""
    import torch
    e = embeddings(w["results"])
    frags = [f for f in w["fragments"] if which is None or block(f) == which]
    keyset = {f["key"] for f in frags}
    crops = [c for c in w["crops"] if c["fragment"] in keyset]
    if not crops:
        return []
    ids = [c["crop_id"] for c in crops]
    x = e["v0"][[e["row"][c] for c in ids]]
    if adapter is not None:
        with torch.no_grad():
            x = adapter(torch.tensor(x)).numpy()
    sim = x @ x.T
    index = {c: i for i, c in enumerate(ids)}
    local: dict = {}
    for c in crops:
        local.setdefault(c["fragment"], []).append(index[c["crop_id"]])
    rows = []
    for q in build_queries(frags, local, after=True, exclude_track=True):
        qc = _pick(q.crops, "spread8")
        best, score, margin = _best(sim, qc, q.galleries)
        h = len(q.crops) // 2
        halves = None
        if h >= 2 and len(q.galleries) > 1:
            halves = _best(sim, _pick(q.crops[:h], "spread8"), q.galleries)[0] == \
                _best(sim, _pick(q.crops[h:], "spread8"), q.galleries)[0]
        rows.append({"match": w["match"], "team": team_key(w, q.team), "seen": seen_team(w, q.team),
                     "identity": q.identity, "known": q.known, "synthetic_unknown": q.synthetic_unknown,
                     "duration_s": q.meta["duration_s"], "best": best, "score": score, "margin": margin,
                     "halves_agree": halves, "candidates": len(q.galleries)})
    return rows


def metric(rows: list) -> dict:
    """M: max coverage of known queries at known precision >= KP and unknown FMR <= FMR."""
    if not rows:
        return {"M": float("nan"), "gate": None, "fmr": float("nan"), "rank1": float("nan"), "n": 0}
    a = Arrays(rows, halves=True)
    sel = np.ones(len(rows), bool)
    gate = a.choose(sel, KP, FMR)
    known = [r for r in rows if r["known"]]
    rank1 = float(np.mean([r["best"] == r["identity"] for r in known])) if known else float("nan")
    if gate is None:
        return {"M": 0.0, "gate": None, "fmr": 0.0, "rank1": rank1, "n": len(rows)}
    s = a.stats(sel, np.array([gate]))
    return {"M": float(s["coverage"][0]), "gate": [float(g) for g in gate], "fmr": float(s["fmr"][0]),
            "rank1": rank1, "n": len(rows)}


def delta(base: list, adapted: list) -> float:
    return metric(adapted)["M"] - metric(base)["M"]


def _resample(rows: list, rng: np.random.Generator) -> list:
    teams = sorted({r["team"] for r in rows})
    by = {t: [r for r in rows if r["team"] == t] for t in teams}
    return [r for t in rng.choice(teams, len(teams)) for r in by[t]]


def paired(base: list, adapted: list) -> list:
    """Rows of the same queries under both models, so a resample keeps them together."""
    assert len(base) == len(adapted)
    return [{"b": b, "a": a, "team": b["team"]} for b, a in zip(base, adapted)]


def boot_ci(pairs: list, stat, reps: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    vals = [stat(_resample(pairs, rng)) for _ in range(reps)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def group_rows(windows: list, adapter, groups: tuple, which: str | None = None) -> dict:
    out = {"base": [], "adapter": []}
    for w in windows:
        if w["group"] in groups:
            out["base"] += rows_for(w, None, which)
            out["adapter"] += rows_for(w, adapter, which)
    return out


def d_of(pairs: list) -> float:
    return delta([p["b"] for p in pairs], [p["a"] for p in pairs])


# ---------------------------------------------------------------- commands

def _report(name: str, rows: dict) -> dict:
    mb, ma = metric(rows["base"]), metric(rows["adapter"])
    print(f"{name:28s} base M={mb['M']:.3f} rank1={mb['rank1']:.3f} | adapter M={ma['M']:.3f} "
          f"rank1={ma['rank1']:.3f} | n={mb['n']}", flush=True)
    return {"base": mb, "adapter": ma, "delta": ma["M"] - mb["M"]}


def cmd_pilot() -> None:
    lock = json.loads(LOCK.read_text())
    model, info = train(lock)
    windows = load_window_sets(("A", "B"))
    rep = {"train": info,
           "B_inner_validation": _report("B inner validation", group_rows(windows, model, ("B",))),
           "A_seen_same_match": _report("A seen same-match block", group_rows(windows, model, ("A",), "heldout")),
           "A_training_material": _report("A training block", group_rows(windows, model, ("A",), "train"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot.json").write_text(json.dumps(rep, indent=1))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_train() -> None:
    import torch
    lock = json.loads(LOCK.read_text())
    path = OUT / "adapter.pt"
    if path.exists():
        sys.exit(f"{path} exists: the frozen run is done")
    model, info = train(lock)
    torch.save(model.state_dict(), path)
    (OUT / "train.json").write_text(json.dumps({**info, "lock_sha256": _sha(LOCK), "adapter_sha256": _sha(path)},
                                               indent=1))
    print("adapter", _sha(path))


def cmd_parity(reps: int = 2000) -> None:
    import torch
    lock = json.loads(LOCK.read_text())
    res_path = OUT / "parity.json"
    if res_path.exists():
        sys.exit(f"{res_path} exists: the parity test runs once")
    model = make_adapter()
    model.load_state_dict(torch.load(OUT / "adapter.pt"))
    windows = load_window_sets()
    cd = group_rows(windows, model, ("C", "D"))
    unseen = paired([r for r in cd["base"] if not r["seen"]], [r for r in cd["adapter"] if not r["seen"]])
    seen = paired([r for r in cd["base"] if r["seen"]], [r for r in cd["adapter"] if r["seen"]])
    d_un, d_seen = d_of(unseen), d_of(seen)
    rng = np.random.default_rng(0)
    g_vals, un_vals = [], []
    for _ in range(reps):
        u, s = _resample(unseen, rng), _resample(seen, rng)
        du = d_of(u)
        un_vals.append(du)
        g_vals.append(d_of(s) - du)
    un_ci = [float(np.percentile(un_vals, 2.5)), float(np.percentile(un_vals, 97.5))]
    g = d_seen - d_un
    g_ci = [float(np.percentile(g_vals, 2.5)), float(np.percentile(g_vals, 97.5))]
    mb_un, ma_un = metric([p["b"] for p in unseen]), metric([p["a"] for p in unseen])
    passes = {"P1": bool(d_un > 0 and un_ci[0] > 0),
              "P2": bool(g <= max(0.01, 0.25 * d_un) and g_ci[1] <= 0.04),
              "P3": bool(ma_un["fmr"] <= mb_un["fmr"] + 0.01)}
    report = {
        "lock_sha256": _sha(LOCK), "adapter_sha256": _sha(OUT / "adapter.pt"),
        "unseen": {"base": mb_un, "adapter": ma_un, "delta": d_un, "ci95": un_ci,
                   "teams": sorted({p["team"] for p in unseen})},
        "seen_new_match": {"base": metric([p["b"] for p in seen]), "adapter": metric([p["a"] for p in seen]),
                           "delta": d_seen, "teams": sorted({p["team"] for p in seen})},
        "G": g, "G_ci95": g_ci,
        "per_team": {t: {"unseen" if not ps[0]["b"]["seen"] else "seen": d_of(ps)}
                     for t, ps in ((t, [p for p in unseen + seen if p["team"] == t])
                                   for t in sorted({p["team"] for p in unseen + seen}))},
        "controls": {"A_seen_same_match": _report("A seen same-match", group_rows(windows, model, ("A",), "heldout")),
                     "A_training_material": _report("A training block", group_rows(windows, model, ("A",), "train")),
                     "B_inner_validation": _report("B inner validation", group_rows(windows, model, ("B",)))},
        "pass": passes, "verdict": "PASS" if all(passes.values()) else "FAIL",
    }
    res_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("G", "G_ci95", "pass", "verdict")}, indent=1))
    print("unseen delta", d_un, un_ci, "seen delta", d_seen)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg), ties count half."""
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def _diag(rows: list) -> dict:
    a = Arrays(rows, halves=True)
    sel = np.ones(len(rows), bool)
    raw = np.array([r["score"] for r in rows], float)
    wrong = a.known & ~a.right
    out = {"n_known": int(a.known.sum()), "n_unknown": int((~a.known).sum()),
           "n_synthetic": int(sum(r["synthetic_unknown"] for r in rows)), "rank1": float(a.right.sum() / a.known.sum()),
           "halves_disagree": {"right": int(sum(r["halves_agree"] is False for r, x in zip(rows, a.right) if x)),
                               "wrong": int(sum(r["halves_agree"] is False for r, x in zip(rows, wrong) if x)),
                               "unknown": int(sum(r["halves_agree"] is False for r in rows if not r["known"]))}}
    for name, kp, fmr in (("both", KP, FMR), ("precision_only", KP, 1.0), ("fmr_only", 0.0, FMR)):
        g = a.choose(sel, kp, fmr)
        s = a.stats(sel, np.array([g])) if g is not None else None
        out[f"M_{name}"] = float(s["coverage"][0]) if s else 0.0
        if name == "both" and s:
            out["gate"] = [float(x) for x in g]
            out["at_gate"] = {k: float(s[k][0]) for k in ("precision", "fmr", "accepted_known", "correct")}
    for v, arr in (("score", raw), ("margin", a.margin)):
        out[v] = {"right_vs_unknown_auc": _auc(arr[a.right], arr[~a.known]),
                  "right_vs_wrong_auc": _auc(arr[a.right], arr[wrong]),
                  "median": {k: float(np.median(arr[m])) if m.any() else None
                             for k, m in (("right", a.right), ("wrong", wrong), ("unknown", ~a.known))}}
    return out


def cmd_diag_b() -> None:
    """Why B loses M while rank-1 rises (plan v3 review, step 2). Reads B rows only."""
    import torch
    model = make_adapter()
    model.load_state_dict(torch.load(OUT / "adapter.pt"))
    windows = load_window_sets(("B",))
    rep = {}
    for scope, ws in [("B", windows)] + [(w["match"], [w]) for w in windows]:
        rows = group_rows(ws, model, ("B",))
        rep[scope] = {"base": _diag(rows["base"]), "adapter": _diag(rows["adapter"])}
    (OUT / "diag_b.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["B"], indent=1))
    for scope in rep:
        print(scope, {c: (round(rep[scope][c]["M_both"], 3), round(rep[scope][c]["rank1"], 3)) for c in rep[scope]})


if __name__ == "__main__":
    {"encode": cmd_encode, "pilot": cmd_pilot, "train": cmd_train, "parity": cmd_parity,
     "diag_b": cmd_diag_b}[sys.argv[1]]()
