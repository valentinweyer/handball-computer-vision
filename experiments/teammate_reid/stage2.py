"""Stage 2: incremental coverage on baseline-unresolved segments.

Two banks, each with its own output folder, galleries and lock (`--bank`):
`test_v1` -- the three split-v1 test matches against `locked/split_v1.json` (the default,
unchanged); `final` -- the four final-test windows against `locked/split_v2.json`
(galleries from both sides without the segment's own track, spread query crops,
half-split guard).

Order, so the matcher can never shape the labels:

1. `population` -- fix the population before any matcher runs. A local segment is one
   tracker id's uninterrupted presence, cut at identity breaks and >1 s gaps (exactly
   `mining.split_fragments`). It is in the population when it is not a goalkeeper, the
   production bundle shows a number on < 50% of its rows (baseline-unresolved), and at
   least 3 clean crops can be sampled from it (`mining.sample_crops`, the matcher's own
   evidence floor). Shorter or crop-poor unresolved segments are listed as
   `insufficient_evidence` with durations, never labelled or scored.
2. `review` -- blind review material per segment: a keyframe sheet (whole-frame context
   before, during and after, plus zoomed crops) and a short mp4. Only the target box is
   drawn, in one neutral colour; no IDs, team colours, numbers or proposals.
3. Tier-1 labels (assistant, blind) go to `annotations_assistant_v1.json` and are
   frozen (hashed) before step 4. Tier 2 is the user's blind check of a compact queue.
4. `propose` -- the split-v1 lock's matcher on the population; written separately.

    python -m experiments.teammate_reid.stage2 population
    python -m experiments.teammate_reid.stage2 review
    python -m experiments.teammate_reid.stage2 propose     # sealed: proposals never enter the queue or page
    python -m experiments.teammate_reid.stage2 queue       # blind mixed queue + local labelling page
    python -m experiments.teammate_reid.stage2 serve       # the page over Tailscale; labels saved here
    python -m experiments.teammate_reid.stage2 score       # only after the labels are frozen (hashed)
    python -m experiments.teammate_reid.stage2 recheck --number 51   # blank re-check page for one label value
    python -m experiments.teammate_reid.stage2 --bank final population   # same steps on the final bank
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments.teammate_reid.mining import load_video, mine_video, sample_crops
from experiments.teammate_reid.run import HELDOUT, VIDEO_SETS, VIDEOS

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
RUNS = REPO / "runs/smart-reid-shadow"
BANKS = {
    "test_v1": {"out": RUNS / "stage2",
                "specs": {"Kiel-Lemgo": VIDEOS[0], "Eisenach-Hamburg": HELDOUT[0], "Gummersbach-Loewen": HELDOUT[1]},
                "galleries": {"Kiel-Lemgo": REPO / "runs/smart-reid-prototype-2026-10-18/results",
                              "Eisenach-Hamburg": RUNS / "stage1", "Gummersbach-Loewen": RUNS / "stage1"},
                "lock": "split_v1.json", "page_key": "stage2_labels_v1", "recheck_key": "stage2_recheck_v1"},
    "final": {"out": RUNS / "stage2_final",
              "specs": {s.match: s for s in VIDEO_SETS["final"]},
              "galleries": {s.match: RUNS / "final" / s.bundle.name for s in VIDEO_SETS["final"]},
              "lock": "split_v2.json", "page_key": "stage2_final_labels_v1", "recheck_key": "stage2_final_recheck_v1"},
}
SHOWN_MAX = 0.5          # baseline shows a number on fewer rows than this -> unresolved
MIN_CROPS = 3            # the matcher's evidence floor
FPS = 25.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    segments, excluded, inputs = [], [], {}
    for match, spec in SPECS.items():
        loaded = load_video(spec)
        frags, _ = mine_video(loaded)
        tracks, numbers = loaded["tracks"], loaded["numbers"]
        by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}
        tbf = {f: g for f, g in tracks.groupby("frame")}
        nbf = {f: list(g.itertuples()) for f, g in numbers.groupby("frame")}
        det = pd.read_parquet(spec.bundle / "detections.parquet")
        refs = {f: list(g.itertuples()) for f, g in det[det["class_name"] == "referee"].groupby("frame")}
        src = loaded["manifest"]["source"]
        inputs[match] = {k: v["sha256"] for k, v in loaded["inputs"].items()}
        for f in frags:
            rows = by_track[f.track_id].loc[f.start:f.end]
            record = {"segment": f"{match}|{f.key}", "match": match, "video": spec.name, "key": f.key,
                      "track_id": f.track_id, "start": f.start, "end": f.end,
                      "duration_s": round(len(rows) / FPS, 2), "opened_by": f.opened_by, "closed_by": f.closed_by}
            if rows["is_goalkeeper"].any():
                continue
            shown = float(rows["number"].notna().mean())
            if shown >= SHOWN_MAX:
                continue                     # baseline-resolved: not in the population
            crops, _ = sample_crops(f, tbf, nbf, set(det["frame"].unique()), src["width"], src["height"],
                                    every=10, cap=60, referees_by_frame=refs)
            record.update(baseline_shown=round(shown, 3), clean_crops=len(crops),
                          own_reads=dict(Counter(f.reads)) if f.reads else {})
            (segments if len(crops) >= MIN_CROPS else excluded).append(record)
    for i, s in enumerate(sorted(segments, key=lambda s: (s["match"], s["start"], s["track_id"]))):
        s["review_id"] = f"S{i:03d}"
    report = {"definition": {"shown_max": SHOWN_MAX, "min_crops": MIN_CROPS, "goalkeepers": "excluded",
                             "segments": "mining.split_fragments (identity breaks, >1 s gaps)"},
              "inputs": inputs, "population": segments, "insufficient_evidence": excluded,
              "summary": {m: {"segments": sum(s["match"] == m for s in segments),
                              "seconds": round(sum(s["duration_s"] for s in segments if s["match"] == m), 1),
                              "insufficient": sum(s["match"] == m for s in excluded),
                              "insufficient_seconds": round(sum(s["duration_s"] for s in excluded if s["match"] == m), 1)}
                          for m in SPECS}}
    (OUT / "population.json").write_text(json.dumps(report, indent=1, default=float))
    return report


BOX = (0, 255, 255)      # one neutral colour for the target; nothing else is drawn


def review() -> None:
    """Blind review sheets and clips; reads only population.json and the bundles."""
    pop = json.loads((OUT / "population.json").read_text())["population"]
    rdir = OUT / "review"
    rdir.mkdir(exist_ok=True)
    for match, spec in SPECS.items():
        segs = [s for s in pop if s["match"] == match]
        if not segs:
            continue
        tracks = pd.read_parquet(spec.bundle / "tracks.parquet", columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        box_of = {(int(r.frame), int(r.track_id)): (r.x1, r.y1, r.x2, r.y2) for r in tracks.itertuples()}
        manifest = json.loads((spec.bundle / "manifest.json").read_text())
        cap = cv2.VideoCapture(str(spec.bundle / manifest["files"]["video"]))
        want: dict = {}
        plans = {}
        for s in segs:
            span = list(range(s["start"], s["end"] + 1))
            inside = [span[i] for i in np.linspace(0, len(span) - 1, 8).round().astype(int)]
            before = [max(0, s["start"] - 50), max(0, s["start"] - 12)]
            after = [s["end"] + 12, s["end"] + 50]
            clip = list(range(max(0, s["start"] - 50), s["end"] + 51, 2))[:450]   # <= 36 s at half rate
            plans[s["review_id"]] = (s, before + inside + after, clip)
            for f in before + inside + after + clip:
                want.setdefault(f, []).append(s["review_id"])
        frames: dict = {}
        writers = {}
        for f in range(max(want) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if f not in want:
                continue
            for rid in want[f]:
                s, keys, clip = plans[rid]
                box = box_of.get((f, s["track_id"]))
                if f in keys:
                    frames.setdefault(rid, {})[f] = (frame.copy(), box)
                if f in clip:
                    small = cv2.resize(frame, (960, 540))
                    if box is not None:   # box is in source pixels; the clip is 960 wide
                        x1, y1, x2, y2 = (int(v * 960 / frame.shape[1]) for v in box)
                        cv2.rectangle(small, (x1, y1), (x2, y2), BOX, 2)
                    cv2.putText(small, f"{rid}  frame {f}", (8, 24), 0, 0.7, (255, 255, 255), 2)
                    if rid not in writers:
                        writers[rid] = cv2.VideoWriter(str(rdir / f"{rid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                       12.5, (960, 540))
                    writers[rid].write(small)
        for w in writers.values():
            w.release()
        with ThreadPoolExecutor(8) as pool:   # OpenCV writes MPEG-4 Part 2, which Firefox/Chrome cannot play
            list(pool.map(_to_h264, (rdir / f"{rid}.mp4" for rid in writers)))
        for rid, (s, keys, _) in plans.items():
            _sheet(rdir / f"{rid}.jpg", rid, s, keys, frames.get(rid, {}))


def _to_h264(clip: Path) -> None:
    tmp = clip.with_suffix(".h264.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(clip)


def _sheet(path: Path, rid: str, s: dict, keys: list, frames: dict) -> None:
    W = 1600
    tiles, zooms = [], []
    for f in keys:
        if f not in frames:
            continue
        frame, box = frames[f]
        ctx = frame.copy()
        tag = "before" if f < s["start"] else "after" if f > s["end"] else "in"
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(ctx, (x1, y1), (x2, y2), BOX, 4)
            if tag == "in":
                bw, bh = x2 - x1, y2 - y1
                zx1, zy1 = max(0, x1 - bw // 2), max(0, y1 - bh // 4)
                zx2, zy2 = min(frame.shape[1], x2 + bw // 2), min(frame.shape[0], y2 + bh // 4)
                z = frame[zy1:zy2, zx1:zx2]
                if z.size:
                    z = cv2.resize(z, (max(1, int(z.shape[1] * 300 / z.shape[0])), 300))
                    cv2.putText(z, str(f), (4, 292), 0, 0.5, (255, 255, 255), 1)
                    zooms.append(z)
        t = cv2.resize(ctx, (384, 216))
        cv2.putText(t, f"{tag} f{f}", (6, 22), 0, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    head = np.full((44, W, 3), 25, np.uint8)
    cv2.putText(head, f"{rid}  ({s['duration_s']:.1f} s)  target = yellow box. Who is it? Number if readable anywhere,"
                      f" else 'unverifiable'; 'mixed' if the box changes person.", (8, 30), 0, 0.6, (255, 255, 255), 1)
    grid = [np.hstack(tiles[i:i + 4] + [np.zeros((216, 384, 3), np.uint8)] * (4 - len(tiles[i:i + 4])))
            for i in range(0, len(tiles), 4)]
    rows = [head] + [np.hstack([g, np.zeros((216, W - g.shape[1], 3), np.uint8)]) for g in grid]
    if zooms:
        z = np.hstack(zooms)
        z = z[:, :W] if z.shape[1] >= W else np.hstack([z, np.zeros((300, W - z.shape[1], 3), np.uint8)])
        rows.append(z)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


# --- proposals (split-v1 lock, applied to unresolved segments) -----------------

M_CROPS, TOP, MIN_SEP = 8, 3, 50


def _supported_numbers(reads: dict) -> set:
    """Numbers this segment's own reads support (>= 2 reads); a trailing single digit folds in."""
    full = {v for v, c in reads.items() if c >= 2 and len(v) > 1}
    return full | {v for v, c in reads.items() if c >= 2 and len(v) == 1 and not any(f.endswith(v) for f in full)}


def gallery_allowed(g: dict, s: dict, v2: bool) -> bool:
    """May vouched fragment `g` serve as gallery for segment `s`? (split v1 / v2 protocol)"""
    before = g["end"] + MIN_SEP <= s["start"]
    after = v2 and g["start"] >= s["end"] + MIN_SEP
    return (before or after) and not (v2 and g["track_id"] == s["track_id"])


def propose() -> dict:
    """Match every population segment against its match's number-vouched galleries.

    Same protocol the lock was validated under: gallery fragments of the same team end
    >= 2 s before the segment starts (team taken from the segment's majority rows; if
    its team is unsettled, both teams' galleries are candidates), first 8 sampled crops,
    top-3 mean. Vetoes before ranking, each logged: co-presence (the candidate is shown
    by the baseline, or has a vouched fragment, on any frame of the segment) and number
    contradiction (the segment's own reads support only other numbers). Accept when
    score >= tau and margin >= m over the runner-up; one candidate left -> abstain.
    Never feeds back into galleries, tracks or labels.

    Under a split-v2 lock the lock's own protocol applies instead: gallery fragments may
    also start >= 2 s after the segment, the segment's own track never serves as a
    gallery, 8 query crops are spread over the segment, and the segment abstains
    (`halves_disagree`) when its first and second half pick different players.
    """
    from experiments.teammate_reid.benchmark import load_reviewed
    from experiments.teammate_reid.encoders import SportsReIDViT
    from experiments.teammate_reid.masking import crop_with_digits
    from experiments.teammate_reid.run import FrameReader
    from experiments.teammate_reid.symmetric import _pick
    if (OUT / "proposals_SEALED.json").exists():
        raise SystemExit(f"{OUT / 'proposals_SEALED.json'} exists; proposals are sealed once")
    lock = json.loads(LOCK.read_text())
    v2 = "gallery" in lock
    if v2:
        assert (lock["encoder"], lock["mask"], lock["gallery"], lock["query_crops"], lock["rule"]) == (
            "sportsreid_vitb16", "raw", "both_notrack", "spread8", "absolute") and lock["min_sep_frames"] == MIN_SEP
    else:
        assert lock["feature"] == "sportsreid" and lock["rule"] == "absolute"
    tau, m_gate = lock["gate"][:2]
    pop = {s["segment"]: s for s in json.loads((OUT / "population.json").read_text())["population"]}
    encoder = SportsReIDViT()
    out_rows = []
    for match, spec in SPECS.items():
        loaded = load_video(spec)
        frags, _ = mine_video(loaded)
        frags = {f.key: f for f in frags}
        tracks, numbers = loaded["tracks"], loaded["numbers"]
        by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}
        tbf = {f: g for f, g in tracks.groupby("frame")}
        nbf = {f: list(g.itertuples()) for f, g in numbers.groupby("frame")}
        det = pd.read_parquet(spec.bundle / "detections.parquet")
        refs = {f: list(g.itertuples()) for f, g in det[det["class_name"] == "referee"].groupby("frame")}
        src = loaded["manifest"]["source"]
        # galleries: the reviewed, number-vouched fragments of this video and their SportsReID features
        gres = GALLERY_RESULTS[match]
        gcrops, gfrags, _ = load_reviewed(gres)
        gfrags = [f for f in gfrags if f["video"] == spec.name]
        gfeat = np.load(gres / "features_sportsreid_vitb16.npz")["raw"]
        gfeat = gfeat / np.clip(np.linalg.norm(gfeat, axis=1, keepdims=True), 1e-8, None)
        gcrops_by_frag: dict = {}
        for c in sorted(gcrops, key=lambda c: c["frame"]):
            gcrops_by_frag.setdefault(c["fragment"], []).append(c["crop_id"])
        shown = tracks.dropna(subset=["number"])
        shown_frames = {(int(t), str(n)): set(g["frame"]) for (t, n), g in shown.groupby(["team", "number"])}
        # query crops for this match's population segments
        segs = [s for s in pop.values() if s["match"] == match]
        records = {}
        for s in segs:
            recs, _ = sample_crops(frags[s["key"]], tbf, nbf, set(det["frame"].unique()), src["width"],
                                   src["height"], every=10, cap=60, referees_by_frame=refs)
            recs = sorted(recs, key=lambda r: r["frame"])
            records[s["segment"]] = recs if v2 else recs[:M_CROPS]
        wanted = sorted({r["frame"] for rs in records.values() for r in rs})
        reader = FrameReader(spec.bundle / loaded["manifest"]["files"]["video"])
        pixels = {}
        for f in wanted:
            frame = reader.read(f)
            for seg, rs in records.items():
                for i, r in enumerate(rs):
                    if r["frame"] == f:
                        pixels[(seg, i)] = crop_with_digits(frame, r["box"], [])[0]
        for s in segs:
            rs = records[s["segment"]]
            q_all = encoder.encode([pixels[(s["segment"], i)] for i in range(len(rs))])
            q = q_all[_pick(list(range(len(rs))), "spread8")] if v2 else q_all
            rows = by_track[s["track_id"]].loc[s["start"]:s["end"]]
            teams = rows["team"].value_counts()
            settled = (rows["team_state"] == "stable").mean() >= 0.9 and len(teams) == 1
            seg_frames = set(range(s["start"], s["end"] + 1)) & set(rows.index)
            supported = _supported_numbers(s.get("own_reads", {}))
            scores, vetoes = {}, {}
            for g in gfrags:
                if settled and g["team"] != int(teams.index[0]):
                    continue
                if not gallery_allowed(g, s, v2):
                    continue
                scores.setdefault(g["identity"], []).extend(gcrops_by_frag.get(g["key"], []))
            for identity in list(scores):
                team, number = int(identity.split("|team")[1].split("|")[0]), identity.split("#")[-1]
                if seg_frames & shown_frames.get((team, number), set()) or any(
                        f["identity"] == identity and f["start"] <= s["end"] and f["end"] >= s["start"]
                        for f in gfrags):
                    vetoes[identity] = "co_presence"
                elif supported and number not in supported:
                    vetoes[identity] = "number_contradiction:" + ",".join(sorted(supported))
            def rank(qx: np.ndarray) -> list:
                out = []
                for identity, ids in scores.items():
                    if identity in vetoes or not ids:
                        continue
                    sims = qx @ gfeat[ids].T
                    k = min(TOP, sims.shape[1])
                    out.append((float(np.sort(sims, axis=1)[:, -k:].mean(1).mean()), identity))
                return sorted(out, reverse=True)

            ranked = rank(q)
            best = ranked[0] if ranked else (float("-inf"), None)
            margin = best[0] - ranked[1][0] if len(ranked) > 1 else float("-inf")
            halves_agree = None
            h = len(rs) // 2
            if v2 and h >= 2 and len(ranked) > 1:
                first, second = (rank(q_all[_pick(list(part), "spread8")])[0][1]
                                 for part in (range(h), range(h, len(rs))))
                halves_agree = first == second
            gate_ok = best[1] is not None and best[0] >= tau and margin >= m_gate
            accepted = gate_ok and halves_agree is not False
            reason = ("accepted" if accepted else "halves_disagree" if gate_ok else "no_candidate" if not ranked else
                      "single_candidate" if len(ranked) == 1 else "below_gate")
            out_rows.append({"review_id": s["review_id"], "segment": s["segment"], "match": match,
                             "duration_s": s["duration_s"], "query_crops": len(rs), "team_settled": bool(settled),
                             "proposal": best[1] if accepted else None, "best": best[1], "score": best[0],
                             "margin": margin, "decision": reason, "candidates": len(ranked),
                             "halves_agree": halves_agree,
                             "vetoed": vetoes, "top3": ranked[:3]})
    protocol = ("split v2: galleries end >= 2 s before or start >= 2 s after the segment, never its own track; "
                "8 spread query crops; half-split guard" if v2 else
                "gallery fragments end >= 2 s before the segment (as validated)")
    sealed = {"lock_sha256": sha256(LOCK), "lock": LOCK.name, "gate": [tau, m_gate], "protocol": protocol,
              "rows": out_rows}
    path = OUT / "proposals_SEALED.json"
    path.write_text(json.dumps(sealed, indent=1, default=float))
    (OUT / "accepted_ids.json").write_text(json.dumps(
        sorted(r["review_id"] for r in out_rows if r["decision"] == "accepted")))
    return {"segments": len(out_rows), "accepted": sum(r["decision"] == "accepted" for r in out_rows),
            "decisions": dict(Counter(r["decision"] for r in out_rows)), "sealed_sha256": sha256(path)}


RANDOM_EXTRA = 40
SEED = 20261019


def team_reference(match: str) -> dict:
    """One clear crop per team from the vouched gallery, so the labeller can name the team.

    Team 0/1 are the anonymous per-video clusters; a reference crop is copied into the
    review folder as `team_<match>_<t>.jpg`.
    """
    from experiments.teammate_reid.benchmark import load_reviewed
    spec, gres = SPECS[match], GALLERY_RESULTS[match]
    crops, frags, _ = load_reviewed(gres)
    keys = {f["key"] for f in frags if f["video"] == spec.name}
    out = {}
    for team in (0, 1):
        cands = [c for c in crops if c["fragment"] in keys and c["team"] == team and c["digit_boxes"]]
        best = max(cands, key=lambda c: (c["box"][3] - c["box"][1]))
        img = cv2.imread(str(gres / "crops" / f"{best['crop_id']}.jpg"))
        name = f"team_{match}_{team}.jpg"
        cv2.imwrite(str(OUT / "review" / name), cv2.resize(img, (int(img.shape[1] * 220 / img.shape[0]), 220)))
        out[team] = name
    return out


def queue() -> dict:
    """Blind queue: every accepted segment plus a seeded random sample of the rest, shuffled.

    Only review ids, matches and durations enter the queue and the page; the sealed
    proposals never do. The labeller cannot tell an accepted segment from a sampled one.
    """
    pop = json.loads((OUT / "population.json").read_text())["population"]
    accepted = set(json.loads((OUT / "accepted_ids.json").read_text()))
    rng = np.random.default_rng(SEED)
    rest = [s["review_id"] for s in pop if s["review_id"] not in accepted]
    sample = set(rng.choice(rest, size=min(RANDOM_EXTRA, len(rest)), replace=False).tolist())
    items = [s for s in pop if s["review_id"] in accepted | sample]
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    refs = {m: team_reference(m) for m in SPECS}
    q = [{"n": i + 1, "review_id": s["review_id"], "match": s["match"], "duration_s": s["duration_s"]}
         for i, s in enumerate(items)]
    membership = {"accepted": sorted(accepted), "random_sample": sorted(sample), "seed": SEED}
    (OUT / "queue.json").write_text(json.dumps(q, indent=1))
    (OUT / "queue_membership_SEALED.json").write_text(json.dumps(membership, indent=1))
    (OUT / "review" / "label.html").write_text(_page(q, refs, BANK["page_key"]))
    return {"queue": len(q), "accepted": len(accepted), "random": len(sample),
            "page": str(OUT / "review" / "label.html")}


def _page(q: list, refs: dict, key: str = "stage2_labels_v1", labels_url: str = "labels") -> str:
    data = json.dumps(q)
    team_refs = json.dumps({m: {str(t): n for t, n in r.items()} for m, r in refs.items()})
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stage 2 labelling</title>
<style>
:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa0a6;--acc:#ffd400;--card:#1f2227}
body{background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif;margin:0;padding:16px}
.wrap{max-width:1650px;margin:auto}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.card{background:var(--card);padding:12px;border-radius:8px}img.sheet{width:100%;border-radius:6px}
video{width:100%;max-width:960px;border-radius:6px;margin-top:8px}
.teams{display:flex;gap:16px;align-items:flex-end}.teams figure{margin:0;text-align:center;color:var(--mut)}
.teams img{height:160px;border-radius:4px;border:3px solid transparent;cursor:pointer}.teams img.sel{border-color:var(--acc)}
input[type=text]{font-size:20px;width:90px;padding:6px;background:#111;color:var(--fg);border:1px solid #444;border-radius:6px}
button{font-size:15px;padding:7px 12px;border-radius:6px;border:1px solid #444;background:#2a2e35;color:var(--fg);cursor:pointer}
button.on{background:var(--acc);color:#000}.mut{color:var(--mut)}textarea{width:100%;background:#111;color:var(--fg);border:1px solid #444}
</style></head><body><div class="wrap">
<div class="bar"><b id="pos"></b><span class="mut" id="meta"></span>
<button onclick="go(-1)">&larr; prev</button><button onclick="go(1)">next &rarr;</button>
<span class="mut" id="done"></span><span class="mut" id="sync"></span><button onclick="exportJson()">Download labels JSON</button></div>
<div class="card">
<p class="mut">Who is the player in the <b style="color:var(--acc)">yellow box</b> for the whole segment? Use any frame
(also before/after) where the number is readable. If the box switches person, choose <i>mixed</i>.
If you cannot tell, choose <i>unclear</i>. Never guess.</p>
<div class="bar"><span>Team:</span><div class="teams" id="teams"></div>
<span>Number:</span><input type="text" id="num" placeholder="e.g. 22" autocomplete="off" inputmode="numeric" oninput="save()">
<button id="b_mixed" onclick="flag('mixed')">mixed</button><button id="b_unclear" onclick="flag('unclear')">unclear</button>
<button id="b_np" onclick="flag('not_a_player')">not a player</button></div>
<div class="bar"><span class="mut">Evidence frames / note (optional):</span></div>
<textarea id="note" rows="2" oninput="save()"></textarea>
<img class="sheet" id="sheet"><video id="vid" controls muted loop playsinline></video></div></div>
<script>
const Q=__DATA__, REFS=__REFS__, KEY="__KEY__", LABELS_URL="__URL__";
let L={}; try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
// Served by `stage2 serve`: the server copy is the source of truth, so any device resumes it.
let SERVED=false, timer=null;
function setSync(t){document.getElementById("sync").textContent=t}
function push(){clearTimeout(timer);if(!SERVED)return;timer=setTimeout(()=>fetch(LABELS_URL,{method:"PUT",
headers:{"Content-Type":"application/json"},body:JSON.stringify({labeller:"human",exported_at:new Date().toISOString(),labels:L})})
.then(r=>{if(!r.ok)throw r.status;setSync("· saved on server")}).catch(e=>setSync("· NOT saved on server ("+e+")")),400)}
let i=0; try{i=+localStorage.getItem(KEY+"_pos")||0}catch(e){}
function cur(){const q=Q[i];return L[q.review_id]||(L[q.review_id]={team:null,number:"",flag:null,note:""})}
function render(){const q=Q[i],l=cur();document.getElementById("pos").textContent=`#${q.n} / ${Q.length}  (${q.review_id})`;
document.getElementById("meta").textContent=`${q.match} · ${q.duration_s.toFixed(1)} s`;
document.getElementById("sheet").src=q.review_id+".jpg";const v=document.getElementById("vid");v.src=q.review_id+".mp4";
const t=document.getElementById("teams");t.innerHTML="";for(const k of ["0","1"]){const f=document.createElement("figure");
const im=document.createElement("img");im.src=REFS[q.match][k];im.className=l.team===k?"sel":"";im.onclick=()=>{l.team=l.team===k?null:k;save();render()};
f.appendChild(im);f.appendChild(document.createTextNode("team "+k));t.appendChild(f)}
document.getElementById("num").value=l.number;document.getElementById("note").value=l.note;
for(const [id,fl] of [["b_mixed","mixed"],["b_unclear","unclear"],["b_np","not_a_player"]])document.getElementById(id).className=l.flag===fl?"on":"";
const n=Object.values(L).filter(x=>x.flag||(x.team!==null&&x.number)).length;document.getElementById("done").textContent=`${n} labelled`}
function save(){const l=cur();l.number=document.getElementById("num").value.trim();l.note=document.getElementById("note").value;
try{localStorage.setItem(KEY,JSON.stringify(L));localStorage.setItem(KEY+"_pos",i)}catch(e){}push();render()}
function flag(f){const l=cur();l.flag=l.flag===f?null:f;save()}
function go(d){i=Math.max(0,Math.min(Q.length-1,i+d));save()}
function exportJson(){const out={labeller:"human",exported_at:new Date().toISOString(),labels:L};
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
a.download="annotations_human_v1.json";a.click()}
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA")return;
if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1)});
fetch(LABELS_URL).then(r=>r.ok?r.json():Promise.reject(r.status)).then(d=>{SERVED=true;
if(Object.keys(d.labels||{}).length)L=d.labels;setSync("· saved on server");render()})
.catch(()=>setSync("· local only: use Download"));render();
</script></body></html>""".replace("__DATA__", data).replace("__REFS__", team_refs).replace("__KEY__", key).replace("__URL__", labels_url)


def configure(bank: str) -> None:
    """Point the module at one bank. Labels freeze with `<labels>.sha256`; edits stop then.
    The adjudication file is not blind: it is made after unsealing."""
    global BANK, OUT, SPECS, GALLERY_RESULTS, LOCK, LABELS, FROZEN, RECHECK, ADJUDICATION, LABEL_FILES
    BANK = BANKS[bank]
    OUT, SPECS, GALLERY_RESULTS = BANK["out"], BANK["specs"], BANK["galleries"]
    LOCK = Path(__file__).parent / "locked" / BANK["lock"]
    LABELS = OUT / "annotations_human_v1.json"
    FROZEN = OUT / "annotations_human_v1.sha256"
    RECHECK = OUT / "annotations_human_v1_recheck.json"
    ADJUDICATION = OUT / "annotations_human_v1_adjudication.json"
    LABEL_FILES = {"/labels": (LABELS, "queue.json"), "/labels_recheck": (RECHECK, "recheck_queue.json")}


configure("test_v1")


def recheck(number: str) -> dict:
    """A blank page for the frozen v1 labels carrying one number, for a second blind
    look. v1 stays as it is; the re-check is its own file and is reported beside it."""
    labels = json.loads(LABELS.read_text())["labels"]
    q = [x for x in json.loads((OUT / "queue.json").read_text())
         if labels.get(x["review_id"], {}).get("number", "").strip() == number]
    q = [{**x, "n": i + 1} for i, x in enumerate(q)]
    (OUT / "recheck_queue.json").write_text(json.dumps(q, indent=1))
    refs = {m: {t: f"team_{m}_{t}.jpg" for t in (0, 1)} for m in SPECS}
    (OUT / "review" / "recheck.html").write_text(_page(q, refs, BANK["recheck_key"], "labels_recheck"))
    return {"items": len(q), "page": "recheck.html"}


class _LabelHandler(SimpleHTTPRequestHandler):
    """The review folder read-only (with byte ranges, which Safari needs for video),
    plus `GET/PUT /labels` for the one labels file. Nothing else is writable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUT / "review"), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route in LABEL_FILES:
            path = LABEL_FILES[route][0]
            return self._json(200, json.loads(path.read_text()) if path.is_file() else {"labels": {}})
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().do_GET()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        size = path.stat().st_size
        if not m or not any(m.groups()):
            return self.send_error(416)
        first, last = m.groups()
        start, end = (int(first), min(int(last), size - 1) if last else size - 1) if first \
            else (max(0, size - int(last)), size - 1)
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            return self.end_headers()
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            self.wfile.write(f.read(end - start + 1))

    def do_PUT(self):
        route = self.path.split("?", 1)[0]
        if route not in LABEL_FILES:
            return self.send_error(405)
        target, queue_file = LABEL_FILES[route]
        if target.with_suffix(".sha256").is_file():
            return self._json(409, {"error": "labels are frozen"})
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= 2_000_000:
            return self._json(413, {"error": "bad size"})
        try:
            body = json.loads(self.rfile.read(n))
            known = {q["review_id"] for q in json.loads((OUT / queue_file).read_text())}
            assert isinstance(body, dict) and isinstance(body.get("labels"), dict)
            assert set(body["labels"]) <= known and all(isinstance(v, dict) for v in body["labels"].values())
        except (ValueError, AssertionError):
            return self._json(400, {"error": "not a labels file for this queue"})
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, indent=1))
        os.replace(tmp, target)
        return self._json(200, {"saved": len(body["labels"])})


def serve(host: str | None, port: int) -> None:
    """Serve the labelling page; default host is this machine's Tailscale IPv4, so the
    page is reachable from the tailnet only."""
    host = host or subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True,
                                  check=True).stdout.split()[0]
    server = ThreadingHTTPServer((host, port), _LabelHandler)
    print(f"http://{host}:{port}/label.html  (labels -> {LABELS})", flush=True)
    server.serve_forever()


def _identity(match: str, team, number: str, gallery_ids: set) -> tuple:
    """A human label as (team, number). A missing team is filled only when the number
    belongs to exactly one team among the match's vouched identities (recorded)."""
    number = number.strip().lstrip("0") or "0"
    if team is not None:
        return (int(team), number), False
    teams = {t for t, n in gallery_ids if n == number}
    return ((teams.pop(), number), True) if len(teams) == 1 else (None, False)


def _frozen_labels(path: Path) -> tuple:
    frozen = path.with_suffix(".sha256").read_text().split()[0]
    if sha256(path) != frozen:
        raise SystemExit(f"{path.name} changed after freezing")
    return json.loads(path.read_text())["labels"], frozen


def score(version: str = "v1") -> dict:
    """Unseal: compare the frozen human labels with the sealed proposals.

    Accepted segment -> correct (same team and number), wrong (a different named
    player, or not a player), or unscored (mixed / unclear / unnamed). Not-accepted
    segments (random sample) -> human-identifiable or not, and whether that player had a
    vouched gallery before the segment (else no proposal could have been correct).
    Nothing is re-tuned: the gate is the split-v1 lock.
    """
    labels, frozen = _frozen_labels(LABELS)
    if version in ("v2", "v3"):   # v1 with the blind re-check replacing the items it covered
        rechecked, frozen_re = _frozen_labels(RECHECK)
        labels = {**labels, **rechecked}
        frozen = {"v1": frozen, "recheck": frozen_re}
    if version == "v3":   # + the user's adjudication of the errors, made after seeing the proposals
        adjudicated, frozen_adj = _frozen_labels(ADJUDICATION)
        labels = {**labels, **adjudicated}
        frozen["adjudication"] = frozen_adj
    proposals = {r["review_id"]: r for r in json.loads((OUT / "proposals_SEALED.json").read_text())["rows"]}
    membership = json.loads((OUT / "queue_membership_SEALED.json").read_text())
    pop = json.loads((OUT / "population.json").read_text())["population"]

    def parse(identity: str) -> tuple:
        return int(identity.split("|team")[1].split("|")[0]), identity.split("#")[-1]

    gallery = {m: set() for m in SPECS}
    for r in proposals.values():
        for _, ident in r["top3"]:
            gallery[r["match"]].add(parse(ident))
        for ident in r["vetoed"]:
            gallery[r["match"]].add(parse(ident))
    items = []
    for rid in membership["accepted"] + membership["random_sample"]:
        lab, prop = labels.get(rid, {}), proposals[rid]
        truth, inferred = (None, False)
        if not lab.get("flag") and lab.get("number"):
            truth, inferred = _identity(prop["match"], lab.get("team"), lab["number"], gallery[prop["match"]])
        candidates = {parse(i) for _, i in prop["top3"]} | {parse(i) for i in prop["vetoed"]}
        named = parse(prop["proposal"]) if prop["proposal"] else None
        best = parse(prop["best"]) if prop["best"] else None
        if named is None:
            outcome = "abstained"
        elif lab.get("flag") == "not_a_player":
            outcome = "wrong"
        elif truth is None:
            outcome = "unscored"
        else:
            outcome = "correct" if named == truth else "wrong"
        items.append({"review_id": rid, "match": prop["match"], "duration_s": prop["duration_s"],
                      "group": "accepted" if rid in membership["accepted"] else "random",
                      "label": lab, "truth": truth, "team_inferred": inferred, "proposal": named,
                      "best": best, "decision": prop["decision"], "score": prop["score"],
                      "margin": prop["margin"], "outcome": outcome,
                      "truth_in_candidates": truth in candidates if truth else None,
                      "truth_vetoed": next((v for i, v in prop["vetoed"].items() if truth and parse(i) == truth), None),
                      "best_is_truth": (best == truth) if truth and best else None, "top3": prop["top3"]})

    def tally(rows: list) -> dict:
        out = {}
        for k in ("correct", "wrong", "unscored", "abstained"):
            sel = [r for r in rows if r["outcome"] == k]
            out[k] = {"segments": len(sel), "seconds": round(sum(r["duration_s"] for r in sel), 1)}
        return out

    acc = [r for r in items if r["group"] == "accepted"]
    rnd = [r for r in items if r["group"] == "random"]
    c, w = (sum(r["outcome"] == k for r in acc) for k in ("correct", "wrong"))
    total_s = sum(s["duration_s"] for s in pop)
    rest_s = sum(s["duration_s"] for s in pop if s["review_id"] not in set(membership["accepted"]))
    named_rnd = [r for r in rnd if r["truth"]]
    report = {
        "labels_sha256": frozen, "proposals_sha256": sha256(OUT / "proposals_SEALED.json"),
        "population": {"segments": len(pop), "seconds": round(total_s, 1)},
        "accepted": {"by_outcome": tally(acc), "precision_scored": c / max(1, c + w), "n_scored": c + w,
                     "per_match": {m: tally([r for r in acc if r["match"] == m]) for m in SPECS}},
        "coverage": {"correct_segments_of_population": c / len(pop),
                     "correct_seconds_of_population": sum(r["duration_s"] for r in acc if r["outcome"] == "correct") / total_s},
        "random_sample": {
            "n": len(rnd), "drawn_from_seconds": round(rest_s, 1),
            "human_named": len(named_rnd),
            "flags": dict(Counter(r["label"].get("flag") or ("named" if r["truth"] else "unnamed") for r in rnd)),
            "truth_had_gallery": sum(bool(r["truth_in_candidates"]) for r in named_rnd),
            "truth_vetoed": sum(bool(r["truth_vetoed"]) for r in named_rnd),
            "best_guess_was_truth": sum(bool(r["best_is_truth"]) for r in named_rnd),
            "decisions": dict(Counter(r["decision"] for r in rnd)),
        },
        "items": items,
    }
    report["labels"] = version
    (OUT / ("score.json" if version == "v1" else f"score_{version}.json")).write_text(
        json.dumps(report, indent=1, default=float))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bank", default="test_v1", choices=tuple(BANKS))
    parser.add_argument("command", choices=("population", "review", "propose", "queue", "serve", "score", "recheck"))
    parser.add_argument("--number", help="recheck: the label value to look at again")
    parser.add_argument("--labels", default="v1", choices=("v1", "v2", "v3"),
                        help="score: v2 = v1 + blind re-check; v3 = v2 + adjudication (not blind)")
    parser.add_argument("--host", help="serve: bind address (default: Tailscale IPv4)")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    configure(args.bank)
    if args.command == "serve":
        return serve(args.host, args.port)
    if args.command == "population":
        print(json.dumps(population()["summary"], indent=1))
    elif args.command == "review":
        review()
    elif args.command == "propose":
        print(json.dumps(propose(), indent=1))
    elif args.command == "recheck":
        print(json.dumps(recheck(args.number), indent=1))
    elif args.command == "score":
        r = score(args.labels)
        print(json.dumps({k: v for k, v in r.items() if k != "items"}, indent=1, default=float))
    else:
        print(json.dumps(queue(), indent=1))


if __name__ == "__main__":
    main()
