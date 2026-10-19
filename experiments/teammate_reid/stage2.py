"""Stage 2: incremental coverage on baseline-unresolved segments (the three split-v1 test matches).

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
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from experiments.teammate_reid.mining import load_video, mine_video, sample_crops
from experiments.teammate_reid.run import HELDOUT, VIDEOS

OUT = Path("/home/valentinweyer/projects/handball-computer-vision/runs/smart-reid-shadow/stage2")
SPECS = {"Kiel-Lemgo": VIDEOS[0], "Eisenach-Hamburg": HELDOUT[0], "Gummersbach-Loewen": HELDOUT[1]}
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
                    if box is not None:
                        x1, y1, x2, y2 = (int(v / 2) for v in box)
                        cv2.rectangle(small, (x1, y1), (x2, y2), BOX, 2)
                    cv2.putText(small, f"{rid}  frame {f}", (8, 24), 0, 0.7, (255, 255, 255), 2)
                    if rid not in writers:
                        writers[rid] = cv2.VideoWriter(str(rdir / f"{rid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                       12.5, (960, 540))
                    writers[rid].write(small)
        for w in writers.values():
            w.release()
        for rid, (s, keys, _) in plans.items():
            _sheet(rdir / f"{rid}.jpg", rid, s, keys, frames.get(rid, {}))


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

LOCK = Path(__file__).parent / "locked/split_v1.json"
REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
GALLERY_RESULTS = {"Kiel-Lemgo": REPO / "runs/smart-reid-prototype-2026-10-18/results",
                   "Eisenach-Hamburg": REPO / "runs/smart-reid-shadow/stage1",
                   "Gummersbach-Loewen": REPO / "runs/smart-reid-shadow/stage1"}
M_CROPS, TOP, MIN_SEP = 8, 3, 50


def _supported_numbers(reads: dict) -> set:
    """Numbers this segment's own reads support (>= 2 reads); a trailing single digit folds in."""
    full = {v for v, c in reads.items() if c >= 2 and len(v) > 1}
    return full | {v for v, c in reads.items() if c >= 2 and len(v) == 1 and not any(f.endswith(v) for f in full)}


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
    """
    from experiments.teammate_reid.benchmark import load_reviewed
    from experiments.teammate_reid.encoders import SportsReIDViT
    from experiments.teammate_reid.masking import crop_with_digits
    from experiments.teammate_reid.run import FrameReader
    lock = json.loads(LOCK.read_text())
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
            records[s["segment"]] = sorted(recs, key=lambda r: r["frame"])[:M_CROPS]
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
            q = encoder.encode([pixels[(s["segment"], i)] for i in range(len(rs))])
            rows = by_track[s["track_id"]].loc[s["start"]:s["end"]]
            teams = rows["team"].value_counts()
            settled = (rows["team_state"] == "stable").mean() >= 0.9 and len(teams) == 1
            seg_frames = set(range(s["start"], s["end"] + 1)) & set(rows.index)
            supported = _supported_numbers(s.get("own_reads", {}))
            scores, vetoes = {}, {}
            for g in gfrags:
                if settled and g["team"] != int(teams.index[0]):
                    continue
                if g["end"] + MIN_SEP > s["start"]:
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
            ranked = []
            for identity, ids in scores.items():
                if identity in vetoes or not ids:
                    continue
                sims = q @ gfeat[ids].T
                k = min(TOP, sims.shape[1])
                ranked.append((float(np.sort(sims, axis=1)[:, -k:].mean(1).mean()), identity))
            ranked.sort(reverse=True)
            best = ranked[0] if ranked else (float("-inf"), None)
            margin = best[0] - ranked[1][0] if len(ranked) > 1 else float("-inf")
            accepted = best[1] is not None and best[0] >= tau and margin >= m_gate
            reason = ("accepted" if accepted else "no_candidate" if not ranked else
                      "single_candidate" if len(ranked) == 1 else "below_gate")
            out_rows.append({"review_id": s["review_id"], "segment": s["segment"], "match": match,
                             "duration_s": s["duration_s"], "query_crops": len(rs), "team_settled": bool(settled),
                             "proposal": best[1] if accepted else None, "best": best[1], "score": best[0],
                             "margin": margin, "decision": reason, "candidates": len(ranked),
                             "vetoed": vetoes, "top3": ranked[:3]})
    sealed = {"lock_sha256": sha256(LOCK), "gate": [tau, m_gate],
              "protocol": "gallery fragments end >= 2 s before the segment (as validated)", "rows": out_rows}
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
    (OUT / "review" / "label.html").write_text(_page(q, refs))
    return {"queue": len(q), "accepted": len(accepted), "random": len(sample),
            "page": str(OUT / "review" / "label.html")}


def _page(q: list, refs: dict) -> str:
    data = json.dumps(q)
    team_refs = json.dumps({m: {str(t): n for t, n in r.items()} for m, r in refs.items()})
    return """<!doctype html><html><head><meta charset="utf-8"><title>Stage 2 labelling</title>
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
<span class="mut" id="done"></span><button onclick="exportJson()">Download labels JSON</button></div>
<div class="card">
<p class="mut">Who is the player in the <b style="color:var(--acc)">yellow box</b> for the whole segment? Use any frame
(also before/after) where the number is readable. If the box switches person, choose <i>mixed</i>.
If you cannot tell, choose <i>unclear</i>. Never guess.</p>
<div class="bar"><span>Team:</span><div class="teams" id="teams"></div>
<span>Number:</span><input type="text" id="num" placeholder="e.g. 22" oninput="save()">
<button id="b_mixed" onclick="flag('mixed')">mixed</button><button id="b_unclear" onclick="flag('unclear')">unclear</button>
<button id="b_np" onclick="flag('not_a_player')">not a player</button></div>
<div class="bar"><span class="mut">Evidence frames / note (optional):</span></div>
<textarea id="note" rows="2" oninput="save()"></textarea>
<img class="sheet" id="sheet"><video id="vid" controls muted loop></video></div></div>
<script>
const Q=__DATA__, REFS=__REFS__, KEY="stage2_labels_v1";
let L={}; try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
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
try{localStorage.setItem(KEY,JSON.stringify(L));localStorage.setItem(KEY+"_pos",i)}catch(e){}render()}
function flag(f){const l=cur();l.flag=l.flag===f?null:f;save()}
function go(d){i=Math.max(0,Math.min(Q.length-1,i+d));save()}
function exportJson(){const out={labeller:"human",exported_at:new Date().toISOString(),labels:L};
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
a.download="annotations_human_v1.json";a.click()}
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA")return;
if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1)});render();
</script></body></html>""".replace("__DATA__", data).replace("__REFS__", team_refs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("population", "review", "propose", "queue"))
    args = parser.parse_args()
    if args.command == "population":
        print(json.dumps(population()["summary"], indent=1))
    elif args.command == "review":
        review()
    elif args.command == "propose":
        print(json.dumps(propose(), indent=1))
    else:
        print(json.dumps(queue(), indent=1))


if __name__ == "__main__":
    main()
