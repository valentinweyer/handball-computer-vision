"""Coverage development on the two spent Stage 2 banks: cache all evidence, then try rules offline.

Both banks (`stage2.BANKS`: test_v1 = Kiel/Eisenach/Gummersbach, final = the four
lower-league windows) were released as development data on 2026-10-20 after their
tests were scored. Their human labels (blind v1, blind re-check, and the
post-unsealing adjudication) are the reference here.

`extract` runs once per bank and keeps everything a rule could use, so variants
never need the GPU again:

- every population segment's crops (up to 60, as Stage 2 samples them) encoded
  with the lock's encoder (SportsReID ViT-B/16, raw crops);
- every same-team (or both-team, if the segment's team is unsettled) gallery
  identity under the split-v2 protocol: vouched fragments ending >= 2 s before
  or starting >= 2 s after the segment, never the segment's own track;
- the evidence behind each veto: frames where the baseline shows the candidate's
  number on another track, how much that box overlaps the segment's box (a
  duplicate box on the same person overlaps; a real second player does not),
  and overlap with the candidate's vouched fragments;
- court positions at the segment's ends and at the candidate's nearest gallery
  fragment on each side, for a speed check.

`evaluate` then scores rule variants on the labelled segments.

    python -m experiments.teammate_reid.coverage extract
    python -m experiments.teammate_reid.coverage evaluate
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.teammate_reid import stage2

OUT = stage2.RUNS / "coverage_dev"
BANKS = ("test_v1", "final")
FPS = 25.0


def _box_iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _end_position(rows: pd.DataFrame, first: bool):
    """Median court position over the first or last 5 rows with a court position."""
    r = rows.dropna(subset=["court_x", "court_y"])
    if r.empty:
        return None
    r = r.head(5) if first else r.tail(5)
    return [float(r["court_x"].median()), float(r["court_y"].median())]


def extract(bank: str) -> dict:
    from experiments.teammate_reid.benchmark import load_reviewed
    from experiments.teammate_reid.encoders import SportsReIDViT
    from experiments.teammate_reid.masking import crop_with_digits
    from experiments.teammate_reid.mining import load_video, mine_video, sample_crops
    from experiments.teammate_reid.run import FrameReader

    stage2.configure(bank)
    pop = json.loads((stage2.OUT / "population.json").read_text())["population"]
    encoder = SportsReIDViT()
    segments, embeddings = [], {}
    for match, spec in stage2.SPECS.items():
        loaded = load_video(spec)
        frags = {f.key: f for f in mine_video(loaded)[0]}
        tracks, numbers = loaded["tracks"], loaded["numbers"]
        court = pd.read_parquet(spec.bundle / "tracks.parquet", columns=["frame", "track_id", "court_x", "court_y"]) \
            if {"court_x", "court_y"} <= set(pd.read_parquet(spec.bundle / "tracks.parquet").columns) else None
        if court is not None:
            tracks = tracks.merge(court, on=["frame", "track_id"], how="left")
        else:
            tracks = tracks.assign(court_x=np.nan, court_y=np.nan)
        by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}
        tbf = {f: g for f, g in tracks.groupby("frame")}
        nbf = {f: list(g.itertuples()) for f, g in numbers.groupby("frame")}
        det = pd.read_parquet(spec.bundle / "detections.parquet")
        det_frames = set(det["frame"].unique())
        refs = {f: list(g.itertuples()) for f, g in det[det["class_name"] == "referee"].groupby("frame")}
        src = loaded["manifest"]["source"]
        gres = stage2.GALLERY_RESULTS[match]
        gcrops, gfrags, _ = load_reviewed(gres)
        gfrags = [f for f in gfrags if f["video"] == spec.name]
        gfeat = np.load(gres / "features_sportsreid_vitb16.npz")["raw"]
        gfeat = gfeat / np.clip(np.linalg.norm(gfeat, axis=1, keepdims=True), 1e-8, None)
        gcrops_by_frag: dict = {}
        for c in sorted(gcrops, key=lambda c: c["frame"]):
            gcrops_by_frag.setdefault(c["fragment"], []).append(c["crop_id"])
        shown = tracks.dropna(subset=["number"])
        shown_by_id = {(int(t), str(n)): g for (t, n), g in shown.groupby(["team", "number"])}
        segs = [s for s in pop if s["match"] == match]
        records = {}
        for s in segs:
            recs, _ = sample_crops(frags[s["key"]], tbf, nbf, det_frames, src["width"], src["height"],
                                   every=10, cap=60, referees_by_frame=refs)
            records[s["review_id"]] = sorted(recs, key=lambda r: r["frame"])
        wanted = sorted({r["frame"] for rs in records.values() for r in rs})
        reader = FrameReader(spec.bundle / loaded["manifest"]["files"]["video"])
        pixels = {}
        for f in wanted:
            frame = reader.read(f)
            for rid, rs in records.items():
                for i, r in enumerate(rs):
                    if r["frame"] == f:
                        pixels[(rid, i)] = crop_with_digits(frame, r["box"], [])[0]
        for s in segs:
            rid = s["review_id"]
            rs = records[rid]
            q = encoder.encode([pixels[(rid, i)] for i in range(len(rs))])
            rows = by_track[s["track_id"]].loc[s["start"]:s["end"]]
            teams = rows["team"].value_counts()
            settled = bool((rows["team_state"] == "stable").mean() >= 0.9 and len(teams) == 1)
            boxes = {int(f): (r.x1, r.y1, r.x2, r.y2) for f, r in rows.iterrows()}
            start_pos, end_pos = _end_position(rows, True), _end_position(rows, False)
            cands: dict = {}
            for g in gfrags:
                if settled and g["team"] != int(teams.index[0]):
                    continue
                if not stage2.gallery_allowed(g, s, True):
                    continue
                c = cands.setdefault(g["identity"], {"identity": g["identity"], "team": g["team"],
                                                     "number": g["identity"].split("#")[-1], "crops": [],
                                                     "before": None, "after": None})
                c["crops"] += gcrops_by_frag.get(g["key"], [])
                grows = by_track.get(g["track_id"])
                if g["end"] < s["start"] and (c["before"] is None or g["end"] > c["before"]["frame"]):
                    pos = _end_position(grows.loc[g["start"]:g["end"]], False) if grows is not None else None
                    c["before"] = {"frame": g["end"], "pos": pos}
                if g["start"] > s["end"] and (c["after"] is None or g["start"] < c["after"]["frame"]):
                    pos = _end_position(grows.loc[g["start"]:g["end"]], True) if grows is not None else None
                    c["after"] = {"frame": g["start"], "pos": pos}
            out_cands = []
            for identity, c in cands.items():
                team, number = int(c["team"]), c["number"]
                # baseline shows this number on another track during the segment
                other = shown_by_id.get((team, number))
                co_frames, dup_frames = 0, 0
                if other is not None:
                    o = other[(other["frame"] >= s["start"]) & (other["frame"] <= s["end"])
                              & (other["track_id"] != s["track_id"])]
                    for r in o.itertuples():
                        if int(r.frame) in boxes:
                            co_frames += 1
                            if _box_iou(boxes[int(r.frame)], (r.x1, r.y1, r.x2, r.y2)) >= 0.5:
                                dup_frames += 1
                frag_overlap = sum(max(0, min(f["end"], s["end"]) - max(f["start"], s["start"]) + 1)
                                   for f in gfrags if f["identity"] == identity and f["track_id"] != s["track_id"])
                sims = q @ gfeat[c["crops"]].T if c["crops"] else np.zeros((len(q), 0), np.float32)
                out_cands.append({"identity": identity, "team": team, "number": number,
                                  "gallery_crops": len(c["crops"]), "co_frames": co_frames, "dup_frames": dup_frames,
                                  "frag_overlap_frames": int(frag_overlap), "before": c["before"], "after": c["after"],
                                  # per query crop: top-3 mean against this candidate's gallery, and the
                                  # top-3 values themselves (so a gallery grown later can be merged exactly)
                                  "crop_scores": (np.sort(sims, axis=1)[:, -min(3, sims.shape[1]):].mean(1).round(5).tolist()
                                                  if sims.shape[1] else []),
                                  "crop_top3": (np.sort(sims, axis=1)[:, -min(3, sims.shape[1]):].round(5).tolist()
                                                if sims.shape[1] else [])})
            embeddings[rid] = q.astype(np.float16)
            segments.append({"bank": bank, "review_id": rid, "match": match, "track_id": s["track_id"],
                             "start": s["start"], "end": s["end"], "duration_s": s["duration_s"],
                             "team": int(teams.index[0]), "settled": settled, "own_reads": s.get("own_reads", {}),
                             "crop_frames": [r["frame"] for r in rs], "start_pos": start_pos, "end_pos": end_pos,
                             "candidates": out_cands})
        print(f"{bank} {match}: {len(segs)} segments", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{bank}_evidence.json").write_text(json.dumps({"bank": bank, "segments": segments}))
    np.savez_compressed(OUT / f"{bank}_query_embeddings.npz", **embeddings)
    return {"segments": len(segments), "candidates": sum(len(s["candidates"]) for s in segments)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("extract", "evaluate"))
    args = parser.parse_args()
    if args.command == "extract":
        for bank in BANKS:
            print(json.dumps(extract(bank)), flush=True)
    else:
        from experiments.teammate_reid.coverage_rules import main as evaluate
        evaluate()


if __name__ == "__main__":
    main()
