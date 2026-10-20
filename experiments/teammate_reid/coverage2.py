"""Coverage round 2: an evidence store that keeps raw embeddings, so gallery and query
selection, crop-level support and digit masking can be tested offline.

`coverage.py` cached per-crop top-3 scores, which fixes the gallery and the score.
This store keeps, per bank:

- for every population segment: every sampled crop (as Stage 2 samples them) encoded
  raw and under the `strict` digit mask (number boxes plus a fixed torso-and-shorts
  band hidden, `masking.py`), whether a number box was detected on the crop, the
  share of rows with an on-court position, and the baseline's shown-number share;
- per match: every reviewed, number-vouched gallery fragment with its crop ids, and
  those crops encoded raw (copied from the results dir, unchanged) and `strict`;
- per segment and candidate: the co-presence evidence as an IoU series summary
  between the segment's box and every other track showing the candidate's number,
  so a duplicate box of the same person can be told from a different player;
- a negative set for the duplicate test: pairs of time-overlapping vouched fragments
  of different players in the same match, with the same IoU summary.

Inputs are read only; everything is written under `runs/smart-reid-shadow/coverage_dev2/`.

    python -m experiments.teammate_reid.coverage2 extract [--banks test_v1 final higher]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from experiments.teammate_reid import stage2

OUT = stage2.RUNS / "coverage_dev2"
COURT = (-100.0, -100.0, 4100.0, 2100.0)   # cm, the 40 x 20 m court with a 1 m margin


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def iou_summary(values: list) -> dict:
    """Co-present frames and how they overlap. A duplicate track of the same person
    overlaps almost everywhere; two players overlap only while they cross."""
    if not values:
        return {"n": 0}
    v = np.asarray(values, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()),
            "frac50": float((v >= 0.5).mean()), "frac70": float((v >= 0.7).mean())}


def _pair_ious(rows_a: pd.DataFrame, rows_b: pd.DataFrame) -> list:
    shared = rows_a.index.intersection(rows_b.index)
    return [_iou(tuple(rows_a.loc[f, ["x1", "y1", "x2", "y2"]]), tuple(rows_b.loc[f, ["x1", "y1", "x2", "y2"]]))
            for f in shared]


def _encode_strict(encoder, imgs_and_boxes: list, seeds: list) -> np.ndarray:
    from experiments.teammate_reid.masking import apply_mask
    masked = [apply_mask(img, boxes, "strict", seed_key=seed)[0] for (img, boxes), seed in zip(imgs_and_boxes, seeds)]
    return encoder.encode(masked)


def extract(bank: str, encoder) -> dict:
    from experiments.teammate_reid.benchmark import load_reviewed
    from experiments.teammate_reid.encoders import local_digit_boxes
    from experiments.teammate_reid.masking import crop_with_digits
    from experiments.teammate_reid.mining import load_video, mine_video, sample_crops
    from experiments.teammate_reid.run import FrameReader

    stage2.configure(bank)
    pop = json.loads((stage2.OUT / "population.json").read_text())["population"]
    arrays, segments, galleries, negatives = {}, [], {}, []
    for match, spec in stage2.SPECS.items():
        loaded = load_video(spec)
        frags = {f.key: f for f in mine_video(loaded)[0]}
        tracks, numbers = loaded["tracks"], loaded["numbers"]
        court = pd.read_parquet(spec.bundle / "tracks.parquet", columns=["frame", "track_id", "court_x", "court_y"])
        tracks = tracks.merge(court, on=["frame", "track_id"], how="left")
        by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}
        tbf = {f: g for f, g in tracks.groupby("frame")}
        nbf = {f: list(g.itertuples()) for f, g in numbers.groupby("frame")}
        det = pd.read_parquet(spec.bundle / "detections.parquet")
        det_frames = set(det["frame"].unique())
        refs = {f: list(g.itertuples()) for f, g in det[det["class_name"] == "referee"].groupby("frame")}
        src = loaded["manifest"]["source"]

        # gallery: reviewed number-vouched fragments of this video; raw features copied, strict encoded here
        gres = stage2.GALLERY_RESULTS[match]
        gcrops, gfrags, _ = load_reviewed(gres)
        gfrags = [f for f in gfrags if f["video"] == spec.name]
        keys = {f["key"] for f in gfrags}
        gcrops = sorted((c for c in gcrops if c["fragment"] in keys), key=lambda c: c["crop_id"])
        raw_all = np.load(gres / "features_sportsreid_vitb16.npz")["raw"]
        ids = [c["crop_id"] for c in gcrops]
        imgs = []
        for c in gcrops:
            img = cv2.cvtColor(cv2.imread(str(gres / "crops" / f"{c['crop_id']}.jpg")), cv2.COLOR_BGR2RGB)
            imgs.append((img, local_digit_boxes(c, img.shape)))
        strict = np.concatenate([_encode_strict(encoder, imgs[i:i + 128], [str(c) for c in ids[i:i + 128]])
                                 for i in range(0, len(imgs), 128)]) if imgs else np.zeros((0, 1), np.float32)
        arrays[f"gal_ids|{match}"] = np.asarray(ids)
        arrays[f"gal_raw|{match}"] = raw_all[ids].astype(np.float16)
        arrays[f"gal_strict|{match}"] = strict.astype(np.float16)
        arrays[f"gal_digit|{match}"] = np.asarray([bool(c["digit_boxes"]) for c in gcrops])
        by_frag: dict = {}
        for c in gcrops:
            by_frag.setdefault(c["fragment"], []).append(c["crop_id"])
        galleries[match] = [{"key": f["key"], "identity": f["identity"], "team": f["team"],
                             "number": f["identity"].split("#")[-1], "track_id": f["track_id"],
                             "start": f["start"], "end": f["end"], "crop_ids": by_frag.get(f["key"], [])}
                            for f in gfrags]

        # negatives for the duplicate test: overlapping vouched fragments of different players
        for i, a in enumerate(gfrags):
            for b in gfrags[i + 1:]:
                if a["identity"] == b["identity"] or a["track_id"] == b["track_id"]:
                    continue
                if a["start"] > b["end"] or b["start"] > a["end"]:
                    continue
                ra = by_track[a["track_id"]].loc[a["start"]:a["end"]]
                rb = by_track[b["track_id"]].loc[b["start"]:b["end"]]
                s = iou_summary(_pair_ious(ra, rb))
                if s["n"] and s["frac50"] > 0:
                    negatives.append({"match": match, "a": a["key"], "b": b["key"], **s})

        shown = tracks.dropna(subset=["number"])
        shown_by_id = {(int(t), str(n)): g for (t, n), g in shown.groupby(["team", "number"])}
        segs = [s for s in pop if s["match"] == match]
        records = {}
        for s in segs:
            recs, _ = sample_crops(frags[s["key"]], tbf, nbf, det_frames, src["width"], src["height"],
                                   every=10, cap=60, referees_by_frame=refs)
            records[s["review_id"]] = sorted(recs, key=lambda r: r["frame"])
        reader = FrameReader(spec.bundle / loaded["manifest"]["files"]["video"])
        pix = {}
        for f in sorted({r["frame"] for rs in records.values() for r in rs}):
            frame = reader.read(f)
            for rid, rs in records.items():
                for i, r in enumerate(rs):
                    if r["frame"] == f:
                        pix[(rid, i)] = crop_with_digits(frame, r["box"], r["digit_boxes"])
        for s in segs:
            rid, rs = s["review_id"], records[s["review_id"]]
            crops = [pix[(rid, i)] for i in range(len(rs))]
            arrays[f"q_raw|{rid}"] = encoder.encode([c for c, _ in crops]).astype(np.float16)
            arrays[f"q_strict|{rid}"] = _encode_strict(encoder, crops, [f"{rid}:{i}" for i in range(len(rs))]).astype(np.float16)
            arrays[f"q_digit|{rid}"] = np.asarray([bool(r["digit_boxes"]) for r in rs])
            rows = by_track[s["track_id"]].loc[s["start"]:s["end"]]
            teams = rows["team"].value_counts()
            settled = bool((rows["team_state"] == "stable").mean() >= 0.9 and len(teams) == 1)
            on = rows.dropna(subset=["court_x", "court_y"])
            inside = ((on["court_x"] >= COURT[0]) & (on["court_x"] <= COURT[2])
                      & (on["court_y"] >= COURT[1]) & (on["court_y"] <= COURT[3]))
            copres = {}
            for g in galleries[match]:
                ident = g["identity"]
                if ident in copres:
                    continue
                other = shown_by_id.get((int(g["team"]), g["number"]))
                ious = []
                if other is not None:
                    o = other[(other["frame"] >= s["start"]) & (other["frame"] <= s["end"])
                              & (other["track_id"] != s["track_id"])]
                    for t, og in o.groupby("track_id"):
                        ious += _pair_ious(rows, og.set_index("frame"))
                overlap = sum(max(0, min(f["end"], s["end"]) - max(f["start"], s["start"]) + 1)
                              for f in galleries[match] if f["identity"] == ident and f["track_id"] != s["track_id"])
                copres[ident] = {**iou_summary(ious), "frag_overlap_frames": int(overlap)}
            segments.append({"bank": bank, "review_id": rid, "match": match, "track_id": s["track_id"],
                             "start": s["start"], "end": s["end"], "duration_s": s["duration_s"],
                             "baseline_shown": s["baseline_shown"], "team": int(teams.index[0]), "settled": settled,
                             "own_reads": s.get("own_reads", {}), "crop_frames": [r["frame"] for r in rs],
                             "court_rows": int(len(on)), "on_court_frac": float(inside.mean()) if len(on) else None,
                             "copresence": copres})
        print(f"{bank} {match}: {len(segs)} segments, {len(gcrops)} gallery crops, {len(negatives)} negatives so far",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{bank}_store.json").write_text(json.dumps({"bank": bank, "segments": segments, "galleries": galleries,
                                                        "negatives": negatives}))
    np.savez_compressed(OUT / f"{bank}_store.npz", **arrays)
    return {"segments": len(segments), "negatives": len(negatives)}


def main() -> None:
    from experiments.teammate_reid.encoders import SportsReIDViT
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("extract",))
    parser.add_argument("--banks", nargs="+", default=["test_v1", "final"])
    args = parser.parse_args()
    encoder = SportsReIDViT()
    for bank in args.banks:
        print(json.dumps({bank: extract(bank, encoder)}), flush=True)


if __name__ == "__main__":
    main()
