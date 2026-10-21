"""Gallery-only extra footage (development data): does more of the same match give more
players a gallery and more correct names, without new wrong ones?

The population is the 66:00-window Stage 2 population of Kiel-Lemgo and Eisenach-Hamburg
(round-2 store `test_v1`). Each segment is decided by the locked split_v2 rule and the
coverage_v1 candidate twice: with the 66:00 galleries only (exactly the store), and with
the number-vouched fragments of the adjacent windows (`heldout.EXTRA_DEV_WINDOWS`) added.

- Team ids are anonymous clusters per video. Each extra window's clusters are mapped to
  the 66:00 window's by the mean torso colour of their gallery crops (label free).
- An extra fragment joins the 66:00 identity with the same (team, number); a player only
  the extra windows know becomes a new candidate. Extra windows never overlap the 66:00
  window, so every extra fragment is "before" or "after" and never the segment's track.
- Co-presence evidence for the new candidates is computed from the 66:00 tracks exactly
  as `coverage2.extract` does for the 66:00 galleries.
- Extra fragments are mined automatically and not contact-sheet reviewed (production would
  not review either); fragments behind new names are listed for inspection.

    python -m experiments.teammate_reid.extra_galleries

Gallery size is a confound: top-3 scoring over more crops raises every candidate's score
against a gate set on one window, so each comparison is also run with a fixed crop budget
per identity (`cap`, crops picked evenly over match time).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from experiments.teammate_reid import stage2
from experiments.teammate_reid.coverage2 import _pair_ious, iou_summary
from experiments.teammate_reid.coverage2_rules import Bank, Rule, _norm, decide
from experiments.teammate_reid.heldout import EXTRA_DEV_WINDOWS

LOCK = Path(__file__).parent / "locked/coverage_v1.json"
EXTRA_RUNS = stage2.RUNS / "extra_dev"
OUT = stage2.RUNS / "extra_galleries"
MAIN_START = 66 * 60   # the 66:00 windows; extra windows start before or after it
FAR = 10 ** 7


def torso_bgr(crop_dir, crop_ids) -> np.ndarray:
    """Mean BGR of the upper body (rows 20-50%, middle half) over the given crops."""
    vals = []
    for c in crop_ids:
        img = cv2.imread(str(crop_dir / "crops" / f"{c}.jpg"))
        h, w = img.shape[:2]
        vals.append(img[h // 5:h // 2, w // 4:3 * w // 4].reshape(-1, 3).mean(0))
    return np.mean(vals, 0)


def team_map(main_colour: dict, extra_colour: dict) -> dict:
    """extra cluster -> main cluster, the assignment with the smaller total colour distance."""
    d = lambda a, b: float(np.linalg.norm(extra_colour[a] - main_colour[b]))
    keep = d(0, 0) + d(1, 1)
    swap = d(0, 1) + d(1, 0)
    return {0: 0, 1: 1} if keep <= swap else {0: 1, 1: 0}


class ExtendedBank(Bank):
    """The `test_v1` store restricted to the matches with extra windows; with `extra=True`
    the extra windows' vouched fragments are added to those matches' galleries."""

    def __init__(self, extra: bool, cap: int | None = None):
        super().__init__("test_v1")
        self.cap = cap
        windows: dict = {}
        for name, match, *_ in EXTRA_DEV_WINDOWS:
            windows.setdefault(match, []).append(name)
        self.segments = {r: s for r, s in self.segments.items() if s["match"] in windows}
        self.population = len(self.segments)
        self.seconds = sum(s["duration_s"] for s in self.segments.values())
        self.alignment, self.added = {}, {}
        if extra:
            for match, names in windows.items():
                self._add(match, names)
        self._cache = {}

    def _add(self, match: str, names: list) -> None:
        from experiments.teammate_reid.benchmark import load_reviewed
        stage2.configure("test_v1")
        main_dir = stage2.GALLERY_RESULTS[match]
        main = self.galleries[match]
        main_colour = {t: torso_bgr(main_dir, [c for f in main if f["team"] == t for c in f["crop_ids"]][:300])
                       for t in (0, 1)}
        by_id = {(int(f["team"]), f["number"]): f["identity"] for f in main}
        rows_raw, new_frags = [], []
        for name in names:
            res = EXTRA_RUNS / name
            crops, frags, _ = load_reviewed(res)
            frags = [f for f in frags if f["video"] == name.lower()]
            keys = {f["key"] for f in frags}
            crops = [c for c in crops if c["fragment"] in keys]
            colour = {t: torso_bgr(res, [c["crop_id"] for c in crops if c["team"] == t][:300]) for t in (0, 1)}
            mapping = team_map(main_colour, colour)
            self.alignment[name] = {"map": mapping, "main_bgr": {t: v.round(1).tolist() for t, v in main_colour.items()},
                                    "extra_bgr": {t: v.round(1).tolist() for t, v in colour.items()}}
            raw = np.load(res / "features_sportsreid_vitb16.npz")["raw"]
            before = int(name.split("_")[-2].split("m")[0]) * 60 < MAIN_START
            by_frag: dict = {}
            for c in crops:
                by_frag.setdefault(c["fragment"], []).append(c["crop_id"])
            for f in frags:
                team, number = mapping[int(f["team"])], f["identity"].split("#")[-1]
                ident = by_id.get((team, number), f"{match}|extra|team{team}|#{number}")
                ids = []
                for c in by_frag.get(f["key"], []):
                    ids.append(f"{name}:{c}")
                    rows_raw.append(raw[c])
                new_frags.append({"key": f"{name}:{f['key']}", "identity": ident, "team": team, "number": number,
                                  "track_id": f"{name}:{f['track_id']}", "start": -FAR if before else FAR,
                                  "end": -FAR if before else FAR, "crop_ids": ids, "window": name})
        n0 = len(self.g_index[match])
        flat = [c for f in new_frags for c in f["crop_ids"]]
        self.g_index[match].update({c: n0 + i for i, c in enumerate(flat)})
        add = _norm(np.asarray(rows_raw, np.float32))
        for e in ("raw", "strict"):   # strict is not used by v2 or coverage_v1 (asserted in main)
            self.g[match][e] = np.concatenate([self.g[match][e], add])
            self.g_keep[match][e] = np.concatenate([self.g_keep[match][e], np.ones(len(add), bool)])
        self.galleries[match] = main + new_frags
        known = set(by_id.values())
        self.added[match] = {"fragments": len(new_frags), "crops": len(flat),
                             "new_identities": sorted({f["identity"] for f in new_frags} - known),
                             "extended_identities": sorted({f["identity"] for f in new_frags} & known)}
        self._copresence(match, set(self.added[match]["new_identities"]))

    def candidates(self, seg: dict) -> dict:
        """The v2 candidates; with `cap`, each identity keeps at most `cap` gallery crops,
        picked evenly over its crops in match time (earlier window, 66:00, later window).
        Top-3 scoring over a bigger gallery inflates every candidate's score, so a fixed
        budget is the fair test of "more of the match" against a gate set on one window."""
        out = super().candidates(seg)
        if self.cap is None:
            return out
        rank = {}
        for f in self.galleries[seg["match"]]:
            r = 0 if "window" not in f else (-1 if f["start"] < 0 else 1)
            for c in f["crop_ids"]:
                rank[self.g_index[seg["match"]][c]] = (r, f["start"] if r == 0 else 0)
        capped = {}
        for ident, (team, number, idx) in out.items():
            idx = sorted(idx, key=lambda i: (rank[i], i))
            if len(idx) > self.cap:
                idx = [idx[i] for i in np.linspace(0, len(idx) - 1, self.cap).round().astype(int)]
            capped[ident] = (team, number, idx)
        return capped

    def _copresence(self, match: str, idents: set) -> None:
        """Co-presence of each new candidate with every 66:00 segment (coverage2.extract logic)."""
        from experiments.teammate_reid.mining import load_video
        tracks = load_video(stage2.SPECS[match])["tracks"]
        by_track = {t: g.set_index("frame") for t, g in tracks.groupby("track_id")}
        shown = tracks.dropna(subset=["number"])
        shown_by_id = {(int(t), str(n)): g for (t, n), g in shown.groupby(["team", "number"])}
        team_num = {f["identity"]: (int(f["team"]), f["number"]) for f in self.galleries[match]}
        for s in self.segments.values():
            if s["match"] != match:
                continue
            rows = by_track[s["track_id"]].loc[s["start"]:s["end"]]
            for ident in idents:
                other = shown_by_id.get(team_num[ident])
                ious = []
                if other is not None:
                    o = other[(other["frame"] >= s["start"]) & (other["frame"] <= s["end"])
                              & (other["track_id"] != s["track_id"])]
                    for _, og in o.groupby("track_id"):
                        ious += _pair_ious(rows, og.set_index("frame"))
                s["copresence"][ident] = {**iou_summary(ious), "frag_overlap_frames": 0}


VERIFY = OUT / "verify"     # blind assistant labels of names no earlier reference covered (frozen)
CAPS = (None, 150, 100)


def references_with_verify() -> tuple:
    """Earlier references (human > adjudicated > assistant, `coverage2_eval`), plus the frozen
    blind labels of this check for segments none of them covers. Controls are not added."""
    from experiments.teammate_reid.coverage2_eval import reference, references
    import hashlib
    refs = references()
    labels_path = VERIFY / "assistant_labels_v1.json"
    sha = labels_path.with_suffix(".sha256").read_text().split()[0]
    assert sha == hashlib.sha256(labels_path.read_bytes()).hexdigest(), "verification labels not frozen"
    labels = json.loads(labels_path.read_text())["labels"]
    for it in json.loads((VERIFY / "queue_SEALED.json").read_text())["items"]:
        if it["group"] != "target":
            continue
        a = labels[it["code"]]
        refs.setdefault(("test_v1", it["review_id"]), []).append(
            {"source": "assistant_extra_v1", "truth": (a["team"], a["number"]) if a.get("number") else None,
             "flag": a.get("flag")})

    def ref(key):
        r = reference(refs, key)
        if r is None:
            extra = [x for x in refs.get(key, []) if x["source"] == "assistant_extra_v1"]
            r = extra[0] if extra else None
        return r
    return refs, ref, sha


def main() -> None:
    from experiments.teammate_reid.coverage2_eval import outcome, reference
    from experiments.teammate_reid.coverage2_stress import synthetic_unknowns
    lock = json.loads(LOCK.read_text())
    rules = {"v2": Rule(**lock["baseline"]["rule"]), "coverage_v1": Rule(**lock["candidate"]["rule"])}
    assert all(r.emb == "raw" and r.tier2 is None and r.lone_strict is None for r in rules.values())
    refs, ref, sha = references_with_verify()
    report: dict = {"verify_labels_sha256": sha, "grid": {}}
    per: dict = {}
    errs = ("wrong", "unknown_named", "mixed")
    for cap in CAPS:
        banks = {"base": ExtendedBank(extra=False, cap=cap), "extra": ExtendedBank(extra=True, cap=cap)}
        if cap is None:
            report.update(alignment=banks["extra"].alignment, added=banks["extra"].added,
                          population=banks["base"].population, seconds=banks["base"].seconds)
        for bname, bank in banks.items():
            for rname, rule in rules.items():
                c, secs = Counter(), 0.0
                for rid, seg in bank.segments.items():
                    d = decide(bank, rid, rule)
                    o = outcome(bank, rid, d["name"], ref(("test_v1", rid)))
                    c[o] += 1
                    secs += seg["duration_s"] * (1 - seg["baseline_shown"]) if o == "correct" else 0.0
                    per.setdefault(rid, {})[f"{cap}|{bname}|{rname}"] = {"name": d["name"], "outcome": o}
                e = sum(c[x] for x in errs)
                report["grid"][f"cap={cap}|{bname}|{rname}"] = {
                    "counts": dict(c), "errors": e, "precision": c["correct"] / max(1, c["correct"] + e),
                    "newly_correct_seconds": round(secs, 1),
                    "synthetic_unknowns": dict(synthetic_unknowns(bank, rule, refs, reference))}
                print(f"cap={cap} {bname} {rname}: {report['grid'][f'cap={cap}|{bname}|{rname}']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({**report, "per_segment": per}, indent=1, default=str))


if __name__ == "__main__":
    main()
