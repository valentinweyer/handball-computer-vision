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

    python -m experiments.teammate_reid.extra_galleries            # gallery-cap grid
    python -m experiments.teammate_reid.extra_galleries variants   # size-robust variants, all mined matches
    python -m experiments.teammate_reid.extra_galleries variants-v1   # the same on Kiel and Eisenach only

Gallery size is a confound: top-3 scoring over more crops raises every candidate's score
against a gate set on one window, so each comparison is also run with a fixed crop budget
per identity (`cap`, crops picked evenly over match time).
"""
from __future__ import annotations

import json
import re
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
FAR = 10 ** 7
# Size-robust scoring (fixed before any run): the reference gallery size is the order of a
# 10-minute window's gallery per player (66:00 medians: Kiel 168, Eisenach 74 crops), and
# FRAC = 3 / REF_SIZE makes "frac" equal to top-3 at that size.
SCORINGS = ("top3", "frac", "partition")
REF_SIZE = 120
FRAC = 3 / REF_SIZE


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


def mined_windows(matches=None) -> dict:
    """match -> extra windows that have been mined (optionally only the given matches)."""
    out: dict = {}
    for name, match, *_ in EXTRA_DEV_WINDOWS:
        if (EXTRA_RUNS / name / "fragments.json").exists() and (matches is None or match in matches):
            out.setdefault(match, []).append(name)
    return out


def _start_min(name: str) -> int:
    """Window start in minutes from its name (`..._86m00_10min`); the Kiel bundle name has none: 66."""
    m = re.search(r"_(\d+)m00_", name)
    return int(m.group(1)) if m else 66


class ExtendedBank(Bank):
    """A development store (`test_v1` or `final`) restricted to the matches with mined extra
    windows; with `extra=True` their vouched fragments are added to those matches' galleries."""

    def __init__(self, extra: bool, cap: int | None = None, new_only: bool = False, scoring: str = "top3",
                 bank: str = "test_v1", matches=None):
        super().__init__(bank)
        assert scoring in SCORINGS
        self.cap, self.new_only, self.scoring = cap, new_only, scoring
        windows = {m: w for m, w in mined_windows(matches).items() if m in self.galleries}
        self.segments = {r: s for r, s in self.segments.items() if s["match"] in windows}
        self.population = len(self.segments)
        self.seconds = sum(s["duration_s"] for s in self.segments.values())
        self.alignment, self.added = {}, {}
        if extra:
            for match, names in windows.items():
                self._add(match, names)
        self._cache, self._rank = {}, {}

    def _add(self, match: str, names: list) -> None:
        from experiments.teammate_reid.benchmark import load_reviewed
        stage2.configure(self.name)
        main_dir = stage2.GALLERY_RESULTS[match]
        main_start = _start_min(stage2.SPECS[match].name)
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
            before = _start_min(name) < main_start
            by_frag: dict = {}
            for c in crops:
                by_frag.setdefault(c["fragment"], []).append(c["crop_id"])
            for f in frags:
                team, number = mapping[int(f["team"])], f["identity"].split("#")[-1]
                if self.new_only and (team, number) in by_id:
                    continue
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

    def _in_time_order(self, match: str, idx: list) -> list:
        """Gallery rows in match time: earlier window, the 66:00 window by fragment start, later window."""
        if match not in self._rank:
            rank = {}
            for f in self.galleries[match]:
                r = 0 if "window" not in f else (-1 if f["start"] < 0 else 1)
                for c in f["crop_ids"]:
                    rank[self.g_index[match][c]] = (r, f["start"] if r == 0 else 0)
            self._rank[match] = rank
        return sorted(idx, key=lambda i: (self._rank[match][i], i))

    def candidates(self, seg: dict) -> dict:
        """The v2 candidates, gallery rows in match time; with `cap`, each identity keeps at
        most `cap` crops picked evenly over match time. Top-3 scoring over a bigger gallery
        inflates every candidate's score, so a fixed budget is the fair comparison against a
        gate set on one window."""
        out = {}
        for ident, (team, number, idx) in super().candidates(seg).items():
            idx = self._in_time_order(seg["match"], idx)
            if self.cap is not None and len(idx) > self.cap:
                idx = [idx[i] for i in np.linspace(0, len(idx) - 1, self.cap).round().astype(int)]
            out[ident] = (team, number, idx)
        return out

    def crop_scores(self, rid: str, emb: str, clean_gallery: bool) -> dict:
        """identity -> per-query-crop score, by `scoring`:

        - "top3": mean of the 3 best gallery crops (the locked score);
        - "frac": mean of the best FRAC share of the gallery (at least 3), so a bigger
          gallery needs proportionally more close crops, not just three lucky ones;
        - "partition": the gallery in match time is split into m = ceil(n / REF_SIZE)
          interleaved subsets (every m-th crop), each about REF_SIZE crops spread over the
          whole footage; the top-3 score is averaged over the subsets."""
        if self.scoring == "top3":
            return super().crop_scores(rid, emb, clean_gallery)
        key = (rid, emb, clean_gallery)
        if key not in self._cache:
            assert not clean_gallery
            seg = self.segments[rid]
            q, G = self.q[emb][rid], self.g[seg["match"]][emb]
            out = {}
            for ident, (team, number, idx) in self.candidates(seg).items():
                if not idx:
                    continue
                sims = q @ G[idx].T
                if self.scoring == "frac":
                    k = max(3, int(round(FRAC * len(idx))))
                    out[ident] = (team, number, np.sort(sims, axis=1)[:, -min(k, sims.shape[1]):].mean(1))
                else:
                    m = -(-len(idx) // REF_SIZE)
                    parts = [sims[:, j::m] for j in range(m)]
                    out[ident] = (team, number, np.mean(
                        [np.sort(p, axis=1)[:, -min(3, p.shape[1]):].mean(1) for p in parts], 0))
            self._cache[key] = out
        return self._cache[key]

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


VERIFY_ROUNDS = (OUT / "verify", OUT / "verify2")   # blind assistant labels of names no earlier reference covered
CAPS = (None, 150, 100)
V1_MATCHES = ("Kiel-Lemgo", "Eisenach-Hamburg")      # the first two matches (cap grid)
BANKS = ("test_v1", "final")
ERRORS = ("wrong", "unknown_named", "mixed")


def references_with_verify() -> tuple:
    """Earlier references (human > adjudicated > assistant, `coverage2_eval`), plus every frozen
    verification round of this check for segments none of them covers. Controls are not added."""
    import hashlib
    from experiments.teammate_reid.coverage2_eval import reference, references
    refs, shas = references(), {}
    for vdir in VERIFY_ROUNDS:
        labels_path = vdir / "assistant_labels_v1.json"
        if not labels_path.exists():
            continue
        sha = labels_path.with_suffix(".sha256").read_text().split()[0]
        assert sha == hashlib.sha256(labels_path.read_bytes()).hexdigest(), f"{labels_path} not frozen"
        shas[vdir.name] = sha
        labels = json.loads(labels_path.read_text())["labels"]
        for it in json.loads((vdir / "queue_SEALED.json").read_text())["items"]:
            if it["group"] != "target":
                continue
            a = labels[it["code"]]
            refs.setdefault((it.get("bank", "test_v1"), it["review_id"]), []).append(
                {"source": f"assistant_extra_{vdir.name}",
                 "truth": (a["team"], a["number"]) if a.get("number") else None, "flag": a.get("flag")})

    def ref(key):
        r = reference(refs, key)
        if r is None:
            extra = [x for x in refs.get(key, []) if x["source"].startswith("assistant_extra")]
            r = extra[0] if extra else None
        return r
    return refs, ref, shas


def _rules() -> dict:
    lock = json.loads(LOCK.read_text())
    rules = {"v2": Rule(**lock["baseline"]["rule"]), "coverage_v1": Rule(**lock["candidate"]["rule"])}
    assert all(r.emb == "raw" and r.tier2 is None and r.lone_strict is None for r in rules.values())
    return rules


def _score(banks: list, rule, refs, ref, per: dict, tag: str) -> dict:
    """Outcomes over several stores: totals, per match, newly correct time, synthetic unknowns."""
    from experiments.teammate_reid.coverage2_eval import outcome, reference
    from experiments.teammate_reid.coverage2_stress import synthetic_unknowns
    c, secs, syn, by_match = Counter(), 0.0, Counter(), {}
    for bank in banks:
        for rid, seg in bank.segments.items():
            d = decide(bank, rid, rule)
            o = outcome(bank, rid, d["name"], ref((bank.name, rid)))
            c[o] += 1
            by_match.setdefault(seg["match"], Counter())[o] += 1
            secs += seg["duration_s"] * (1 - seg["baseline_shown"]) if o == "correct" else 0.0
            per.setdefault(f"{bank.name}|{rid}", {})[tag] = {"name": d["name"], "outcome": o}
        syn.update(synthetic_unknowns(bank, rule, refs, reference))
        for m, cm in _syn_by_match(bank, rule, refs, reference).items():
            by_match.setdefault(m, Counter()).update(cm)
    e = sum(c[x] for x in ERRORS)
    return {"counts": dict(c), "errors": e, "precision": c["correct"] / max(1, c["correct"] + e),
            "newly_correct_seconds": round(secs, 1), "synthetic_unknowns": dict(syn),
            "by_match": {m: dict(v) for m, v in sorted(by_match.items())}}


def _syn_by_match(bank, rule, refs, reference) -> dict:
    """Synthetic false matches per match (same procedure as `coverage2_stress.synthetic_unknowns`)."""
    from experiments.teammate_reid.coverage2_stress import synthetic_unknowns
    out = {}
    for match in sorted({s["match"] for s in bank.segments.values()}):
        keep = bank.segments
        bank.segments = {r: s for r, s in keep.items() if s["match"] == match}
        r = synthetic_unknowns(bank, rule, refs, reference)
        bank.segments = keep
        out[match] = Counter({"syn_trials": r["trials"], "syn_false_match": r.get("false_match", 0)})
    return out


def main() -> None:
    """Gallery-cap grid on the first two matches (Kiel, Eisenach), top-3 scoring."""
    rules = _rules()
    refs, ref, shas = references_with_verify()
    report: dict = {"verify_labels_sha256": shas, "grid": {}}
    per: dict = {}
    for cap in CAPS:
        banks = {"base": ExtendedBank(extra=False, cap=cap, matches=V1_MATCHES),
                 "extra": ExtendedBank(extra=True, cap=cap, matches=V1_MATCHES)}
        if cap is None:
            report.update(alignment=banks["extra"].alignment, added=banks["extra"].added,
                          population=banks["base"].population, seconds=banks["base"].seconds)
        for bname, bank in banks.items():
            for rname, rule in rules.items():
                row = _score([bank], rule, refs, ref, per, f"{cap}|{bname}|{rname}")
                report["grid"][f"cap={cap}|{bname}|{rname}"] = row
                print(f"cap={cap} {bname} {rname}: correct={row['counts'].get('correct', 0)} errors={row['errors']} "
                      f"synthFM={row['synthetic_unknowns'].get('false_match', 0)}/{row['synthetic_unknowns']['trials']}",
                      flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({**report, "per_segment": per}, indent=1, default=str))


VARIANTS = {   # name -> ExtendedBank options; each is scored on the window's galleries and with the extra windows
    "locked_top3": {},
    "new_players_only": {"new_only": True},
    "frac_top": {"scoring": "frac"},
    "partition": {"scoring": "partition"},
}


def variants(matches=None, out_name: str = "variants.json") -> dict:
    """Size-robust variants over every development match with mined extra windows (both stores):
    correct names, errors, newly correct time and synthetic false matches, base vs extra, per match."""
    rules = _rules()
    refs, ref, shas = references_with_verify()
    report: dict = {"verify_labels_sha256": shas, "ref_size": REF_SIZE, "frac": FRAC,
                    "matches": mined_windows(matches), "results": {}}
    per: dict = {}
    for vname, opts in VARIANTS.items():
        for extra in (False, True):
            if not extra and opts.get("new_only"):
                continue   # identical to locked_top3 base
            banks = [ExtendedBank(extra=extra, bank=b, matches=matches, **opts) for b in BANKS]
            banks = [b for b in banks if b.segments]
            if extra and vname == "locked_top3":
                report["added"] = {m: a for b in banks for m, a in b.added.items()}
                report["alignment"] = {w: a for b in banks for w, a in b.alignment.items()}
            for rname, rule in rules.items():
                tag = f"{vname}|{'extra' if extra else 'base'}|{rname}"
                row = _score(banks, rule, refs, ref, per, tag)
                report["results"][tag] = row
                c = row["counts"]
                print(f"{vname:17s} {'extra' if extra else 'base ':5s} {rname:11s} correct={c.get('correct', 0):3d} "
                      f"errors={row['errors']} unreadable={c.get('unreadable', 0):3d} unlabelled={c.get('unlabelled', 0)} "
                      f"newly={row['newly_correct_seconds']:5.0f}s synthFM={row['synthetic_unknowns'].get('false_match', 0)}/"
                      f"{row['synthetic_unknowns'].get('trials', 0)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / out_name).write_text(json.dumps({**report, "per_segment": per}, indent=1, default=str))
    return report


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["variants"]:
        variants()
    elif sys.argv[1:] == ["variants-v1"]:
        variants(V1_MATCHES, "variants_v1.json")
    else:
        main()
