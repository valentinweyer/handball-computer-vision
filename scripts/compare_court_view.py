"""Re-fit the court on cached keypoints and camera motion, old rule against new.

Reads `data/interim/<clip>/keypoints.json`, `motion.json` and `meta.json` (written
by `process_video`), so no model or video pass is needed. For each clip it
re-runs `fit_court` with carried confirmation off (the old rule, checked against
the bundle's own `court.parquet` when one exists) and on, and reports how often
the court is fitted, carried or declared out of view. Writes a JSON summary with
per-frame sources so stretches can be inspected.

    python -m scripts.compare_court_view Wuerzburg_Regensburg_66m00_10min Kiel_Lemgo_66m00_10min
    python -m scripts.compare_court_view --all
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from handball_cv.court.sequence import fit_court

REPO = Path(__file__).resolve().parents[1]
# worktrees share the main checkout's artifacts
DATA = Path("/home/valentinweyer/projects/handball-computer-vision")


def refit(clip: str, **options) -> list[str]:
    interim = DATA / "data/interim" / clip
    keypoints = json.loads((interim / "keypoints.json").read_text())["keypoints"]
    motion = json.loads((interim / "motion.json").read_text())
    meta = json.loads((interim / "meta.json").read_text())
    return [fit.source if fit.court_in_view else "out_of_view"
            for fit in fit_court(keypoints, motion, (meta["width"], meta["height"]), **options)]


def longest(sources: list[str], kind: str) -> float:
    best = run = 0
    for s in sources:
        run = run + 1 if s == kind else 0
        best = max(best, run)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--all", action="store_true", help="every clip with cached keypoints and motion")
    parser.add_argument("--out", type=Path, default=DATA / "runs/court_carried_view/compare.json")
    args = parser.parse_args()
    clips = args.clips or sorted(p.name for p in (DATA / "data/interim").iterdir()
                                 if (p / "keypoints.json").exists() and (p / "motion.json").exists()
                                 and (p / "meta.json").exists())
    report = {}
    print(f"{'clip':44s} {'rule':4s} {'fitted':>7s} {'carried':>7s} {'no fit':>7s} {'no court':>8s} "
          f"{'longest no-court s':>18s} {'matches bundle':>14s}")
    for clip in clips:
        fps = json.loads((DATA / "data/interim" / clip / "meta.json").read_text()).get("fps", 25.0)
        old, new = refit(clip, min_confirmed=None), refit(clip)
        bundle = DATA / "runs/bundles" / clip / "court.parquet"
        same = None
        if bundle.exists():
            b = pd.read_parquet(bundle, columns=["source", "court_in_view"])
            ref = [s if v else "out_of_view" for s, v in zip(b.source, b.court_in_view)]
            same = ref == old
        report[clip] = {}
        for rule, sources in (("old", old), ("new", new)):
            n = len(sources)
            c = Counter(sources)
            row = {k: c.get(k, 0) / n for k in ("keypoints", "propagated", "none", "out_of_view")}
            row["longest_out_of_view_s"] = longest(sources, "out_of_view") / fps
            report[clip][rule] = {**row, "sources": sources}
            print(f"{clip[:44]:44s} {rule:4s} {row['keypoints']:7.3f} {row['propagated']:7.3f} {row['none']:7.3f} "
                  f"{row['out_of_view']:8.3f} {row['longest_out_of_view_s']:18.1f} "
                  f"{'' if rule == 'new' or same is None else str(same):>14s}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
