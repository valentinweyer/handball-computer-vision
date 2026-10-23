"""Shared paths and scoring for the SAM2 throughput experiments.

Two questions, both on the production SAM2 tracker
(`scripts.run_sam2_reprompt_tracker`, JPEG frames, `_v1` detections, the
cached team models -- the configuration §8 scores):

  1. Concurrency: do two tracking processes side by side finish more frames
     per hour than one, with byte-identical output?
  2. Half frame rate: does propagating every second frame (checkpoints kept
     at the same real-time interval, skipped frames interpolated) keep
     identity quality?

All outputs go under `runs/sam2_throughput/`, never over the stored
`runs/sam2_reprompt` baselines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.box_tracker_oracle.common import Clip
from scripts.evaluate_tracker_identity import Sam2ReplayTracker, run_tracker, score

MAIN = Path("/home/valentinweyer/projects/handball-computer-vision")
OUT = MAIN / "runs/sam2_throughput"
CHECKPOINT = MAIN / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"

# clip -> (frame-cache dir name, detections, team model)
CASES = {
    "felix": ("FelixClaar", "outputs/team_comparison/.FelixClaar_detections_v1.npz",
              "outputs/team_comparison/.FelixClaar_team.pkl"),
    "han": ("Han-Ber4_cached", "outputs/team_dataset/.Han-Ber4_detections_v1.npz",
            "outputs/team_correction_mcbyte/.Han-Ber4_team.pkl"),
    "bhc": ("BHC-FAG_window_cached", "outputs/team_dataset/.BHC-FAG_window_detections_v1.npz",
            "outputs/team_confidence_v2/.BHC-FAG_team.pkl"),
}


def score_dump(clip_name: str, dump: Path) -> dict:
    """Score a full-rate tracker dump with the unchanged §8 scorer."""
    clip = Clip(clip_name)
    matches = run_tracker(lambda: Sam2ReplayTracker(dump), clip.reference, clip.video,
                          clip.detection_cache, "cpu")
    m = score(matches, clip.keep)
    players = sum(sum(i in clip.keep for i in t) for t in clip.reference.values())
    m["recall_pct"] = 100 * m["matched_detections"] / players
    m["correct_pct"] = 100 * (m["matched_detections"] - m["wrong_identity_frames"]) / players
    return m


def same_output(a: Path, b: Path) -> bool:
    with np.load(a) as x, np.load(b) as y:
        return all(np.array_equal(x[k], y[k]) for k in ("frame_index", "tracker_id", "boxes"))
