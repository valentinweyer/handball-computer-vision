"""Held-out match windows for the smart re-ID transfer test.

Kiel-Lemgo chose the prototype's method and thresholds, so it cannot test them.
These windows come from matches whose teams appear nowhere in the prototype's
data (Kiel, Lemgo, BHC, FAG, Melsungen, Berlin). The detector-unseen `eval/`
matches are listed first.

The window rule was fixed before any of them was processed: 10 minutes from
66:00 of the broadcast file, at 25 fps. This is the same rule as
`Kiel_Lemgo_66m00_10min.mp4`. Frames at 66-76 min were checked only for live
play.

    python -m experiments.teammate_reid.heldout cut
    python -m scripts.process_video data/raw/<window>.mp4     # from the main checkout

`process_video` runs unmodified, with production defaults (SAM2), so the
bundles are also the production baseline that Stage 2 needs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from handball_cv.video import VideoFrames, ffmpeg_exe

REPO = Path("/home/valentinweyer/projects/handball-computer-vision")
SOURCES = Path("/home/valentinweyer/projects/rfdetr-handball-finetune/datasets/videos")
START_S = 66 * 60
MINUTES = 10
FPS = 25

# (window name, match = split unit, source, detector exposure)
WINDOWS = [
    ("Eisenach_Hamburg_66m00_10min", "Eisenach-Hamburg",
     "eval/2026-06-07_1284788_Eisenach_Hamburg.mp4", "detector-unseen"),
    ("VfL_GM_Loewen_66m00_10min", "Gummersbach-Loewen",
     "eval/2026-06-07_1284786_VfL_GM_Loewen.mp4", "detector-unseen"),
    # Fresh test for rules locked after Stage 1 (the two above are development data
    # now). RNL-SGF shares the Loewen (RNL) players with the match above.
    ("Ludwigshafen_Wetzlar_67m00_10min", "Ludwigshafen-Wetzlar",
     "train/Ludwigshafen-Wetzlar.mp4", "detector-train"),
    ("RNL_SGF_66m00_10min", "RNL-SGF", "train/RNL-SGF.mp4", "detector-train"),
]
# Development bank (2026-10-19): `train/` matches, only the detector trained on them.
# Their teams overlap earlier development data only, never the test or sealed sets.
# Tuning and leave-one-match-out checks run here in seconds once processed.
DEV_WINDOWS = [
    ("BAL_THW_66m00_10min", "Balingen-Kiel", "train/BAL-THW.mp4", "detector-train"),
    ("BER_BHC_66m00_10min", "Berlin-BHC", "train/BER-BHC.mp4", "detector-train"),
    ("TBV_SCM_66m00_10min", "Lemgo-Magdeburg", "train/TBV-SCM.mp4", "detector-train"),
    ("FAG_LEI_66m00_10min", "Goeppingen-Leipzig", "train/FAG-LEI.mp4", "detector-train"),
    ("DRE_MTM_66m00_10min", "Dresden-Melsungen", "train/DRE-MTM.mp4", "detector-train"),
    ("Vinnhorst_Potsdam_66m00_10min", "Vinnhorst-Potsdam",
     "train/sl_1435862_2026-08-21_1435862_tus_vinnhorst-1__vfl_potsdam_720p24_4001kbps.mp4", "detector-train"),
]
# Sealed test bank: every team is new to all development data. Processed now, mined
# only after a rule is locked; nothing in them may inform a method or threshold choice.
SEALED_WINDOWS = [
    ("Altenholz_Luebeck_66m00_10min", "Altenholz-Luebeck", "train/Altenholz-Lübeck.mp4", "detector-train"),
    ("Pforzheim_Bietigheim_66m00_10min", "Pforzheim-Bietigheim", "train/Pfortzheim-BBM.mp4", "detector-train"),
]
# 2-minute previews of the sealed matches (same 66:00 start), for an early look at
# the locked rule. Indicative only: the verdict is the full sealed windows.
PREVIEW_WINDOWS = [(n.replace("10min", "2min"), m, src, e) for n, m, src, e in SEALED_WINDOWS]
# Final test bank (2026-10-19): new lower-league matches the user downloaded for the
# one-shot test of `locked/split_v2.json` (locked before any of them was processed).
# Detector-unseen, every team new to all tuning and test data. TV floors with only
# handball lines.
FINAL_WINDOWS = [
    ("Saarlouis_Muenster_66m00_10min", "Saarlouis-Muenster",
     "eval/smart-id-eval/sl_1444078_2026-10-10_1444078_hg_saarlouis-tsg_muenster_720p24_3054kbps.mp4",
     "detector-unseen"),
    ("Bielefeld_Aurich_86m00_10min", "Bielefeld-Aurich",
     "eval/smart-id-eval/sl_1444091_2026-10-16_1444091_tsg_a-h_bielefeld-ohv_aurich_1080p50_7059kbps.mp4",
     "detector-unseen"),
    ("BalingenII_Erlangen_66m00_10min", "BalingenII-Erlangen",
     "eval/smart-id-eval/sl_1444115_2026-10-18_1444115_hbw_balingen-weilstetten_ii-tv_1861_erlangen-bruck_720p24_2996kbps.mp4",
     "detector-unseen"),
    ("Wuerzburg_Regensburg_66m00_10min", "Wuerzburg-Regensburg",
     "eval/smart-id-eval/sl_1478998_2026-10-17_1478998_woelfe_wuerzburg-sg_regensburg_adler_1080p30_3990kbps.mp4",
     "detector-unseen"),
]
# Higher-league fresh bank (2026-10-20): three HBF broadcasts the user added for the
# test of the coverage candidate. Untouched by any selection, labelling or inspection
# when registered (audit in docs/smart-reid-coverage2-2026-10-20.md); detector-unseen.
# Clubs Melsungen, Loewen (RNL), Magdeburg and Wetzlar also appear in development data
# (per-match galleries, frozen encoder: no player identity carries across matches).
HIGHER_WINDOWS = [
    ("Melsungen_Loewen_66m00_10min", "Melsungen-Loewen",
     "eval/smart-id-eval/2026-09-22_1447412_Melsungen_Loewen.mp4", "detector-unseen"),
    ("Magdeburg_Wetzlar_66m00_10min", "Magdeburg-Wetzlar",
     "eval/smart-id-eval/2026-09-23_1447416_Magdeburg_Wetzlar.mp4", "detector-unseen"),
    ("Stuttgart_Erlangen_66m00_10min", "Stuttgart-Erlangen",
     "eval/smart-id-eval/2026-09-24_1447415_Stuttgart_Erlangen.mp4", "detector-unseen"),
]
# Extra development footage (2026-10-21): windows of two development matches adjacent
# to, never overlapping, their 66:00 windows. Gallery-only: they add vouched fragments
# for the 66:00 population, never population segments of their own. Cut from the full
# broadcasts in data/raw/ by runs/smart-reid-shadow/run_extra_windows.sh (source column informational).
EXTRA_DEV_WINDOWS = [
    ("Kiel_Lemgo_56m00_10min", "Kiel-Lemgo", "eval/2026-06-07_1284785_Kiel_Lemgo.mp4", "detector-unseen"),
    ("Kiel_Lemgo_76m00_10min", "Kiel-Lemgo", "eval/2026-06-07_1284785_Kiel_Lemgo.mp4", "detector-unseen"),
    ("Eisenach_Hamburg_57m00_9min", "Eisenach-Hamburg", "eval/2026-06-07_1284788_Eisenach_Hamburg.mp4",
     "detector-unseen"),
    ("Eisenach_Hamburg_76m00_10min", "Eisenach-Hamburg", "eval/2026-06-07_1284788_Eisenach_Hamburg.mp4",
     "detector-unseen"),
]
# Second batch (2026-10-21): the other five development matches, two windows each, so the
# gallery-size variants can be compared leave-one-match-out. Starts checked for live play on
# 30-second timelines of the full broadcasts. Bielefeld: half-time ~73:30-84:00, signal lost at
# 108:00, so 62:00 (first half) and 96:00. All others: 56:00 and 76:00 around the 66:00 window.
EXTRA_DEV_WINDOWS_2 = [
    (f"{prefix}_{start}m00_10min", match, src, "detector-unseen")
    for prefix, match, src, starts in [
        ("VfL_GM_Loewen", "Gummersbach-Loewen", "eval/2026-06-07_1284786_VfL_GM_Loewen.mp4", (56, 76)),
        ("Saarlouis_Muenster", "Saarlouis-Muenster", FINAL_WINDOWS[0][2], (56, 76)),
        ("Bielefeld_Aurich", "Bielefeld-Aurich", FINAL_WINDOWS[1][2], (62, 96)),
        ("BalingenII_Erlangen", "BalingenII-Erlangen", FINAL_WINDOWS[2][2], (56, 76)),
        ("Wuerzburg_Regensburg", "Wuerzburg-Regensburg", FINAL_WINDOWS[3][2], (56, 76))]
    for start in starts]
EXTRA_DEV_WINDOWS += EXTRA_DEV_WINDOWS_2
# Former reserve: DEV_WINDOWS cut on 2026-10-19, never processed for smart re-ID until 2026-10-22,
# when the user released both as adapter TRAINING data (locked/finetune_v1.json, split A).
# They are no longer an independent test.
RESERVE_WINDOWS = [w for w in DEV_WINDOWS if w[0] in ("DRE_MTM_66m00_10min", "Vinnhorst_Potsdam_66m00_10min")]
# DHB-Pokal pilot (2026-10-24): does older 720p Pokal footage (all with TV floor) yield enough
# number-vouched identities to serve as adapter TRAINING data? Development only, never a test.
# Full broadcasts in data/raw/pokal (absolute sources). Starts checked on 2-minute timelines for
# live play before cutting; the 66:00 rule unless a START_OVERRIDE_S entry says otherwise.
POKAL_DIR = REPO / "data/raw/pokal"
POKAL_PILOT_WINDOWS = [
    (name, match, str(POKAL_DIR / src), "detector-unseen")
    for name, match, src in [
        ("Pokal16_Kiel_Fredenbeck_66m00_10min", "Pokal2016-Kiel-Fredenbeck",
         "2016-08-28_154952_thw_kiel-vfl_fredenbeck_ev-720p-3140kbps.mp4"),
        ("Pokal16_Wetzlar_Coburg_66m00_10min", "Pokal2016-Wetzlar-Coburg",
         "2016-08-28_155037_hsg_wetzlar-hsc_2000_coburg-720p-3000kbps.mp4"),
        ("Pokal16_Erlangen_Loewen_66m00_10min", "Pokal2016-Erlangen-Loewen",
         "2016-10-27_183756_hc_erlangen-rhein-neckar_loewen-720p-3092kbps.mp4"),
        ("Pokal19_Wetzlar_Kiel_66m00_10min", "Pokal2019-Wetzlar-Kiel",
         "2019-10-02_636411_hsg_wetzlar-thw_kiel-720p-3000kbps.mp4"),
        ("Pokal19_Loewen_Goeppingen_66m00_10min", "Pokal2019-Loewen-Goeppingen",
         "2019-10-02_636408_rhein-neckar_loewen-frisch_auf_goeppingen-720p-3000kbps.mp4"),
        ("Pokal22_Flensburg_Hamburg_66m00_10min", "Pokal2022-Flensburg-Hamburg",
         "sl_979487_2022-12-22_979487_sg_flensburg-handewitt-handball_sport_verein_hamburg_1080p30_3896kbps.mp4")]]
WINDOW_SETS = {"test": WINDOWS, "dev": DEV_WINDOWS, "sealed": SEALED_WINDOWS, "preview": PREVIEW_WINDOWS,
               "final": FINAL_WINDOWS, "higher": HIGHER_WINDOWS,
               "pokal_pilot": POKAL_PILOT_WINDOWS}
MINUTES_OVERRIDE = {n: 2 for n, *_ in PREVIEW_WINDOWS}
# The rule's one allowed nudge: step forward in whole minutes to the first live play.
# Ludwigshafen-Wetzlar is in the half-time break at 66:00; the second-half throw-off
# is at 67:00 (checked 66:00-70:30 for live play only).
START_OVERRIDE_S = {"Ludwigshafen_Wetzlar_67m00_10min": 67 * 60,
                    # the stream opens with ~30 min of drone footage: at 66:00 the first
                    # half ends ~6.5 min in and half-time ads follow, so the window moves
                    # to the second half (live from ~84.5 min; signal lost at ~108 min).
                    # Checked on a 4-minute timeline for live play only.
                    "Bielefeld_Aurich_86m00_10min": 86 * 60}
MANIFEST = REPO / "runs/smart-reid-shadow/heldout_windows.json"


def cut(name: str, source: Path, out: Path) -> dict:
    start = START_OVERRIDE_S.get(name, START_S)
    minutes = MINUTES_OVERRIDE.get(name, MINUTES)
    if not out.is_file():
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start), "-i", str(source),
                        "-t", str(minutes * 60), "-vf", f"fps={FPS}", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)],
                       check=True)
    frames = VideoFrames(out)
    return {"window": name, "video": str(out), "source": str(source),
            "source_bytes": source.stat().st_size, "start_s": start, "minutes": minutes,
            "frames": len(frames), "fps": frames.fps}


NO_PAUSE_SUFFIX = "_nocourtpause"


def run_without_court_pause(name: str, check_every: int = 10, ocr_every: int = 10) -> Path:
    """Track one window with the court pause disabled, then bundle it (experiment only).

    The court model does not recognise lower-league halls: on the sealed previews it
    saw a court on 4% and 24% of live-play frames, so production `process_video`
    paused SAM2 almost throughout. This reuses that run's prepare/scan/teams outputs,
    tracks with an all-True court-in-view flag and the same tracker flags
    `process_video` passes, and bundles into `runs/bundles/<name>_nocourtpause`.
    Production code is not changed.
    """
    source_work = REPO / "data/interim" / name
    work = REPO / "data/interim" / (name + NO_PAUSE_SUFFIX)
    work.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "detections.npz", "keypoints.json", "motion.json", "team.pkl"):
        if not (work / f).is_file():
            shutil.copy2(source_work / f, work / f)
    meta = json.loads((work / "meta.json").read_text())
    np.save(work / "court_in_view.npy", np.ones(meta["frames"], dtype=bool))
    if not (work / "track.json").is_file():
        command = [sys.executable, "-m", "scripts.render_full_pipeline", meta["video"],
                   "--tracker", "sam2", "--detections", str(work / "detections.npz"),
                   "--number-detections", str(work / "detections.npz"),
                   "--team-model", str(work / "team.pkl"), "--output", str(work / "track.mp4"),
                   "--frame-source", "video", "--no-video", "--device", "cuda",
                   "--court-in-view", str(work / "court_in_view.npy"),
                   "--check-every", str(check_every), "--ocr-every", str(ocr_every)]
        env = {**os.environ, "SAM2_UPSTREAM_DIR": os.environ.get("SAM2_UPSTREAM_DIR", str(REPO / "sam2-upstream"))}
        with (work / "track.log").open("w") as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    bundle = REPO / "runs/bundles" / (name + NO_PAUSE_SUFFIX)
    # every stage output now exists, so process_video only builds the bundle
    subprocess.run([sys.executable, "-m", "scripts.process_video", meta["video"], "--work", str(work),
                    "--out", str(bundle)], cwd=REPO, check=True)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("cut", "nopause"))
    parser.add_argument("--window", help="nopause: window name")
    parser.add_argument("--count", type=int, default=None, help="how many windows from the top of the set")
    parser.add_argument("--set", dest="window_set", choices=tuple(WINDOW_SETS), default="test")
    args = parser.parse_args()
    if args.command == "nopause":
        print(run_without_court_pause(args.window))
        return
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {"windows": {}}
    manifest["rule"] = f"{MINUTES} min from {START_S // 60}:00 of the broadcast, {FPS} fps (fixed before processing)"
    for name, match, rel, exposure in WINDOW_SETS[args.window_set][:args.count]:
        started = time.monotonic()
        entry = cut(name, SOURCES / rel, REPO / "data/raw" / f"{name}.mp4")
        entry.update(match=match, detector_exposure=exposure, role=args.window_set, cut_s=round(time.monotonic() - started, 1))
        manifest["windows"][name] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
