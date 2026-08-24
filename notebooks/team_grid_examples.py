"""Reproduces the source notebook's team-classification visualization (cells 73
& 75): a grid of player crops per predicted team, via sv.plot_images_grid.

Run standalone as a diagnostic whenever team_model.py's crop geometry or model
changes -- this is the fastest way to see whether a fit is actually separating
kit colours, faster than watching a full pipeline render.

    python team_grid_examples.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import supervision as sv
from inference import get_model

from team_model import MIN_TEAM_VOTE_CONFIDENCE, TeamModel, torso_boxes

SOURCE_VIDEO_PATH = Path("/home/valentinweyer/projects/handball-computer-vision/source/Han-Ber4.mp4")
TEAM_MODEL_CACHE = SOURCE_VIDEO_PATH.parent / f".{SOURCE_VIDEO_PATH.stem}_team.pkl"
GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID = 1, 2
PLAYER_CLASS_IDS = [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]
OUT_DIR = SOURCE_VIDEO_PATH.parent / "team_classification_examples"
TEST_FRAME_IDX = 100

PLAYER_MODEL = get_model(model_id="player-and-handball-detection-3z9xf/3")


def detect_players(frame_rgb: np.ndarray) -> sv.Detections:
    result = PLAYER_MODEL.infer(frame_rgb, confidence=0.5, iou_threshold=0.9)[0]
    det = sv.Detections.from_inference(result)
    return det[np.isin(det.class_id, PLAYER_CLASS_IDS)]


def save_grid(images, title, path, cols=10):
    n = min(len(images), 50)
    if n == 0:
        print(f"skip (no images): {title}")
        return
    # sv.plot_images_grid does plt.subplots(rows, cols).flat, which breaks for a
    # 1x1 grid (a bare Axes has no .flat) -- never let both dims collapse to 1.
    cols = max(cols, 2) if n == 1 else cols
    rows = max((n + cols - 1) // cols, 1)
    # plot_images_grid internally does cv2.cvtColor(BGR2RGB) -- our crops are
    # already RGB (matching what the real pipeline feeds SigLIP), so convert
    # to BGR here just for display, or colours invert.
    bgr_images = [img[:, :, ::-1] for img in images[:n]]
    sv.plot_images_grid(images=bgr_images, grid_size=(rows, cols), size=(cols, rows))
    fig = plt.gcf()
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    team_model = TeamModel.load_or_fit(
        TEAM_MODEL_CACHE, SOURCE_VIDEO_PATH, detect_players,
        exclude_class_ids=(GOALKEEPER_CLASS_ID,), device="cuda",
    )

    # ── crops used to fit the model, grouped by predicted team ────────────────
    # (field players only -- goalkeepers were excluded from the fit and are
    # shown separately below, since they still get assigned to a nearest
    # cluster at predict time even though the fit never saw them)

    crops, is_gk = [], []
    for idx, frame_bgr in enumerate(sv.get_video_frames_generator(str(SOURCE_VIDEO_PATH))):
        if idx % 15 != 0:
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        det = detect_players(frame_rgb)
        if len(det) == 0:
            continue
        for box, cid in zip(torso_boxes(det.xyxy), det.class_id):
            crop = sv.crop_image(frame_rgb, box)
            if crop.size == 0:
                continue
            crops.append(crop)
            is_gk.append(cid == GOALKEEPER_CLASS_ID)
        if len(crops) >= 300:
            break

    is_gk = np.array(is_gk)
    field_crops = [c for c, gk in zip(crops, is_gk) if not gk]
    gk_crops = [c for c, gk in zip(crops, is_gk) if gk]

    field_teams, field_conf = team_model.predict_crops(field_crops)
    print(f"field-player crops: n={len(field_crops)}  "
          f"team0={int((field_teams == 0).sum())}  team1={int((field_teams == 1).sum())}  "
          f"mean_confidence={field_conf.mean():.3f}  "
          f"below_gate(<{MIN_TEAM_VOTE_CONFIDENCE})={int((field_conf < MIN_TEAM_VOTE_CONFIDENCE).sum())}/{len(field_conf)}")
    print(f"goalkeeper crops collected separately: n={len(gk_crops)}")

    save_grid(
        [c for c, t in zip(field_crops, field_teams) if t == 0],
        "Team 0 -- crops used to fit (field players only)", OUT_DIR / "team_0_fit.png",
    )
    save_grid(
        [c for c, t in zip(field_crops, field_teams) if t == 1],
        "Team 1 -- crops used to fit (field players only)", OUT_DIR / "team_1_fit.png",
    )

    # goalkeepers: excluded from the fit, but still get assigned to whichever of
    # the two components they're nearest to at predict time (no 3rd component
    # this pass -- see the plan's "Deferred" section)
    if gk_crops:
        gk_teams, gk_conf = team_model.predict_crops(gk_crops)
        save_grid(
            gk_crops,
            f"Goalkeepers (excluded from fit) -- labels={gk_teams.tolist()} "
            f"confidences={np.round(gk_conf, 2).tolist()}",
            OUT_DIR / "goalkeepers.png", cols=max(len(gk_crops), 1),
        )

    # ── test the fitted model on ONE fresh frame ───────────────────────────────

    for idx, frame_bgr in enumerate(sv.get_video_frames_generator(str(SOURCE_VIDEO_PATH))):
        if idx == TEST_FRAME_IDX:
            break
    frame_rgb = frame_bgr[:, :, ::-1]
    det = detect_players(frame_rgb)
    teams, conf = team_model.predict(frame_rgb, det.xyxy)
    test_crops = [sv.crop_image(frame_rgb, b) for b in torso_boxes(det.xyxy)]

    t0 = [c for c, t in zip(test_crops, teams) if t == 0]
    t1 = [c for c, t in zip(test_crops, teams) if t == 1]
    print(f"frame {TEST_FRAME_IDX} test: team_0={len(t0)}  team_1={len(t1)}  "
          f"confidences={np.round(conf, 2).tolist()}")

    save_grid(t0, f"Frame {TEST_FRAME_IDX} test -- predicted team 0",
              OUT_DIR / "frame_test_0.png", cols=max(len(t0), 1))
    save_grid(t1, f"Frame {TEST_FRAME_IDX} test -- predicted team 1",
              OUT_DIR / "frame_test_1.png", cols=max(len(t1), 1))


if __name__ == "__main__":
    main()
