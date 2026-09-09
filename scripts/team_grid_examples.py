"""Grid of player crops per predicted team, via sv.plot_images_grid.

Run standalone as a diagnostic whenever team_model.py's crop geometry or model
changes -- this is the fastest way to see whether a fit is actually separating
kit colours, faster than watching a full pipeline render. Crops are ordered by
confidence and annotated with it, so a cluster that is a real kit reads as a
solid block of colour and a cluster that is a 2-means artefact does not.

Defaults reproduce the source notebook (cells 73 & 75): fit from the Roboflow
detector on HANDBALL_CV_VIDEO. Pass --detections and --team-model to inspect a
model the pipeline actually ran with, using the identical boxes:

    python -m scripts.team_grid_examples \
        --video data/interim/overnight/Kiel_Lemgo_10min.mp4 \
        --detections data/interim/overnight/.Kiel_Lemgo_10min_det.npz \
        --team-model data/interim/overnight/.Kiel_Lemgo_10min_team.pkl
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import supervision as sv

from handball_cv.teams.model import (
    MIN_TEAM_VOTE_CONFIDENCE, TeamModel, crop_quality, torso_boxes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = Path(
    os.getenv("HANDBALL_CV_VIDEO", PROJECT_ROOT / "data/raw/Han-Ber4.mp4")
)
GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID = 1, 2
PLAYER_CLASS_IDS = [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]
ANNOTATED_HEIGHT = 96
FIT_STRIDE = 10  # must match fit_from_video's stride argument


# Both detectors take (frame_index, frame_rgb) so the caller never has to know
# which one it holds: the cache is addressed by index, Roboflow by pixels.
def roboflow_detector():
    """Lazy: the cached-detection path must not require Roboflow or a GPU model."""
    from inference import get_model

    model = get_model(model_id="player-and-handball-detection-3z9xf/3")

    def detect(frame_index: int, frame_rgb: np.ndarray) -> sv.Detections:
        del frame_index
        result = model.infer(frame_rgb, confidence=0.5, iou_threshold=0.9)[0]
        det = sv.Detections.from_inference(result)
        return det[np.isin(det.class_id, PLAYER_CLASS_IDS)]

    return detect


def cached_detector(cache_path: Path):
    """Boxes exactly as the pipeline saw them, so the grid explains its output."""
    from scripts.render_raw_team_classification import (
        load_detection_cache, person_detections,
    )

    cache = load_detection_cache(cache_path)

    def detect(frame_index: int, frame_rgb: np.ndarray) -> sv.Detections:
        del frame_rgb
        return person_detections(cache, frame_index)

    return detect


def annotate(crop: np.ndarray, text: str) -> np.ndarray:
    """Scale to a common height and burn the confidence in, so grids are readable."""
    height, width = crop.shape[:2]
    if height < 1 or width < 1:
        return np.zeros((ANNOTATED_HEIGHT, ANNOTATED_HEIGHT, 3), dtype=np.uint8)
    scale = ANNOTATED_HEIGHT / height
    scaled = cv2.resize(
        crop, (max(int(round(width * scale)), 8), ANNOTATED_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.zeros(
        (ANNOTATED_HEIGHT + 16, scaled.shape[1], 3), dtype=np.uint8
    )
    canvas[:ANNOTATED_HEIGHT] = scaled
    cv2.putText(
        canvas, text, (2, ANNOTATED_HEIGHT + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


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


def collect_crops(video_path: Path, detect, stride: int, limit: int):
    """Torso crops on the fit's geometry, keeping the goalkeeper flag."""
    crops, is_gk = [], []
    for idx, frame_bgr in enumerate(
        sv.get_video_frames_generator(str(video_path))
    ):
        if idx % stride != 0:
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        det = detect(idx, frame_rgb)
        if len(det) == 0:
            continue
        class_ids = (
            det.class_id if det.class_id is not None
            else np.full(len(det), FIELD_PLAYER_CLASS_ID)
        )
        for box, cid in zip(torso_boxes(det.xyxy), class_ids):
            crop = sv.crop_image(frame_rgb, box)
            if crop.size == 0:
                continue
            crops.append(crop)
            is_gk.append(cid == GOALKEEPER_CLASS_ID)
        if len(crops) >= limit:
            break
    return crops, np.array(is_gk, dtype=bool)


def report(name: str, teams, confidence) -> None:
    below = int((confidence < MIN_TEAM_VOTE_CONFIDENCE).sum())
    print(
        f"{name}: n={len(teams)}  team0={int((teams == 0).sum())}  "
        f"team1={int((teams == 1).sum())}  "
        f"mean_confidence={confidence.mean():.3f}  "
        f"median={float(np.median(confidence)):.3f}  "
        f"below_gate(<{MIN_TEAM_VOTE_CONFIDENCE})={below}/{len(confidence)} "
        f"({100.0 * below / max(len(confidence), 1):.0f}%)"
    )


def cluster_grid(crops, teams, confidence, team_id: int, out_dir: Path) -> None:
    """Most confident first: a real kit stays one colour all the way down."""
    members = [
        (float(c), crop) for crop, t, c in zip(crops, teams, confidence)
        if t == team_id
    ]
    members.sort(key=lambda item: -item[0])
    save_grid(
        [annotate(crop, f"{conf:.2f}") for conf, crop in members],
        f"Team {team_id} -- field players, most confident first",
        out_dir / f"team_{team_id}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--detections", type=Path,
        help="cached detection npz; omit to run the Roboflow detector",
    )
    parser.add_argument(
        "--team-model", type=Path,
        help="fitted model to inspect; omit to fit (and cache) from the video",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-crops", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = args.out or PROJECT_ROOT / "runs/team_grid" / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    detect = (
        cached_detector(args.detections) if args.detections
        else roboflow_detector()
    )
    if args.team_model:
        team_model = TeamModel.load(args.team_model, device=args.device)
    else:
        cache = PROJECT_ROOT / "data/cache/team_models" / f"{args.video.stem}.pkl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        # fit_from_video hands detect_fn the frame, not the index, and calls
        # it only every stride-th frame, so the adapter steps its own counter
        # by the same stride -- counting calls puts frame n's boxes on frame
        # n*stride (see run_overnight_batch.fit_team_model).
        state = {"index": -FIT_STRIDE}

        def fit_detect(frame_rgb: np.ndarray) -> sv.Detections:
            state["index"] += FIT_STRIDE
            return detect(state["index"], frame_rgb)

        team_model = TeamModel.load_or_fit(
            cache, args.video, fit_detect, stride=FIT_STRIDE,
            exclude_class_ids=(GOALKEEPER_CLASS_ID,), device=args.device,
        )

    crops, is_gk = collect_crops(args.video, detect, args.stride, args.max_crops)
    field_crops = [c for c, gk in zip(crops, is_gk) if not gk]
    gk_crops = [c for c, gk in zip(crops, is_gk) if gk]

    field_teams, field_conf = team_model.predict_crops(field_crops)
    report("field players", field_teams, field_conf)
    rejected = sum(1 for c in field_crops if not crop_quality(c).accepted)
    print(
        f"  crop quality rejected {rejected}/{len(field_crops)}; "
        f"visual_color_agreement={team_model.visual_color_agreement:.3f}"
    )
    print(f"goalkeeper crops collected separately: n={len(gk_crops)}")

    for team_id in (0, 1):
        cluster_grid(field_crops, field_teams, field_conf, team_id, out_dir)

    contested = [
        (float(c), crop) for crop, c in zip(field_crops, field_conf)
        if c < MIN_TEAM_VOTE_CONFIDENCE
    ]
    contested.sort(key=lambda item: item[0])
    save_grid(
        [annotate(crop, f"{conf:.2f}") for conf, crop in contested],
        f"Below the vote gate ({MIN_TEAM_VOTE_CONFIDENCE}) -- least confident first",
        out_dir / "contested.png",
    )

    # goalkeepers: excluded from the fit, but still get assigned to whichever of
    # the two components they're nearest to at predict time
    if gk_crops:
        gk_teams, gk_conf = team_model.predict_crops(gk_crops)
        report("goalkeepers", gk_teams, gk_conf)
        save_grid(
            [
                annotate(crop, f"t{t} {c:.2f}")
                for crop, t, c in zip(gk_crops, gk_teams, gk_conf)
            ],
            "Goalkeepers (excluded from fit)", out_dir / "goalkeepers.png",
        )


if __name__ == "__main__":
    main()
