import os
import cv2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = (
    os.environ["CUDA_HOME"] + "/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)

import numpy as np
import torch
from tqdm import tqdm

import supervision as sv
from inference import get_model
from sports import (
    clean_paths,
    MeasurementUnit,
    TeamClassifier,
    ViewTransformer,
)
from sports.handball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
)

# ── paths ─────────────────────────────────────────────────────────────────────

HOME               = Path.cwd()
SOURCE_VIDEO_PATH  = Path("/home/valentinweyer/projects/handball-computer-vision/source/Han-Ber4.mp4")
OUTPUT_VIDEO       = SOURCE_VIDEO_PATH.parent / f"{SOURCE_VIDEO_PATH.stem}-map-side.mp4"
OUTPUT_COMPRESSED  = SOURCE_VIDEO_PATH.parent / f"{SOURCE_VIDEO_PATH.stem}-map-side-compressed.mp4"

# ── model config ──────────────────────────────────────────────────────────────

PLAYER_MODEL_ID            = "player-and-handball-detection-3z9xf/3"
PLAYER_CONFIDENCE          = 0.5
PLAYER_IOU                 = 0.9
GOALKEEPER_CLASS_ID        = 1
FIELD_PLAYER_CLASS_ID      = 2
PLAYER_CLASS_IDS           = [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]

KEYPOINT_MODEL_ID          = "keypointv333-uwois-xprdi/4"
KEYPOINT_CONFIDENCE        = 0.5
KEYPOINT_ANCHOR_CONFIDENCE = 0.5

# ── team config ───────────────────────────────────────────────────────────────

TEAM_NAMES  = {0: "Porto", 1: "SCM"}
TEAM_COLORS = {"Porto": "#0000FF", "SCM": "#FF0000"}

# ── court render config ───────────────────────────────────────────────────────

COURT_SCALE          = 0.5
COURT_PADDING        = 50
COURT_LINE_THICKNESS = 6

# ── load models ───────────────────────────────────────────────────────────────

PLAYER_MODEL   = get_model(model_id=PLAYER_MODEL_ID)
KEYPOINT_MODEL = get_model(model_id=KEYPOINT_MODEL_ID)
config         = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)

# ── SAM2 tracker ──────────────────────────────────────────────────────────────

SAM2_ROOT       = Path("/home/valentinweyer/projects/handball-computer-vision/segment-anything-2-real-time")
SAM2_CHECKPOINT = str(SAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

import sys
sys.path.insert(0, str(SAM2_ROOT))
from sam2.build_sam import build_sam2_camera_predictor

import os as _os
_os.chdir(SAM2_ROOT)
predictor = build_sam2_camera_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
_os.chdir(HOME)


class SAM2Tracker:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")
        if detections.tracker_id is None:
            detections.tracker_id = list(range(1, len(detections) + 1))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(obj_id),
                    bbox=np.asarray([xyxy], dtype=np.float32),
                )
        self._prompted = True

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = self.predictor.track(frame)
        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        masks = np.array([
            sv.filter_segments_by_distance(mask, relative_distance=0.03, mode="edge")
            for mask in masks
        ])
        xyxy = sv.mask_to_xyxy(masks=masks)
        return sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)


# ── fit team classifier on first frame ───────────────────────────────────────

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame           = next(frame_generator)

result     = PLAYER_MODEL.infer(frame, confidence=PLAYER_CONFIDENCE, iou_threshold=PLAYER_IOU)[0]
detections = sv.Detections.from_inference(result)
detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
detections.tracker_id = np.arange(1, len(detections) + 1)

boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
crops = [sv.crop_image(frame, box) for box in boxes]

team_classifier = TeamClassifier(device="cuda")
team_classifier.fit(crops)

TEAMS = np.array(team_classifier.predict(crops))  # stable — SAM2 IDs don't change
P     = len(detections)

# ── prompt SAM2 ───────────────────────────────────────────────────────────────

tracker = SAM2Tracker(predictor)
tracker.prompt_first_frame(frame, detections)

# ── single pass: SAM2 + homography + cache boxes ──────────────────────────────

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

video_xy    = []          # (T, P, 2) court coords
video_boxes = []          # (T, P, 4) xyxy boxes for annotation — NaN if not tracked
video_keypoints = []      # (T,) list of [(class_id, x, y, conf), ...] for overlay

print("Pass 1: tracking + collecting positions...")
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
next(frame_generator)  # skip first frame (already used for prompting)

for frame in tqdm(frame_generator, total=video_info.total_frames - 1, desc="SAM2 + homography"):
    detections = tracker.propagate(frame)

    # cache boxes — aligned to player index (tracker_id - 1)
    boxes_frame = np.full((P, 4), np.nan)
    for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
        idx = int(tid) - 1
        if 0 <= idx < P:
            boxes_frame[idx] = xyxy
    video_boxes.append(boxes_frame)

    # homography
    kp_result     = KEYPOINT_MODEL.infer(frame, confidence=KEYPOINT_CONFIDENCE)[0]
    raw_kps       = kp_result.predictions[0].keypoints if kp_result.predictions else []
    confident_kps = [kp for kp in raw_kps if kp.confidence > KEYPOINT_ANCHOR_CONFIDENCE]

    seen, deduped = set(), []
    for kp in confident_kps:
        pt = tuple(config.vertices[int(kp.class_name) - 1])
        if pt not in seen:
            seen.add(pt)
            deduped.append(kp)
    confident_kps = deduped

    if len(confident_kps) < 4:
        video_xy.append(np.full((P, 2), np.nan))
        video_keypoints.append([
            (int(kp.class_name), float(kp.x), float(kp.y), float(kp.confidence))
            for kp in confident_kps
        ])
        continue

    landmark_indices = np.array([int(kp.class_name) - 1 for kp in confident_kps])
    court_landmarks  = np.array(config.vertices)[landmark_indices]
    frame_landmarks  = np.array([[kp.x, kp.y] for kp in confident_kps], dtype=np.float32)

    # cache keypoints for overlay in pass 3
    video_keypoints.append([
        (int(kp.class_name), float(kp.x), float(kp.y), float(kp.confidence))
        for kp in confident_kps
    ])

    try:
        transformer = ViewTransformer(source=frame_landmarks, target=court_landmarks)
        frame_xy    = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        court_xy    = transformer.transform_points(points=frame_xy)
    except Exception:
        video_xy.append(np.full((P, 2), np.nan))
        continue

    row = np.full((P, 2), np.nan)
    for tid, xy in zip(detections.tracker_id, court_xy):
        idx = int(tid) - 1
        if 0 <= idx < P:
            row[idx] = xy
    video_xy.append(row)

video_xy    = np.stack(video_xy)    # (T, P, 2)
video_boxes = np.stack(video_boxes) # (T, P, 4)

# ── pass 2: clean and smooth ──────────────────────────────────────────────────

print("Pass 2: cleaning paths...")
cleaned_xy, _ = clean_paths(
    video_xy,
    jump_sigma=5.0,
    min_jump_dist=1.5,
    max_jump_run=10,
    pad_around_runs=1,
    smooth_window=5,
    smooth_poly=2,
)

# ── pass 3: render side-by-side from cache (no SAM2) ─────────────────────────

court_ref = draw_court(
    config=config, scale=COURT_SCALE,
    padding=COURT_PADDING, line_thickness=COURT_LINE_THICKNESS,
)
court_h, court_w, _ = court_ref.shape

orig_h      = video_info.height
orig_w      = video_info.width
scale_orig  = court_h / orig_h
new_orig_w  = int(orig_w * scale_orig)
canvas_w    = new_orig_w + court_w
canvas_h    = court_h

out_info        = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
out_info.width  = canvas_w
out_info.height = canvas_h

team_colors   = sv.ColorPalette.from_hex([TEAM_COLORS[TEAM_NAMES[0]], TEAM_COLORS[TEAM_NAMES[1]]])
box_annotator = sv.BoxAnnotator(color=team_colors, thickness=2)

print("Pass 3: rendering side-by-side...")
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
next(frame_generator)  # skip first frame

with sv.VideoSink(OUTPUT_VIDEO, out_info) as sink:
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=len(cleaned_xy), desc="rendering")):
        # annotate original with cached boxes
        boxes_frame = video_boxes[frame_idx]  # (P, 4)
        valid_mask  = np.isfinite(boxes_frame).all(axis=1)

        if valid_mask.any():
            valid_xyxy  = boxes_frame[valid_mask]
            valid_teams = TEAMS[valid_mask]
            det_for_ann = sv.Detections(xyxy=valid_xyxy)
            annotated   = box_annotator.annotate(
                scene=frame.copy(),
                detections=det_for_ann,
                custom_color_lookup=valid_teams,
            )
        else:
            annotated = frame.copy()

        # overlay keypoints
        for class_id, kp_x, kp_y, kp_conf in video_keypoints[frame_idx]:
            px, py = int(round(kp_x)), int(round(kp_y))
            cv2.circle(annotated, (px, py), 6, (0, 255, 0), -1)
            label = str(class_id)
            font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
            tx, ty = px + 8, py + th // 2
            cv2.rectangle(annotated, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(annotated, label, (tx, ty), font, fs, (0, 255, 0), thick)

        orig_resized = cv2.resize(annotated, (new_orig_w, court_h))

        # court map from cleaned positions
        court    = draw_court(
            config=config, scale=COURT_SCALE,
            padding=COURT_PADDING, line_thickness=COURT_LINE_THICKNESS,
        )
        frame_xy = cleaned_xy[frame_idx]  # (P, 2)
        for team_id in [0, 1]:
            pts   = frame_xy[TEAMS == team_id]
            valid = np.isfinite(pts).all(axis=1)
            if valid.any():
                court = draw_points_on_court(
                    config=config,
                    xy=pts[valid],
                    fill_color=sv.Color.from_hex(TEAM_COLORS[TEAM_NAMES[team_id]]),
                    court=court,
                    scale=COURT_SCALE,
                    padding=COURT_PADDING,
                    line_thickness=COURT_LINE_THICKNESS,
                )

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :new_orig_w] = orig_resized
        canvas[:, new_orig_w:] = court
        sink.write_frame(canvas)

os.system(f"ffmpeg -y -loglevel error -i {OUTPUT_VIDEO} -vcodec libx264 -crf 28 {OUTPUT_COMPRESSED}")
print("done →", OUTPUT_COMPRESSED)