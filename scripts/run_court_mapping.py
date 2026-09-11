import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")

import cv2
import numpy as np
from tqdm import tqdm

import supervision as sv
from inference import get_model
from sports import MeasurementUnit, ViewTransformer, TeamClassifier
from sports.handball import CourtConfiguration, League, draw_court, draw_points_on_court

from handball_cv.court.keypoints import court_points

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO_PATH = Path(os.getenv("HANDBALL_CV_VIDEO", PROJECT_ROOT / "data/raw/Han-Ber4.mp4"))
OUTPUT_PATH = PROJECT_ROOT / "runs/court_mapping" / f"{SOURCE_VIDEO_PATH.stem}-court-map{SOURCE_VIDEO_PATH.suffix}"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PLAYER_DETECTION_MODEL_ID = "player-and-handball-detection-3z9xf/3"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.5
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9
GOALKEEPER_CLASS_ID = 1
FIELD_PLAYER_CLASS_ID = 2
PLAYER_CLASS_IDS = [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]

# Version 3, not 4. Version 4 is served as `rfdetr-keypoint-preview`, a type the
# pinned `inference` 0.62.0 has no implementation class for, so `get_model`
# raises KeyError before any inference happens. Version 3 loads and runs locally
# on the same 892 images, and Roboflow reports it as the better model besides:
# mAP 99.5 / precision 99.96 / recall 100.0, against 97.0 / 98.6 / 97.9 for
# version 4 (which was trained from scratch rather than fine-tuned).
KEYPOINT_DETECTION_MODEL_ID = "keypointv333-uwois-xprdi/3"
KEYPOINT_DETECTION_MODEL_CONFIDENCE = 0.5
KEYPOINT_ANCHOR_CONFIDENCE = 0.5

TEAM_COLORS = {
    "Porto": "#1e6ab0",
    "SCM":   "#d12b2b",
}
TEAM_NAMES = {0: "Porto", 1: "SCM"}

COURT_SCALE = 0.5
COURT_PADDING = 50

# ── Models ────────────────────────────────────────────────────────────────────

player_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID)
keypoint_model = get_model(model_id=KEYPOINT_DETECTION_MODEL_ID)

config = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)

# TeamClassifier: fit on a sample of crops from the video
STRIDE = 30
FILTERED_CLASS_IDS = [FIELD_PLAYER_CLASS_ID]

crops = []
for frame in sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE):
    result = player_model.infer(frame, confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD)[0]
    dets = sv.Detections.from_inference(result)
    dets = dets[np.isin(dets.class_id, FILTERED_CLASS_IDS)]
    boxes = sv.scale_boxes(xyxy=dets.xyxy, factor=0.4)
    crops += [sv.crop_image(frame, box) for box in boxes]

print(f"Fitting team classifier on {len(crops)} crops…")
team_classifier = TeamClassifier()
team_classifier.fit(crops)

# ── Helpers ───────────────────────────────────────────────────────────────────

def keypoint_slot(kp):
    """The model's 0-based keypoint slot for a prediction.

    Read `class_id`, not `class_name`. They are different numberings: `class_id`
    is the slot, while `class_name` is the label the annotator typed, and the
    two agree on only 6 of 31 landmarks (measured against the labelled export).
    This script used `int(kp.class_name) - 1` and indexed `config.vertices` with
    it, which got both halves wrong -- the wrong field, then no translation from
    slot order to the template's vertex order. On one Melsungen frame that
    produced 4 RANSAC inliers of 11 keypoints, against 8 once corrected.

    Still not a court vertex index: `handball_cv.court.keypoints` holds that
    translation.
    """
    return kp.class_id


def get_confident_keypoints(result):
    raw = result.predictions[0].keypoints if result.predictions else []
    kps = [kp for kp in raw if kp.confidence > KEYPOINT_ANCHOR_CONFIDENCE]
    # deduplicate keypoints that map to identical court coordinates
    seen = set()
    deduped = []
    for kp in kps:
        pt = tuple(court_points([keypoint_slot(kp)], config.vertices)[0])
        if pt not in seen:
            seen.add(pt)
            deduped.append(kp)
    return deduped


def build_transformer(kps):
    court_pts = court_points([keypoint_slot(kp) for kp in kps], config.vertices)
    frame_pts = np.array([[kp.x, kp.y] for kp in kps], dtype=np.float32)
    return ViewTransformer(source=frame_pts, target=court_pts)

# ── Main loop ─────────────────────────────────────────────────────────────────

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

with sv.VideoSink(str(OUTPUT_PATH), video_info) as sink:
    for frame in tqdm(sv.get_video_frames_generator(SOURCE_VIDEO_PATH),
                      total=video_info.total_frames):

        # detect players
        result = player_model.infer(frame,
                                    confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                                    iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]

        # classify teams
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
        crops = [sv.crop_image(frame, box) for box in boxes]
        teams = np.array(team_classifier.predict(crops)) if crops else np.array([], dtype=int)

        # detect court keypoints
        kp_result = keypoint_model.infer(frame, confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE)[0]
        confident_kps = get_confident_keypoints(kp_result)

        if len(confident_kps) < 4 or len(detections) == 0:
            sink.write_frame(frame)
            continue

        transformer = build_transformer(confident_kps)
        frame_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        court_xy = transformer.transform_points(points=frame_xy)

        # draw court minimap
        court_img = draw_court(config=config, scale=COURT_SCALE)
        for team_id, color_key in TEAM_NAMES.items():
            mask = teams == team_id
            if mask.any():
                court_img = draw_points_on_court(
                    config=config,
                    xy=court_xy[mask],
                    fill_color=sv.Color.from_hex(TEAM_COLORS[color_key]),
                    court=court_img,
                    scale=COURT_SCALE,
                )

        # embed minimap in top-right corner of frame
        ch, cw = court_img.shape[:2]
        fh, fw = frame.shape[:2]
        x0 = fw - cw - COURT_PADDING
        y0 = COURT_PADDING
        annotated = frame.copy()
        annotated[y0:y0 + ch, x0:x0 + cw] = court_img

        sink.write_frame(annotated)

print("Done →", OUTPUT_PATH)
