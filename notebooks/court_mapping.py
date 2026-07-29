import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

import cv2
import numpy as np
from tqdm import tqdm

import supervision as sv
from inference import get_model
from sports import MeasurementUnit, ViewTransformer, TeamClassifier
from sports.handball import CourtConfiguration, League, draw_court, draw_points_on_court

# ── Config ────────────────────────────────────────────────────────────────────

HOME = Path(__file__).parent
SOURCE_VIDEO_PATH = HOME / "Han-Ber4.mp4"
OUTPUT_PATH = HOME / f"{SOURCE_VIDEO_PATH.stem}-court-map{SOURCE_VIDEO_PATH.suffix}"

PLAYER_DETECTION_MODEL_ID = "player-and-handball-detection-3z9xf/3"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.5
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9
GOALKEEPER_CLASS_ID = 1
FIELD_PLAYER_CLASS_ID = 2
PLAYER_CLASS_IDS = [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]

KEYPOINT_DETECTION_MODEL_ID = "keypointv333-uwois-xprdi/4"
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

def get_confident_keypoints(result):
    raw = result.predictions[0].keypoints if result.predictions else []
    kps = [kp for kp in raw if kp.confidence > KEYPOINT_ANCHOR_CONFIDENCE]
    # deduplicate keypoints that map to identical court coordinates
    seen = set()
    deduped = []
    for kp in kps:
        idx = int(kp.class_name) - 1
        pt = tuple(config.vertices[idx])
        if pt not in seen:
            seen.add(pt)
            deduped.append(kp)
    return deduped


def build_transformer(kps):
    indices = np.array([int(kp.class_name) - 1 for kp in kps])
    court_pts = np.array(config.vertices)[indices]
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
