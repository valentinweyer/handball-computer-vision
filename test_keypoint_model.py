"""
Quick test: run the RF-DETR keypoint model (keypointv333-uwois-xprdi/4) on a video
and write an annotated output video with court landmarks overlaid.

Uses Roboflow's hosted inference API (cloud) so no model weights are downloaded locally.

Usage:
    conda run -n handball-computer-vision python test_keypoint_model.py --api-key YOUR_KEY
    conda run -n handball-computer-vision python test_keypoint_model.py --api-key YOUR_KEY --video Hannover.mp4
"""
import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # suppress onnxruntime C++ warnings
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")
warnings.filterwarnings("ignore")

import cv2
import supervision as sv
from inference import get_model

SOURCE_DIR = Path(__file__).parent

KEYPOINT_DETECTION_MODEL_ID = "keypointv333-uwois-xprdi/4"
KEYPOINT_DETECTION_MODEL_CONFIDENCE = 0.5
KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE = 0.5
KEYPOINT_COLOR = sv.Color.from_hex("#FF1493")


def main(video_path: Path, max_frames: int, api_key: str) -> None:
    print(f"Loading model {KEYPOINT_DETECTION_MODEL_ID} ...")
    model = get_model(model_id=KEYPOINT_DETECTION_MODEL_ID, api_key=api_key)
    print("Model loaded.")

    vertex_annotator = sv.VertexAnnotator(color=KEYPOINT_COLOR, radius=8)

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    out_path = SOURCE_DIR / f"{video_path.stem}-keypoint-test.mp4"

    frame_generator = sv.get_video_frames_generator(str(video_path))

    total = min(max_frames, video_info.total_frames)
    detected_count = 0

    with sv.VideoSink(str(out_path), video_info) as sink:
        for frame_idx, frame in enumerate(frame_generator):
            if frame_idx >= max_frames:
                break

            result = model.infer(frame, confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE)[0]
            key_points = sv.KeyPoints.from_inference(result)

            # filter to high-confidence keypoints only
            if key_points.confidence is not None and len(key_points) > 0:
                mask = key_points.confidence[0] > KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE
                key_points_filtered = key_points[:, mask]
                n_detected = int(mask.sum())
            else:
                key_points_filtered = key_points
                n_detected = 0

            if n_detected > 0:
                detected_count += 1

            annotated = frame.copy()
            annotated = vertex_annotator.annotate(scene=annotated, key_points=key_points_filtered)

            cv2.putText(
                annotated,
                f"frame {frame_idx+1}/{total}  landmarks: {n_detected}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            sink.write_frame(annotated)

            if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                print(f"  frame {frame_idx+1}/{total}  landmarks detected: {n_detected}")

    print(f"\nDone. {detected_count}/{total} frames had ≥1 landmark detected.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="FelixClaar.mp4", help="Video filename in project root")
    parser.add_argument("--frames", type=int, default=200, help="Max frames to process")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""), help="Roboflow API key")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY env var or pass --api-key <key>")

    video_path = SOURCE_DIR / args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    main(video_path, args.frames, args.api_key)
