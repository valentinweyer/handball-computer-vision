"""Run native CAMELTrack plus GPU cues against cached RF-DETR detections."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['felix', 'han', 'bhc'], required=True)
    parser.add_argument('--max-frames', type=int, default=None, help='prefix from frame zero; default entire clip')
    parser.add_argument('--warmup-frames', type=int, default=3, help='synthetic frames, then reset all tracks')
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwrite a run')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.warmup_frames < 0:
        parser.error('--warmup-frames must be nonnegative')
    if args.output.exists():
        parser.error(f'output exists: {args.output}')
    from experiments.cameltrack.runtime import CamelTracker, bootstrap, provenance, sha256
    from experiments.cameltrack.cases import inputs
    bootstrap()
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from scripts.render_raw_team_classification import load_detection_cache, person_detections

    paths = inputs(args.case)
    cache = load_detection_cache(paths['detections'])
    frames = sorted(paths['frames'].glob('*.jpg'))
    if not frames:
        raise FileNotFoundError(paths['frames'])
    count = len(frames) if args.max_frames is None else args.max_frames
    if count > len(frames) or count > len(cache['offsets']) - 1:
        raise ValueError('Requested frame count exceeds cached frames or detections')
    frames = frames[:count]
    assert [int(path.stem) for path in frames] == list(range(count)), 'Frame cache must start at zero with no gaps'
    # Fingerprinting, model loading and synthetic warmup are outside the loop timer.
    manifest = {'backend': 'cameltrack', 'case': args.case, 'status': 'initializing',
                'requested_frames': count, 'processed_frames': 0,
                'video': str(paths['video'].resolve()), 'video_sha256': sha256(paths['video']),
                'detections': str(paths['detections'].resolve()),
                'detections_sha256': sha256(paths['detections']), **provenance(),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'onnxruntime': ort.__version__, 'precision': 'FP32; no AMP or quantization',
                'live_detection': False, 'ocr': False, 'team_classification': False,
                'masks': False, 'warmup_frames': args.warmup_frames,
                'timing_scope': 'JPEG decode, cached-detection filtering, RTMPose, KPR, CAMEL association/lifecycle, replay collection; excludes model build, warmup, live RF-DETR, output encoding',
                'thresholds': {'min_det_conf': .1, 'min_init_det_conf': .4,
                               'min_num_hits': 0, 'max_wo_hits': 150, 'max_track_gallery_size': 50}}
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    started = perf_counter()
    tracker = CamelTracker('cuda')
    manifest['model_build_seconds'] = perf_counter() - started
    manifest['gpu'] = torch.cuda.get_device_name(0)
    manifest['loading'] = tracker.loading
    frame = np.full((384, 640, 3), 127, np.uint8)
    boxes = np.array([[80, 30, 190, 360], [350, 30, 470, 360]], np.float32)
    for _ in range(args.warmup_frames):
        tracker.update(frame, boxes, np.array([.95, .95], np.float32))
    tracker.reset()
    frame_index, tracker_id, replay_boxes, timings = [], [], [], []
    stages = {'pose_seconds': 0., 'appearance_seconds': 0., 'association_seconds': 0.}
    processed = 0
    torch.cuda.synchronize()
    started = perf_counter()
    try:
        for index, path in enumerate(frames):
            frame_started = perf_counter()
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Unreadable frame: {path}')
            detections = person_detections(cache, index)
            ids, output = tracker.update(image, detections.xyxy, detections.confidence)
            assert len(ids) == len(output) and len(set(ids)) == len(ids)
            frame_index.extend([index] * len(ids))
            tracker_id.extend(ids.tolist())
            replay_boxes.extend(output.tolist())
            for key, value in tracker.timings.items():
                stages[key] += value
            timings.append(perf_counter() - frame_started)
            processed += 1
            if processed % 25 == 0:
                print(f'{processed}/{count} frames', flush=True)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = repr(error)
        raise
    finally:
        elapsed = perf_counter() - started
        manifest.update(processed_frames=processed, loop_seconds=elapsed,
                        loop_fps=processed / elapsed if elapsed else None,
                        stage_seconds=stages,
                        frame_latency_ms_median=float(np.median(timings)*1000) if timings else None,
                        frame_latency_ms_p95=float(np.percentile(timings, 95)*1000) if timings else None)
        np.savez_compressed(args.output / 'replay.npz',
                            frame_index=np.asarray(frame_index, dtype=np.int32),
                            tracker_id=np.asarray(tracker_id, dtype=np.int64),
                            boxes=np.asarray(replay_boxes, dtype=float).reshape(-1, 4),
                            source=np.asarray(['cameltrack'] * len(tracker_id)))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"{processed} frames, {manifest['loop_fps']:.2f} FPS; {args.output}")


if __name__ == '__main__':
    main()
