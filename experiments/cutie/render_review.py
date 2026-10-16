"""Render Cutie, SAM2 or Lean-SAM2 tracking on an unscored clip for a by-hand identity review.

Every tracked player gets a fixed colour and a large ID label; the frame number
is printed so a switch can be reported by frame. Tracking uses the same
production TrackManager as run_comparison.py. Masks are drawn exactly as Cutie
returns them; nothing is smoothed or filtered for display.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile


def colour(oid):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((oid * 0.618034) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def annotate(frame, frame_idx, ids, masks, boxes):
    """Draw in place, touching only each mask's box.

    `boxes` are the driver's inclusive xyxy mask boxes. Scanning every 1080p
    mask in full (fill + nonzero) was ~110 ms of a ~390 ms frame; inside the
    boxes it is a few ms. Cutie's masks are disjoint (argmax), so blending each
    box separately gives the same picture as one full-frame blend.
    """
    import cv2
    import numpy as np
    for oid, mask, box in zip(ids, masks, boxes):
        x0, y0, x1, y1 = (int(v) for v in box)
        m = mask[y0:y1 + 1, x0:x1 + 1]
        if not m.any():
            continue
        region = frame[y0:y1 + 1, x0:x1 + 1]
        c = np.array(colour(int(oid)), dtype=np.float32)
        region[m] = (region[m] * 0.55 + c * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
        cv2.drawContours(frame, contours, -1, colour(int(oid)), 2)
        ys, xs = np.nonzero(m)
        text = str(int(oid))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        x, y = x0 + int(np.median(xs)) - tw // 2, max(th + 4, y0 + int(ys.min()) - 8)
        cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour(int(oid)), 3)
    cv2.rectangle(frame, (0, 0), (260, 48), (0, 0, 0), -1)
    cv2.putText(frame, f"frame {frame_idx}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True, help='directory of 00000.jpg ...')
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--team', type=Path, required=True)
    parser.add_argument('--backend', choices=['cutie', 'sam2', 'lean_sam2'], default='cutie')
    parser.add_argument('--lean-method', choices=['memory', 'off'], default='memory')
    parser.add_argument('--cutie-size', type=int, choices=[480, 1024], default=480)
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory must be new')
    from experiments.cutie.adapter import CHECKPOINT, SAM2_CHECKPOINT, bootstrap, build_models
    bootstrap()
    if args.backend == 'lean_sam2':
        from experiments.lean_sam2 import adapter as lean
        lean.bootstrap()
    import cv2
    import numpy as np
    import torch
    from handball_cv.teams.model import TeamModel
    from handball_cv.tracking.sam2_driver import drive_sam2
    from scripts.render_raw_team_classification import load_detection_cache, person_detections
    from experiments.cutie.run_comparison import json_default

    files = sorted(args.frames.glob('*.jpg'), key=lambda p: int(p.stem))
    assert [int(p.stem) for p in files] == list(range(len(files))), 'frames must be contiguous from 0'
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    if args.backend == 'cutie':
        predictor, _ = build_models(max_internal_size=args.cutie_size)
        checkpoint, factory = CHECKPOINT, (lambda _: predictor)
    elif args.backend == 'lean_sam2':
        lean.revision()
        checkpoint, factory = SAM2_CHECKPOINT, (lambda ckpt: lean.build_predictor(ckpt, args.lean_method)[0])
    else:
        checkpoint, factory = SAM2_CHECKPOINT, None    # the production SAM2 path
    team = TeamModel.load(args.team, device='cuda')
    cache = load_detection_cache(args.detections)
    raw = args.output / 'raw.mp4'
    h, w = cv2.imread(str(files[0])).shape[:2]
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    rows = []
    with tempfile.TemporaryDirectory(prefix='cutie-review-') as tmp:
        for path in files:
            (Path(tmp) / path.name).symlink_to(path.resolve())
        manager, seeds, frames = drive_sam2(
            args.frames, lambda fid: person_detections(cache, fid), team, str(checkpoint),
            frame_cache_dir=Path(tmp), predictor_factory=factory, desc=f'{args.backend} review')
        rows.extend((0, int(oid), box.copy()) for oid, box in seeds.items())
        # Frame 0 is the seed frame; it has no propagated masks, so show the
        # seed boxes' frame plainly rather than invent masks for it.
        # Read, draw and encode on one worker thread so the GPU keeps tracking
        # meanwhile; one worker keeps frames in order, two in flight bound memory.
        def draw(fid, ids, masks, boxes):
            writer.write(annotate(cv2.imread(str(files[fid])), fid, ids, masks, boxes))
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = [pool.submit(draw, 0, [], [], [])]
            for result in frames:
                rows.extend((result.frame_idx, int(oid), box.copy())
                            for oid, box in zip(result.player_ids, result.boxes))
                pending.append(pool.submit(draw, result.frame_idx, result.player_ids,
                                           result.masks, result.boxes))
                if len(pending) > 2:
                    pending.pop(0).result()
            for job in pending:
                job.result()
    writer.release()
    np.savez_compressed(args.output / 'replay.npz',
                        frame_index=np.array([r[0] for r in rows], dtype=int),
                        tracker_id=np.array([r[1] for r in rows], dtype=int),
                        boxes=np.array([r[2] for r in rows], dtype=float).reshape(-1, 4))
    (args.output / 'events.json').write_text(json.dumps(manager.events, indent=2, default=json_default) + '\n')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '23', str(args.output / 'review.mp4')], check=True)
    raw.unlink()
    print(args.output / 'review.mp4')


if __name__ == '__main__':
    main()
