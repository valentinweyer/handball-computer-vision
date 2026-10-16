"""Render saved benchmark results without loading any tracking model."""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 960, 540
HEADER, FOOTER = 58, 38


def color(oid):
    rgb = colorsys.hsv_to_rgb((int(oid) * 0.61803398875) % 1, 0.8, 1.0)
    return tuple(int(x * 255) for x in rgb[::-1])


def label(image, text, point, scale=0.58, ink=(255, 255, 255)):
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)


def boxes_for(run):
    with np.load(run / 'replay.npz') as data:
        result = {}
        for fid, oid, box in zip(data['frame_index'], data['tracker_id'], data['boxes']):
            result.setdefault(int(fid), []).append((int(oid), box.copy()))
    return result


def panel(frame, run, fid, mode, boxes):
    image = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    sx, sy = WIDTH / frame.shape[1], HEIGHT / frame.shape[0]
    annotations = []
    if mode == 'masks':
        with np.load(run / 'masks' / f'{fid:05d}.npz') as data:
            masks = np.unpackbits(data['masks'], axis=-1, count=int(data['width']))
            for oid, mask in zip(data['ids'], masks):
                mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
                active = mask.astype(bool)
                if not active.any():
                    continue
                ink = color(oid)
                image[active] = (0.6 * image[active] + 0.4 * np.asarray(ink)).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, ink, 1, cv2.LINE_AA)
                ys, xs = np.where(active)
                annotations.append((int(oid), int(np.median(xs)), max(18, int(ys.min()) - 5)))
    else:
        for oid, box in boxes.get(fid, []):
            x1, y1, x2, y2 = np.rint(box * [sx, sy, sx, sy]).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color(oid), 2, cv2.LINE_AA)
            annotations.append((oid, x1, max(18, y1 - 5)))
    for oid, x, y in annotations:
        label(image, f'ID {oid}', (max(0, min(WIDTH-72, x)), y), ink=color(oid))
    return image


def render(output, name, stem, arms, first, last, mode, slow=False, note=''):
    frame_dir = ROOT / 'data/cache/frames' / stem
    cap = cv2.VideoCapture(str(ROOT / 'data/raw' / f'{stem}.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert fps > 0
    playback = fps / 2 if slow else fps
    width, height = WIDTH * len(arms), HEIGHT + HEADER + FOOTER
    target = output / f'{name}.mp4'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-n', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', str(playback), '-i', '-', '-an', '-c:v', 'libx264',
               '-threads', '2', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(target)]
    replays = [boxes_for(run) if mode == 'boxes' else {} for title, run in arms]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for fid in range(first, last + 1):
                frame = cv2.imread(str(frame_dir / f'{fid:05d}.jpg'))
                if frame is None:
                    raise FileNotFoundError(frame_dir / f'{fid:05d}.jpg')
                canvas = np.full((height, width, 3), 20, dtype=np.uint8)
                for index, ((title, run), boxes) in enumerate(zip(arms, replays)):
                    x = index * WIDTH
                    canvas[HEADER:HEADER+HEIGHT, x:x+WIDTH] = panel(frame, run, fid, mode, boxes)
                    label(canvas, title, (x+15, 25), scale=0.68)
                    label(canvas, note or 'Saved tracker boxes and original tracker IDs', (x+15, 48), scale=0.48)
                speed = '0.5x slow motion' if slow else '1x source speed'
                label(canvas, f'{stem} | frame {fid} | {speed} | playback speed is not inference speed',
                      (15, height-13), scale=0.5)
                encoder.stdin.write(canvas.tobytes())
                if fid == (first+last)//2:
                    cv2.imwrite(str(output / f'{name}.jpg'), canvas)
        finally:
            encoder.stdin.close()
        assert encoder.wait() == 0, target
    capture = cv2.VideoCapture(str(target))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, sample = capture.read()
    capture.release()
    assert ok and count == last-first+1 and sample.shape[:2] == (height, width), target
    print(f'{target}: {count} frames, {count/playback:.2f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    full = ROOT / 'runs/sam31_full_20261015'
    short = ROOT / 'runs/sam31_quality_20261015'
    render(args.output, 'felix_sam2_vs_sam31', 'FelixClaar',
           [('SAM2.1-L', full/'felix_sam2_baseline'), ('SAM3.1 multiplex', full/'felix_sam3')],
           0, 248, 'boxes')
    render(args.output, 'han_sam2_vs_sam31', 'Han-Ber4_cached',
           [('SAM2.1-L', full/'han_sam2_baseline'), ('SAM3.1 multiplex', full/'han_sam3')],
           0, 198, 'boxes')
    render(args.output, 'felix_masks_sam2_vs_sam31', 'FelixClaar',
           [('SAM2.1-L masks', short/'sam2_felix141'), ('SAM3.1 multiplex masks', short/'sam3_felix141')],
           90, 140, 'masks', slow=True, note='Saved predicted masks | overlap excerpt, frames 90-140')
    render(args.output, 'bhc_sam31_masks_excerpt', 'BHC-FAG_window_cached',
           [('SAM3.1 multiplex masks | BHC excerpt', full/'bhc_sam3')],
           90, 140, 'masks', slow=True, note='INCOMPLETE benchmark: later failed at frame 370 | excerpt only')


if __name__ == '__main__':
    main()
