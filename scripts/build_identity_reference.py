"""Build per-frame identity ground truth by seed-once mask propagation.

Produces the same artefact as the existing `source/.Han-Ber4_sam2_masks`
reference -- one `label` map per frame, where pixel values are persistent
identity ids -- but for any clip with cached detections.

Why this is a usable reference for scoring trackers: identities are assigned
once, on the seed frame, and thereafter carried purely by mask propagation.
Nothing re-associates them to detections, so the result is independent of the
box-association logic every tracker under test is built on. It is not
independent of SAM/Cutie, so it must still be verified by eye before use --
`scripts/label_tracklet_identity.py`-style sheets, exactly as the Han-Ber
reference was verified (14/14 clean).

Known limits, inherited from seeding once:
  * players who enter after the seed frame are never represented
  * a player who leaves and returns may not be recovered
  * propagation drift is possible and is precisely what verification catches

Usage:
    python -m scripts.build_identity_reference data/raw/FelixClaar.mp4 \\
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \\
        --out source/.FelixClaar_ref_masks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from trackers.core.mcbyte.masks.base import TrackletSnapshot
from trackers.core.mcbyte.masks.cutie import CutieMaskPropagator
from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator

from scripts.render_raw_team_classification import (
    frame_detections,
    load_detection_cache,
)

ROOT = Path(__file__).resolve().parents[1]
MIN_MASK_PIXELS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--seed-frame", type=int, default=0,
        help="frame whose detections define the identity set",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sam-checkpoint", type=Path, default=ROOT / "models/sam/sam_vit_b_01ec64.pth")
    parser.add_argument(
        "--cutie-weights", type=Path, default=ROOT / "models/cutie/cutie-base-mega.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = load_detection_cache(args.detections)
    info = sv.VideoInfo.from_video_path(str(args.video))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    generator = SAMBoxMaskGenerator(
        checkpoint_path=args.sam_checkpoint, model_type="vit_b", device=args.device,
    )
    propagator = CutieMaskPropagator(
        weights_path=args.cutie_weights, device=args.device,
    )

    seed = frame_detections(cache, args.seed_frame)
    if not len(seed):
        raise SystemExit(f"no detections on seed frame {args.seed_frame}")
    # Identity ids are 1-based so 0 can mean background in the label map.
    snapshots = [
        TrackletSnapshot(tracker_id=index + 1, xyxy=np.asarray(box, dtype=float))
        for index, box in enumerate(seed.xyxy)
    ]
    print(f"seeding {len(snapshots)} identities from frame {args.seed_frame}")

    written = 0
    seeded = False
    manifest = {"video": str(args.video), "seed_frame": args.seed_frame,
                "identities": len(snapshots), "frames": {}}
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(args.video)),
        total=info.total_frames, desc="propagate",
    )):
        if frame_index < args.seed_frame:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not seeded:
            mask_output = generator.generate(frame_rgb, snapshots)
            propagator.initialize(frame_rgb, mask_output)
            seeded = True
        else:
            mask_output = propagator.propagate(frame_rgb)
        if mask_output is None or mask_output.masks is None:
            continue

        masks = np.asarray(mask_output.masks, dtype=bool)
        label = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        present = []
        # Paint smallest-last so a small mask is not buried by a large one it
        # overlaps; Cutie masks are mutually exclusive, this only guards ties.
        order = sorted(
            mask_output.tracklet_mask_dict.items(),
            key=lambda kv: -int(masks[int(kv[1])].sum())
            if 0 <= int(kv[1]) < len(masks) else 0,
        )
        for identity, row in order:
            row = int(row)
            if not (0 <= row < len(masks)):
                continue
            mask = masks[row]
            if mask.sum() < MIN_MASK_PIXELS:
                continue
            label[mask] = int(identity)
            present.append(int(identity))
        np.savez_compressed(args.out / f"{frame_index:05d}.npz", label=label)
        manifest["frames"][frame_index] = sorted(present)
        written += 1

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = {}
    for present in manifest["frames"].values():
        for identity in present:
            counts[identity] = counts.get(identity, 0) + 1
    print(f"\nwrote {written} label maps to {args.out}")
    print(f"{'id':>3} {'frames':>7}")
    for identity in sorted(counts):
        print(f"{identity:>3} {counts[identity]:>7}")
    print("\nVERIFY BEFORE USE: render sheets and check each id follows one player.")


if __name__ == "__main__":
    main()
