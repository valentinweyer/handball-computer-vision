"""Native-resolution jersey-number eval set: does resolution move readability at all?

The RF-DETR training COCO export is 640x640 for every image, and readability barely
moved across that export's box-height bands (30% at <16px to 42% at >=24px -- see
runs/jersey_audit). That export's source clips can't be recovered at higher resolution
(it's a Roboflow Universe dataset, not ours), except for `data/raw/FelixClaar.mp4`,
which happens to be native 1080p and already the best-reading clip in the 640x640
export. This builds an independent eval sample directly from that raw video, at full
resolution, to test whether resolution is the ceiling or whether blur/angle/occlusion
caps readability regardless of pixel count.

Not a paired comparison: the Roboflow "FelixClaar_mp4-NNNN" frame numbering does not
line up with this video's own frame indices (checked directly -- same broadcast, several
seconds apart at matching indices), so this is a fresh, independent sample rather than a
same-instant resolution pair. It answers "does resolution matter here at all", not
"how much would this exact number's crop improve".

Every candidate is genuinely new -- there are no existing labels for this raw video to
match against, so (unlike generate_jersey_candidates.py) there is no recovered/missed
distinction. One pass: sample frames, SAM3 propose, render crops in the same geometry
`label_jersey_numbers.py` already serves, then Qwen-read every crop. Human review still
happens in that same reviewer, unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
from PIL import Image

from scripts.build_jersey_audit_set import render_crop_pair
from scripts.benchmark_qwen_jersey_ocr import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    VARIANTS,
    parse_prediction,
    request_prediction,
)
from scripts.generate_jersey_candidates import _import_sam3, propose_for_image
from scripts.label_jersey_numbers import utc_now, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "data/raw/FelixClaar.mp4"
DEFAULT_OUTPUT_DIR = ROOT / "runs/jersey_native_eval"
QWEN_VARIANT = "context"


def sample_frame_indices(frame_count: int, stride: int) -> list[int]:
    return list(range(0, frame_count, stride))


def propose_candidates(video_path: Path, frame_indices: list[int], device: str) -> list[dict]:
    Sam3Processor, build_sam3_image_model = _import_sam3()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    print(f"loading SAM3 image model on {device} ...", flush=True)
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)

    candidates = []
    started = time.monotonic()
    try:
        for position, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            proposals = propose_for_image(processor, image, referee_boxes=[])
            for proposal in proposals:
                candidates.append({"frame_index": frame_index, **proposal})

            print(
                f"[{position}/{len(frame_indices)}] frame {frame_index}: "
                f"{len(proposals)} proposals ({time.monotonic() - started:.1f}s elapsed)",
                flush=True,
            )
    finally:
        capture.release()
    return candidates


def render_dataset(video_path: Path, candidates: list[dict], output_dir: Path) -> dict:
    crops_dir, contexts_dir = output_dir / "crops", output_dir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    rendered = []
    try:
        by_frame: dict[int, list[dict]] = {}
        for candidate in candidates:
            by_frame.setdefault(candidate["frame_index"], []).append(candidate)

        index = 0
        for frame_index in sorted(by_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {frame_index} from {video_path}")

            for candidate in by_frame[frame_index]:
                crop_info = render_crop_pair(frame_bgr, candidate["box"], index, crops_dir, contexts_dir)
                rendered.append({
                    "index": index,
                    "frame": frame_index,
                    "box": candidate["box"],
                    "predictions": {},
                    "sample_id": f"{video_path.stem}-f{frame_index:04d}-{index:04d}",
                    **crop_info,
                    "source_kind": "sam3_native",
                    "sam3_confidence": candidate["confidence"],
                })
                index += 1
    finally:
        capture.release()

    return {
        "schema_version": 1,
        "video": str(video_path),
        "source_kind": "native_resolution_eval",
        "created_at": utc_now(),
        "samples": rendered,
    }


def run_qwen(dataset: dict, output_dir: Path, base_url: str, model: str, max_tokens: int) -> dict:
    variant = VARIANTS[QWEN_VARIANT]
    pending = [s for s in dataset["samples"] if "qwen_context" not in s["predictions"]]
    started = time.monotonic()
    for position, sample in enumerate(pending, start=1):
        image_path = output_dir / sample[variant["path_key"]]
        content, _usage = request_prediction(
            base_url, model, image_path, variant["prompt"], timeout=120, max_tokens=max_tokens,
        )
        prediction, parse_status = parse_prediction(content)
        sample["predictions"]["qwen_context"] = prediction
        sample["predictions"]["qwen_parse_status"] = parse_status
        write_json_atomic(output_dir / "dataset.json", dataset)
        print(
            f"[{position}/{len(pending)}] sample={sample['index']} "
            f"qwen={prediction or '<abstain>'} parse={parse_status} "
            f"({time.monotonic() - started:.1f}s elapsed)",
            flush=True,
        )
    dataset["qwen_model"] = model
    dataset["qwen_variant"] = QWEN_VARIANT
    dataset["verified_at"] = utc_now()
    write_json_atomic(output_dir / "dataset.json", dataset)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-tokens", type=int, default=192,
        help="Qwen completion budget. Must clear the model's internal reasoning trace: a "
        "reasoning-enabled llama-server spends ~77 tokens thinking before emitting the "
        "digits, and truncating mid-trace yields empty content that parses as "
        "'malformed' -- silently looking like a total abstention rather than a "
        "misconfiguration. 16 is enough only when reasoning is effectively disabled.",
    )
    parser.add_argument(
        "--skip-qwen", action="store_true",
        help="build crops and stop -- useful if the Qwen server isn't up yet",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="re-run SAM3 and re-render crops even if dataset.json already exists, "
        "discarding any Qwen predictions in it. Off by default so a re-run resumes the "
        "Qwen pass instead of silently destroying the predictions run_qwen skips on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    dataset_path = output_dir / "dataset.json"
    if dataset_path.is_file() and not args.rebuild:
        # Reuse rather than regenerate: re-rendering would reset every sample's
        # predictions, defeating run_qwen's resume and (worse) discarding a partially
        # complete Qwen pass. Pass --rebuild to force a fresh proposal pass.
        dataset = json.loads(dataset_path.read_text())
        done = sum("qwen_context" in s["predictions"] for s in dataset["samples"])
        print(
            f"reusing {dataset_path} ({len(dataset['samples'])} samples, "
            f"{done} already have Qwen predictions; pass --rebuild to regenerate)"
        )
    else:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise FileNotFoundError(f"could not open video: {args.video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        frame_indices = sample_frame_indices(frame_count, args.frame_stride)
        print(f"sampling {len(frame_indices)} of {frame_count} frames (stride {args.frame_stride})")

        candidates = propose_candidates(args.video, frame_indices, args.device)
        dataset = render_dataset(args.video, candidates, output_dir)
        write_json_atomic(dataset_path, dataset)
        print(f"\n{len(dataset['samples'])} candidates -> {dataset_path}")

    if args.skip_qwen:
        return
    dataset = run_qwen(dataset, output_dir, args.base_url, args.model, args.max_tokens)
    non_abstain = sum(bool(s["predictions"]["qwen_context"]) for s in dataset["samples"])
    malformed = sum(
        s["predictions"].get("qwen_parse_status") == "malformed" for s in dataset["samples"]
    )
    print(f"qwen: {non_abstain}/{len(dataset['samples'])} non-abstain, {malformed} malformed")
    if malformed > len(dataset["samples"]) * 0.1:
        print(
            f"WARNING: {malformed} malformed responses -- likely --max-tokens "
            f"({args.max_tokens}) truncating the model's reasoning trace, not genuine "
            f"abstention. Check a raw response before trusting these numbers.",
            flush=True,
        )


if __name__ == "__main__":
    main()
