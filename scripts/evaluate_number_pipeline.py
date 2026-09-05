"""Measure end-to-end jersey-number accuracy PER TRACKED PLAYER, not per crop.

Per-crop reader accuracy is the wrong metric for this pipeline: numbers are read every
N frames and voted over a tracklet, so a mediocre reader can still yield a correct
player number as long as its errors are unsystematic. Nothing so far has measured what
actually comes out the end. This does.

    cached player detections -> McByte tracking -> IdentityManager (stable player_id)
        -> cached RF-DETR number boxes (see cache_number_detections.py)
        -> mask-IoS match number->player -> reader -> NumberVoter per player_id

Two readers are selectable so the output is a *curve*, not a point: EasyOCR is the
current production reader (~35% per crop) and Qwen is the strongest available (~70%).
Running both shows how much tracklet accuracy is lost per point of per-crop accuracy,
which is the spec for any smaller model trained later -- it says whether 60% or 85%
per-crop is needed to hit a target, before anything is built.

Ground truth is per player, not per crop: after a run, `crops/player_<id>/` holds sample
crops for each tracked identity, so a human can say "player 4 is number 16" in a few
minutes. Pass that mapping back with --truth to get scored.

    # 1. cache number boxes (sibling venv -- rfdetr isn't installed here)
    /home/.../rfdetr-handball-finetune/.venv/bin/python3 scripts/cache_number_detections.py \
        data/raw/FelixClaar.mp4 --output outputs/number_cache/.FelixClaar_numbers_v1.npz
    # 2. run the pipeline (this venv)
    python -m scripts.evaluate_number_pipeline data/raw/FelixClaar.mp4 \
        --detections outputs/team_comparison/.FelixClaar_detections_v1.npz \
        --number-detections outputs/number_cache/.FelixClaar_numbers_v1.npz \
        --team-model outputs/team_dataset/felix_runtime_team_model.json \
        --reader qwen --output-dir runs/number_pipeline/felix_qwen
    # 3. label runs/.../crops/player_*/ then score
    python -m scripts.evaluate_number_pipeline ... --truth truth.json --score-only
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
import mimetypes
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import supervision as sv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from trackers import McByteMaskConfig, McByteTracker

from handball_cv.jersey.identity import (
    OCR_EVERY_N_FRAMES,
    NumberVoter,
    _Votes,
    match_numbers_to_players,
    read_numbers,
)
from handball_cv.teams.model import TeamModel
from handball_cv.tracking.identity import TEAM_SWITCH_OBSERVATIONS, IdentityManager
from scripts.benchmark_qwen_jersey_ocr import (
    VARIANTS,
    parse_prediction,
    request_prediction,
)
from scripts.render_full_pipeline import (
    GOALKEEPER_CLASS_ID,
    readable_number_crop,
    tracklet_masks,
    unique_pairs,
)
from scripts.label_jersey_numbers import clipped_box, utc_now, write_json_atomic
from scripts.render_raw_team_classification import frame_detections, load_detection_cache


ROOT = Path(__file__).resolve().parents[1]
MAX_CROPS_PER_PLAYER = 6  # enough for a human to identify the player, few enough to skim


def read_with_easyocr(ocr_model, frame_rgb: np.ndarray, boxes: np.ndarray) -> list[str]:
    return read_numbers(ocr_model, frame_rgb, boxes)


def read_with_qwen(
    frame_bgr: np.ndarray, boxes: np.ndarray, scratch: Path, base_url: str,
    model: str, max_tokens: int,
) -> list[str]:
    """Render each box as the padded red-box context crop Qwen measured best on, and read it.

    Uses the identical crop geometry as label_jersey_numbers/build_jersey_audit_set so
    the accuracy measured here is comparable to the per-crop numbers from those runs.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    height, width = frame_bgr.shape[:2]
    out = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = clipped_box(list(map(float, box)), width, height)
        if x2 <= x1 or y2 <= y1:
            out.append("")
            continue
        pad = max(24, round(max(x2 - x1, y2 - y1) * 0.8))
        cx1, cy1, cx2, cy2 = clipped_box(list(map(float, box)), width, height, pad)
        context = frame_bgr[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(
            context, (x1 - cx1, y1 - cy1), (x2 - cx1 - 1, y2 - cy1 - 1),
            (0, 0, 255), max(1, round(max(context.shape[:2]) / 180)),
        )
        path = scratch / f"ctx_{i:03d}.jpg"
        cv2.imwrite(str(path), context)
        content, _ = request_prediction(
            base_url, model, path, VARIANTS["context"]["prompt"],
            timeout=120, max_tokens=max_tokens,
        )
        prediction, _status = parse_prediction(content)
        out.append(prediction)
    return out


def process_numbers_for_frame(
    frame_index: int,
    frame_bgr: np.ndarray,
    frame_rgb: np.ndarray,
    player_ids: np.ndarray,
    boxes_for_crop: np.ndarray,
    masks: list,
    number_cache: dict,
    args: argparse.Namespace,
    ocr_model,
    output_dir: Path,
    voter: "NumberVoter",
    reads_per_player: dict,
    crops_saved: dict,
    crops_dir: Path,
) -> tuple[bool, int]:
    """Match this frame's cached number boxes to players, read, and vote.

    Shared by both trackers -- everything downstream of "here is one frame's
    player_ids/boxes/masks" is identical regardless of which tracker produced
    them. `masks[i]` may be None (a player McByte's mask manager has no mask
    for this frame); SAM2 never leaves a row None, so its caller passes a
    plain list of arrays and every row survives the filter below unchanged.

    Returns (had_number_detections_this_frame, reads_added_this_frame); the
    caller keeps its own frames_with_numbers/total_reads running totals since
    each tracker's checkpoint dict already owns those independently.
    """
    masked_rows = [i for i, mask in enumerate(masks) if mask is not None]
    if not masked_rows:
        return False, 0
    number_xyxy = frame_detections(number_cache, frame_index).xyxy
    if not len(number_xyxy):
        return False, 0

    stacked = np.stack([masks[i] for i in masked_rows])
    pairs = unique_pairs(
        match_numbers_to_players(stacked, number_xyxy, frame_bgr.shape)
    )
    pairs = [
        (p, n) for p, n in pairs
        if readable_number_crop(frame_rgb, number_xyxy[n])
    ]
    if not pairs:
        return True, 0

    wanted = sorted({n for _, n in pairs})
    boxes = number_xyxy[wanted]
    if args.reader == "easyocr":
        texts = read_with_easyocr(ocr_model, frame_rgb, boxes)
    else:
        texts = read_with_qwen(
            frame_bgr, boxes, output_dir / "_scratch",
            args.base_url, args.model, args.max_tokens,
        )
    text_by_row = dict(zip(wanted, texts))

    reads_added = 0
    for local_row, number_row in pairs:
        player_id = int(player_ids[masked_rows[local_row]])
        raw = text_by_row.get(number_row, "")
        if raw:
            reads_added += 1
            voter.observe(player_id, raw)
            reads_per_player[player_id].append(raw)
        # Save a few crops per player so a human can identify who this is. Crop the
        # PLAYER box, not the number box, with the number outlined: the ground-truth
        # question is "what number does this player wear", which needs the whole
        # torso -- a tight number crop is exactly the view that is too ambiguous to
        # answer it, and would just reproduce the reader's own uncertainty.
        if crops_saved[player_id] < MAX_CROPS_PER_PLAYER:
            pdir = crops_dir / f"player_{player_id:03d}"
            pdir.mkdir(parents=True, exist_ok=True)
            px1, py1, px2, py2 = clipped_box(
                list(map(float, boxes_for_crop[masked_rows[local_row]])),
                frame_bgr.shape[1], frame_bgr.shape[0], pad=20,
            )
            if px2 > px1 and py2 > py1:
                view = frame_bgr[py1:py2, px1:px2].copy()
                nx1, ny1, nx2, ny2 = clipped_box(
                    list(map(float, number_xyxy[number_row])),
                    frame_bgr.shape[1], frame_bgr.shape[0],
                )
                cv2.rectangle(
                    view, (nx1 - px1, ny1 - py1), (nx2 - px1 - 1, ny2 - py1 - 1),
                    (0, 0, 255), 1,
                )
                scale = max(1, round(180 / max(view.shape[:2])))
                if scale > 1:
                    view = cv2.resize(
                        view, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                    )
                cv2.imwrite(
                    str(pdir / f"f{frame_index:05d}_read-{raw or 'none'}.jpg"), view
                )
                crops_saved[player_id] += 1
    return True, reads_added


def finalize_report(
    args: argparse.Namespace, source: Path, checkpoint_path: Path,
    reads_per_player: dict, voter: "NumberVoter",
    frames_with_numbers: int, total_reads: int,
) -> dict:
    """Shared report shape for both trackers: build it, write it, drop the checkpoint."""
    checkpoint_path.unlink(missing_ok=True)
    players = {}
    for player_id, reads in sorted(reads_per_player.items()):
        number, votes, margin = voter.best(player_id)
        players[str(player_id)] = {
            "voted_number": number,
            "votes": votes,
            "margin": round(margin, 3),
            "n_reads": len(reads),
            "read_distribution": dict(sorted(
                ((v, reads.count(v)) for v in set(reads)), key=lambda kv: -kv[1]
            )),
        }

    report = {
        "schema_version": 1,
        "video": str(source),
        "tracker": args.tracker,
        "reader": args.reader,
        "ocr_every": args.ocr_every,
        "created_at": utc_now(),
        "frames_with_number_detections": frames_with_numbers,
        "total_reads": total_reads,
        "tracked_players_with_reads": len(players),
        "players": players,
    }
    write_json_atomic(args.output_dir.resolve() / "report.json", report)
    return report


def run_mcbyte(args: argparse.Namespace) -> dict:
    source = args.video.resolve()
    cache = load_detection_cache(args.detections)
    number_cache = load_detection_cache(args.number_detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    info = sv.VideoInfo.from_video_path(str(source))
    output_dir = args.output_dir.resolve()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    ocr_model = None
    if args.reader == "easyocr":
        import easyocr
        ocr_model = easyocr.Reader(
            ["en"], gpu=args.device != "cpu", detector=False, verbose=False
        )

    enable_masks = not args.no_masks
    if enable_masks and GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    tracker = McByteTracker(
        frame_rate=info.fps,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        enable_mask_manager=enable_masks,
        mask_config=McByteMaskConfig(device=args.device) if enable_masks else None,
        minimum_mask_average_confidence=0.6,
        minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )
    identity = IdentityManager(
        team_model,
        goalkeeper_class_id=GOALKEEPER_CLASS_ID,
        team_switch_observations=TEAM_SWITCH_OBSERVATIONS,
    )
    voter = NumberVoter(
        min_votes=args.min_votes, min_margin=args.min_margin,
        min_promote_votes=args.min_promote_votes,
    )

    # Checkpointed every frame that yields a read: a Qwen call every ~5th frame makes a
    # full pass ~20 minutes on a 999-frame clip, and this session has already lost two
    # in-progress runs to the (externally managed) model server dying mid-run. Losing
    # nothing more than the current frame on a crash is worth the periodic write.
    checkpoint_path = output_dir / "_checkpoint.json"
    reads_per_player: dict[int, list[str]] = defaultdict(list)
    crops_saved: dict[int, int] = defaultdict(int)
    frames_with_numbers = total_reads = 0
    start_frame = 0
    if checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text())
        if saved.get("video") == str(source) and saved.get("reader") == args.reader:
            reads_per_player = defaultdict(list, {int(k): v for k, v in saved["reads_per_player"].items()})
            frames_with_numbers = saved["frames_with_numbers"]
            total_reads = saved["total_reads"]
            start_frame = saved["next_frame"]
            for player_id, reads in reads_per_player.items():
                for raw in reads:
                    voter.observe(player_id, raw)
            print(f"resuming from checkpoint at frame {start_frame} ({total_reads} reads so far)", flush=True)

    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(source)),
        total=info.total_frames, desc=f"number pipeline ({args.reader})",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = frame_detections(cache, frame_index)
        tracked = tracker.update(detections, frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        player_ids = identity.update(frame_index, frame_rgb, tracked)
        alive = getattr(tracker, "tracked_objects", sv.Detections.empty())
        alive_ids = (
            alive.tracker_id if getattr(alive, "tracker_id", None) is not None
            else np.empty(0, dtype=int)
        )
        identity.retire_missing(frame_index, alive_ids)

        # Tracking state (tracker_id/player_id continuity) can't be resumed from a
        # checkpoint, so every frame up to start_frame still runs tracker.update()/
        # identity.update() above -- only the reader calls, the actually expensive
        # part, are skipped for frames already checkpointed.
        if frame_index < start_frame:
            continue
        if frame_index % args.ocr_every or not len(tracked):
            continue
        # Masks come from McByte's SAM/Cutie mask output keyed by tracker_id, not from
        # `tracked.mask` (which this tracker leaves unset) -- same source as
        # render_full_pipeline, so number->player matching behaves identically.
        masks = (
            tracklet_masks(getattr(tracker, "_last_mask_output", None), tracked.tracker_id)
            if enable_masks else [None] * len(tracked)
        )
        had_numbers, reads_added = process_numbers_for_frame(
            frame_index, frame_bgr, frame_rgb, player_ids, tracked.xyxy, masks,
            number_cache, args, ocr_model, output_dir, voter,
            reads_per_player, crops_saved, crops_dir,
        )
        if not had_numbers:
            continue
        frames_with_numbers += 1
        total_reads += reads_added

        write_json_atomic(checkpoint_path, {
            "video": str(source),
            "reader": args.reader,
            "next_frame": frame_index + 1,
            "frames_with_numbers": frames_with_numbers,
            "total_reads": total_reads,
            "reads_per_player": reads_per_player,
        })

    return finalize_report(
        args, source, checkpoint_path, reads_per_player, voter,
        frames_with_numbers, total_reads,
    )


def run_sam2(args: argparse.Namespace) -> dict:
    """Same per-player number pipeline, driven by SAM2 + periodic reprompting.

    Uses `handball_cv.tracking.sam2_driver.drive_sam2`, the shared loop
    `scripts.run_sam2_reprompt_tracker` also drives -- both need the identical
    predictor/TrackManager sequencing, only what they do with each frame
    differs. `TrackManager`'s registry `player_id` IS the SAM2 `obj_id`
    directly (see `sam2_manager`'s module docstring): no separate identity
    translation layer exists here the way McByte needs one.

    Every propagated frame has a mask by construction (never None the way
    McByte's mask-manager output can be), so number matching never has to
    filter rows for missing masks -- `process_numbers_for_frame` still runs
    that filter, it just never drops anything here.

    True resumability -- skipping SAM2 propagation itself for an
    already-processed prefix -- is not implemented: the predictor session
    holds no state across process restarts. A resumed run still re-propagates
    from frame 0, but (matching run_mcbyte's own comment on this) skips the
    reader calls for frames already checkpointed, which is the actually
    expensive and failure-prone part.
    """
    sam2_upstream = Path(os.getenv("SAM2_UPSTREAM_DIR", ROOT / "sam2-upstream"))
    if not sam2_upstream.is_dir():
        raise FileNotFoundError(
            "Needs an external facebookresearch/sam2 checkout. Set SAM2_UPSTREAM_DIR "
            "to its path before running --tracker sam2."
        )
    sys.path.insert(0, str(sam2_upstream))
    from handball_cv.tracking.sam2_driver import drive_sam2

    source = args.video.resolve()
    number_cache = load_detection_cache(args.number_detections)
    cache = load_detection_cache(args.detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    output_dir = args.output_dir.resolve()
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    ocr_model = None
    if args.reader == "easyocr":
        import easyocr
        ocr_model = easyocr.Reader(
            ["en"], gpu=args.device != "cpu", detector=False, verbose=False
        )

    voter = NumberVoter(
        min_votes=args.min_votes, min_margin=args.min_margin,
        min_promote_votes=args.min_promote_votes,
    )
    checkpoint_path = output_dir / "_checkpoint.json"
    reads_per_player: dict[int, list[str]] = defaultdict(list)
    crops_saved: dict[int, int] = defaultdict(int)
    frames_with_numbers = total_reads = 0
    start_frame = 0
    if checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text())
        if saved.get("video") == str(source) and saved.get("reader") == args.reader:
            reads_per_player = defaultdict(list, {int(k): v for k, v in saved["reads_per_player"].items()})
            frames_with_numbers = saved["frames_with_numbers"]
            total_reads = saved["total_reads"]
            start_frame = saved["next_frame"]
            for player_id, reads in reads_per_player.items():
                for raw in reads:
                    voter.observe(player_id, raw)
            print(f"resuming from checkpoint at frame {start_frame} ({total_reads} reads so far)", flush=True)

    frame_cache_dir = args.frame_cache_dir or (ROOT / "data/cache/frames" / source.stem)
    _track_manager, _seed_boxes, frames = drive_sam2(
        source, lambda idx: frame_detections(cache, idx), team_model,
        checkpoint=args.checkpoint, check_every=args.check_every,
        frame_cache_dir=frame_cache_dir, max_frames=args.max_frames,
        desc=f"number pipeline sam2 ({args.reader})",
    )

    for result in frames:
        if result.frame_idx < start_frame:
            continue
        if result.frame_idx % args.ocr_every or not len(result.player_ids):
            continue
        frame_rgb = result.read_frame()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        had_numbers, reads_added = process_numbers_for_frame(
            result.frame_idx, frame_bgr, frame_rgb, result.player_ids, result.boxes,
            list(result.masks), number_cache, args, ocr_model, output_dir, voter,
            reads_per_player, crops_saved, crops_dir,
        )
        if not had_numbers:
            continue
        frames_with_numbers += 1
        total_reads += reads_added

        write_json_atomic(checkpoint_path, {
            "video": str(source),
            "reader": args.reader,
            "next_frame": result.frame_idx + 1,
            "frames_with_numbers": frames_with_numbers,
            "total_reads": total_reads,
            "reads_per_player": reads_per_player,
        })

    return finalize_report(
        args, source, checkpoint_path, reads_per_player, voter,
        frames_with_numbers, total_reads,
    )


def run(args: argparse.Namespace) -> dict:
    return run_sam2(args) if args.tracker == "sam2" else run_mcbyte(args)


def recompute_votes(report: dict, min_votes: int, min_margin: float, min_promote_votes: int) -> dict:
    """Rebuild voted_number/votes/margin from each player's stored read_distribution.

    read_distribution is NumberVoter's full sufficient statistic, so a report can be
    re-scored under a changed voting algorithm (e.g. the suffix-folding rule added
    after this repo's first Qwen run) without re-running the reader -- the reads
    themselves haven't changed, only how they're tallied. Without this, --score-only
    silently scores whatever voting logic was live at run time, not the current one.
    """
    for entry in report["players"].values():
        votes = _Votes(counts=dict(entry["read_distribution"]))
        number, count, margin = votes.best(min_promote_votes)
        entry["voted_number"] = number if (count >= min_votes and margin >= min_margin) else None
        entry["votes"] = count
        entry["margin"] = round(margin, 3)
    return report


def score(report: dict, truth_path: Path) -> dict:
    """Score voted numbers against a human `{player_id: number}` mapping."""
    truth = json.loads(truth_path.read_text())
    correct = wrong = no_verdict = unlabeled = 0
    rows = []
    for player_id, entry in report["players"].items():
        expected = truth.get(player_id)
        if expected is None:
            unlabeled += 1
            continue
        got = entry["voted_number"]
        if got is None:
            outcome = "NO VERDICT"; no_verdict += 1
        elif str(got) == str(expected):
            outcome = "CORRECT"; correct += 1
        else:
            outcome = "WRONG"; wrong += 1
        rows.append((player_id, expected, got, entry["n_reads"], outcome))

    scored = correct + wrong + no_verdict
    print(f"{'player':<9}{'truth':<7}{'voted':<7}{'reads':>6}  outcome")
    for pid, exp, got, n, outcome in rows:
        print(f"{pid:<9}{str(exp):<7}{str(got):<7}{n:>6}  {outcome}")
    print()
    print(f"correct {correct}/{scored}" + (f" ({100*correct/scored:.0f}%)" if scored else ""))
    print(f"wrong {wrong}, no verdict {no_verdict}, unlabeled players skipped {unlabeled}")
    return {"correct": correct, "wrong": wrong, "no_verdict": no_verdict, "scored": scored}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--number-detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tracker", choices=("mcbyte", "sam2"), default="mcbyte")
    parser.add_argument("--reader", choices=("easyocr", "qwen"), default="easyocr")
    parser.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAMES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-masks", action="store_true",
                         help="mcbyte only; sam2 always has masks")
    parser.add_argument("--checkpoint", default=str(
        ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
    ), help="sam2 only")
    parser.add_argument("--check-every", type=int, default=10, help="sam2 only")
    parser.add_argument("--frame-cache-dir", type=Path, default=None, help="sam2 only")
    parser.add_argument("--max-frames", type=int, default=None, help="sam2 only; smoke-test truncation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", default="qwen38")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--truth", type=Path,
        help='JSON {"<player_id>": "<number>"} to score the run against',
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="skip the pipeline and score an existing report.json in --output-dir",
    )
    parser.add_argument(
        "--recompute-votes", action="store_true",
        help="with --score-only, rebuild voted_number from each player's stored "
        "read_distribution using the CURRENT NumberVoter code before scoring, rather "
        "than trusting whatever voting logic was live when the report was generated. "
        "Costs nothing (no reader calls) since the reads are already stored.",
    )
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--min-margin", type=float, default=0.2)
    parser.add_argument("--min-promote-votes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_only:
        report = json.loads((args.output_dir / "report.json").read_text())
        if args.recompute_votes:
            before = {pid: e["voted_number"] for pid, e in report["players"].items()}
            report = recompute_votes(
                report, args.min_votes, args.min_margin, args.min_promote_votes
            )
            for pid, e in report["players"].items():
                if before[pid] != e["voted_number"]:
                    print(f"  recomputed player {pid}: {before[pid]!r} -> {e['voted_number']!r}")
    else:
        report = run(args)
        print(json.dumps({k: v for k, v in report.items() if k != "players"}, indent=2))
        print(f"\n{len(report['players'])} tracked players with reads:")
        for pid, e in report["players"].items():
            print(
                f"  player {pid:>4}: voted={str(e['voted_number']):<6} "
                f"votes={e['votes']:<3} margin={e['margin']:<5} {e['read_distribution']}"
            )
        print(f"\ncrops for labeling: {args.output_dir/'crops'}")
    if args.truth:
        score(report, args.truth)


if __name__ == "__main__":
    main()
