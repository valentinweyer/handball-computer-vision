"""The track stage: SAM2 tracking with detector reprompting, re-ID and jersey numbers.

Reads a video with its cached detections and per-video team model, and writes
the run summary, the reader's output, and per-frame geometry that the bundle
stage joins into tables. Run as a module, it is the default pipeline's track
stage; `track_sam2` is the same pass for callers that bring their own reader
or draw each frame.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import supervision as sv
import torch

from handball_cv.detection.cache import (
    GOALKEEPER_CLASS_ID, load_detection_cache, number_detections, person_detections,
)
from handball_cv.jersey.identity import (
    IDENTITY_BREAK_EVENTS,
    OCR_EVERY_N_FRAMES,
    NumberIdentityResolver,
    NumberVoter,
    match_numbers_to_players,
    read_numbers_parseq,
    read_weight,
    small_crop_read_ok,
)
from handball_cv.teams.model import (
    MIN_CROP_CONTRAST, MIN_CROP_SHARPNESS, MIN_CROP_SIDE, TeamModel, crop_quality,
)
from handball_cv.tracking.geometry_cache import GeometryCache, as_crop
from handball_cv.tracking.sam2_driver import CHECK_EVERY_DEFAULT
from handball_cv.visualization.overlay import number_label
from handball_cv.visualization.palette import color_index

ROOT = Path(__file__).resolve().parents[3]


def unique_pairs(pairs: list) -> list:
    """Greedy one-to-one filter over (player_row, number_row), best score first.

    `match_numbers_to_players` returns every pair above the IoS floor, so a
    player whose mask happens to cover an occluded neighbour's torso collects
    both numbers in one frame and votes for both. Mask IoS is already sorted
    best-first, so first-come wins.
    """
    seen_players, seen_numbers, kept = set(), set(), []
    for player_row, number_row in pairs:
        if player_row in seen_players or number_row in seen_numbers:
            continue
        seen_players.add(player_row)
        seen_numbers.add(number_row)
        kept.append((player_row, number_row))
    return kept


def _number_crop(frame_rgb: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = np.rint(box).astype(int)
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_rgb[y1:y2, x1:x2]


def readable_number_crop(frame_rgb: np.ndarray, box: np.ndarray) -> bool:
    """Whether a number crop carries enough signal to be worth OCR-ing."""
    crop = _number_crop(frame_rgb, box)
    return crop is not None and crop_quality(crop).accepted


def small_number_crop(frame_rgb: np.ndarray, box: np.ndarray) -> bool:
    """Refused by `readable_number_crop` for its area alone; every other check passes.

    The area gate is the torso crop's, in absolute pixels, so at 720p it drops
    most number crops: 37% pass it against 89% at 1080p. Reads from these crops
    count only through `small_crop_read_ok`.
    """
    crop = _number_crop(frame_rgb, box)
    if crop is None:
        return False
    quality = crop_quality(crop)
    return (not quality.accepted and min(quality.width, quality.height) >= MIN_CROP_SIDE
            and quality.contrast >= MIN_CROP_CONTRAST and quality.sharpness >= MIN_CROP_SHARPNESS)


def voted_labels(player_ids, color_lookup, voter, roster: dict, canonical=None) -> list:
    """`#number name` per player from a live voter, or `P<id>` where it abstains."""
    resolve = canonical or int
    labels = []
    for color, player_id in zip(color_lookup, player_ids):
        # Number evidence can fold two allocated ids into one player after the
        # fact (`NumberIdentityResolver`); the label follows the fold so the
        # overlay shows one person, not two boxes disagreeing about a number.
        player_id = resolve(int(player_id))
        number, _votes, _margin = voter.best(player_id)
        labels.append(number_label(number, int(color), roster, player_id))
    return labels


def reads_provenance_path(reads_cache_path: Path) -> Path:
    return reads_cache_path.with_name(reads_cache_path.stem + ".provenance.json")


def load_reads_cache(reads_cache_path: Path, provenance: dict) -> tuple[dict, bool]:
    """`(reads, provenance_known)` -- `{frame: {number_row: text}}`, empty if none yet.

    A cache from another reader is refused: replaying it would score this run's
    voting on that reader's text and label the run with this reader's name. A
    cache written before provenance was recorded is replayed as before, but
    stays unvouched, so reads added to it are not passed off as all one reader's.
    """
    if not reads_cache_path.is_file():
        return {}, True  # a record left beside no cache describes nothing
    reads = json.loads(reads_cache_path.read_text())
    record = reads_provenance_path(reads_cache_path)
    if not record.is_file():
        print(f"[reads] replaying cached reads from {reads_cache_path} "
              "(no provenance recorded; assuming it is this reader's)")
        return reads, False
    recorded = json.loads(record.read_text())
    if recorded != provenance:
        raise SystemExit(
            f"reads cache {reads_cache_path} was made by {recorded}, but this run reads "
            f"with {provenance}; pass another --reads-cache, or move that one aside"
        )
    print(f"[reads] replaying cached reads from {reads_cache_path}")
    return reads, True


def cached_read(known: dict, frame_idx: int, rows, read_rows) -> tuple[dict, dict, int]:
    """`(text_by_row, confidence_by_row, added)` for `rows`, reading only what `known` lacks.

    `known` is the reads cache, `{frame: {row: text}}`, and gains what is read.
    A row the cache lacks is read, not taken as a blank read: a cache made at
    another `--ocr-every`, or under masks that paired other number boxes, holds
    no answer for it. Only rows read now can carry a confidence, and only
    from a reader that reports one.
    """
    hit = known.setdefault(str(frame_idx), {})
    missing = [n for n in rows if str(n) not in hit]
    confidence_by_row: dict = {}
    if missing:
        confidences: list = []
        texts = read_rows(missing, confidences)
        hit.update({str(n): t for n, t in zip(missing, texts)})
        confidence_by_row = dict(zip(missing, confidences))
    return {n: hit[str(n)] for n in rows}, confidence_by_row, len(missing)


def identity_report(
    everyone, events, voter, number_reads, presence, teams, aliases,
) -> dict:
    """The half of the run summary both tracker paths produce identically.

    `presence` and `teams` are what make the number evidence checkable rather
    than merely suggestive. Two identities resolving to the same number are
    either one fragmented player or two different people, and only their teams
    and whether they were ever live in the same frame can tell those apart --
    a question the earlier output could not answer, leaving read co-occurrence
    as the only proxy, which is far too sparse to conclude from.
    """
    resolved = {p.player_id: voter.best(p.player_id)[0] for p in everyone}
    # Raw histograms make a poor number yield diagnosable: too few reads per
    # player, or enough reads that disagree and never clear the vote margin.
    raw_votes = {
        pid: dict(sorted(v.counts.items(), key=lambda kv: -kv[1]))
        for pid, v in voter._votes.items()
    }
    return {
        "numbers_resolved": {k: v for k, v in resolved.items() if v is not None},
        "number_votes_raw": raw_votes,
        "number_reads": number_reads,
        "player_teams": teams,
        "player_aliases": aliases,
        "frame_players": presence,
        "label_changes_total": sum(p.team_switches for p in everyone),
        # Counters alone cannot be investigated: a switch is only actionable with
        # its frame and player_id, so the events themselves travel with the run.
        "identity_events": [
            e for e in events
            if e["type"] in (*IDENTITY_BREAK_EVENTS, "team_switch", "link", "pause")
        ],
    }


def _json_default(value):
    """numpy scalars and arrays in the run's event logs, as plain JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def court_in_view_fn(path: Path | None):
    """`frame -> bool` from a saved per-frame flag; frames past its end count as in view."""
    if path is None:
        return None
    flags = np.load(path).astype(bool)
    return lambda frame: bool(flags[frame]) if frame < len(flags) else True


def use_sam2_upstream() -> None:
    """Put the external facebookresearch/sam2 checkout first on the import path.

    Called before any other model loads, so every later import sees the same
    path order whichever entry point runs the tracker.
    """
    sam2_upstream = Path(os.getenv("SAM2_UPSTREAM_DIR", ROOT / "sam2-upstream"))
    if not sam2_upstream.is_dir():
        raise FileNotFoundError(
            "Needs an external facebookresearch/sam2 checkout. Set SAM2_UPSTREAM_DIR "
            "before running --tracker sam2."
        )
    if str(sam2_upstream) not in sys.path:
        sys.path.insert(0, str(sam2_upstream))


def parseq_provenance(checkpoint: Path) -> dict:
    """What decides the parseq reader's output, recorded beside its reads cache."""
    checkpoint = Path(checkpoint).resolve()
    return {"reader": "parseq", "parseq_checkpoint": str(checkpoint),
            "parseq_checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None}


def parseq_reader(checkpoint: Path, device: str):
    """`(frame_bgr, frame_rgb, boxes, confidences) -> [text]` from the baudm/parseq checkpoint."""
    from handball_cv.jersey.parseq_backend import load_jersey_parseq

    model = load_jersey_parseq(checkpoint, device)
    return lambda bgr, rgb, boxes, confidences=None: read_numbers_parseq(model, rgb, boxes, confidences)


def prtreid_encoder(root: Path, checkpoint: Path, device: str):
    """`crops_rgb -> (N, D)` person re-ID features, used to recognise a returning player."""
    from handball_cv.embeddings.prtreid import PRTReIDBackend

    backend = PRTReIDBackend(source_root=root, checkpoint=checkpoint, device=device,
                             feature_kind="global")
    return backend.encode_images


def track_sam2(video: Path, detections: dict, number_detections_cache: dict, team_model, read, *,
               run_path: Path, reads_path: Path, geometry_path: Path, extras_path: Path,
               reader: str, reader_provenance: dict, reid_embedding: str, reid_encoder=None,
               number_detections_path: Path, checkpoint, check_every: int, ocr_every: int,
               sam2_stride: int = 1, small_number_crops: bool = False, court_in_view=None,
               frame_source: str = "video", frame_cache_dir: Path | None = None,
               roster: dict | None = None, needs_bgr: bool = False, on_frame=None,
               head: dict | None = None) -> dict:
    """Track players with SAM2 and periodic detector reprompting, and read their numbers.

    Writes the run summary to `run_path`, the reader's output to `reads_path`,
    per-frame boxes, ids and masks to `geometry_path`, and each geometry row's
    team confidence to `extras_path`; returns the summary.

    `read(frame_bgr, frame_rgb, boxes, confidences)` reads number crops, and
    appends a confidence per box to `confidences` if it has one. Reads already
    in `reads_path` are replayed rather than read again, as long as the file was
    written with the same `reader_provenance`. `on_frame(frame_idx, frame_bgr,
    boxes, crops, color_lookup, labels)`, if given, runs for every frame in
    order, on a worker thread, while SAM2 propagates the next one. `head` holds
    extra summary fields, placed after `source`.
    """
    use_sam2_upstream()
    from handball_cv.tracking.sam2_driver import drive_sam2

    source = Path(video).resolve()
    info = sv.VideoInfo.from_video_path(str(source))
    voter = NumberVoter()
    roster = roster or {}

    track_manager, _seed_boxes, frames = drive_sam2(
        source, lambda idx: person_detections(detections, idx), team_model,
        checkpoint=checkpoint, check_every=check_every,
        frame_cache_dir=frame_cache_dir or (ROOT / "runs/cache/frames" / source.stem),
        goalkeeper_class_id=GOALKEEPER_CLASS_ID,
        desc=f"full pipeline sam2 {source.stem}",
        reid_encoder=reid_encoder,
        prefetch_frames=4,
        frame_source=frame_source,
        court_in_view=court_in_view,
        sam2_stride=sam2_stride,
        # Jersey reads match number boxes to masks; keep those frames real.
        real_mask_every=(ocr_every,),
    )
    registry = track_manager.registry

    # Reader output is the most expensive part of a pass and the part that does
    # not change when the voting rules do. It is cached keyed by (frame_index,
    # number-box row) -- both stable, since the number boxes come from a fixed
    # detection cache -- so iterating on NumberVoter costs one SAM2 pass instead
    # of a full reader pass. Rows missing from the cache -- a different
    # ocr_every, or masks pairing other number boxes -- are read now and added,
    # never replayed as blank.
    known_reads, provenance_known = load_reads_cache(reads_path, reader_provenance)
    reads_added = 0

    geometry = GeometryCache(info.width, info.height, info.total_frames)

    frames_written = ocr_frames = ocr_reads = rejected_crops = 0
    number_reads: list[dict] = []
    presence: dict[str, list[int]] = {}
    resolver = NumberIdentityResolver(registry, voter)

    # Geometry packing and `on_frame` run on one worker, in frame order, so SAM2
    # propagates the next frame meanwhile: the generator only advances when this
    # loop asks for the next frame. Everything handed over is final for its
    # frame -- labels are computed here because they read vote state the next
    # frame changes.
    # Each geometry row's team confidence, in the order the rows are added --
    # which is the cache's own row order, since frames arrive in order.
    team_confidence_rows: list[float] = []

    def draw(frame_idx, frame_bgr, player_ids, boxes, masks, color_lookup, team_ids, labels):
        # Crop once: the cache stores the crops and a drawer draws from them.
        crops = [as_crop(mask) for mask in masks]
        geometry.add(frame_idx, player_ids, boxes, crops, color_lookup, team_ids)
        if on_frame is not None:
            on_frame(frame_idx, frame_bgr, boxes, crops, color_lookup, labels)

    drawer = ThreadPoolExecutor(max_workers=1)
    in_flight: deque = deque()
    for result in frames:
        frame_rgb = result.read_frame()
        # Only a drawer and some readers take BGR; skip the conversion otherwise.
        frame_bgr = (cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                     if on_frame is not None or needs_bgr else None)
        player_ids = result.player_ids
        masks = result.masks
        players = [
            registry.live.get(int(pid))
            or next(p for p in reversed(registry.retired) if p.player_id == int(pid))
            for pid in player_ids
        ]
        resolver.begin_frame(player_ids)

        # Interpolated frames carry display geometry only, never evidence; the
        # schedule keeps every read frame real, this is the guard if it did not.
        if result.frame_idx % ocr_every == 0 and len(player_ids) and not result.interpolated:
            number_xyxy = number_detections(number_detections_cache, result.frame_idx)
            if len(number_xyxy):
                ocr_frames += 1
                pairs = unique_pairs(match_numbers_to_players(
                    masks, number_xyxy, frame_rgb.shape, reject_ambiguous=True,
                ))
                small = {n for _, n in pairs if small_number_crops
                         and small_number_crop(frame_rgb, number_xyxy[n])}
                pairs = [
                    (p, n) for p, n in pairs
                    if n in small or readable_number_crop(frame_rgb, number_xyxy[n])
                ]
                rejected_crops += len(number_xyxy) - len(pairs)
                if pairs:
                    wanted = sorted({n for _, n in pairs})

                    def read_rows(rows, confidences):
                        # drive_sam2 yields from inside its BF16 autocast, which
                        # stays on in this loop; the reader was measured in
                        # FP32, so it runs outside it.
                        with torch.autocast("cuda", enabled=False):
                            texts = read(frame_bgr, frame_rgb, number_xyxy[rows], confidences)
                        # Judged before the reads cache keeps it: the cache holds
                        # no confidence, so a replay could not judge it again.
                        return [text if row not in small or small_crop_read_ok(text, confidence)
                                else "" for row, text, confidence
                                in zip(rows, texts, confidences or [None] * len(rows))]

                    text_by_row, confidence_by_row, added = cached_read(
                        known_reads, result.frame_idx, wanted, read_rows)
                    reads_added += added
                    for local_row, number_row in pairs:
                        raw = text_by_row.get(number_row, "")
                        if raw:
                            ocr_reads += 1
                            # Votes belong to the canonical identity, not the id
                            # the tracker happens to use.
                            pid = registry.canonical(int(player_ids[local_row]))
                            voter.observe(pid, raw, read_weight(number_xyxy[number_row]))
                            entry = {
                                "frame": result.frame_idx,
                                "player_id": pid, "value": raw,
                                "track_id": int(player_ids[local_row]),
                                "number_row": int(number_row),
                                "box": number_xyxy[number_row].tolist(),
                            }
                            if number_row in confidence_by_row:
                                entry["confidence"] = confidence_by_row[number_row]
                            if number_row in small:
                                entry["small_crop"] = True
                            number_reads.append(entry)

        presence[str(result.frame_idx)] = sorted(int(p) for p in player_ids)
        if result.frame_idx % ocr_every == 0:
            resolver.resolve(result.frame_idx)
        color_lookup = [color_index(player) for player in players]
        team_confidence_rows.extend(float(player.team_confidence) for player in players)
        in_flight.append(drawer.submit(
            draw, result.frame_idx, frame_bgr, player_ids, result.boxes, masks, color_lookup,
            [player.voted_team_id for player in players],
            voted_labels(player_ids, color_lookup, voter, roster,
                         canonical=registry.canonical),
        ))
        frames_written += 1
        while len(in_flight) > 2:
            in_flight.popleft().result()  # bounded, and re-raises a worker error

    while in_flight:
        in_flight.popleft().result()
    drawer.shutdown()

    np.savez_compressed(extras_path, team_confidence=np.asarray(team_confidence_rows, np.float32))

    # Written even when empty: `process_video` takes the file as the track pass's.
    if reads_added or not reads_path.is_file():
        reads_path.write_text(json.dumps({f: rows for f, rows in known_reads.items() if rows}))
        if provenance_known:
            reads_provenance_path(reads_path).write_text(json.dumps(reader_provenance, indent=2))
        print(f"[reads] added {reads_added} reads -> {reads_path}")

    geometry.save(geometry_path)
    print(f"[geometry] wrote {geometry_path} ({geometry_path.stat().st_size / 1e6:.1f} MB)")

    everyone = list(registry.live.values()) + registry.retired
    result_dict = {
        "source": str(source),
        **(head or {}),
        "geometry_cache": str(geometry_path),
        "extras": str(extras_path),
        "frame_source": frame_source,
        "sam2_stride": sam2_stride,
        "check_every": check_every,
        "ocr_every": ocr_every,
        "small_number_crops": small_number_crops,
        # Where the read number boxes came from: the detector's own, or sheets.
        "number_detections": str(number_detections_path),
        "tracker": "sam2",
        "reid_embedding": reid_embedding,
        "reader": reader,
        "frames": frames_written,
        "numbers_enabled": True,
        "ocr_frames": ocr_frames,
        "ocr_reads": ocr_reads,
        "rejected_number_crops": rejected_crops,
        **identity_report(
            everyone, registry.events, voter, number_reads, presence,
            registry.team_by_player_id(), dict(registry.alias),
        ),
        **registry.summary(),
        # The tracker's own decisions -- reprompt, reset, remove, pause -- and
        # every re-ID decision with what it compared: what explains a tracker
        # failure after the fact, without re-running it.
        "lifecycle_events": track_manager.events,
        "reid_decisions": registry.reid_log,
    }
    run_path.write_text(json.dumps(result_dict, indent=2, default=_json_default))
    return result_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The track stage of process_video: SAM2 tracking, re-ID and jersey numbers.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--number-detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path,
                        help="run summary JSON; the reads, geometry and extras files sit beside it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--court-in-view", type=Path, default=None,
                        help=".npy of one bool per frame; tracking pauses where it is False")
    parser.add_argument("--check-every", type=int, default=CHECK_EVERY_DEFAULT)
    parser.add_argument("--ocr-every", type=int, default=OCR_EVERY_N_FRAMES)
    parser.add_argument("--sam2-stride", type=int, default=1)
    parser.add_argument("--small-number-crops", action="store_true")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models/sam2/sam2.1_hiera_large.pt")
    parser.add_argument("--prtreid-root", type=Path, default=ROOT / "prtreid-upstream")
    parser.add_argument("--prtreid-checkpoint", type=Path,
                        default=ROOT / "models/prtreid/prtreid-soccernet-baseline.pth.tar")
    parser.add_argument("--parseq-checkpoint", type=Path,
                        default=ROOT / "models/jersey_parseq/parseq_original.ckpt")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    use_sam2_upstream()
    run = args.run
    result = track_sam2(
        args.video, load_detection_cache(args.detections),
        load_detection_cache(args.number_detections),
        TeamModel.load(args.team_model, device=args.device),
        parseq_reader(args.parseq_checkpoint, args.device),
        run_path=run, reads_path=run.with_name(run.stem + "_reads.json"),
        geometry_path=run.with_name(run.stem + "_geometry.npz"),
        extras_path=run.with_name(run.stem + "_extras.npz"),
        reader="parseq", reader_provenance=parseq_provenance(args.parseq_checkpoint),
        reid_embedding="prtreid",
        reid_encoder=prtreid_encoder(args.prtreid_root, args.prtreid_checkpoint, args.device),
        number_detections_path=args.number_detections, checkpoint=str(args.checkpoint),
        check_every=args.check_every, ocr_every=args.ocr_every, sam2_stride=args.sam2_stride,
        small_number_crops=args.small_number_crops,
        court_in_view=court_in_view_fn(args.court_in_view),
    )
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
