"""End-to-end handball pipeline: tracking + team + stable identity + jersey number.

Runs every stage the project has built so far in one pass, so the combined
result can be reviewed as a whole rather than one component at a time:

    cached RF-DETR detections
        -> MCByte tracking (boxes + SAM/Cutie masks)
        -> frame-local TeamModel  (team, confidence, quality)
        -> IdentityManager        (stable player_id, re-ID, reversible team)
        -> jersey-number OCR      (optional; class-4 boxes -> EasyOCR -> vote)
        -> notebook-style overlay (team-colored mask + rich number label)

The overlay deliberately matches the original notebook's final "Player
recognition" cell (cell 96): an `sv.MaskAnnotator` tinted by team plus an
`sv.RichLabelAnnotator` drawing the jersey number under each player.

Number votes are keyed on `player_id`, not on the tracker's own id. That is the
whole reason `IdentityManager` exists: when McByte fragments a track and re-ID
stitches it back, the number evidence accumulated before the gap continues into
the same `player_id` automatically, with no explicit vote merge.

The number stage needs the Roboflow-hosted detector for class-4 number boxes,
so it requires ROBOFLOW_API_KEY. OCR itself runs locally with EasyOCR. Without
the detector key the script still runs every other stage and simply labels
players `P<id>` -- see --numbers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from trackers import McByteMaskConfig, McByteTracker

from handball_cv.jersey.identity import (
    read_numbers_doctr,
    read_numbers_parseq,
    NUMBER_CLASS_ID,
    OCR_EVERY_N_FRAMES,
    NumberVoter,
    match_numbers_to_players,
    read_numbers,
)
from handball_cv.teams.model import MIN_STABLE_TEAM_CONFIDENCE, TeamModel, crop_quality
from handball_cv.tracking.identity import TEAM_SWITCH_OBSERVATIONS, IdentityManager
from scripts.render_raw_team_classification import (
    number_detections,
    person_detections,
    frame_detections,
    load_detection_cache,
)

ROOT = Path(__file__).resolve().parents[1]
PLAYER_MODEL_ID = "player-and-handball-detection-3z9xf/3"
GOALKEEPER_CLASS_ID = 1

# One palette, indexed by the lookup below. Team A/B first so a roster-less
# clip still reads like the notebook's two-team overlay.
TEAM_A, TEAM_B, NEW_UNCERTAIN, CONTESTED, GOALKEEPER = 0, 1, 2, 3, 4
PALETTE_HEX = [
    "#00BEFF",  # A
    "#FF7800",  # B
    "#FFDC00",  # new / never had a qualified read
    "#DC0000",  # established but actively contested
    "#B4B4B4",  # goalkeeper
]
LEGEND = [
    (TEAM_A, "Team A"),
    (TEAM_B, "Team B"),
    (NEW_UNCERTAIN, "New / uncertain"),
    (CONTESTED, "Contested"),
    (GOALKEEPER, "GK"),
]
DEFAULT_FONT = ROOT / "notebooks" / "fonts" / "ConcertOne-Regular.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--team-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reid-embedding", default="prtreid", choices=("prtreid", "team-model"),
        help="appearance features used to recognise a returning player. The "
             "team model's own features cannot separate teammates (0.35 rank-1 "
             "against a 0.19 chance floor, versus 0.55 for prtreid); they remain "
             "selectable so the pipeline runs without the prtreid checkout",
    )
    parser.add_argument("--prtreid-root", type=Path, default=ROOT / "prtreid-upstream")
    parser.add_argument(
        "--prtreid-checkpoint", type=Path,
        default=ROOT / "models/prtreid/prtreid-soccernet-baseline.pth.tar",
    )
    parser.add_argument(
        "--tracker", choices=("sam2", "mcbyte"), default="sam2",
        help="SAM2 with periodic detector reprompting wins on all three clips "
             "scored against ground truth -- 93.8/89.2/98.9%% correct, against "
             "81.5/86.7/95.9%% for the best box tracker on each; see "
             "docs/tracking-evaluation.md section 8. It costs ~1s/frame against "
             "McByte's ~0.1s and needs --number-detections and a sam2 checkout",
    )
    parser.add_argument(
        "--number-detections", type=Path, default=None,
        help="cached number-detector npz; required by --tracker sam2, which "
             "does not call the hosted class-4 detector",
    )
    parser.add_argument("--checkpoint", default=str(
        ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
    ), help="sam2 only")
    parser.add_argument("--check-every", type=int, default=10, help="sam2 only")
    parser.add_argument(
        "--reads-cache", type=Path, default=None,
        help="sam2 only; JSON of per-frame reader output. Written on the first run "
             "and replayed on later ones, so changing the voting rules does not "
             "re-pay for the reader. Defaults to <output stem>_reads.json",
    )
    parser.add_argument(
        "--reader", choices=("parseq", "doctr", "qwen", "easyocr"), default="parseq",
        help="jersey-number recogniser, honoured by both trackers. Accuracy on "
             "the 323 labelled 1080p crops: parseq 0.858, qwen 0.628, "
             "doctr 0.622, easyocr 0.365. `parseq` needs a baudm/parseq checkout "
             "and models/jersey_parseq/parseq_original.ckpt",
    )
    parser.add_argument("--doctr-arch", default="parseq",
                         help="--reader doctr only; docTR recogniser architecture")
    parser.add_argument(
        "--parseq-checkpoint", type=Path,
        default=ROOT / "models/jersey_parseq/parseq_original.ckpt",
        help="--reader parseq only",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1",
                         help="--reader qwen only")
    parser.add_argument("--model", default="qwen38-flash-next",
                         help="--reader qwen only")
    parser.add_argument("--max-tokens", type=int, default=1024,
                         help="--reader qwen only")
    parser.add_argument("--frame-cache-dir", type=Path, default=None, help="sam2 only")
    parser.add_argument("--no-masks", action="store_true",
                         help="mcbyte only; sam2 always has masks")
    parser.add_argument(
        "--numbers", choices=("auto", "on", "off"), default="auto",
        help="jersey-number OCR: 'auto' enables it only when ROBOFLOW_API_KEY "
             "is set, 'on' fails loudly if it cannot start, 'off' skips it",
    )
    parser.add_argument(
        "--ocr-every", type=int, default=OCR_EVERY_N_FRAMES,
        help="run number detection/OCR every N frames",
    )
    parser.add_argument(
        "--roster", type=Path, default=None,
        help='optional JSON mapping number -> name, either {"7": "Name"} or '
             '{"A": {"7": "Name"}, "B": {...}} to disambiguate per team',
    )
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    return parser.parse_args()


def load_roster(path: Path | None) -> dict:
    """-> {team_index_or_None: {number: name}}. Empty dict when unavailable."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if all(isinstance(v, dict) for v in data.values()):
        by_team = {"A": TEAM_A, "B": TEAM_B, "0": TEAM_A, "1": TEAM_B}
        return {
            by_team.get(str(key).upper(), None): {str(n): v for n, v in value.items()}
            for key, value in data.items()
        }
    return {None: {str(k): v for k, v in data.items()}}


def roster_name(roster: dict, team_index: int, number: str) -> str | None:
    if not roster:
        return None
    for key in (team_index, None):
        if key in roster and number in roster[key]:
            return roster[key][number]
    return None


# The notebook calls load_dotenv() from notebooks/, so the key has historically
# lived at notebooks/.env. Scripts run from the repo root and would miss it, so
# check both -- root first, matching .env.example.
ENV_CANDIDATES = (ROOT / ".env", ROOT / "notebooks" / ".env")


def load_env() -> Path | None:
    """Load the first .env found and return it, or None. Never logs values."""
    for candidate in ENV_CANDIDATES:
        if candidate.is_file():
            load_dotenv(candidate)
            # SigLIP is pulled from HuggingFace by TeamModel; without this the
            # download falls back to unauthenticated and rate-limited requests.
            token = os.getenv("HF_TOKEN", "").strip()
            if token:
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            return candidate
    return None


def load_number_detector(mode: str):
    """The Roboflow class-4 number detector, or None.

    Recognition is chosen separately by `--reader`; this used to build an EasyOCR
    reader here as well, which is why the McByte path ignored that flag.
    Returning None rather than raising keeps the rest of the pipeline runnable on
    a machine without a detector key.
    """
    if mode == "off":
        return None
    env_path = load_env()
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        searched = " or ".join(str(c) for c in ENV_CANDIDATES)
        message = (
            f"jersey-number OCR needs ROBOFLOW_API_KEY (looked in {searched}); "
            "the class-4 number detector is Roboflow-hosted"
        )
        if mode == "on":
            raise RuntimeError(message)
        print(f"[numbers] disabled: {message}")
        return None
    if env_path is not None:
        print(f"[numbers] credentials loaded from {env_path}")
    try:
        from inference import get_model

        player_model = get_model(model_id=PLAYER_MODEL_ID, api_key=api_key)
    except Exception as error:  # network, auth, or missing model access
        if mode == "on":
            raise
        print(f"[numbers] disabled: could not load the number detector ({error})")
        return None
    return player_model


def tracklet_masks(mask_output, tracker_ids) -> list:
    """One boolean (H, W) mask per tracker_id, or None where McByte has none."""
    if mask_output is None or mask_output.masks is None or len(mask_output.masks) == 0:
        return [None] * len(tracker_ids)
    masks = np.asarray(mask_output.masks, dtype=bool)
    row_by_tracker = mask_output.tracklet_mask_dict
    out = []
    for tracker_id in tracker_ids:
        row = row_by_tracker.get(int(tracker_id))
        out.append(masks[int(row)] if row is not None and 0 <= int(row) < len(masks) else None)
    return out


def color_index(player) -> int:
    """Which palette entry this player draws with.

    Keeps yesterday's distinction between a brand-new player still building
    evidence and an established one whose team is under active opposition --
    the second is the interesting failure, the first is just bootstrapping.
    """
    if player.is_goalkeeper:
        return GOALKEEPER
    if player.team_is_provisional:
        return NEW_UNCERTAIN
    if player.team_confidence < MIN_STABLE_TEAM_CONFIDENCE:
        return CONTESTED
    return TEAM_A if player.voted_team_id == 0 else TEAM_B


# Matches the notebook's PLAYER_DETECTION_MODEL_CONFIDENCE / IOU_THRESHOLD.
# At 0.3 the class-4 head also fired on sponsor lettering on the boards -- the
# OCR then dutifully "read" it, poisoning the vote.
NUMBER_DETECTION_CONFIDENCE = 0.5
NUMBER_DETECTION_IOU = 0.9
# OCR can still emit a high-confidence digit for an unreadable crop. Junk must
# be rejected before it reaches the model, not only confidence-filtered after.
MIN_NUMBER_BOX_AREA = 120



def detect_number_boxes(player_model, frame_bgr: np.ndarray) -> np.ndarray:
    """Class-4 jersey-number boxes for this frame, as (N, 4) xyxy.

    The cached detections were built for the team work and only kept the
    player classes, so numbers need their own detector call.
    """
    result = player_model.infer(
        frame_bgr,
        confidence=NUMBER_DETECTION_CONFIDENCE,
        iou_threshold=NUMBER_DETECTION_IOU,
    )[0]
    detections = sv.Detections.from_inference(result)
    boxes = detections[detections.class_id == NUMBER_CLASS_ID].xyxy
    if not len(boxes):
        return boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return boxes[(widths * heights) >= MIN_NUMBER_BOX_AREA]


def unique_pairs(pairs: list) -> list:
    """Greedy one-to-one filter over (player_row, number_row), best score first.

    `match_numbers_to_players` returns every pair above the IoS floor, so a
    player whose mask happens to cover an occluded neighbour's torso collects
    both numbers in one frame and votes for both. Measured at 15% of matches on
    Felix. Mask IoS is already sorted best-first, so first-come wins.
    """
    seen_players, seen_numbers, kept = set(), set(), []
    for player_row, number_row in pairs:
        if player_row in seen_players or number_row in seen_numbers:
            continue
        seen_players.add(player_row)
        seen_numbers.add(number_row)
        kept.append((player_row, number_row))
    return kept


def readable_number_crop(frame_rgb: np.ndarray, box: np.ndarray) -> bool:
    """Whether a number crop carries enough signal to be worth OCR-ing."""
    x1, y1, x2, y2 = np.rint(box).astype(int)
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    return crop_quality(frame_rgb[y1:y2, x1:x2]).accepted


def draw_legend(frame: np.ndarray, palette: sv.ColorPalette) -> None:
    height, width = frame.shape[:2]
    scale = max(width / 1920.0, 1.0)
    x, y = round(18 * scale), height - round(18 * scale)
    chip = round(14 * scale)
    for index, label in reversed(LEGEND):
        color = palette.by_idx(index).as_bgr()
        cv2.rectangle(frame, (x, y - chip), (x + chip, y), color, -1)
        cv2.rectangle(frame, (x, y - chip), (x + chip, y), (20, 20, 20), 1)
        cv2.putText(
            frame, label, (x + chip + round(6 * scale), y - round(2 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255),
            max(1, round(scale)), cv2.LINE_AA,
        )
        y -= chip + round(10 * scale)


def annotate_frame(
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    player_ids,
    players: list,
    masks: list,
    voter,
    roster: dict,
    mask_annotator,
    label_annotator,
    canonical=None,
) -> np.ndarray:
    """Team-tinted mask fill plus a `#number` label per player.

    Shared by both tracker paths: everything here works off one frame's
    (boxes, player_ids, PlayerRecords, masks), regardless of whether McByte or
    SAM2 produced them. `masks` entries may be None (McByte's mask manager has
    no mask for that tracklet this frame); SAM2 never leaves one None.
    """
    annotated = frame_bgr.copy()
    if not len(boxes_xyxy):
        return annotated

    lookup = np.array([color_index(player) for player in players], dtype=int)
    labels = []
    resolve = canonical or int
    for player, player_id in zip(players, player_ids):
        # Number evidence can fold two allocated ids into one player after the
        # fact (`NumberIdentityResolver`); the label follows the fold so the
        # overlay shows one person, not two boxes disagreeing about a number.
        player_id = resolve(int(player_id))
        number, _votes, _margin = voter.best(player_id)
        if number is None:
            labels.append(f"P{player_id}")
            continue
        name = roster_name(roster, color_index(player), number)
        labels.append(f"#{number} {name}" if name else f"#{number}")

    drawn = sv.Detections(
        xyxy=np.asarray(boxes_xyxy, dtype=float).copy(),
        class_id=lookup.copy(),
        tracker_id=np.asarray(player_ids, dtype=int),
    )
    if any(mask is not None for mask in masks):
        blank = np.zeros(frame_bgr.shape[:2], dtype=bool)
        drawn.mask = np.stack([blank if mask is None else mask for mask in masks])
        annotated = mask_annotator.annotate(
            scene=annotated, detections=drawn, custom_color_lookup=lookup
        )
    return label_annotator.annotate(
        scene=annotated, detections=drawn, labels=labels,
        custom_color_lookup=lookup,
    )


class NumberIdentityResolver:
    """Applies number evidence to identities once the reads have accumulated.

    Two jobs, in this order, both of which need the *same* two facts -- what
    team an identity is on, and whether two identities were ever on screen at
    the same instant -- which is why they live together:

      1. `link`: same team, same resolved number, never simultaneously live.
         That is one player the tracker split in two, and the halves are folded.
      2. `arbitrate`: same team, same resolved number, but they *were* on screen
         together. That cannot be one player, so the weaker claim is withheld
         rather than displayed as a second copy of someone else's number.

    Order matters. Linking first means arbitration only ever sees duplicates
    that genuinely coexisted, so it never suppresses a fragment it should have
    merged instead.
    """

    def __init__(self, registry, voter):
        self.registry = registry
        self.voter = voter
        self.co_live: dict[int, set] = {}
        self._events_seen = 0

    def begin_frame(self, player_ids) -> None:
        """Note who shared this frame, and un-vouch anything re-ID just moved.

        Called before this frame's reads are voted, because a revival makes the
        identity's existing number stale from that instant: the verdict was
        earned by a fragment that has ended, and the new fragment may well be a
        different person.
        """
        for event in self.registry.events[self._events_seen:]:
            if event["type"] == "reid":
                self.voter.suspend(event["player_id"])
        self._events_seen = len(self.registry.events)

        ids = {int(p) for p in player_ids}
        for player_id in ids:
            self.co_live.setdefault(player_id, set()).update(ids - {player_id})

    def _ever_together(self, a: int, b: int) -> bool:
        """True if any identity folded into `a` shared a frame with one in `b`."""
        canonical = self.registry.canonical
        group_a = {p for p in self.co_live if canonical(p) == canonical(a)}
        group_b = {p for p in self.co_live if canonical(p) == canonical(b)}
        return any(self.co_live.get(p, set()) & group_b for p in group_a)

    def resolve(self, frame_idx: int) -> None:
        teams = self.registry.team_by_player_id()
        claims: dict = {}
        for player_id, team_id in teams.items():
            value, count, _margin = self.voter.claim(player_id)
            if value is not None:
                claims.setdefault((team_id, value), []).append((count, player_id))

        for holders in claims.values():
            if len(holders) < 2:
                continue
            # Fold into the best-evidenced holder so the surviving identity is
            # the one whose number is least likely to be wrong.
            holders.sort(key=lambda h: (-h[0], h[1]))
            keeper = holders[0][1]
            for _count, other in holders[1:]:
                if self._ever_together(keeper, other):
                    continue
                if self.registry.link(other, keeper, frame_idx):
                    self.voter.merge(other, keeper)

        self.voter.arbitrate({
            self.registry.canonical(pid): team for pid, team in teams.items()
        })


def build_ocr_model(args):
    """(recogniser, qwen_callable) for `--reader`; exactly one is not None."""
    if args.reader == "easyocr":
        import easyocr
        return easyocr.Reader(
            ["en"], gpu=args.device != "cpu", detector=False, verbose=False
        ), None
    if args.reader == "doctr":
        import torch
        from doctr.models import recognition_predictor
        model = recognition_predictor(args.doctr_arch, pretrained=True).eval()
        if args.device != "cpu" and torch.cuda.is_available():
            model = model.cuda()
        return model, None
    if args.reader == "parseq":
        from handball_cv.jersey.parseq_backend import load_jersey_parseq
        return load_jersey_parseq(args.parseq_checkpoint, args.device), None
    # Deferred: evaluate_number_pipeline imports this module at module level, so
    # importing it back at import time would be circular. By call time both
    # modules are loaded and this resolves cleanly.
    from scripts.evaluate_number_pipeline import read_with_qwen
    return None, read_with_qwen


def build_reader(args, ocr_model, read_with_qwen=None):
    """`(frame_bgr, frame_rgb, boxes) -> [text]` for the selected reader.

    Both tracker paths used to choose a reader independently, and the McByte one
    simply called EasyOCR whatever `--reader` said. That was survivable while
    EasyOCR was the default and everything agreed; it stops being survivable the
    moment the default is a reader that is twice as accurate, because the flag
    and the behaviour would silently disagree.
    """
    if args.reader == "easyocr":
        return lambda bgr, rgb, boxes: read_numbers(ocr_model, rgb, boxes)
    if args.reader == "doctr":
        return lambda bgr, rgb, boxes: read_numbers_doctr(ocr_model, rgb, boxes)
    if args.reader == "parseq":
        return lambda bgr, rgb, boxes: read_numbers_parseq(ocr_model, rgb, boxes)
    return lambda bgr, rgb, boxes: read_with_qwen(
        bgr, boxes, args.output.parent / "_scratch",
        args.base_url, args.model, args.max_tokens,
    )


def build_reid_encoder(args: argparse.Namespace):
    """`crops_rgb -> (N, D)` for identity, or None to reuse the team features.

    Measured on the labelled 1080p set (`scripts.measure_reid_discriminability`),
    rank-1 within a team over an identical 256 queries: team-model features 0.35,
    person re-ID 0.55, chance 0.19. The encoder only runs on people the tracker
    has not seen before, so it is a small batch a few times a second.
    """
    if args.reid_embedding == "team-model":
        return None
    from handball_cv.embeddings.prtreid import PRTReIDBackend
    backend = PRTReIDBackend(
        source_root=args.prtreid_root, checkpoint=args.prtreid_checkpoint,
        device=args.device, feature_kind="global",
    )
    return backend.encode_images


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
            if e["type"] in ("suspected_id_switch", "team_switch", "reid", "link")
        ],
    }


def render(args: argparse.Namespace) -> dict:
    source = args.video.resolve()
    cache = load_detection_cache(args.detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    info = sv.VideoInfo.from_video_path(str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = args.output.with_name(f"{args.output.stem}_preview.jpg")

    enable_masks = not args.no_masks
    player_model = load_number_detector(args.numbers)
    numbers_enabled = player_model is not None
    read = None
    if numbers_enabled:
        ocr_model, read_with_qwen = build_ocr_model(args)
        read = build_reader(args, ocr_model, read_with_qwen)
    roster = load_roster(args.roster)
    voter = NumberVoter()

    if not args.font.is_file():
        raise FileNotFoundError(f"font not found: {args.font}")
    palette = sv.ColorPalette.from_hex(PALETTE_HEX)
    mask_annotator = sv.MaskAnnotator(
        color=palette, opacity=0.5, color_lookup=sv.ColorLookup.INDEX
    )
    label_annotator = sv.RichLabelAnnotator(
        font_path=str(args.font),
        font_size=round(34 * max(info.width / 1920.0, 1.0)),
        color=palette,
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_offset=(0, 10),
        color_lookup=sv.ColorLookup.INDEX,
    )

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
        reid_encoder=build_reid_encoder(args),
    )
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {args.output}")

    preview = None
    frames_written = 0
    ocr_frames = 0
    ocr_reads = 0
    number_reads: list[dict] = []
    presence: dict[str, list[int]] = {}
    resolver = NumberIdentityResolver(identity.registry, voter)
    rejected_crops = 0
    for frame_index, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(str(source)),
        total=info.total_frames,
        desc=f"full pipeline {source.stem}",
    )):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = person_detections(cache, frame_index)
        tracked = tracker.update(detections, frame=frame_rgb)
        tracked = tracked[tracked.tracker_id >= 0]
        player_ids = identity.update(frame_index, frame_rgb, tracked)
        alive = getattr(tracker, "tracked_objects", sv.Detections.empty())
        alive_ids = (
            alive.tracker_id
            if getattr(alive, "tracker_id", None) is not None
            else np.empty(0, dtype=int)
        )
        identity.retire_missing(frame_index, alive_ids)

        masks = (
            tracklet_masks(getattr(tracker, "_last_mask_output", None), tracked.tracker_id)
            if enable_masks else [None] * len(tracked)
        )
        resolver.begin_frame(player_ids)

        players = []
        for player_id in player_ids:
            player = identity.players.get(int(player_id))
            if player is None:
                player = next(
                    item for item in reversed(identity.retired)
                    if item.player_id == int(player_id)
                )
            players.append(player)

        # Numbers: detect class-4 boxes, match them to player masks by mask IoS,
        # OCR the crops, and vote per stable player_id.
        if numbers_enabled and frame_index % args.ocr_every == 0 and len(tracked):
            masked_rows = [i for i, mask in enumerate(masks) if mask is not None]
            if masked_rows:
                number_xyxy = detect_number_boxes(player_model, frame_bgr)
                if len(number_xyxy):
                    ocr_frames += 1
                    stacked = np.stack([masks[i] for i in masked_rows])
                    pairs = unique_pairs(match_numbers_to_players(
                        stacked, number_xyxy, frame_bgr.shape
                    ))
                    # Drop unreadable crops before OCR: confidence filtering
                    # alone cannot reliably identify every junk prediction.
                    pairs = [
                        (player_row, number_row) for player_row, number_row in pairs
                        if readable_number_crop(frame_rgb, number_xyxy[number_row])
                    ]
                    rejected_crops += len(number_xyxy) - len(pairs)
                    if pairs:
                        wanted = sorted({number_row for _, number_row in pairs})
                        texts = read(frame_bgr, frame_rgb, number_xyxy[wanted])
                        text_by_row = dict(zip(wanted, texts))
                        for local_row, number_row in pairs:
                            raw = text_by_row.get(number_row, "")
                            if raw:
                                ocr_reads += 1
                                # Against the canonical identity: once two ids
                                # are folded, the tracker keeps emitting the
                                # folded one, and votes filed under it would
                                # never reach the identity that speaks.
                                _pid = identity.registry.canonical(
                                    int(player_ids[masked_rows[local_row]])
                                )
                                voter.observe(_pid, raw)
                                # Timestamped so a read can be placed before or
                                # after an identity switch; aggregate vote counts
                                # cannot tell which physical player produced them.
                                number_reads.append({
                                    "frame": frame_index, "player_id": _pid,
                                    "value": raw, "number_row": int(number_row),
                                    "box": number_xyxy[number_row].tolist(),
                                })

        presence[str(frame_index)] = sorted(int(p) for p in player_ids)
        if numbers_enabled and frame_index % args.ocr_every == 0:
            resolver.resolve(frame_index)
        annotated = annotate_frame(
            frame_bgr, tracked.xyxy, player_ids, players, masks,
            voter, roster, mask_annotator, label_annotator,
            canonical=identity.registry.canonical,
        )
        draw_legend(annotated, palette)
        writer.write(annotated)
        frames_written += 1
        if frame_index == info.total_frames // 2:
            preview = annotated.copy()

    writer.release()
    if preview is not None:
        if preview.shape[1] > 1600:
            ratio = 1600 / preview.shape[1]
            preview = cv2.resize(
                preview, (1600, round(preview.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(preview_path), preview)

    everyone = list(identity.players.values()) + identity.retired
    result = {
        "source": str(source),
        "output": str(args.output),
        "preview": str(preview_path),
        "frames": frames_written,
        "tracker": "mcbyte",
        "reid_embedding": args.reid_embedding,
        "reader": "easyocr",   # the mcbyte path has no reader choice
        "mcbyte_masks": enable_masks,
        "numbers_enabled": numbers_enabled,
        "ocr_frames": ocr_frames,
        "ocr_reads": ocr_reads,
        "rejected_number_crops": rejected_crops,
        **identity_report(
            everyone, identity.events, voter, number_reads, presence,
            identity.team_by_player_id(), dict(identity.registry.alias),
        ),
        **identity.summary(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def render_sam2(args: argparse.Namespace) -> dict:
    """Same overlay, driven by SAM2 + periodic reprompting instead of McByte.

    Numbers come from a cached number-detector npz rather than the live
    Roboflow class-4 head: this path exists to review the measured-best
    tracker end to end, and re-running a hosted detector per frame would make
    the render depend on network state for no benefit -- the cache is the same
    detector's output.
    """
    sam2_upstream = Path(os.getenv("SAM2_UPSTREAM_DIR", ROOT / "sam2-upstream"))
    if not sam2_upstream.is_dir():
        raise FileNotFoundError(
            "Needs an external facebookresearch/sam2 checkout. Set SAM2_UPSTREAM_DIR "
            "before running --tracker sam2."
        )
    sys.path.insert(0, str(sam2_upstream))
    from handball_cv.tracking.sam2_driver import drive_sam2

    if args.number_detections is None:
        raise ValueError("--tracker sam2 requires --number-detections (a cached npz)")

    source = args.video.resolve()
    cache = load_detection_cache(args.detections)
    number_cache = load_detection_cache(args.number_detections)
    team_model = TeamModel.load(args.team_model, device=args.device)
    info = sv.VideoInfo.from_video_path(str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = args.output.with_name(f"{args.output.stem}_preview.jpg")

    ocr_model, read_with_qwen = build_ocr_model(args)
    read = build_reader(args, ocr_model, read_with_qwen)
    roster = load_roster(args.roster)
    voter = NumberVoter()

    if not args.font.is_file():
        raise FileNotFoundError(f"font not found: {args.font}")
    palette = sv.ColorPalette.from_hex(PALETTE_HEX)
    mask_annotator = sv.MaskAnnotator(
        color=palette, opacity=0.5, color_lookup=sv.ColorLookup.INDEX
    )
    label_annotator = sv.RichLabelAnnotator(
        font_path=str(args.font),
        font_size=round(34 * max(info.width / 1920.0, 1.0)),
        color=palette,
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER,
        text_offset=(0, 10),
        color_lookup=sv.ColorLookup.INDEX,
    )

    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {args.output}")

    track_manager, _seed_boxes, frames = drive_sam2(
        source, lambda idx: person_detections(cache, idx), team_model,
        checkpoint=args.checkpoint, check_every=args.check_every,
        frame_cache_dir=args.frame_cache_dir or (ROOT / "data/cache/frames" / source.stem),
        goalkeeper_class_id=GOALKEEPER_CLASS_ID,
        desc=f"full pipeline sam2 {source.stem}",
        reid_encoder=build_reid_encoder(args),
    )
    registry = track_manager.registry

    # Reader output is by far the most expensive part of this render (a Qwen call
    # is ~6.6s, versus ~0.9s/frame for everything else combined), and it is also
    # the part that does NOT change when the voting rules do. Cache it keyed by
    # (frame_index, number-box row) -- both stable, since the number boxes come
    # from a fixed detection cache -- so iterating on NumberVoter costs one SAM2
    # pass instead of a full reader pass.
    reads_cache_path = args.reads_cache or args.output.with_name(
        args.output.stem + "_reads.json"
    )
    cached_reads = None
    if reads_cache_path.is_file():
        cached_reads = json.loads(reads_cache_path.read_text())
        print(f"[reads] replaying cached reads from {reads_cache_path}")
    reads_log: dict[str, dict[str, str]] = {}

    preview = None
    frames_written = ocr_frames = ocr_reads = rejected_crops = 0
    number_reads: list[dict] = []
    presence: dict[str, list[int]] = {}
    resolver = NumberIdentityResolver(registry, voter)
    for result in frames:
        frame_rgb = result.read_frame()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        player_ids = result.player_ids
        masks = list(result.masks)
        players = [
            registry.live.get(int(pid))
            or next(p for p in reversed(registry.retired) if p.player_id == int(pid))
            for pid in player_ids
        ]
        resolver.begin_frame(player_ids)

        if result.frame_idx % args.ocr_every == 0 and len(player_ids):
            number_xyxy = number_detections(number_cache, result.frame_idx)
            if len(number_xyxy):
                ocr_frames += 1
                pairs = unique_pairs(match_numbers_to_players(
                    np.stack(masks), number_xyxy, frame_bgr.shape
                ))
                pairs = [
                    (p, n) for p, n in pairs
                    if readable_number_crop(frame_rgb, number_xyxy[n])
                ]
                rejected_crops += len(number_xyxy) - len(pairs)
                if pairs:
                    wanted = sorted({n for _, n in pairs})
                    if cached_reads is not None:
                        hit = cached_reads.get(str(result.frame_idx), {})
                        texts = [hit.get(str(n), "") for n in wanted]
                    else:
                        texts = read(frame_bgr, frame_rgb, number_xyxy[wanted])
                    text_by_row = dict(zip(wanted, texts))
                    reads_log.setdefault(str(result.frame_idx), {}).update(
                        {str(n): t for n, t in zip(wanted, texts)}
                    )
                    for local_row, number_row in pairs:
                        raw = text_by_row.get(number_row, "")
                        if raw:
                            ocr_reads += 1
                            # See the McByte path: votes belong to the canonical
                            # identity, not the id the tracker happens to use.
                            pid = registry.canonical(int(player_ids[local_row]))
                            voter.observe(pid, raw)
                            number_reads.append({
                                "frame": result.frame_idx,
                                "player_id": pid, "value": raw,
                                "number_row": int(number_row),
                                "box": number_xyxy[number_row].tolist(),
                            })

        presence[str(result.frame_idx)] = sorted(int(p) for p in player_ids)
        if result.frame_idx % args.ocr_every == 0:
            resolver.resolve(result.frame_idx)
        annotated = annotate_frame(
            frame_bgr, result.boxes, player_ids, players, masks,
            voter, roster, mask_annotator, label_annotator,
            canonical=registry.canonical,
        )
        draw_legend(annotated, palette)
        writer.write(annotated)
        frames_written += 1
        if preview is None and result.frame_idx >= info.total_frames // 2:
            preview = annotated.copy()

    writer.release()
    if preview is not None:
        if preview.shape[1] > 1600:
            ratio = 1600 / preview.shape[1]
            preview = cv2.resize(
                preview, (1600, round(preview.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(preview_path), preview)

    if cached_reads is None:
        reads_cache_path.write_text(json.dumps(reads_log))
        print(f"[reads] wrote {sum(len(v) for v in reads_log.values())} reads -> {reads_cache_path}")

    everyone = list(registry.live.values()) + registry.retired
    result_dict = {
        "source": str(source),
        "output": str(args.output),
        "preview": str(preview_path),
        "tracker": "sam2",
        "reid_embedding": args.reid_embedding,
        # Two different recognisers are both called "parseq" -- docTR's
        # reimplementation (`--reader doctr --doctr-arch parseq`, 0.622 on the
        # labelled set) and the original baudm weights (`--reader parseq`,
        # 0.858). A run that does not say which it used cannot be compared.
        "reader": (
            f"doctr:{args.doctr_arch}" if args.reader == "doctr" else args.reader
        ),
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
    }
    args.output.with_suffix(".json").write_text(json.dumps(result_dict, indent=2))
    return result_dict


def main() -> None:
    args = parse_args()
    runner = render_sam2 if args.tracker == "sam2" else render
    print(json.dumps(runner(args), indent=2))


if __name__ == "__main__":
    main()
