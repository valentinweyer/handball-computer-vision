import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_UPSTREAM_DIR = Path(os.getenv("SAM2_UPSTREAM_DIR", PROJECT_ROOT / "sam2-upstream"))
if not SAM2_UPSTREAM_DIR.is_dir():
    raise FileNotFoundError(
        "The archived SAM2 baseline needs an external facebookresearch/sam2 "
        "checkout. Set SAM2_UPSTREAM_DIR to its path before running this module."
    )

# `inference` (imported below via sports/roboflow) depends on RF-SAM-2, whose
# .pth file puts segment-anything-2-real-time on sys.path at interpreter
# startup. If that import happens before this insert, `sam2` gets cached in
# sys.modules pointing at the wrong package and this insert becomes a no-op --
# so this must run before ANY other import in this file, not just before
# `from sam2...`.
sys.path.insert(0, str(SAM2_UPSTREAM_DIR))

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = (
    os.environ["CUDA_HOME"] + "/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)

import subprocess

import cv2
import numpy as np
import torch
from tqdm import tqdm

import supervision as sv
from inference import get_model
from sports import (
    clean_paths,
    MeasurementUnit,
    ViewTransformer,
)
from sports.handball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
)

from handball_cv.teams.model import TeamModel
from handball_cv.tracking.sam2_manager import TrackManager
from handball_cv.tracking.mask_cache import MaskCache
from sam2.build_sam import build_sam2_video_predictor

# ── paths ─────────────────────────────────────────────────────────────────────

SOURCE_VIDEO_PATH = Path(os.getenv("HANDBALL_CV_VIDEO", PROJECT_ROOT / "data/raw/Han-Ber4.mp4"))
EXPERIMENT_DIR    = PROJECT_ROOT / "runs/sam2_baseline" / SOURCE_VIDEO_PATH.stem
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_CACHE_DIR   = PROJECT_ROOT / "data/cache/frames" / SOURCE_VIDEO_PATH.stem
OUTPUT_VIDEO      = EXPERIMENT_DIR / "overlay.mp4"
OUTPUT_COMPRESSED = EXPERIMENT_DIR / "overlay-h264.mp4"

# ── model config ──────────────────────────────────────────────────────────────

PLAYER_MODEL_ID            = "player-and-handball-detection-3z9xf/3"
PLAYER_CONFIDENCE          = 0.5
PLAYER_IOU                 = 0.9
GOALKEEPER_CLASS_ID        = 1
FIELD_PLAYER_CLASS_ID      = 2
PLAYER_CLASS_IDS           = [GOALKEEPER_CLASS_ID, FIELD_PLAYER_CLASS_ID]

KEYPOINT_MODEL_ID          = "keypointv333-uwois-xprdi/4"
KEYPOINT_CONFIDENCE        = 0.5
KEYPOINT_ANCHOR_CONFIDENCE = 0.5

# ── team config ───────────────────────────────────────────────────────────────
# Cluster 0 / cluster 1 are arbitrary KMeans labels, not a known mapping to a
# real side -- HAN vs BER is this clip's actual matchup (was previously the
# placeholder "Porto"/"SCM" from an unrelated game).

TEAM_NAMES  = {0: "HAN", 1: "BER"}
TEAM_COLORS = {"HAN": "#0000FF", "BER": "#FF0000"}
TEAM_MODEL_CACHE = PROJECT_ROOT / "data/cache/team_models" / f"{SOURCE_VIDEO_PATH.stem}.pkl"

# ── court render config ───────────────────────────────────────────────────────

COURT_SCALE          = 0.5
COURT_PADDING        = 50
COURT_LINE_THICKNESS = 6

# ── track lifecycle config ────────────────────────────────────────────────────

CHECK_EVERY      = 10    # detector checkpoint cadence, in frames
COURT_MARGIN_CM  = 300   # slack around the court polygon for the "is this a
                         # player, not bench/crowd" test used to gate new adds

# ── overlay config ────────────────────────────────────────────────────────────
# SAM2's masks were previously discarded after boxes were derived from them.
# They are cached to disk here so pass 3 can draw them, matching the McByte
# pipeline's overlay so the two outputs are directly comparable.

DRAW_MASKS     = os.getenv("DRAW_MASKS", "1") == "1"
DRAW_KEYPOINTS = os.getenv("DRAW_KEYPOINTS", "0") == "1"  # court landmarks, debug only
MASK_OUTLINE   = 3      # masks drawn as outlines, not fills -- a team-coloured
                        # fill hides the jersey, which is the only cue that says
                        # whether an ID is sitting on the right player
MASK_CACHE_DIR = PROJECT_ROOT / "data/cache/masks" / SOURCE_VIDEO_PATH.stem / "sam2"
RUN_NAME       = os.getenv("RUN_NAME", "sam2")

# ── load models ───────────────────────────────────────────────────────────────

PLAYER_MODEL   = get_model(model_id=PLAYER_MODEL_ID)
KEYPOINT_MODEL = get_model(model_id=KEYPOINT_MODEL_ID)
config         = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)

_court_vertices = np.array(config.vertices)
COURT_MIN = _court_vertices.min(axis=0) - COURT_MARGIN_CM
COURT_MAX = _court_vertices.max(axis=0) + COURT_MARGIN_CM

# ── SAM2 (upstream) ───────────────────────────────────────────────────────────
# Offline predictor, not the real-time camera predictor: e2e.py already reads
# a file and makes multiple full passes over it, so streaming bought nothing.
# Upstream adds what the real-time fork never implemented -- add_new_object
# after tracking starts and remove_object -- which the lifecycle manager below
# depends on.

SAM2_CHECKPOINT = os.getenv(
    "SAM2_CHECKPOINT",
    str(PROJECT_ROOT / "segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"),
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)


def detect_players(frame: np.ndarray) -> sv.Detections:
    result = PLAYER_MODEL.infer(frame, confidence=PLAYER_CONFIDENCE, iou_threshold=PLAYER_IOU)[0]
    det = sv.Detections.from_inference(result)
    return det[np.isin(det.class_id, PLAYER_CLASS_IDS)]


def compute_homography(frame: np.ndarray):
    """Returns (ViewTransformer | None, keypoint_cache_for_overlay)."""
    kp_result     = KEYPOINT_MODEL.infer(frame, confidence=KEYPOINT_CONFIDENCE)[0]
    raw_kps       = kp_result.predictions[0].keypoints if kp_result.predictions else []
    confident_kps = [kp for kp in raw_kps if kp.confidence > KEYPOINT_ANCHOR_CONFIDENCE]

    seen, deduped = set(), []
    for kp in confident_kps:
        pt = tuple(config.vertices[int(kp.class_name) - 1])
        if pt not in seen:
            seen.add(pt)
            deduped.append(kp)
    confident_kps = deduped

    kp_cache = [
        (int(kp.class_name), float(kp.x), float(kp.y), float(kp.confidence))
        for kp in confident_kps
    ]
    if len(confident_kps) < 4:
        return None, kp_cache

    landmark_indices = np.array([int(kp.class_name) - 1 for kp in confident_kps])
    court_landmarks  = np.array(config.vertices)[landmark_indices]
    frame_landmarks  = np.array([[kp.x, kp.y] for kp in confident_kps], dtype=np.float32)
    try:
        transformer = ViewTransformer(source=frame_landmarks, target=court_landmarks)
    except Exception:
        return None, kp_cache
    return transformer, kp_cache


def make_court_test(transformer):
    if transformer is None:
        # can't verify court membership this checkpoint -- reject new adds
        # rather than risk seeding a bench/crowd detection as a player
        return lambda xyxy: False

    def test(xyxy: np.ndarray) -> bool:
        cx = (xyxy[0] + xyxy[2]) / 2.0
        by = xyxy[3]
        pt = transformer.transform_points(points=np.array([[cx, by]], dtype=np.float32))[0]
        return bool(np.all(COURT_MIN <= pt) and np.all(pt <= COURT_MAX))

    return test


def masks_from_logits(mask_logits: torch.Tensor) -> np.ndarray:
    """(N, 1, H, W) logits -> (N, H, W) bool, edge-fragment filtered."""
    masks = (mask_logits > 0.0).squeeze(1).cpu().numpy().astype(bool)
    return np.array([
        sv.filter_segments_by_distance(m, relative_distance=0.03, mode="edge")
        for m in masks
    ])


# ── extract frames to a JPEG cache ────────────────────────────────────────────
# decord (SAM2's mp4 backend) has no reliable ARM64 wheel here; init_state
# accepts a JPEG directory instead.

if not FRAME_CACHE_DIR.exists() or not any(FRAME_CACHE_DIR.glob("*.jpg")):
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(SOURCE_VIDEO_PATH),
         "-q:v", "2", "-start_number", "0", str(FRAME_CACHE_DIR / "%05d.jpg")],
        check=True,
    )

frame_files = sorted(FRAME_CACHE_DIR.glob("*.jpg"), key=lambda p: int(p.stem))
num_frames  = len(frame_files)


def read_frame(idx: int) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(frame_files[idx])), cv2.COLOR_BGR2RGB)


# ── fit team classifier + detect players + seed tracks on frame 0 ────────────
# Fit across the whole clip, not just frame 0 -- a dozen crops is below what a
# stable fit needs, and goalkeepers (a third kit) are excluded so their colour
# can't drag the fit off the two real team colours. Cached to disk so repeat
# runs are both cheaper and directly comparable (same fit -> same cluster
# labels); load_or_fit refits automatically if the cache predates the current
# crop geometry or model (team_model.MODEL_SCHEMA_VERSION).

team_model = TeamModel.load_or_fit(
    TEAM_MODEL_CACHE, SOURCE_VIDEO_PATH, detect_players,
    exclude_class_ids=(GOALKEEPER_CLASS_ID,), device="cuda",
)

frame0     = read_frame(0)
detections = detect_players(frame0)
if len(detections) == 0:
    raise RuntimeError("no players detected on frame 0")

track_manager = TrackManager(team_model, court_test_fn=lambda b: True)
is_gk0        = detections.class_id == GOALKEEPER_CLASS_ID
obj_ids0      = track_manager.seed(0, detections.xyxy, frame0, is_gk0)

# obj_id -> materialized-array column, in creation order (never reused/reordered)
obj_id_to_col: dict[int, int] = {oid: i for i, oid in enumerate(obj_ids0)}

mask_cache = MaskCache(MASK_CACHE_DIR, enabled=DRAW_MASKS)

state = predictor.init_state(video_path=str(FRAME_CACHE_DIR))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for oid, xyxy in zip(obj_ids0, detections.xyxy):
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=oid, box=np.asarray(xyxy, dtype=np.float32)
        )

# ── pass 1: SAM2 + homography + track lifecycle ──────────────────────────────
# Rendering (pass 3, below) mirrors the original script and starts at frame 1
# -- frame 0 is used only for seeding/team-fitting, never rendered -- so only
# frames [1, num_frames) are materialized into the output arrays.

per_frame_xy:    dict[int, dict[int, tuple]] = {}
per_frame_boxes: dict[int, dict[int, np.ndarray]] = {}
per_frame_kps:   dict[int, list] = {}

next_new_fid = 1  # frame 0 already handled above; skip re-recording it
last_masks_by_id: dict[int, np.ndarray] = {}

print("Pass 1: tracking + lifecycle management + homography...")
pbar = tqdm(total=num_frames - 1, desc="SAM2 + homography")
t = 0
while t < num_frames - 1:
    chunk_len = min(CHECK_EVERY, num_frames - 1 - t)
    chunk_end = t

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for fid, obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=t, max_frame_num_to_track=chunk_len):
            if fid < next_new_fid:
                continue  # repeated boundary frame from the previous chunk

            masks = masks_from_logits(mask_logits)
            track_manager.update_from_propagation(fid, obj_ids, masks)

            boxes_this = sv.mask_to_xyxy(masks=masks)
            per_frame_boxes[fid] = dict(zip(obj_ids, boxes_this))
            mask_cache.save(fid, masks, [obj_id_to_col.get(int(o)) for o in obj_ids])

            frame = read_frame(fid)
            transformer, kp_cache = compute_homography(frame)
            per_frame_kps[fid] = kp_cache
            if transformer is not None:
                det_tmp = sv.Detections(xyxy=boxes_this)
                anchors = det_tmp.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                court_xy = transformer.transform_points(points=anchors)
                per_frame_xy[fid] = dict(zip(obj_ids, map(tuple, court_xy)))
            else:
                per_frame_xy[fid] = {}

            chunk_end = fid
            if fid == t + chunk_len:  # last frame of this chunk
                last_masks_by_id = dict(zip(obj_ids, masks))
                last_frame, last_transformer = frame, transformer
            pbar.update(1)

    next_new_fid = chunk_end + 1
    t = chunk_end

    # ── detector checkpoint: decide add / remove / reprompt ──
    live_obj_ids = list(last_masks_by_id.keys())
    live_masks   = np.array([last_masks_by_id[oid] for oid in live_obj_ids]) \
        if live_obj_ids else np.empty((0, 0, 0), dtype=bool)
    det = detect_players(last_frame)
    det_is_goalkeeper = det.class_id == GOALKEEPER_CLASS_ID

    track_manager.court_test_fn = make_court_test(last_transformer)
    actions = track_manager.checkpoint(
        chunk_end, last_frame, live_obj_ids, live_masks, det.xyxy, det_is_goalkeeper)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for action in actions:
            if action["type"] == "remove":
                predictor.remove_object(state, obj_id=action["obj_id"])
            elif action["type"] == "reprompt":
                predictor.add_new_points_or_box(
                    state, frame_idx=chunk_end, obj_id=action["obj_id"],
                    box=np.asarray(action["box"], dtype=np.float32),
                    clear_old_points=True,
                )
            elif action["type"] == "add":
                oid = action["obj_id"]
                predictor.add_new_points_or_box(
                    state, frame_idx=chunk_end, obj_id=oid,
                    box=np.asarray(action["box"], dtype=np.float32),
                )
                if oid not in obj_id_to_col:
                    obj_id_to_col[oid] = len(obj_id_to_col)

pbar.close()

print(f"track lifecycle events: {len(track_manager.events)}")
for e in track_manager.events:
    print(f"  frame {e['frame']:>4}  {e['type']:<16} obj_id={e['obj_id']}")

# obj_id -> current best (voted) team id. Computed once, at the end, rather than
# captured at creation time -- `Track.voted_team_id` accumulates evidence across
# the whole run, so this is more reliable than any single frame's read. Every
# obj_id ever allocated stays in `track_manager.tracks` even after a "remove"
# action (only the SAM2 predictor's session forgets it), so this is safe to do
# for every column, including ones later removed from the live set.
team_by_obj_id: dict[int, int] = {
    oid: track_manager.tracks[oid].voted_team_id for oid in obj_id_to_col
}

# ── materialize dynamic per-frame dicts into fixed arrays ────────────────────

P_total = len(obj_id_to_col)
T       = num_frames - 1  # frames [1, num_frames), matching pass-3 rendering

video_xy    = np.full((T, P_total, 2), np.nan)
video_boxes = np.full((T, P_total, 4), np.nan)
video_keypoints = [per_frame_kps.get(fid, []) for fid in range(1, num_frames)]

for t_idx, fid in enumerate(range(1, num_frames)):
    for oid, xyxy in per_frame_boxes.get(fid, {}).items():
        col = obj_id_to_col.get(oid)
        if col is not None:
            video_boxes[t_idx, col] = xyxy
    for oid, xy in per_frame_xy.get(fid, {}).items():
        col = obj_id_to_col.get(oid)
        if col is not None:
            video_xy[t_idx, col] = xy

TEAMS = np.array([team_by_obj_id[oid] for oid, _ in
                   sorted(obj_id_to_col.items(), key=lambda kv: kv[1])])
# goalkeeper status per column -- from the detector's own class at track
# creation, never re-guessed. Needed downstream (team_labels.py) to keep
# excluding keepers from the team fit consistently with fit_from_video.
IS_GOALKEEPER = np.array([track_manager.tracks[oid].is_goalkeeper for oid, _ in
                           sorted(obj_id_to_col.items(), key=lambda kv: kv[1])])

# ── dump the run for compare_trackers.py ─────────────────────────────────────
# Eyeballing catches fragmentation but not ID swaps. Persisting boxes + ids lets
# one offline comparator score both pipelines the same way.

RUN_DIR = PROJECT_ROOT / "runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RUN_DIR / f"{RUN_NAME}.npz",
    boxes=video_boxes,
    teams=TEAMS,
    is_goalkeeper=IS_GOALKEEPER,
    track_ids=np.array([oid for oid, _ in sorted(obj_id_to_col.items(), key=lambda kv: kv[1])], dtype=int),
    source=str(SOURCE_VIDEO_PATH),
)
print("run dumped ->", RUN_DIR / f"{RUN_NAME}.npz")


# ── duplicate-track sanity check ──────────────────────────────────────────────
# A real duplicate needs one checkpoint to be detected, a second to confirm
# (MIN_CONFIRM_CHECKPOINTS=2), plus up to CHECK_EVERY frames of latency before
# either checkpoint fires -- worst case ~3*CHECK_EVERY frames of continuous
# overlap. Anything sustained longer than that slipped past the lifecycle
# manager. Checked over every frame (not just checkpoints) against box IoU as
# a proxy for the mask-based rule the manager itself uses.

_MAX_OVERLAP_FRAMES = 3 * CHECK_EVERY
_consec_overlap: dict[tuple, int] = {}
_unresolved = []
for t_idx in range(T):
    boxes_t = video_boxes[t_idx]
    valid   = np.isfinite(boxes_t).all(axis=1)
    idxs    = np.nonzero(valid)[0]
    seen_pairs = set()
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            a, b = int(idxs[i]), int(idxs[j])
            iou = sv.box_iou_batch(boxes_t[a][None], boxes_t[b][None])[0, 0]
            key = (a, b)
            seen_pairs.add(key)
            if iou > 0.6:
                _consec_overlap[key] = _consec_overlap.get(key, 0) + 1
                if _consec_overlap[key] > _MAX_OVERLAP_FRAMES:
                    _unresolved.append((t_idx + 1, a, b, float(iou)))
            else:
                _consec_overlap[key] = 0
    for key in list(_consec_overlap):
        if key not in seen_pairs:
            _consec_overlap[key] = 0

if _unresolved:
    print(f"WARNING: duplicate track pairs survived >{_MAX_OVERLAP_FRAMES} frames:")
    for fid, a, b, iou in _unresolved[:10]:
        print(f"  frame {fid}: columns {a} & {b}, IoU={iou:.2f}")
assert not _unresolved, f"{len(_unresolved)} duplicate track pairs were not resolved"

# ── pass 2: clean and smooth ──────────────────────────────────────────────────

print("Pass 2: cleaning paths...")
cleaned_xy, _ = clean_paths(
    video_xy,
    jump_sigma=5.0,
    min_jump_dist=1.5,
    max_jump_run=10,
    pad_around_runs=1,
    smooth_window=5,
    smooth_poly=2,
)

# ── pass 3: render side-by-side from cache (no SAM2) ─────────────────────────

court_ref = draw_court(
    config=config, scale=COURT_SCALE,
    padding=COURT_PADDING, line_thickness=COURT_LINE_THICKNESS,
)
court_h, court_w, _ = court_ref.shape

video_info  = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
orig_h      = video_info.height
orig_w      = video_info.width
scale_orig  = court_h / orig_h
new_orig_w  = int(orig_w * scale_orig)
canvas_w    = new_orig_w + court_w
canvas_h    = court_h

out_info        = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
out_info.width  = canvas_w
out_info.height = canvas_h

team_colors    = sv.ColorPalette.from_hex([TEAM_COLORS[TEAM_NAMES[0]], TEAM_COLORS[TEAM_NAMES[1]]])
box_annotator  = sv.BoxAnnotator(color=team_colors, thickness=2)
mask_annotator = sv.PolygonAnnotator(color=team_colors, thickness=MASK_OUTLINE)
label_annotator = sv.LabelAnnotator(
    color=team_colors, text_color=sv.Color.WHITE, text_scale=0.7,
    text_thickness=2, text_position=sv.Position.TOP_CENTER,
)

# column -> the obj_id it renders, for the on-box label
col_to_obj  = {col: oid for oid, col in obj_id_to_col.items()}
frame_shape = (video_info.height, video_info.width)

print("Pass 3: rendering side-by-side...")
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
next(frame_generator)  # skip first frame

with sv.VideoSink(OUTPUT_VIDEO, out_info) as sink:
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=len(cleaned_xy), desc="rendering")):
        boxes_frame = video_boxes[frame_idx]  # (P_total, 4)
        valid_mask  = np.isfinite(boxes_frame).all(axis=1)

        annotated = frame.copy()

        if valid_mask.any():
            valid_cols  = np.nonzero(valid_mask)[0]
            valid_xyxy  = boxes_frame[valid_mask]
            valid_teams = TEAMS[valid_mask]
            fid         = frame_idx + 1  # video_boxes row 0 is frame 1

            masks = mask_cache.load(fid, valid_cols, frame_shape)
            det_for_ann = sv.Detections(xyxy=valid_xyxy, mask=masks)

            if masks is not None:
                annotated = mask_annotator.annotate(
                    scene=annotated, detections=det_for_ann,
                    custom_color_lookup=valid_teams,
                )
            annotated = box_annotator.annotate(
                scene=annotated, detections=det_for_ann,
                custom_color_lookup=valid_teams,
            )
            annotated = label_annotator.annotate(
                scene=annotated, detections=det_for_ann,
                labels=[str(col_to_obj[int(c)]) for c in valid_cols],
                custom_color_lookup=valid_teams,
            )

        if DRAW_KEYPOINTS:
            for class_id, kp_x, kp_y, kp_conf in video_keypoints[frame_idx]:
                px, py = int(round(kp_x)), int(round(kp_y))
                cv2.circle(annotated, (px, py), 6, (0, 255, 0), -1)
                label = str(class_id)
                font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
                tx, ty = px + 8, py + th // 2
                cv2.rectangle(annotated, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(annotated, label, (tx, ty), font, fs, (0, 255, 0), thick)

        orig_resized = cv2.resize(annotated, (new_orig_w, court_h))

        court    = draw_court(
            config=config, scale=COURT_SCALE,
            padding=COURT_PADDING, line_thickness=COURT_LINE_THICKNESS,
        )
        frame_xy = cleaned_xy[frame_idx]  # (P_total, 2)
        for team_id in [0, 1]:
            pts   = frame_xy[TEAMS == team_id]
            valid = np.isfinite(pts).all(axis=1)
            if valid.any():
                court = draw_points_on_court(
                    config=config,
                    xy=pts[valid],
                    fill_color=sv.Color.from_hex(TEAM_COLORS[TEAM_NAMES[team_id]]),
                    court=court,
                    scale=COURT_SCALE,
                    padding=COURT_PADDING,
                    line_thickness=COURT_LINE_THICKNESS,
                )

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :new_orig_w] = orig_resized
        canvas[:, new_orig_w:] = court
        sink.write_frame(canvas)

os.system(f"ffmpeg -y -loglevel error -i {OUTPUT_VIDEO} -vcodec libx264 -crf 28 {OUTPUT_COMPRESSED}")
print("done →", OUTPUT_COMPRESSED)
