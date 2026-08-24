"""End-to-end handball pipeline with McByte (roboflow/trackers 2.6.0) as the tracker.

A/B counterpart to e2e.py, which uses the SAM2 offline video predictor as the
tracker. Same detector, same homography, same court rendering -- only pass 1
differs. McByte is tracking-by-detection, so this runs as a single streaming pass
over the mp4: no JPEG frame cache, no chunked propagation, no add/remove/reprompt
lifecycle. Masks are still involved, but internally: McByte box-prompts SAM for a
new tracklet and propagates that mask with Cutie, using it as an association cue.

Correction happens in two places, both outside the tracker:
  - `filter_to_court` drops off-court detections *before* update(), every frame
    (e2e.py could only gate new adds, at 10-frame checkpoints)
  - IdentityManager remaps tracker_ids to stable player_ids *after* update()
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", ""))
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider]")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = (
    os.environ["CUDA_HOME"] + "/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)

import cv2
import numpy as np
from hydra.core.global_hydra import GlobalHydra
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
from trackers import McByteMaskConfig

from handball_cv.teams.model import TeamModel
from handball_cv.tracking.identity import IdentityManager
from handball_cv.tracking.mask_cache import MaskCache
from experiments.team_gated_tracking.tracker import (
    TEAM_PROBABILITY_KEY,
    TEAM_QUALITY_KEY,
    TeamGatedMcByteTracker,
)
from handball_cv.teams.calibration import (
    ROLE_REFEREE,
    CalibratedTeamObserver,
    PrototypeTeamCalibrator,
)

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_VIDEO_PATH = Path(os.getenv("HANDBALL_CV_VIDEO", PROJECT_ROOT / "data/raw/Han-Ber4.mp4"))
EXPERIMENT_DIR = PROJECT_ROOT / "runs/team_gated_tracking" / SOURCE_VIDEO_PATH.stem
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
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
# placeholder "Porto"/"SCM" from an unrelated game). Same cache path as
# e2e.py -- both pipelines fit on the same clip, so sharing one fitted model
# means their team colours are directly comparable instead of each run
# producing its own arbitrary cluster-to-side assignment.

TEAM_NAMES  = {0: "HAN", 1: "BER"}
TEAM_COLORS = {"HAN": "#0000FF", "BER": "#FF0000"}
TEAM_MODEL_CACHE = PROJECT_ROOT / "data/cache/team_models" / f"{SOURCE_VIDEO_PATH.stem}.pkl"
TEAM_CALIBRATION_CACHE = Path(os.getenv(
    "TEAM_CALIBRATION",
    PROJECT_ROOT / "data/cache/team_calibration" / f"{SOURCE_VIDEO_PATH.stem}.pkl",
))

# ── court render config ───────────────────────────────────────────────────────

COURT_SCALE          = 0.5
COURT_PADDING        = 50
COURT_LINE_THICKNESS = 6

# ── tracker config ────────────────────────────────────────────────────────────

ENABLE_MASKS    = os.getenv("MCBYTE_MASKS", "1") == "1"  # 0 = pure IoU, no SAM/Cutie
MASK_DEVICE     = "cuda"

# Env-overridable so compare_trackers.py can sweep without editing the file.
# minimum_mask_coverage is the one that bites in a scrum: it is the fraction of
# the whole propagated mask that must fall inside the detection box, and under
# occlusion the mask spills past the (smaller) visible box and the evidence gets
# discarded exactly when association is ambiguous.
LOST_TRACK_BUFFER    = int(os.getenv("MCBYTE_LOST_BUFFER", "30"))
TRACK_ACTIVATION     = float(os.getenv("MCBYTE_ACTIVATION", "0.7"))
MIN_MASK_COVERAGE    = float(os.getenv("MCBYTE_MASK_COVERAGE", "0.9"))
MIN_MASK_FILL_RATIO  = float(os.getenv("MCBYTE_MASK_FILL", "0.05"))
MIN_MASK_AVG_CONF    = float(os.getenv("MCBYTE_MASK_CONF", "0.6"))
ISOLATED_MASK_MATCH  = os.getenv("MCBYTE_ISOLATED", "0") == "1"
SKIP_RENDER          = os.getenv("SKIP_RENDER", "0") == "1"  # pass 1 + dump only

# ── overlay config ────────────────────────────────────────────────────────────

DRAW_MASKS     = ENABLE_MASKS and os.getenv("DRAW_MASKS", "1") == "1"
DRAW_KEYPOINTS = os.getenv("DRAW_KEYPOINTS", "0") == "1"  # court landmarks, debug only
MASK_OUTLINE   = 3      # masks drawn as outlines, not fills -- a team-coloured
                        # fill hides the jersey, which is the only cue that says
                        # whether an ID is sitting on the right player
# Masks are (K, 1080, 1920) bool per frame -- ~25 MB/frame dense, so they are
# cached to disk as a uint8 label map (0 = background, col+1 = player column)
# rather than held in RAM. Mostly-zero maps compress to a few KB each, which is
# what makes this survive a full match rather than a 199-frame clip.
MASK_CACHE_DIR = SOURCE_VIDEO_PATH.parent / f".{SOURCE_VIDEO_PATH.stem}_mcbyte_masks"
RUN_NAME       = os.getenv("RUN_NAME", "mcbyte")
COURT_MARGIN_CM = 300    # slack around the court polygon for the "player, not
                         # bench/crowd" test, now applied to every detection

# ── load models ───────────────────────────────────────────────────────────────

PLAYER_MODEL   = get_model(model_id=PLAYER_MODEL_ID)
KEYPOINT_MODEL = get_model(model_id=KEYPOINT_MODEL_ID)
config         = CourtConfiguration(league=League.IHF, measurement_unit=MeasurementUnit.CENTIMETERS)

_court_vertices = np.array(config.vertices)
COURT_MIN = _court_vertices.min(axis=0) - COURT_MARGIN_CM
COURT_MAX = _court_vertices.max(axis=0) + COURT_MARGIN_CM


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


def project_to_court(transformer, det: sv.Detections) -> np.ndarray:
    """Bottom-centre of each box, projected into court coordinates."""
    anchors = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    return transformer.transform_points(points=anchors)


def filter_to_court(det: sv.Detections, transformer) -> sv.Detections:
    """Drop detections whose court projection falls outside the padded court.

    e2e.py's `make_court_test` rejected everything when homography failed, because
    it only gated *new* SAM2 prompts and a false negative merely delayed an add.
    Here the test runs on every detection every frame, so failing open is the safe
    direction -- rejecting all would starve the tracker for the whole frame.
    """
    if transformer is None or len(det) == 0:
        return det
    pts  = project_to_court(transformer, det)
    keep = np.all(pts >= COURT_MIN, axis=1) & np.all(pts <= COURT_MAX, axis=1)
    return det[keep]


def cache_masks(fid: int, mask_output, identity, player_id_to_col: dict) -> None:
    """Flatten McByte's per-tracklet masks into the shared cache.

    `mask_output.masks` is (K, H, W) bool with `tracklet_mask_dict` mapping
    tracker_id -> row. Resolving that to player columns here means pass 3 never
    needs to know about tracker_ids. Note masks only appear once a tracklet has
    survived `minimum_mask_creation_frames` (3), and occluded tracklets sit in a
    pending pool without one -- so K is normally smaller than the tracked count.
    """
    if mask_output is None or mask_output.masks is None:
        return
    rows, cols = [], []
    for tracker_id, row in mask_output.tracklet_mask_dict.items():
        pid = identity.tracker_to_player.get(int(tracker_id))
        rows.append(row)
        cols.append(player_id_to_col.get(pid) if pid is not None else None)
    mask_cache.save(fid, mask_output.masks[rows], cols)


# ── video ─────────────────────────────────────────────────────────────────────

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
num_frames = video_info.total_frames

mask_cache = MaskCache(MASK_CACHE_DIR, enabled=DRAW_MASKS)

# `import inference` initialises Hydra globally (it pulls in RF-SAM-2), and Cutie's
# config loader calls initialize_config_dir, which refuses to run against a live
# GlobalHydra. Both detector models are already built above and this script never
# builds a SAM2/SAM3 model, so nothing depends on that global state -- clear it.
# Cutie's own initialize_config_dir is a scoped `with`, so it cleans up after itself.
if ENABLE_MASKS and GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

tracker = TeamGatedMcByteTracker(
    frame_rate=video_info.fps,
    lost_track_buffer=LOST_TRACK_BUFFER,
    track_activation_threshold=TRACK_ACTIVATION,
    enable_mask_manager=ENABLE_MASKS,
    mask_config=McByteMaskConfig(device=MASK_DEVICE) if ENABLE_MASKS else None,
    minimum_mask_average_confidence=MIN_MASK_AVG_CONF,
    minimum_mask_coverage=MIN_MASK_COVERAGE,
    minimum_mask_fill_ratio=MIN_MASK_FILL_RATIO,
    enable_isolated_mask_matching=ISOLATED_MASK_MATCH,
)
print(f"McByte config: buffer={LOST_TRACK_BUFFER} activation={TRACK_ACTIVATION} "
      f"mask_coverage={MIN_MASK_COVERAGE} mask_fill={MIN_MASK_FILL_RATIO} "
      f"isolated={ISOLATED_MASK_MATCH}")

# ── fit team classifier ───────────────────────────────────────────────────────
# Fit across the whole clip, not just frame 0 -- a dozen crops is below what a
# stable fit needs, and goalkeepers (a third kit) are excluded so their colour
# can't drag the fit off the two real team colours. Cached to disk so repeat
# runs are both cheaper and directly comparable (same fit -> same cluster
# labels), and shared with e2e.py via the same cache path. load_or_fit refits
# automatically if the cache predates the current crop geometry or model
# (team_model.MODEL_SCHEMA_VERSION).

team_model = TeamModel.load_or_fit(
    TEAM_MODEL_CACHE, SOURCE_VIDEO_PATH, detect_players,
    exclude_class_ids=(GOALKEEPER_CLASS_ID,), device="cuda",
)

team_calibrator = (
    PrototypeTeamCalibrator.load(TEAM_CALIBRATION_CACHE)
    if TEAM_CALIBRATION_CACHE.exists() else None
)
team_observer = (
    CalibratedTeamObserver(team_model, team_calibrator)
    if team_calibrator is not None else team_model
)
identity = IdentityManager(
    team_observer, goalkeeper_class_id=GOALKEEPER_CLASS_ID
)
print(
    "team calibration:",
    TEAM_CALIBRATION_CACHE
    if team_calibrator is not None else "none (gate disabled)",
)

# ── pass 1: detect → court gate → McByte → identity → homography ─────────────
# Frame 0 primes the tracker but is never rendered, matching e2e.py: the output
# covers frames [1, num_frames).

per_frame_xy:    dict[int, dict[int, tuple]] = {}
per_frame_boxes: dict[int, dict[int, np.ndarray]] = {}
per_frame_kps:   dict[int, list] = {}

player_id_to_col: dict[int, int] = {}

print(f"Pass 1: McByte tracking (masks={'on' if ENABLE_MASKS else 'off'}) + homography...")
t_start = time.perf_counter()

for fid, frame_bgr in enumerate(tqdm(
        sv.get_video_frames_generator(SOURCE_VIDEO_PATH),
        total=num_frames, desc="McByte + homography")):
    # the detector, TeamModel and McByte's SAM/Cutie backends all consume RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    det = detect_players(frame_rgb)
    transformer, kp_cache = compute_homography(frame_rgb)

    if fid == 0 and len(det) == 0:
        raise RuntimeError("no players detected on frame 0")

    det = filter_to_court(det, transformer)
    if team_calibrator is not None and len(det):
        probability_b, _certainty, role, role_confidence, quality = (
            team_calibrator.predict_frame(frame_rgb, det.xyxy, det.xyxy)
        )
        is_goalkeeper = det.class_id == GOALKEEPER_CLASS_ID
        probability_b = probability_b.astype(float)
        quality = quality.astype(float)
        probability_b[is_goalkeeper] = np.nan
        quality[is_goalkeeper] = 0.0
        det.data[TEAM_PROBABILITY_KEY] = probability_b
        det.data[TEAM_QUALITY_KEY] = quality
        excluded_role = (
            (det.class_id == FIELD_PLAYER_CLASS_ID)
            & (role == ROLE_REFEREE)
            & (role_confidence >= 0.15)
            & (quality > 0)
        )
        det = det[~excluded_role]
    det = tracker.update(det, frame=frame_rgb)
    det = det[det.tracker_id >= 0]

    player_ids = identity.update(fid, frame_rgb, det)

    for pid in player_ids:
        if int(pid) not in player_id_to_col:
            player_id_to_col[int(pid)] = len(player_id_to_col)

    # Cache this frame's masks before retiring anyone -- retire_missing drops the
    # tracker_id -> player_id entries the mask dict is keyed through.
    if DRAW_MASKS:
        cache_masks(fid, tracker._last_mask_output, identity, player_id_to_col)

    identity.retire_missing(fid, tracker.tracked_objects.tracker_id)

    if fid == 0:
        continue  # primed the tracker and the classifier; nothing to render

    per_frame_kps[fid]   = kp_cache
    per_frame_boxes[fid] = {int(p): b for p, b in zip(player_ids, det.xyxy)}
    if transformer is not None and len(det):
        court_xy = project_to_court(transformer, det)
        per_frame_xy[fid] = {int(p): tuple(xy) for p, xy in zip(player_ids, court_xy)}
    else:
        per_frame_xy[fid] = {}

pass1_seconds = time.perf_counter() - t_start

# ── tracking summary ──────────────────────────────────────────────────────────

summary = identity.summary()
print(f"\npass 1 wall clock: {pass1_seconds:.1f}s ({num_frames / pass1_seconds:.2f} fps)")
print("identity summary:")
for k, v in summary.items():
    print(f"  {k:<26} {v}")
print(f"identity events: {len(identity.events)}")
for e in identity.events:
    print(f"  frame {e['frame']:>4}  {e['type']:<8} player_id={e['player_id']} tracker_id={e['tracker_id']}")

# ── materialize dynamic per-frame dicts into fixed arrays ────────────────────

P_total = len(player_id_to_col)
T       = num_frames - 1  # frames [1, num_frames)

video_xy        = np.full((T, P_total, 2), np.nan)
video_boxes     = np.full((T, P_total, 4), np.nan)
video_keypoints = [per_frame_kps.get(fid, []) for fid in range(1, num_frames)]

for t_idx, fid in enumerate(range(1, num_frames)):
    for pid, xyxy in per_frame_boxes.get(fid, {}).items():
        col = player_id_to_col.get(pid)
        if col is not None:
            video_boxes[t_idx, col] = xyxy
    for pid, xy in per_frame_xy.get(fid, {}).items():
        col = player_id_to_col.get(pid)
        if col is not None:
            video_xy[t_idx, col] = xy

team_by_player_id = identity.team_by_player_id()
TEAMS = np.array([team_by_player_id[pid] for pid, _ in
                  sorted(player_id_to_col.items(), key=lambda kv: kv[1])])

# ── dump the run for compare_trackers.py ─────────────────────────────────────
# Eyeballing catches fragmentation but not ID swaps. Persisting boxes + ids lets
# one offline comparator score both pipelines the same way.

RUN_DIR = PROJECT_ROOT / "runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    RUN_DIR / f"{RUN_NAME}.npz",
    boxes=video_boxes,
    teams=TEAMS,
    track_ids=np.array([pid for pid, _ in sorted(player_id_to_col.items(), key=lambda kv: kv[1])], dtype=int),
    source=str(SOURCE_VIDEO_PATH),
)
print("run dumped ->", RUN_DIR / f"{RUN_NAME}.npz")


# ── duplicate-track check (A/B metric, not an invariant) ─────────────────────
# McByte associates 1:1 per frame, so sustained overlap between two player_ids
# means the identity layer split one player in two -- reported, not fatal.

_MAX_OVERLAP_FRAMES = 30
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

print(f"duplicate-overlap frames (>{_MAX_OVERLAP_FRAMES} consecutive, IoU>0.6): {len(_unresolved)}")
for fid, a, b, iou in _unresolved[:10]:
    print(f"  frame {fid}: columns {a} & {b}, IoU={iou:.2f}")

if SKIP_RENDER:
    print("SKIP_RENDER set -- stopping after the run dump")
    raise SystemExit(0)

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

# ── pass 3: render side-by-side ───────────────────────────────────────────────

court_ref = draw_court(
    config=config, scale=COURT_SCALE,
    padding=COURT_PADDING, line_thickness=COURT_LINE_THICKNESS,
)
court_h, court_w, _ = court_ref.shape

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

# column -> the stable player_id it renders, for the on-box label
col_to_player = {col: pid for pid, col in player_id_to_col.items()}
frame_shape   = (video_info.height, video_info.width)

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
                labels=[str(col_to_player[int(c)]) for c in valid_cols],
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
