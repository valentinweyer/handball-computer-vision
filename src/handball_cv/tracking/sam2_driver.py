"""Reusable SAM2 predictor + TrackManager checkpoint driver.

Extracted from `scripts.run_sam2_reprompt_tracker`, which originally inlined
this whole loop for one purpose: dumping a tracking-only `(frame_index,
tracker_id, box)` replay for `evaluate_tracker_identity`. A second consumer
needs the same propagation + periodic reprompting -- reading jersey numbers
per player -- and duplicating a ~140-line predictor/state-machine loop for
that would leave two copies of the exact re-ID/lifecycle sequencing this
project has already spent real effort getting right. This module is that loop,
factored so both consumers drive it once.

`SAM2_UPSTREAM_DIR` must already be on `sys.path` before importing this module
(see the guard at the top of `scripts.run_sam2_reprompt_tracker` -- SAM2 has no
importable package name until its checkout is added to the path, so the guard
has to run in the entry-point script, before any import of this module).
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import functools
from pathlib import Path
import subprocess
from typing import Callable, Iterator

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from handball_cv.video import VideoFrames, ffmpeg_exe  # ffmpeg_exe: re-exported for existing callers
from handball_cv.tracking.sam2_acceleration import install_fused_rope, install_small_module_graphs
from handball_cv.tracking.sam2_manager import TrackManager

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECK_EVERY_DEFAULT = 10  # matches run_pipeline.py's detector-checkpoint cadence
# How long the court must be back before a paused tracker resumes. A lights-off
# timeout on the first broadcast was broken by three ~1 s stretches where a
# spotlight lit enough floor for the court fit to see a court -- still the
# huddle, not play resuming -- and restarting for each would create and revive
# players inside it. 2 s at 25 fps clears them.
RESUME_AFTER_FRAMES = 50


def ensure_frame_cache(video: Path, frame_cache_dir: Path) -> list[Path]:
    """Extract `video` to `frame_cache_dir` as numbered JPGs if not already done."""
    if not frame_cache_dir.exists() or not any(frame_cache_dir.glob("*.jpg")):
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
             "-q:v", "2", "-start_number", "0", str(frame_cache_dir / "%05d.jpg")],
            check=True,
        )
    return sorted(frame_cache_dir.glob("*.jpg"), key=lambda p: int(p.stem))


def filter_edge_fragments(
    mask: np.ndarray, relative_distance: float, connectivity: int = 8
) -> np.ndarray:
    """Byte-identical, ~9x faster `sv.filter_segments_by_distance(mode="edge")`.

    Keeps the largest connected component, plus every component whose nearest
    edge lies within `relative_distance * image_diagonal` of it.

    Supervision spends ~31ms per 1080p mask inside its `_chamfer_distances`, a
    pure-Python row loop (2*H iterations of full-width int64 numpy ops) that
    reimplements `cv2.distanceTransform(..., cv2.DIST_L2, 3)`: its fixed-point
    weights 62587/65536 and 89738/65536 are exactly OpenCV's DIST_L2 maskSize=3
    constants, and the two agree bit-for-bit. That loop is O(H*W) per object
    regardless of what the mask contains, so at 14 players it costs ~434ms per
    frame against a ~1s/frame propagation budget -- a cost no faster
    segmentation model can touch, since it runs after the model.

    Three changes, none of them semantic:

      * `cv2.distanceTransform` in place of the Python chamfer loop;
      * one `np.unique` pass instead of a full-image comparison per component;
      * work inside the mask's bounding box -- every component is made of mask
        pixels so none is lost, and a chamfer path between two points inside a
        rectangle stays inside it -- while the threshold still comes from the
        *full* image diagonal, not the crop's.

    Equivalence with supervision is pinned by
    `tests/unit/test_filter_edge_fragments.py`; keep it that way, so this stays
    a speed change and never a silent change to the boxes the identity layer
    consumes.
    """
    if mask.dtype != bool:
        raise TypeError("mask must be boolean")
    if not mask.any():
        return mask.copy()

    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask[y0:y1, x0:x1].astype(np.uint8), connectivity=connectivity
    )
    if num_labels <= 1:
        return mask.copy()

    # Largest component, ties broken by position then label -- the crop shifts
    # every component's coordinates equally, so this picks what supervision picks.
    areas = stats[1:, cv2.CC_STAT_AREA]
    candidates = 1 + np.flatnonzero(areas == int(areas.max()))
    main_label = min(
        (int(label) for label in candidates),
        key=lambda label: (int(stats[label, 0]), int(stats[label, 1]), label),
    )

    threshold = float(relative_distance) * float(np.hypot(height, width))
    main_mask = labels == main_label
    distances = cv2.distanceTransform((~main_mask).astype(np.uint8), cv2.DIST_L2, 3)

    keep = np.zeros(num_labels, dtype=bool)
    keep[np.unique(labels[distances <= threshold])] = True
    keep[main_label] = True
    keep[0] = False  # background touches the threshold band; it is not a component

    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = keep[labels]
    return out


_PINNED_MASKS: torch.Tensor | None = None


def _to_host(masks: torch.Tensor) -> np.ndarray:
    """Copy a CUDA bool stack to host through a reused pinned buffer.

    On GB10 a plain `.cpu()` of the 13 x 1080p stack measured ~67 ms -- it
    allocates and faults in fresh pageable memory every frame -- against ~1 ms
    into page-locked memory, with identical bytes. The returned array views the
    buffer, so it is only valid until the next call; `masks_from_logits` copies
    out of it before returning.
    """
    global _PINNED_MASKS
    if not masks.is_cuda:
        return masks.numpy()
    if (_PINNED_MASKS is None or _PINNED_MASKS.shape[1:] != masks.shape[1:]
            or _PINNED_MASKS.shape[0] < masks.shape[0]):
        rows = max(masks.shape[0], 32)
        _PINNED_MASKS = torch.empty((rows, *masks.shape[1:]), dtype=torch.bool, pin_memory=True)
    view = _PINNED_MASKS[:masks.shape[0]]
    view.copy_(masks)
    return view.numpy()


def masks_from_logits(mask_logits: torch.Tensor) -> np.ndarray:
    """(N, 1, H, W) logits -> (N, H, W) bool, edge-fragment filtered."""
    masks = _to_host((mask_logits > 0.0).squeeze(1))
    # reshape keeps (0, H, W) when nothing is tracked; np.array of an empty
    # list would be (0,), which breaks every consumer of the stack.
    return np.array([
        filter_edge_fragments(m, relative_distance=0.03) for m in masks
    ], dtype=bool).reshape(masks.shape)


def rgb_to_sam2_tensor(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """Upstream `_load_img_as_tensor`'s steps, from a decoded RGB array.

    Same PIL `convert("RGB").resize`, same /255 in float64, same permute; only
    the opening of a file is missing, because there is no file.
    """
    from PIL import Image

    img_np = np.array(Image.fromarray(rgb).convert("RGB").resize((image_size, image_size)))
    return torch.from_numpy(img_np / 255.0).permute(2, 0, 1)


class LazyFrames:
    """SAM2's `inference_state["images"]`, decoded when SAM2 asks for a frame.

    Upstream `init_state` decodes every JPEG of the clip before tracking starts
    and holds them as one float32 (N, 3, 1024, 1024) tensor: 12.6 MB a frame,
    about 19 GB and most of a minute for a 1500-frame window, and the reason
    long windows run out of memory. SAM2 only ever indexes it
    (`_get_image_feature`), so this decodes on demand, a few frames ahead on a
    thread, and keeps nothing behind but the last two frames.

    The values are upstream's to the bit: the same `_load_img_as_tensor` (PIL
    resize, /255 in float64), the same float64 -> float32 cast the preload's
    assignment performs, then the same elementwise `-= mean` and `/= std` on the
    compute device, applied per frame instead of to the whole stack.

    `source` is the JPEG folder's paths, or a `VideoFrames` decoding the video
    itself. For the latter the decoded RGB array goes through the same PIL
    steps `_load_img_as_tensor` applies to an opened JPEG, so the two paths
    differ only in the pixels they start from.
    """

    MEAN = (0.485, 0.456, 0.406)  # load_video_frames' defaults, which init_state uses
    STD = (0.229, 0.224, 0.225)

    def __init__(self, source: "list[Path] | VideoFrames", image_size: int, device,
                 ahead: int = 2) -> None:
        from PIL import Image
        from sam2.utils.misc import _load_img_as_tensor

        self._load_img = _load_img_as_tensor
        self.image_size = image_size
        self.device = device
        self.ahead = ahead
        if isinstance(source, VideoFrames):
            self.video = source
            self.paths = None
            self.video_width, self.video_height = source.width, source.height
        else:
            self.video = None
            self.paths = [str(p) for p in source]
            with Image.open(self.paths[0]) as first:
                self.video_width, self.video_height = first.size
        self.mean = torch.tensor(self.MEAN, dtype=torch.float32)[:, None, None].to(device)
        self.std = torch.tensor(self.STD, dtype=torch.float32)[:, None, None].to(device)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: dict[int, Future] = {}
        self._recent: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.video) if self.video is not None else len(self.paths)

    def _decode(self, idx: int) -> torch.Tensor:
        if self.video is None:
            img, _, _ = self._load_img(self.paths[idx], self.image_size)
        else:
            img = rgb_to_sam2_tensor(self.video.rgb(idx), self.image_size)
        return img.to(torch.float32)

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx = int(idx)
        if idx in self._recent:
            return self._recent[idx]
        future = self._pending.pop(idx, None)
        image = future.result() if future is not None else self._decode(idx)
        for later in range(idx + 1, min(idx + 1 + self.ahead, len(self))):
            if later not in self._pending:
                self._pending[later] = self._pool.submit(self._decode, later)
        image = image.to(self.device)
        image -= self.mean
        image /= self.std
        self._recent = {k: v for k, v in self._recent.items() if k == idx - 1}
        self._recent[idx] = image
        return image


def install_rope_repeat_cache() -> None:
    """Make SAM2's memory attention reuse its repeated RoPE frequencies.

    Every cross-attention call in `RoPEAttention` passes `repeat_freqs_k=True`,
    and upstream `apply_rotary_enc` then rebuilds `freqs_cis.repeat(..., r, 1)`
    -- the same 7x complex tensor -- on every call: 4 layers x ~13 objects x
    every frame. This wraps it so the repeat is built once per (freqs tensor,
    r) and otherwise runs upstream's arithmetic unchanged; outputs are
    identical (torch.equal). Idempotent.
    """
    try:
        import sam2.modeling.sam.transformer as transformer
        from sam2.modeling.position_encoding import reshape_for_broadcast
    except ImportError:
        return  # not upstream SAM2 (a test double, or another backend)

    if getattr(transformer.apply_rotary_enc, "_repeat_cached", False):
        return
    upstream = transformer.apply_rotary_enc
    repeats: dict = {}

    def apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
        if not repeat_freqs_k or xk.shape[-2] == 0 or not freqs_cis.is_cuda:
            return upstream(xq, xk, freqs_cis, repeat_freqs_k)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
        r = xk_.shape[-2] // xq_.shape[-2]
        key = (id(freqs_cis), freqs_cis.data_ptr(), tuple(freqs.shape), r)
        if key not in repeats:
            if len(repeats) >= 16:  # one entry per attention layer is all a model needs
                repeats.clear()
            # The entry holds freqs_cis itself, so its id cannot be reused while cached.
            repeats[key] = (freqs_cis, freqs.repeat(*([1] * (freqs.ndim - 2)), r, 1))
        xk_out = torch.view_as_real(xk_ * repeats[key][1]).flatten(3)
        return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

    apply_rotary_enc._repeat_cached = True
    transformer.apply_rotary_enc = apply_rotary_enc


def init_state_lazily(predictor, frames: "Path | VideoFrames") -> dict:
    """`predictor.init_state` with `LazyFrames` in place of the whole-clip preload.

    `frames` is a JPEG folder, or a `VideoFrames` decoding the video directly.
    Only upstream's own `SAM2VideoPredictor` loads frames through the function
    replaced here; anything else gets its plain `init_state`, which needs a
    folder.
    """
    video = frames if isinstance(frames, VideoFrames) else None
    try:
        import sam2.sam2_video_predictor as video_predictor
    except ImportError:
        video_predictor = None
    if video_predictor is None or not isinstance(predictor, video_predictor.SAM2VideoPredictor):
        if video is not None:
            raise ValueError("decoding the video directly needs upstream SAM2VideoPredictor; "
                             "other backends read a JPEG folder")
        return predictor.init_state(video_path=str(frames))

    if video is not None:
        source, video_path = video, video.video
    else:
        source = sorted(
            (p for p in Path(frames).iterdir() if p.suffix in (".jpg", ".jpeg", ".JPG", ".JPEG")),
            key=lambda p: int(p.stem),  # upstream's frame order
        )
        video_path = str(frames)

    def load_lazily(video_path, image_size, offload_video_to_cpu, async_loading_frames=False,
                    compute_device=torch.device("cuda"), **_):
        lazy = LazyFrames(source, image_size, compute_device)
        return lazy, lazy.video_height, lazy.video_width

    preload = video_predictor.load_video_frames
    video_predictor.load_video_frames = load_lazily
    try:
        return predictor.init_state(video_path=video_path)
    finally:
        video_predictor.load_video_frames = preload


# Frames kept beyond the farthest one SAM2's memory attention reads; slack only.
PRUNE_MARGIN_FRAMES = 4


def memory_window(predictor) -> int | None:
    """How many frames back forward propagation reads stored per-frame outputs.

    Mask memory reads the last `num_maskmem - 1` frames at
    `memory_temporal_stride_for_eval`; object pointers read the last
    `max_obj_ptrs_in_encoder - 1`. None for a predictor without these
    attributes, which is then never pruned.
    """
    try:
        masks = (predictor.num_maskmem - 1) * max(1, predictor.memory_temporal_stride_for_eval)
        pointers = (predictor.max_obj_ptrs_in_encoder - 1) if predictor.use_obj_ptrs_in_encoder else 0
    except AttributeError:
        return None
    return max(masks, pointers)


def prune_frame_outputs(state: dict, frame_idx: int, window: int) -> int:
    """Drop stored per-frame outputs older than `window` + margin frames before `frame_idx`.

    Upstream keeps every frame's non-conditioning output for every object for the
    whole session -- about 1 MB per object per frame, so memory, not compute,
    capped clip length -- while forward propagation only ever reads the last
    `window` of them. Conditioning (prompted) frames are never touched: memory
    attention reads all of them. The per-frame "already tracked" records go
    too; a prompt only ever lands on the current frame. Returns entries dropped.
    """
    oldest = frame_idx - window - PRUNE_MARGIN_FRAMES
    stores = [outputs["non_cond_frame_outputs"]
              for outputs in state.get("output_dict_per_obj", {}).values()]
    stores += list(state.get("frames_tracked_per_obj", {}).values())
    dropped = 0
    for store in stores:
        for t in [t for t in store if t < oldest]:
            del store[t]
            dropped += 1
    return dropped


@dataclass
class Sam2FrameResult:
    """One propagated frame, in the order live objects were returned.

    `player_id` is SAM2's own `obj_id` -- `TrackManager` uses it directly as
    the registry's stable id (see `sam2_manager`'s module docstring), so no
    separate tracker_id/player_id translation exists here the way McByte needs
    one. `read_frame` is lazy: the tracking-only consumer never needs pixel
    data outside checkpoint frames, so eagerly decoding every propagated
    frame's JPEG would cost real time for no benefit to that caller.
    """
    frame_idx: int
    player_ids: np.ndarray   # (N,) int
    masks: np.ndarray        # (N, H, W) bool
    boxes: np.ndarray        # (N, 4) xyxy, mask-derived
    read_frame: Callable[[], np.ndarray]  # -> RGB frame, decoded once and cached


def _apply_reset_reseed(predictor, state, frame_idx, actions, masks_by_id) -> None:
    """Apply a checkpoint by discarding SAM2's memory bank and re-seeding.

    McByte++'s `mask_manager__edgetam.reseed_at_frame` does this because
    EdgeTAM's predictor raises "Cannot add new object id ... after tracking
    starts"; `reset_state` clears `tracking_has_started`, so re-adding everyone
    is the way around it. This reproduces that pattern on SAM2 so its cost can
    be measured before anyone ports anything.

    `TrackManager`'s decisions are honoured exactly as the default policy
    honours them, so the only variable is memory:

    - removed objects are dropped, as they would be by `remove_object`;
    - an object the manager wants corrected (`reprompt` or `reset`) is
      re-seeded from that action's detector box, the same correction it would
      have received -- never from its current mask, which for a `reset` is on
      the wrong player entirely;
    - every other survivor is re-seeded from its own mask at this frame, which
      preserves its shape where a box would not;
    - new objects enter from their detector box.
    """
    corrections = {
        int(a["obj_id"]): a["box"]
        for a in actions if a["type"] in ("reprompt", "reset")
    }
    removed = {int(a["obj_id"]) for a in actions if a["type"] == "remove"}
    additions = {int(a["obj_id"]): a["box"] for a in actions if a["type"] == "add"}

    survivors = [
        (int(oid), mask) for oid, mask in masks_by_id.items()
        if int(oid) not in removed
    ]

    predictor.reset_state(state)

    for obj_id, mask in survivors:
        if obj_id in corrections:
            predictor.add_new_points_or_box(
                state, frame_idx=frame_idx, obj_id=obj_id,
                box=np.asarray(corrections[obj_id], dtype=np.float32),
            )
        elif mask.any():
            predictor.add_new_mask(
                state, frame_idx=frame_idx, obj_id=obj_id, mask=mask
            )
        # A survivor whose mask has collapsed to nothing and that the manager
        # did not flag has no usable seed; it is dropped rather than re-added
        # from an empty mask, which SAM2 would treat as an empty object.

    for obj_id, box in additions.items():
        predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=obj_id,
            box=np.asarray(box, dtype=np.float32),
        )


def drive_sam2(
    video: Path,
    frame_detections_fn: Callable[[int], "sv.Detections"],
    team_model,
    checkpoint: str,
    check_every: int = CHECK_EVERY_DEFAULT,
    frame_cache_dir: Path | None = None,
    max_frames: int | None = None,
    goalkeeper_class_id: int = 1,
    desc: str = "SAM2 reprompt",
    reid_encoder=None,
    checkpoint_policy: str = "reprompt",
    predictor_factory: Callable[[str], object] | None = None,
    prefetch_frames: int = 0,
    frame_source: str = "jpeg",
    prune_memory: bool = True,
    court_in_view: Callable[[int], bool] | None = None,
) -> tuple[TrackManager, dict[int, np.ndarray], Iterator[Sam2FrameResult]]:
    """Seed and propagate SAM2 with periodic detector-checkpoint reprompting.

    `frame_detections_fn(frame_index) -> sv.Detections` is the same cached
    per-frame detection lookup every tracker in this project's evaluation
    scripts already uses (`scripts.render_raw_team_classification.
    frame_detections`, bound to a loaded `.npz` cache) -- both the frame-0 seed
    and every checkpoint's reprompt boxes come from it, so SAM2 is not given a
    stronger detection input than a caller comparing it against other trackers
    would give them.

    `predictor_factory(checkpoint)` optionally constructs an API-compatible video
    predictor. The default still builds the current SAM2 model. Construction is
    the only backend seam; all lifecycle decisions and output processing remain
    shared. Callers must provide compatible, explicitly configured weights.

    Returns `(track_manager, seed_boxes, frames)`:

    - `track_manager` is usable immediately (its `.events`/`.registry` fill in
      as `frames` is consumed).
    - `seed_boxes` is `{obj_id: xyxy}` at frame 0, straight from the detector
      -- there is no mask yet at the seed frame, so it is kept separate from
      `frames` rather than forced into a `Sam2FrameResult` with an empty
      `masks` array that would not line up in length with `player_ids`.
    - `frames` yields every propagated frame from frame 1 onward, each with
      masks/boxes/player_ids of matching length.

    `court_test_fn` is left permissive (accepts any new detection), matching
    the box trackers in `evaluate_tracker_identity`, which also add every
    unmatched detection without a court-membership check.

    `prefetch_frames` > 0 decodes that many upcoming frames on a background
    thread, for a consumer that calls `read_frame` on every frame (the render).
    The default 0 decodes lazily, which suits tracking-only callers that read
    pixels only at checkpoints. The decoded frames are identical either way.

    `frame_source` is where frames come from. `"jpeg"` (default) extracts the
    video to a JPEG folder once and reads that -- every scored result so far
    was made this way. `"video"` decodes the video directly (`VideoFrames`):
    nothing written to disk and no second lossy encode, but different input
    pixels from the scored runs, so it is not byte-identical to them.

    `prune_memory` drops each object's stored per-frame outputs once they fall
    out of the window forward propagation reads (`prune_frame_outputs`), so
    memory stays flat however long the clip. Output is unchanged by it.

    `court_in_view(frame)` says whether a court is on screen at all
    (`CourtFit.court_in_view`). A checkpoint without one pauses tracking:
    every live player is retired (`TrackManager.pause`) and SAM2 runs empty,
    as it already does after a crowd pan. Tracking resumes once the court has
    been back for `RESUME_AFTER_FRAMES`, and players re-enter through the
    usual confirmation and re-ID. None, the default, never pauses.

    `checkpoint_policy` selects how `TrackManager`'s decisions reach the
    predictor. It does not change what those decisions are.

    - `"reprompt"` (default): today's behaviour. Corrections are applied to
      individual objects and SAM2's memory bank survives the checkpoint.
    - `"reset_reseed"`: `reset_state` first, then every surviving object is
      re-added at this frame. The memory bank is discarded wholesale.

    The second exists to price one specific cost. EdgeTAM's predictor refuses
    to add an object once tracking has started, so McByte++ reaches around it
    by tearing the session down and re-seeding everyone each time it needs a
    new target -- meaning any EdgeTAM port here would have to pay that too. Run
    both policies through `evaluate_tracker_identity` and the difference is
    what discarding mask memory costs, measured on this project's own tracker
    and references rather than inferred from another paper's benchmark. It is
    an experiment seam, not a supported configuration: §8.3 of
    `docs/tracking-evaluation.md` traces this tracker's recall advantage to
    propagating on accumulated memory, which is exactly what the second policy
    throws away.
    """
    if checkpoint_policy not in ("reprompt", "reset_reseed"):
        raise ValueError(
            f"checkpoint_policy must be 'reprompt' or 'reset_reseed', got {checkpoint_policy!r}"
        )

    if frame_source == "video":
        # Straight from the video: no JPEG folder written, one decode per frame
        # shared by SAM2 and the consumer.
        video_frames = VideoFrames(video)
        frames_for_sam2 = video_frames
        num_frames = len(video_frames)
        read_frame = video_frames.rgb
    elif frame_source == "jpeg":
        frame_cache_dir = frame_cache_dir or (
            video.resolve().parent / "_sam2_frame_cache" / video.stem
        )
        frame_files = ensure_frame_cache(video, frame_cache_dir)
        frames_for_sam2 = frame_cache_dir
        num_frames = len(frame_files)

        def read_frame(idx: int) -> np.ndarray:
            return cv2.cvtColor(cv2.imread(str(frame_files[idx])), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"frame_source must be 'jpeg' or 'video', got {frame_source!r}")
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    if predictor_factory is None:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(SAM2_CONFIG, checkpoint)
        # build_sam2_video_predictor sets fill_hole_area=8, but hole filling needs
        # the compiled sam2._C extension, which this checkout does not have; it
        # warns and returns the mask unchanged. Every scored result was made that
        # way, so pin it: a checkout that does build _C must not silently start
        # filling holes and change the tracker's output.
        predictor.fill_hole_area = 0
        install_rope_repeat_cache()
        # Output-identical speedups (~1.16x on GB10); each falls back to the
        # upstream path for anything it cannot reproduce exactly.
        install_fused_rope()
        install_small_module_graphs(predictor)
    else:
        predictor = predictor_factory(checkpoint)

    frame0 = read_frame(0)
    det0 = frame_detections_fn(0)
    if len(det0) == 0:
        raise RuntimeError("no cached detections on frame 0")

    track_manager = TrackManager(
        team_model, court_test_fn=lambda box: True, reid_encoder=reid_encoder
    )
    is_gk0 = det0.class_id == goalkeeper_class_id
    obj_ids0 = track_manager.seed(0, det0.xyxy, frame0, is_gk0)

    # A SAM2VideoPredictor -- the default, or an injected SAM2 fork such as
    # Lean-SAM2 -- gets frames on demand (LazyFrames, identical values), so
    # forks are timed on the same frame loading; any other backend keeps its
    # own init_state untouched (init_state_lazily falls back to it).
    state = init_state_lazily(predictor, frames_for_sam2)
    window = memory_window(predictor) if prune_memory else None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for oid, xyxy in zip(obj_ids0, det0.xyxy):
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=oid, box=np.asarray(xyxy, dtype=np.float32)
            )

    seed_boxes = dict(zip(obj_ids0, det0.xyxy))

    # Frames the consumer will read, decoded ahead on one thread (cv2 releases
    # the GIL) so the decode overlaps propagation instead of stalling it.
    reader_pool = ThreadPoolExecutor(max_workers=1) if prefetch_frames else None
    pending: dict[int, Future] = {}

    def frame_reader(fid: int) -> Callable[[], np.ndarray]:
        if reader_pool is None:
            return functools.lru_cache(maxsize=1)(lambda: read_frame(fid))
        for ahead in range(fid, min(fid + prefetch_frames, num_frames)):
            if ahead not in pending:
                pending[ahead] = reader_pool.submit(read_frame, ahead)
        return pending.pop(fid).result

    def propagate(t: int, chunk_len: int):
        # With every player removed (the camera left the court: VfL frame 610
        # is a crowd pan) SAM2 has reset its session and refuses to propagate.
        # Nobody is tracked, so the chunk is empty frames; the checkpoint after
        # it still runs and adds players back once play returns. Backends
        # without `obj_id_to_idx` handle an empty state themselves.
        if state.get("obj_id_to_idx", True):
            yield from predictor.propagate_in_video(
                state, start_frame_idx=t, max_frame_num_to_track=chunk_len)
            return
        empty = torch.empty((0, 1, *frame0.shape[:2]))
        for fid in range(t, t + chunk_len + 1):
            yield fid, [], empty

    def frames() -> Iterator[Sam2FrameResult]:
        next_new_fid = 1
        in_view_since: int | None = 0
        last_masks_by_id: dict[int, np.ndarray] = {}
        last_frame = frame0

        pbar = tqdm(total=num_frames - 1, desc=desc)
        t = 0
        while t < num_frames - 1:
            chunk_len = min(check_every, num_frames - 1 - t)
            chunk_end = t

            # Each chunk restarts on frame t, which the previous chunk already
            # yielded. Skipping that recompute (starting at t + 1) looks free --
            # same memory, same frame -- but it is not bit-identical: the stored
            # and recomputed frame-t outputs differ, and on FelixClaar skipping
            # it cost 2 wrong-identity frames (95.01% -> 94.92%). Keep it.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for fid, obj_ids, mask_logits in propagate(t, chunk_len):
                    if fid < next_new_fid or fid >= num_frames:
                        continue  # repeated boundary frame, or past our truncation

                    masks = masks_from_logits(mask_logits)
                    track_manager.update_from_propagation(fid, obj_ids, masks)

                    boxes_this = sv.mask_to_xyxy(masks=masks)
                    reader = frame_reader(fid)
                    yield Sam2FrameResult(
                        frame_idx=fid,
                        player_ids=np.asarray(obj_ids, dtype=int),
                        masks=masks,
                        boxes=boxes_this,
                        read_frame=reader,
                    )

                    chunk_end = fid
                    if window is not None:
                        prune_frame_outputs(state, fid, window)
                    if fid == t + chunk_len:
                        last_masks_by_id = dict(zip(obj_ids, masks))
                        last_frame = reader()  # decoded once, shared with the consumer
                    pbar.update(1)

            next_new_fid = chunk_end + 1
            t = chunk_end
            if t >= num_frames - 1:
                break

            # ── detector checkpoint: decide add / remove / reprompt ──
            live_obj_ids = list(last_masks_by_id.keys())
            if court_in_view is not None:
                in_view_since = (in_view_since if in_view_since is not None else chunk_end) \
                    if court_in_view(chunk_end) else None
                if in_view_since is None:
                    if track_manager.paused_since is None:
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            for action in track_manager.pause(chunk_end, live_obj_ids):
                                predictor.remove_object(state, obj_id=action["obj_id"])
                        last_masks_by_id = {}
                    continue
                if (track_manager.paused_since is not None
                        and chunk_end - in_view_since < RESUME_AFTER_FRAMES):
                    continue
                track_manager.resume(chunk_end)
            live_masks = (
                np.array([last_masks_by_id[oid] for oid in live_obj_ids])
                if live_obj_ids else np.empty((0, 0, 0), dtype=bool)
            )
            det = frame_detections_fn(chunk_end)
            det_is_goalkeeper = det.class_id == goalkeeper_class_id

            actions = track_manager.checkpoint(
                chunk_end, last_frame, live_obj_ids, live_masks, det.xyxy, det_is_goalkeeper
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if checkpoint_policy == "reset_reseed":
                    _apply_reset_reseed(
                        predictor, state, chunk_end, actions, last_masks_by_id
                    )
                    continue
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
                        predictor.add_new_points_or_box(
                            state, frame_idx=chunk_end, obj_id=action["obj_id"],
                            box=np.asarray(action["box"], dtype=np.float32),
                        )
                    elif action["type"] == "reset":
                        # Body-swap, not drift: memory is contaminated with the
                        # wrong player's appearance, so reprompting in place
                        # would seed the "correction" from that wrong mask.
                        # Tear the object down and re-add it fresh under the
                        # same obj_id.
                        if len(state["obj_id_to_idx"]) > 1:
                            predictor.remove_object(state, obj_id=action["obj_id"])
                            predictor.add_new_points_or_box(
                                state, frame_idx=chunk_end, obj_id=action["obj_id"],
                                box=np.asarray(action["box"], dtype=np.float32),
                            )
                        else:
                            # Removing the only live object resets the whole
                            # session (see SAM2VideoPredictor.remove_object) --
                            # fall back to an in-place reprompt instead.
                            predictor.add_new_points_or_box(
                                state, frame_idx=chunk_end, obj_id=action["obj_id"],
                                box=np.asarray(action["box"], dtype=np.float32),
                                clear_old_points=True,
                            )
        pbar.close()
        if reader_pool is not None:
            reader_pool.shutdown(cancel_futures=True)

    return track_manager, seed_boxes, frames()
