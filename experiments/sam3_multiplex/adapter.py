"""Adapt the released SAM3.1 multiplex tracker to the production driver.

Use the trained interactive head to turn RF-DETR boxes into masks, then insert
them together through the official mask API. Sequential add_new_points calls
prefer new buckets upstream, defeating the multi-object sharing under test.
No text detector, weight approximation, frame skipping or global memory reset.

Players are tracked in groups, each an independent upstream state of at most
one bucket's worth of objects. Upstream cannot grow a state's bucket count
once tracking has started: every stored frame keeps the bucket count it was
computed with, and the next frame's torch.cat of object pointers across frames
fails ([1,16,256] against [2,16,256]; BHC-FAG crashed at frame 370). Removal is
no escape -- a removed slot is marked removed, never reused, so player churn
alone exhausts a bucket. Buckets are separate batch rows that never attend to
each other, so a separate state per group computes what a bucket would, and a
group created mid-video starts with no history instead of a fabricated one.
New players fill a group's never-used slots first; a group whose last player
leaves is dropped. All groups share one frame decode and one image encoding.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "sam3-checkpoints/sam3.1_multiplex.pt"


def _tracking_image_feature(model, state, frame_idx, batch_size):
    from sam3.model.data_misc import NestedTensor

    cache = state["cached_features"]  # shared by every group's state; mutate, never rebind
    image, encoded = cache.get(frame_idx, (None, None))
    if encoded is None:
        image = state["images"][frame_idx].cuda().float().unsqueeze(0)
        # The standalone demo requests the unused SAM3 detection neck, whose
        # tensor-valued outputs break its forward_image dictionary loop. Request
        # only the two tracker necks, as the combined predictor does for tracking.
        encoded = model.forward_image(
            NestedTensor(tensors=image, mask=None), need_sam3_out=False,
            need_interactive_out=True, need_propagation_out=True,
        )
        cache.clear()
        cache[frame_idx] = (image, encoded)
    return image, model._prepare_backbone_features(encoded)


def build_tracker(checkpoint=CHECKPOINT):
    from sam3.model_builder import build_sam3_multiplex_video_model

    model = build_sam3_multiplex_video_model(
        load_from_HF=False, device="cpu", compile=False, use_fa3=False,
    )
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    raw = raw.get("model", raw)
    weights = {}
    for key, value in raw.items():
        if key.startswith("tracker.model."):
            weights[key.removeprefix("tracker.model.")] = value
        elif key.startswith("detector.backbone.vision_backbone."):
            weights["backbone.vision_backbone." + key.removeprefix(
                "detector.backbone.vision_backbone.")] = value
    # Includes the shared vision encoder and both trained mask heads. A missing
    # key must fail rather than leave any randomly initialized tracker weights.
    model.load_state_dict(weights, strict=True, assign=True)
    loaded = {"loaded_tensors": len(weights),
              "loaded_parameters": sum(p.numel() for p in model.parameters()),
              "missing_keys": [], "unexpected_keys": []}
    del weights, raw
    model.eval().to("cuda")
    model._get_image_feature = MethodType(_tracking_image_feature, model)
    assert all(p.is_cuda for p in model.parameters())
    return model, loaded


class _Frames:
    """One-frame CPU cache, using upstream PIL resize and FP16 normalization."""

    def __init__(self, files, size):
        self.files, self.size = files, size
        self.last = None
        self.tensor = None

    def __getitem__(self, index):
        if index != self.last:
            with Image.open(self.files[index]) as image:
                pixels = np.array(image.convert("RGB").resize((self.size, self.size)))
            self.tensor = torch.from_numpy(pixels / 255.0).permute(2, 0, 1).half()
            self.tensor.sub_(0.5).div_(0.5)
            self.last = index
        return self.tensor


class MultiplexAdapter:
    def __init__(self, model):
        self.model = model
        self.image_size = model.image_size

    # ── groups ────────────────────────────────────────────────────────────
    def _new_group(self, state):
        group = self.model.init_state(state["video_height"], state["video_width"], state["num_frames"])
        group["images"] = state["images"]
        group["cached_features"] = state["cached_features"]
        state["groups"].append(group)
        return group

    @staticmethod
    def _free_slots(group):
        multiplex = group.get("multiplex_state")
        return 0 if multiplex is None or multiplex.assignments is None else multiplex.available_slots

    def _group_of(self, state, obj_id):
        return next((g for g in state["groups"] if obj_id in g["obj_id_to_idx"]), None)

    def _sync_ids(self, state):
        ids = [oid for g in state["groups"] for oid in g["obj_ids"]]
        state["obj_id_to_idx"] = {oid: i for i, oid in enumerate(ids)}

    def bucket_count(self, state):
        return sum(g["multiplex_state"].num_buckets for g in state["groups"]
                   if g.get("multiplex_state") is not None and g["multiplex_state"].assignments)

    # ── predictor API used by drive_sam2 ─────────────────────────────────
    def init_state(self, video_path):
        files = sorted(Path(video_path).glob("*.jpg"), key=lambda p: int(p.stem))
        with Image.open(files[0]) as image:
            width, height = image.size
        state = {"video_height": height, "video_width": width, "num_frames": len(files),
                 "images": _Frames(files, self.image_size), "cached_features": {},
                 "groups": [], "obj_id_to_idx": {}, "experiment_pending_boxes": {}}
        # Hosts box-to-mask inference before any group exists.
        state["feature_host"] = {"images": state["images"], "cached_features": state["cached_features"]}
        self.last_state = state
        return state

    def add_new_points_or_box(self, state, frame_idx, obj_id, box,
                              clear_old_points=True):
        # The production driver ignores the immediate prompt output. Buffer all
        # decisions at this frame and perform their mask inference as one batch.
        state["experiment_pending_boxes"][(frame_idx, int(obj_id))] = np.asarray(box)

    def remove_object(self, state, obj_id):
        obj_id = int(obj_id)
        for key in list(state["experiment_pending_boxes"]):
            if key[1] == obj_id:
                del state["experiment_pending_boxes"][key]
        group = self._group_of(state, obj_id)
        if group is None:
            return
        if len(group["obj_ids"]) == 1:
            state["groups"].remove(group)  # upstream would leave an empty, invalid state
        else:
            self.model.remove_object(group, obj_id=obj_id, need_output=False)
        self._sync_ids(state)

    def _box_masks(self, state, fid, boxes):
        _, features = self.model._get_image_feature(state["feature_host"], fid, len(boxes))
        interactive = features["interactive"]
        feats, sizes = interactive["vision_feats"], interactive["feat_sizes"]
        high_res = [x.permute(1, 2, 0).reshape(x.size(1), x.size(2), *size)
                    for x, size in zip(feats[:-1], sizes[:-1])]
        pixels = self.model._get_interactive_pix_mem(feats, sizes)
        corners = torch.as_tensor(np.asarray(boxes), device="cuda", dtype=torch.float32).reshape(-1, 2, 2)
        corners = corners / corners.new_tensor([state["video_width"], state["video_height"]]) * self.image_size
        points = {"point_coords": corners,
                  "point_labels": torch.tensor([2, 3], dtype=torch.int32, device="cuda").expand(len(boxes), -1)}
        out = self.model._forward_sam_heads(
            pixels, point_inputs=points, interactive_high_res_features=high_res,
            multimask_output=self.model._use_multimask(True, points),
            multiplex_state=None, objects_to_interact=list(range(len(boxes))),
        )
        masks = F.interpolate(out["low_res_masks"],
                              size=(state["video_height"], state["video_width"]),
                              mode="bilinear", align_corners=False)[:, 0] > 0
        return masks

    def _flush(self, state):
        pending = state["experiment_pending_boxes"]
        if not pending:
            return
        frames = {fid for fid, _ in pending}
        assert len(frames) == 1, "cannot defer prompts across checkpoint frames"
        fid = next(iter(frames))
        ids = [oid for _, oid in pending]
        masks = self._box_masks(state, fid, list(pending.values()))
        pending.clear()

        # Corrections stay in their player's group.
        for group in state["groups"]:
            rows = [i for i, oid in enumerate(ids) if oid in group["obj_id_to_idx"]]
            if rows:
                self.model.add_new_masks(group, fid, [ids[i] for i in rows], masks[rows],
                                         reconditioning=True)

        # New players: never-used slots in existing groups first, never a new bucket.
        new = [i for i, oid in enumerate(ids) if self._group_of(state, oid) is None]
        for group in list(state["groups"]):
            take, new = new[:self._free_slots(group)], new[self._free_slots(group):]
            if not take:
                continue
            # A late target's first prompt must remain a conditioning anchor.
            # Otherwise upstream classifies it as non-conditioning because this
            # frame was already propagated for other players.
            previous = self.model.add_all_frames_to_correct_as_cond
            self.model.add_all_frames_to_correct_as_cond = True
            try:
                self.model.add_new_masks(group, fid, [ids[i] for i in take], masks[take])
            finally:
                self.model.add_all_frames_to_correct_as_cond = previous
        if new:
            # The rest seed a fresh group: initial conditioning frame, no history.
            self.model.add_new_masks(self._new_group(state), fid, [ids[i] for i in new], masks[new])
        self._sync_ids(state)

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
        self._flush(state)
        streams = []
        for group in state["groups"]:
            self.model.propagate_in_video_preflight(group)
            streams.append(self.model.propagate_in_video(
                group, start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=False, tqdm_disable=True,
            ))
        for outputs in zip(*streams):
            fids = {fid for fid, *_ in outputs}
            assert len(fids) == 1, f"groups out of step: {fids}"
            ids = [oid for _, group_ids, *_ in outputs for oid in group_ids]
            masks = torch.cat([group_masks for _, _, _, group_masks, _ in outputs])
            assert masks.is_cuda
            yield fids.pop(), ids, masks
