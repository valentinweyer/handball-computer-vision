"""Backend construction must preserve the shared dynamic-object output contract."""
from contextlib import nullcontext
import sys
from types import ModuleType

import cv2
import numpy as np
import pytest
import supervision as sv
import torch

from handball_cv.tracking import sam2_driver as driver


@pytest.mark.parametrize("scores,winner", [([.800, .443], 2), ([.443, .800], 1)])
def test_conflicting_roles_keep_the_stronger_detection(scores, winner):
    # Kiel f3540: the exact same box was emitted as player (.800) and keeper (.443).
    detections = sv.Detections(
        xyxy=np.array([[718.429, 350.026, 808.330, 593.015]] * 2),
        class_id=np.array([2, 1]), confidence=np.array(scores),
        data={"source_row": np.array([10, 11])},
    )
    kept = driver._deduplicate_role_detections(detections, 1)
    assert kept.class_id.tolist() == [winner]
    assert kept.confidence.tolist() == [max(scores)]
    assert kept.data["source_row"].tolist() == [10 if winner == 2 else 11]
    assert len(detections) == 2  # raw input is preserved


@pytest.mark.parametrize("classes,boxes", [
    ([2, 2], [[10, 10, 50, 100], [10, 10, 50, 100]]),
    ([1, 2], [[10, 10, 50, 100], [25, 10, 65, 100]]),
])
def test_same_role_or_ordinary_overlap_is_not_suppressed(classes, boxes):
    detections = sv.Detections(xyxy=np.array(boxes, float),
                              class_id=np.array(classes), confidence=np.array([.8, .9]))
    kept = driver._deduplicate_role_detections(detections, 1)
    np.testing.assert_array_equal(kept.xyxy, detections.xyxy)
    np.testing.assert_array_equal(kept.class_id, detections.class_id)


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("duplicate_roles", [False, True])
def test_factory_preserves_lifecycle_and_output_order(tmp_path, monkeypatch, custom, duplicate_roles):
    for frame in range(4):
        cv2.imwrite(str(tmp_path / f"{frame:05d}.jpg"), np.zeros((8, 12, 3), np.uint8))
    memories = {}
    states = []

    class Predictor:
        def init_state(self, video_path):
            state = {"obj_id_to_idx": {}}
            states.append(state)
            return state

        def add_new_points_or_box(self, state, *, obj_id, **kwargs):
            if obj_id not in state["obj_id_to_idx"]:
                state["obj_id_to_idx"][obj_id] = len(state["obj_id_to_idx"])
                memories[obj_id] = object()

        def remove_object(self, state, *, obj_id):
            del state["obj_id_to_idx"][obj_id]
            del memories[obj_id]
            state["obj_id_to_idx"] = {oid: i for i, oid in enumerate(state["obj_id_to_idx"])}

        def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
            for frame in range(start_frame_idx, start_frame_idx+max_frame_num_to_track+1):
                ids = list(state["obj_id_to_idx"])
                masks = torch.full((len(ids), 1, 8, 12), -1.)
                for i in range(len(ids)):
                    masks[i, 0, 1:5, 1+i*4:4+i*4] = 1
                yield frame, ids, masks

    initial = {}
    survivor = {}

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def seed(self, frame, boxes, image, is_goalkeeper):
            np.testing.assert_array_equal(boxes, [[1, 1, 4, 5], [5, 1, 8, 5]])
            assert not is_goalkeeper.any()
            return [11, 22]

        def update_from_propagation(self, frame, ids, masks):
            assert masks.shape == (len(ids), 8, 12)

        def checkpoint(self, frame, *args):
            np.testing.assert_array_equal(args[3], [[1, 1, 4, 5], [5, 1, 8, 5]])
            assert not args[4].any()
            if frame == 1:
                initial[11] = memories[11]
                return [{"type": "remove", "obj_id": 22},
                        {"type": "add", "obj_id": 33, "box": [5, 1, 8, 5]}]
            survivor[33] = memories[33]
            return [{"type": "reset", "obj_id": 11, "box": [1, 1, 4, 5]}]

    def factory(checkpoint):
        assert checkpoint == "weights.pt"
        return Predictor()

    # Default construction is checked without importing heavyweight SAM2.
    module = ModuleType("sam2.build_sam")
    def build(config, checkpoint):
        assert config == driver.SAM2_CONFIG
        return factory(checkpoint)
    module.build_sam2_video_predictor = build
    monkeypatch.setitem(sys.modules, "sam2.build_sam", module)
    monkeypatch.setattr(driver, "TrackManager", Manager)
    monkeypatch.setattr(driver.torch, "autocast", lambda *a, **k: nullcontext())
    det = sv.Detections(xyxy=np.array([[1, 1, 4, 5], [5, 1, 8, 5]], float),
                        class_id=np.array([2, 2]), confidence=np.array([.9, .8]))
    if duplicate_roles:
        det = sv.Detections.merge([det, sv.Detections(
            xyxy=det.xyxy.copy(), class_id=np.array([1, 1]), confidence=np.array([.4, .3]),
        )])
    _, seed, frames = driver.drive_sam2(
        tmp_path / "unused.mp4", lambda fid: det, None, "weights.pt",
        check_every=1, frame_cache_dir=tmp_path,
        predictor_factory=factory if custom else None,
    )
    results = list(frames)
    assert list(seed) == [11, 22]
    assert [r.frame_idx for r in results] == [1, 2, 3]
    assert [r.player_ids.tolist() for r in results] == [[11, 22], [11, 33], [33, 11]]
    assert memories[11] is not initial[11]
    assert memories[33] is survivor[33]
    for result in results:
        assert result.masks.dtype == bool
        np.testing.assert_array_equal(result.boxes, sv.mask_to_xyxy(result.masks))
        assert result.read_frame().shape == (8, 12, 3)


def test_tracking_continues_when_every_player_leaves_the_frame(tmp_path, monkeypatch):
    # A crowd pan removes every player; real SAM2 then resets its session and
    # raises on propagate. The driver must yield empty frames and keep running
    # checkpoints so players come back when play does.
    for frame in range(7):
        cv2.imwrite(str(tmp_path / f"{frame:05d}.jpg"), np.zeros((8, 12, 3), np.uint8))

    class Predictor:
        def init_state(self, video_path):
            return {"obj_id_to_idx": {}}

        def add_new_points_or_box(self, state, *, obj_id, **kwargs):
            state["obj_id_to_idx"].setdefault(obj_id, len(state["obj_id_to_idx"]))

        def remove_object(self, state, *, obj_id):
            del state["obj_id_to_idx"][obj_id]

        def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
            if not state["obj_id_to_idx"]:
                raise RuntimeError("No input points or masks are provided for any object")
            for frame in range(start_frame_idx, start_frame_idx + max_frame_num_to_track + 1):
                ids = list(state["obj_id_to_idx"])
                masks = torch.full((len(ids), 1, 8, 12), -1.)
                masks[:, 0, 1:5, 1:4] = 1
                yield frame, ids, masks

    class Manager:
        def __init__(self, *args, **kwargs):
            pass

        def seed(self, *args):
            return [11]

        def update_from_propagation(self, frame, ids, masks):
            assert masks.shape == (len(ids), 8, 12)

        def checkpoint(self, frame, *args):
            if frame == 2:
                return [{"type": "remove", "obj_id": 11}]
            if frame == 4:
                return [{"type": "add", "obj_id": 12, "box": [1, 1, 4, 5]}]
            return []

    monkeypatch.setattr(driver, "TrackManager", Manager)
    monkeypatch.setattr(driver.torch, "autocast", lambda *a, **k: nullcontext())
    det = sv.Detections(xyxy=np.array([[1, 1, 4, 5]], float), class_id=np.array([2]))
    _, _, frames = driver.drive_sam2(
        tmp_path / "unused.mp4", lambda fid: det, None, "weights.pt",
        check_every=2, frame_cache_dir=tmp_path, predictor_factory=lambda _: Predictor(),
    )
    results = list(frames)
    assert [r.player_ids.tolist() for r in results] == [[11], [11], [], [], [12], [12]]
    assert results[2].masks.shape == (0, 8, 12) and results[2].boxes.shape == (0, 4)
