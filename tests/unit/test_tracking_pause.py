"""Tracking pauses where no court is in view (a timeout huddle) and restarts after."""
from contextlib import nullcontext

import cv2
import numpy as np
import supervision as sv
import torch

from handball_cv.tracking import sam2_driver as driver
from handball_cv.tracking.identity import PlayerRegistry
from handball_cv.tracking.sam2_manager import TrackManager


def test_a_pause_does_not_age_the_reid_gallery():
    registry = PlayerRegistry(reid_max_gap_frames=300)
    player = registry.create(100, None, 0, np.ones(4), False)
    registry.retire(player.player_id, 100)
    # 1500 frames later: far past the gap, unless most of it was a timeout.
    assert registry.reid_match(np.ones(4), 1600, set()) is None
    registry.pause(150, 1500)
    assert registry.play_frames(100, 1600) == 150
    assert registry.reid_match(np.ones(4), 1600, set()) is player


def test_pause_retires_everyone_and_resume_records_the_stretch():
    manager = TrackManager(team_model=None, court_test_fn=lambda box: True)
    ids = [manager.registry.create(0, None, 0, np.ones(4), False).player_id for _ in range(3)]
    manager._pending_new_boxes.append([np.zeros(4), 1])

    actions = manager.pause(40, ids)
    assert [a["obj_id"] for a in actions] == ids and all(a["type"] == "remove" for a in actions)
    assert not manager.registry.live and len(manager.registry.retired) == 3
    assert manager._pending_new_boxes == []  # a candidate from before the pause is stale

    manager.resume(90)
    assert manager.paused_since is None and manager.registry.paused == [(40, 90)]
    manager.resume(95)  # resuming while not paused changes nothing
    assert manager.registry.paused == [(40, 90)]


def test_the_driver_pauses_while_no_court_is_in_view(tmp_path, monkeypatch):
    for frame in range(11):
        cv2.imwrite(str(tmp_path / f"{frame:05d}.jpg"), np.zeros((8, 12, 3), np.uint8))

    class Predictor:
        def init_state(self, video_path):
            return {"obj_id_to_idx": {}}

        def add_new_points_or_box(self, state, *, obj_id, **kwargs):
            state["obj_id_to_idx"].setdefault(obj_id, len(state["obj_id_to_idx"]))

        def remove_object(self, state, *, obj_id):
            del state["obj_id_to_idx"][obj_id]

        def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
            for frame in range(start_frame_idx, start_frame_idx + max_frame_num_to_track + 1):
                ids = list(state["obj_id_to_idx"])
                masks = torch.full((len(ids), 1, 8, 12), -1.)
                masks[:, 0, 1:5, 1:4] = 1
                yield frame, ids, masks

    checkpoints, calls = [], []

    class Manager:
        def __init__(self, *args, **kwargs):
            self.paused_since = None

        def seed(self, *args):
            return [11]

        def update_from_propagation(self, frame, ids, masks):
            pass

        def pause(self, frame, live):
            calls.append(("pause", frame, list(live)))
            self.paused_since = frame
            return [{"type": "remove", "obj_id": oid} for oid in live]

        def resume(self, frame):
            if self.paused_since is not None:
                calls.append(("resume", frame))
            self.paused_since = None

        def checkpoint(self, frame, *args):
            checkpoints.append(frame)
            return [{"type": "add", "obj_id": 12, "box": [1, 1, 4, 5]}] if frame == 8 else []

    monkeypatch.setattr(driver, "TrackManager", Manager)
    monkeypatch.setattr(driver, "RESUME_AFTER_FRAMES", 0)  # resume at once; see the next test
    monkeypatch.setattr(driver.torch, "autocast", lambda *a, **k: nullcontext())
    det = sv.Detections(xyxy=np.array([[1, 1, 4, 5]], float), class_id=np.array([2]))
    _, _, frames = driver.drive_sam2(
        tmp_path / "unused.mp4", lambda fid: det, None, "weights.pt",
        check_every=2, frame_cache_dir=tmp_path, predictor_factory=lambda _: Predictor(),
        court_in_view=lambda frame: not 4 <= frame <= 6,
    )
    results = list(frames)
    assert calls == [("pause", 4, [11]), ("resume", 8)]
    assert checkpoints == [2, 8]  # nothing is decided from frames with no court
    assert [r.player_ids.tolist() for r in results] == [[11]] * 4 + [[]] * 4 + [[12]] * 2


def test_a_short_return_of_the_court_does_not_resume(tmp_path, monkeypatch):
    """A ~1 s replay inside a timeout shows the court but is not play resuming."""
    calls = run_with_court(tmp_path, monkeypatch, frames=13, resume_after=4,
                           in_view=lambda f: f not in (4, 8, 10))
    # Back at 6 for only 2 frames, out again at 8 and 10, back at 12: never 4 frames in view.
    assert calls == [("pause", 4)]
    calls = run_with_court(tmp_path, monkeypatch, frames=13, resume_after=4,
                           in_view=lambda f: f != 4)
    assert calls == [("pause", 4), ("resume", 10)]


def run_with_court(tmp_path, monkeypatch, frames, resume_after, in_view):
    """Drive a stub tracker through `frames` frames; return its pause/resume calls."""
    for frame in range(frames):
        cv2.imwrite(str(tmp_path / f"{frame:05d}.jpg"), np.zeros((8, 12, 3), np.uint8))

    class Predictor:
        def init_state(self, video_path):
            return {"obj_id_to_idx": {}}

        def add_new_points_or_box(self, state, *, obj_id, **kwargs):
            state["obj_id_to_idx"].setdefault(obj_id, len(state["obj_id_to_idx"]))

        def remove_object(self, state, *, obj_id):
            del state["obj_id_to_idx"][obj_id]

        def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
            for frame in range(start_frame_idx, start_frame_idx + max_frame_num_to_track + 1):
                yield frame, list(state["obj_id_to_idx"]), torch.full(
                    (len(state["obj_id_to_idx"]), 1, 8, 12), 1.)

    calls = []

    class Manager:
        def __init__(self, *args, **kwargs):
            self.paused_since = None

        def seed(self, *args):
            return [11]

        def update_from_propagation(self, *args):
            pass

        def pause(self, frame, live):
            calls.append(("pause", frame))
            self.paused_since = frame
            return [{"type": "remove", "obj_id": oid} for oid in live]

        def resume(self, frame):
            if self.paused_since is not None:
                calls.append(("resume", frame))
            self.paused_since = None

        def checkpoint(self, *args):
            return []

    monkeypatch.setattr(driver, "TrackManager", Manager)
    monkeypatch.setattr(driver, "RESUME_AFTER_FRAMES", resume_after)
    monkeypatch.setattr(driver.torch, "autocast", lambda *a, **k: nullcontext())
    det = sv.Detections(xyxy=np.array([[1, 1, 4, 5]], float), class_id=np.array([2]))
    _, _, results = driver.drive_sam2(
        tmp_path / "unused.mp4", lambda fid: det, None, "weights.pt",
        check_every=2, frame_cache_dir=tmp_path, predictor_factory=lambda _: Predictor(),
        court_in_view=in_view,
    )
    list(results)
    return calls
