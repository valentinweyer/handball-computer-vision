"""A clipped or single-frame team guess must not resurrect an old identity."""
import numpy as np
import pytest

from handball_cv.tracking.sam2_manager import TrackManager


class Observations:
    def __init__(self, teams, qualities):
        self.teams = iter(teams)
        self.qualities = iter(qualities)

    def observe(self, image, boxes):
        n = len(boxes)
        return (np.tile([1., 0., 0., 0.], (n, 1)),
                np.full(n, next(self.teams)), np.full(n, .748),
                np.full(n, next(self.qualities)))


@pytest.mark.parametrize("boxes,teams,qualities,revive", [
    # The #61 re-entry at f5970 was only 30px wide at the left image edge.
    ([[.4, 526, 30.6, 750], [.4, 526, 30.6, 750]], [0, 0], [.813, .813], False),
    # A later clear crop alone must not qualify evidence from a clipped entry.
    ([[.4, 526, 30.6, 750], [30, 526, 70, 750]], [0, 0], [.813, .813], False),
    # One clean read after a bad or contradictory observation is insufficient.
    ([[100, 200, 140, 400]] * 2, [0, 0], [0., .813], False),
    ([[100, 200, 140, 400]] * 2, [1, 0], [.813, .813], False),
    # Two clean agreeing checkpoints preserve a legitimate re-identification.
    ([[100, 200, 140, 400]] * 2, [0, 0], [.813, .813], True),
])
def test_reid_requires_clean_consistent_entry_evidence(boxes, teams, qualities, revive):
    manager = TrackManager(Observations(teams, qualities), court_test_fn=lambda box: True)
    old = manager.registry.create(5800, None, 0, np.array([1., 0., 0., 0.]), False)
    for _ in range(6):
        old.record_team_vote(0, .9, 1.)
    manager.registry.retire(old.player_id, 5900)
    actions = []
    for fid, box in zip([5960, 5970], boxes):
        actions += manager.checkpoint(
            fid, np.zeros((1080, 1920, 3), np.uint8), [], np.empty((0, 0, 0), bool),
            np.array([box]), np.array([False]),
        )
    added = [a['obj_id'] for a in actions if a['type'] == 'add']
    assert len(added) == 1  # tracking still starts even if re-ID must abstain
    assert (added[0] == old.player_id) == revive
    assert (old in manager.registry.retired) != revive


def test_two_detections_in_one_checkpoint_cannot_confirm_an_identity():
    manager = TrackManager(Observations([0], [1.]), court_test_fn=lambda box: True)
    actions = manager.checkpoint(
        5960, np.zeros((1080, 1920, 3), np.uint8), [], np.empty((0, 0, 0), bool),
        np.array([[100, 200, 140, 400], [130, 200, 170, 400]]), np.array([False, False]),
    )
    assert not any(a['type'] == 'add' for a in actions)
    assert len(manager._pending_new_boxes) == 2
    assert [entry[3] for entry in manager._pending_new_boxes] == [1, 1]
