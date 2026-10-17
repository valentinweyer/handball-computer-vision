"""Regression cases independently confirmed in the Kiel identity audit."""
import numpy as np
import pytest

from handball_cv.jersey.identity import NumberVoter, causal_labels, match_numbers_to_players
from handball_cv.pipeline.tables import reads_table
from handball_cv.pipeline.result import load, write_bundle
from handball_cv.tracking.identity import PlayerRegistry
from scripts.render_full_pipeline import NumberIdentityResolver, unique_pairs


def player(registry, team=0, embedding=(1., 0.)):
    return registry.create(0, None, team, np.array(embedding), False, 1., 1.)


def votes(voter, pid, value, n=4):
    for _ in range(n):
        voter.observe(pid, value)


@pytest.mark.parametrize("candidate_is_root", [True, False])
def test_reid_cannot_enter_a_family_already_live(candidate_is_root):
    r = PlayerRegistry()
    root, member = player(r), player(r)
    r.link(member.player_id, root.player_id, 1)
    retired = root if candidate_is_root else member
    r.retire(retired.player_id, 2)
    assert r.reid_match(np.array([1., 0.]), 3, set()) is None


def test_reid_reserves_whole_alias_chain_within_one_checkpoint():
    r = PlayerRegistry()
    root = player(r)
    middle, member = player(r, embedding=(0., 1.)), player(r, embedding=(0., 1.))
    r.link(member.player_id, middle.player_id, 1)
    r.link(middle.player_id, root.player_id, 1)
    for p in (root, middle, member):
        r.retire(p.player_id, 2)
    assert r.reid_match(np.array([1., 0.]), 3, {member.player_id}) is None


def test_unoccupied_linked_family_can_still_be_revived():
    r = PlayerRegistry()
    root, member = player(r), player(r, embedding=(0., 1.))
    r.link(member.player_id, root.player_id, 1)
    r.retire(root.player_id, 2)
    r.retire(member.player_id, 2)
    assert r.reid_match(np.array([1., 0.]), 3, set()) is root


def test_mixed_contradictions_keep_the_reads_that_actually_happened():
    v = NumberVoter()
    votes(v, 1, "61", 9)
    v.suspend(1)
    v.observe(1, "24")
    v.observe(1, "35")
    assert v._votes[1].counts == {"24": 1, "35": 1}
    v.observe(1, "35")
    assert v.best(1)[0] is None  # only two actual 35 reads, not three
    v.observe(1, "35")
    assert v.best(1)[0] == "35"


def test_identical_contradictions_and_partial_recovery_still_work():
    v = NumberVoter()
    votes(v, 1, "61", 9)
    v.suspend(1)
    votes(v, 1, "35", 3)
    assert v.best(1)[0] == "35"
    v.suspend(1)
    v.observe(1, "5")
    assert v.best(1)[0] == "35"


def test_visible_family_member_owns_its_team_for_arbitration():
    r = PlayerRegistry()
    retired = player(r, team=1)
    visible = player(r, team=0)
    other = player(r, team=1)
    r.retire(retired.player_id, 1)
    r.link(visible.player_id, retired.player_id, 2)
    v = NumberVoter()
    votes(v, retired.player_id, "7")
    votes(v, other.player_id, "7", 6)
    resolver = NumberIdentityResolver(r, v)
    resolver.begin_frame([visible.player_id, other.player_id])
    assert v.best(retired.player_id)[0] == "7"
    assert v.best(other.player_id)[0] == "7"
    # A real same-team conflict must still suppress the weaker number claim.
    visible.team_id = 1
    resolver.begin_frame([visible.player_id, other.player_id])
    assert v.best(retired.player_id)[0] is None


def test_causal_replay_ignores_offscreen_member_team():
    reads = ([{"frame": 0, "player_id": 1, "value": "7"}] * 4
             + [{"frame": 0, "player_id": 3, "value": "7"}] * 6)
    timeline = causal_labels(
        reads, [{"frame": 2, "type": "link", "player_id": 2, "into_player_id": 1}],
        {1: {0, 1}, 2: {2, 3}, 3: {0, 1, 2, 3}}, teams={2: 0, 3: 1, 1: 1},
    )
    assert timeline[2] == ((2, "7"),)


@pytest.mark.parametrize("dtype", [bool, np.uint8])
def test_tied_number_owner_abstains_independent_of_mask_order(dtype):
    masks = np.ones((2, 20, 20), dtype=dtype)
    boxes = np.array([[3., 3., 8., 8.]])
    assert match_numbers_to_players(masks, boxes, (20, 20)) == [(0, 0), (1, 0)]
    for ordered in (masks, masks[::-1]):
        assert unique_pairs(match_numbers_to_players(
            ordered, boxes, (20, 20), reject_ambiguous=True,
        )) == []


def test_number_owner_with_a_clear_best_match_is_preserved():
    masks = np.ones((2, 20, 20), dtype=bool)
    masks[1, 3, 3:8] = False
    boxes = np.array([[3., 3., 12., 12.]])
    assert unique_pairs(match_numbers_to_players(
        masks, boxes, (20, 20), reject_ambiguous=True,
    )) == [(0, 0)]


def test_ambiguous_number_does_not_consume_a_players_unambiguous_read():
    masks = np.ones((2, 20, 20), dtype=bool)
    masks[1, :, 10:] = False
    boxes = np.array([[3., 3., 6., 6.], [12., 3., 16., 6.]])
    assert unique_pairs(match_numbers_to_players(
        masks, boxes, (20, 20), reject_ambiguous=True,
    )) == [(0, 1)]


def test_read_provenance_is_optional_for_old_runs_and_survives_links():
    old = {"frame": 0, "player_id": 65, "value": "8", "number_row": 0, "box": [1, 2, 3, 4]}
    new = {**old, "frame": 1, "track_id": 165}
    table = reads_table({"number_reads": [old, new]})
    assert table.player_id.tolist() == [65, 65]
    assert table.track_id.isna().tolist() == [True, False]
    assert table.loc[1, "track_id"] == 165


def test_read_provenance_round_trips_without_breaking_legacy_bundles(tmp_path):
    row = {"frame": 0, "player_id": 65, "track_id": 165, "value": "8",
           "number_row": 0, "box": [1, 2, 3, 4]}
    table = reads_table({"number_reads": [row]})
    path = write_bundle(tmp_path / "new", {"reads": table}, files={}, info={})
    assert load(path).reads.loc[0, "track_id"] == 165
    legacy = table.drop(columns="track_id")
    path = write_bundle(tmp_path / "old", {"reads": legacy}, files={}, info={})
    assert load(path).reads.equals(legacy)
