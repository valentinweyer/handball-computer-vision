"""Historical labels belong to their source track, even after later links."""
import copy

import numpy as np

from handball_cv.jersey.identity import Segment
from handball_cv.pipeline.tables import _supported_track_segments, tracks_table
from handball_cv.tracking.geometry_cache import GeometryCache
from handball_cv.visualization.palette import TEAM_A


def assemble(tmp_path, run):
    total = max(int(f) for f in run['frame_players']) + 2
    cache = GeometryCache(40, 30, total)
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:20, 10:20] = True
    for frame, ids in run['frame_players'].items():
        cache.add(int(frame), ids, [[10, 5, 20, 20]] * len(ids), [mask] * len(ids),
                  color_index=[TEAM_A] * len(ids), team_id=[0] * len(ids))
    geometry = GeometryCache.load(cache.save(tmp_path / 'geometry.npz'))
    return tracks_table(run, geometry, [], fps=25, ocr_every=1)


def read(frame, track, value, owner=None):
    return dict(frame=frame, player_id=track if owner is None else owner,
                track_id=track, value=value)


def future_fold_run():
    """Old family #3, later #4 track, and a separate genuine #3 together."""
    presence = {str(f): [71] for f in range(1, 13)}
    presence.update({str(f): [23] for f in range(13, 16)})
    presence.update({str(f): [114, 113] for f in range(16, 24)})
    presence.update({str(f): [113] for f in range(24, 27)})
    presence['30'] = []
    reads = [read(f, 71, '3') for f in range(1, 13)]
    reads += [read(f, 23, '4') for f in range(13, 16)]
    reads += [read(f, 114, '4', 23 if f > 18 else None) for f in range(16, 24)]
    reads += [read(f, 113, '3') for f in range(16, 19)]
    return dict(frame_players=presence, number_reads=reads,
                identity_events=[dict(frame=18, type='link', player_id=114, into_player_id=23),
                                 dict(frame=30, type='link', player_id=23, into_player_id=71)],
                player_teams={'71': 0, '23': 0, '114': 0, '113': 0},
                player_aliases={'114': 23, '23': 71})


def test_later_fold_cannot_hide_independently_read_three_and_four(tmp_path):
    run = future_fold_run()
    before = copy.deepcopy(run)
    tracks = assemble(tmp_path, run)
    for track, number in [(114, '4'), (113, '3')]:
        rows = tracks[(tracks.track_id == track) & tracks.frame.between(16, 23)]
        assert rows.number.tolist() == [number] * 8
        # The existing causal column and identity fold keep their old meaning.
        assert rows.number_live.tolist() == [None, None] + [number] * 6
    assert set(tracks.loc[tracks.track_id == 114, 'player_id']) == {71}
    assert run == before


def test_legacy_reads_without_source_provenance_keep_existing_behavior(tmp_path):
    run = future_fold_run()
    for row in run['number_reads']:
        row.pop('track_id')
    tracks = assemble(tmp_path, run)
    # Source ownership cannot be reconstructed safely from final aliases.
    assert tracks.loc[tracks.frame.between(16, 23), 'number'].isna().all()


def test_independent_source_verdict_cannot_bypass_real_duplicate_guard(tmp_path):
    presence = {'1': [1, 2], '2': [1, 2]}
    presence.update({str(f): [1] for f in range(3, 6)})
    presence.update({str(f): [2] for f in range(6, 9)})
    run = dict(frame_players=presence,
               number_reads=[read(f, 1, '4') for f in range(3, 6)] +
                            [read(f, 2, '4') for f in range(6, 9)],
               identity_events=[], player_teams={'1': 0, '2': 0}, player_aliases={})
    tracks = assemble(tmp_path, run)
    assert tracks.loc[tracks.frame.isin([1, 2]), 'number'].isna().all()


def test_trailing_digit_does_not_erase_full_linked_number(tmp_path):
    presence = {str(f): [1] for f in range(1, 13)}
    presence.update({str(f): [2] for f in range(13, 17)})
    run = dict(frame_players=presence,
               number_reads=[read(f, 1, '23') for f in range(1, 13)] +
                            [read(f, 2, '3') for f in range(13, 17)],
               identity_events=[dict(frame=17, type='link', player_id=2, into_player_id=1)],
               player_teams={'1': 0, '2': 0}, player_aliases={'2': 1})
    tracks = assemble(tmp_path, run)
    assert tracks.loc[tracks.track_id == 2, 'number'].tolist() == ['23'] * 4


def test_correct_family_backfill_survives_revised_local_reads(tmp_path):
    presence = {str(f): [1] for f in range(1, 21)}
    presence.update({str(f): [2] for f in range(21, 35)})
    run = dict(frame_players=presence,
               number_reads=[read(f, 1, '21') for f in range(1, 21)] +
                            [read(f, 2, '2' if f <= 23 else '21') for f in range(21, 35)],
               identity_events=[dict(frame=35, type='link', player_id=2, into_player_id=1)],
               player_teams={'1': 0, '2': 0}, player_aliases={'2': 1})
    tracks = assemble(tmp_path, run)
    assert tracks.loc[tracks.track_id == 2, 'number'].tolist() == ['21'] * 14


def test_source_reads_before_revival_cannot_support_later_segment():
    old = Segment(1, 1, 3, (1, 2, 3), ((3, '4'),), 3, 1.)
    later = Segment(1, 10, 12, (10, 11, 12), ((10, '4'),), 3, 1.)
    run = {'number_reads': [read(f, 1, '4') for f in [1, 2, 3]]}
    assert _supported_track_segments(run, {1: [old, later]}) == {(1, 1)}


def test_two_reads_cannot_override_a_family_verdict():
    segment = Segment(1, 1, 3, (1, 2, 3), ((1, '4'),), 3, 1.)
    run = {'number_reads': [read(f, 1, '4') for f in [1, 2]]}
    assert _supported_track_segments(run, {1: [segment]}) == set()
