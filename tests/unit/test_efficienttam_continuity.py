from experiments.efficienttam.score_comparison import continuity, handoffs


def test_clean_fragmentation_is_an_id_transition_even_without_mixed_tracklets():
    refs = {f: {1: []} for f in range(4)}
    report = continuity(refs, [(0, 11, 1), (1, 11, 1), (2, 22, 1), (3, 22, 1)], {1})[1]
    assert report['longest_constant_id_match_run'] == 2
    assert len(report['predicted_id_transitions']) == 1
    assert report['missing_frames'] == 0


def test_gap_under_same_reused_id_is_not_uninterrupted():
    refs = {f: {1: []} for f in range(5)}
    report = continuity(refs, [(0, 11, 1), (1, 11, 1), (4, 11, 1)], {1})[1]
    assert report['longest_constant_id_match_run'] == 2
    assert report['missing_runs'] == [{'start': 2, 'end': 3, 'id': None}]
    assert report['predicted_id_transitions'] == []


def test_untrusted_gap_does_not_create_miss_or_bridge_runs():
    report = continuity({0: {1: []}, 4: {1: []}}, [(0, 11, 1), (4, 22, 1)], {1})[1]
    assert report['longest_constant_id_match_run'] == 1
    assert report['missing_frames'] == 0
    assert report['predicted_id_transitions'] == []


def test_same_team_switch_needs_verified_team_labels():
    matches = [(0, 11, 1), (1, 11, 2), (2, 11, 3)]
    events = handoffs(matches, {1, 2, 3}, {'1': 'A', '2': 'A'})
    assert [e['same_team'] for e in events] == [True, None]
