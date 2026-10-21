"""Contract and failure-case tests for retrospective team overlap repair."""
from copy import deepcopy

import pytest

from experiments.arc_team_repair.repair import repair_tracks


def fixture():
    lines, reads = [], {}
    for f in range(25):
        if f < 8:
            a, b = 0, 30
        elif f == 8:
            a, b = 8, 14
        elif f == 9:
            a, b = 9, 13
        elif f == 10:
            a, b = 12, 10    # Wrong route begins here.
        elif f == 11:
            a, b = 13, 9
        elif f == 12:
            a, b = 16, 6
        else:
            a, b = 30, 0
        for tid, x in [(1, a), (2, b)]:
            lines.append(f"{f+1},{tid},{x},0,10,20,1,-1,-1,-1")
        if f < 8:
            reads[f, 1], reads[f, 2] = 0, 1
        elif f >= 13:
            reads[f, 1], reads[f, 2] = 1, 0
    return lines, reads


def test_disabled_is_exact_passthrough_even_on_swap():
    lines, reads = fixture()
    output, audit = repair_tracks(lines, reads)
    assert output == lines
    assert audit["corrections"] == []


def test_repair_rewinds_to_supported_switch_without_changing_boxes_or_reads():
    lines, reads = fixture()
    original = deepcopy((lines, reads))
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert len(audit["corrections"]) == 1
    assert audit["corrections"][0]["boundary"]["frame"] == 10
    assert output[:20] == lines[:20]  # Correct prefix survives.
    assert output[20].split(",")[1] == "2"
    assert output[21].split(",")[1] == "1"
    assert audit["changed_rows"] == 30
    assert all(a.split(",")[2:] == b.split(",")[2:] for a, b in zip(lines, output))
    assert (lines, reads) == original


@pytest.mark.parametrize("case", ["same_team", "weak_post", "already_consistent", "later_contradiction"])
def test_unproven_team_swap_is_not_repaired(case):
    lines, reads = fixture()
    if case == "same_team":
        reads = {key: 0 for key in reads}
    elif case == "weak_post":
        reads = {key: value for key, value in reads.items() if key[0] < 8 or key[0] == 13}
    elif case == "already_consistent":
        reads = {(f, t): t - 1 for f, t in reads}
    elif case == "later_contradiction":
        for f in (23, 24):
            reads[f, 1] = 0
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert output == lines
    assert not audit["corrections"]


def test_third_player_at_contact_is_not_forced_into_pair_swap():
    lines, reads = fixture()
    lines.append("10,3,10,0,10,20,1,-1,-1,-1")
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert output == lines
    assert not audit["corrections"]
    assert any(e["decision"] == "third_participant" for e in audit["encounters"])


def test_missing_track_during_contact_is_not_stitched():
    lines, reads = fixture()
    lines = [line for line in lines if not line.startswith("10,2,")]
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert output == lines
    assert not audit["corrections"]


def test_team_reversal_without_geometric_switch_evidence_abstains():
    lines, reads = fixture()
    # Hold both paths steady through the contact: no uniquely supported
    # rerouting point, despite contradictory clear endpoint labels.
    rewritten = []
    for line in lines:
        p = line.split(",")
        if 8 <= int(p[0]) - 1 <= 12:
            p[2] = "10"
        rewritten.append(",".join(p))
    output, audit = repair_tracks(rewritten, reads, enabled=True)
    assert output == rewritten
    assert not audit["corrections"]


def handover_fixture():
    lines, reads = [], {}
    for f in range(25):
        a, b = (0, 30) if f < 8 else (10 + min(f - 8, 3), 14 - (f - 8))
        lines.append(f"{f+1},1,{a},0,10,20,1,-1,-1,-1")
        if f <= 9:
            lines.append(f"{f+1},2,{b},0,10,20,1,-1,-1,-1")
        if f < 8:
            reads[f, 1], reads[f, 2] = 0, 1
        elif f >= 13:
            reads[f, 1] = 1
    return lines, reads


def test_lost_partner_handover_reconnects_tail_without_guessing_inside_contact():
    lines, reads = handover_fixture()
    original = deepcopy((lines, reads))
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert len(audit["corrections"]) == 1
    correction = audit["corrections"][0]
    assert correction["kind"] == "lost_partner"
    assert correction["boundary"]["frame"] == 13
    assert output[:23] == lines[:23]  # Includes the ambiguous post-loss gap.
    assert all(line.split(",")[1] == "2" for line in output[23:])
    assert all(a.split(",")[2:] == b.split(",")[2:] for a, b in zip(lines, output))
    assert (lines, reads) == original


def test_handover_starts_at_a_read_supporting_the_confirmed_team():
    lines, reads = handover_fixture()
    # A single leading contrary read is tolerated by the aggregate purity
    # checks on a long tail, but must not move the correction onto that row.
    reads[13, 1] = 0
    for f in range(25, 50):
        lines.append(f"{f+1},1,13,0,10,20,1,-1,-1,-1")
        reads[f, 1] = 1
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert len(audit["corrections"]) == 1
    assert audit["corrections"][0]["boundary"]["frame"] == 14
    assert output[:24] == lines[:24]


@pytest.mark.parametrize("case", ["returning_partner", "same_team", "weak_post",
                                 "later_contradiction", "no_bridge", "competing_partner",
                                 "unknown_competing_partner", "unknown_competing_survivor"])
def test_unsupported_handover_abstains(case):
    lines, reads = handover_fixture()
    if case == "returning_partner":
        lines.append("25,2,100,0,10,20,1,-1,-1,-1")
    elif case == "same_team":
        reads = {key: 1 for key in reads}
    elif case == "weak_post":
        reads = {key: value for key, value in reads.items() if key[0] < 8 or key[0] == 13}
    elif case == "later_contradiction":
        reads[23, 1] = reads[24, 1] = 0
    elif case == "no_bridge":
        rewritten = []
        for line in lines:
            p = line.split(",")
            if int(p[0]) >= 11:
                p[2] = "100"
            rewritten.append(",".join(p))
        lines = rewritten
    elif case in ("competing_partner", "unknown_competing_partner"):
        for line in list(lines):
            p = line.split(",")
            if p[1] == "2":
                p[1] = "3"
                lines.append(",".join(p))
        for (f, t), team in list(reads.items()):
            if t == 2 and case == "competing_partner":
                reads[f, 3] = team
    elif case == "unknown_competing_survivor":
        for line in list(lines):
            p = line.split(",")
            if p[1] == "1":
                p[1] = "3"
                lines.append(",".join(p))
    output, audit = repair_tracks(lines, reads, enabled=True)
    assert output == lines
    assert not audit["corrections"]
