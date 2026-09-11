"""What the redraw pass draws, given segment verdicts instead of a live voter.

The causal render earns the right to say a number partway through a player's
span; these cover the rules that decide how far back that right reaches.
"""
import unittest

from handball_cv.jersey.identity import segment_verdicts
from scripts.render_full_pipeline import backfilled_labels


def reads(*rows):
    return [{"frame": f, "player_id": p, "value": v} for f, p, v in rows]


def index(segments):
    by_player = {}
    for segment in segments:
        by_player.setdefault(segment.player_id, []).append(segment)
    return by_player


def labels_at(by_player, frame, player_ids, teams=None, causal=False):
    return backfilled_labels(
        by_player, frame, player_ids, [0] * len(player_ids),
        roster={}, canonical=int, causal=causal,
    )


class BackfilledLabelTests(unittest.TestCase):
    def test_a_number_reaches_back_to_the_start_of_its_segment(self):
        by_player = index(segment_verdicts(
            reads((10, 1, "7"), (20, 1, "7"), (30, 1, "7")),
            events=[], live_frames={1: range(1, 51)},
        ))
        # Frame 5 drew P1 in the causal render; the same evidence explains it.
        self.assertEqual(labels_at(by_player, 5, [1], [0]), ["#7"])
        self.assertEqual(labels_at(by_player, 45, [1], [0]), ["#7"])

    def test_causal_mode_withholds_the_backfill(self):
        """The check that the redraw reproduces the render it replaces."""
        by_player = index(segment_verdicts(
            reads((10, 1, "7"), (20, 1, "7"), (30, 1, "7")),
            events=[], live_frames={1: range(1, 51)},
        ))
        self.assertEqual(labels_at(by_player, 5, [1], [0], causal=True), ["P1"])
        self.assertEqual(labels_at(by_player, 29, [1], [0], causal=True), ["P1"])
        self.assertEqual(labels_at(by_player, 30, [1], [0], causal=True), ["#7"])

    def test_a_revised_segment_stays_causal_even_when_backfilling(self):
        """p20 read 6, then 15. Frame 900 showed 6 and still shows 6.

        Backfill is withheld from a revised segment, but so is the revision
        itself: applying the final answer from the segment's first labelled
        frame would be backfill by another name.
        """
        by_player = index(segment_verdicts(
            reads(
                (10, 20, "6"), (20, 20, "6"), (30, 20, "6"),
                *[(f, 20, "15") for f in range(40, 130, 10)],
            ),
            events=[], live_frames={20: range(1, 201)},
        ))
        self.assertEqual(labels_at(by_player, 5, [20], [0]), ["P20"])
        self.assertEqual(labels_at(by_player, 35, [20], [0]), ["#6"])
        self.assertEqual(labels_at(by_player, 150, [20], [0]), ["#15"])

    def test_backfill_stops_at_an_identity_break(self):
        by_player = index(segment_verdicts(
            reads(
                (10, 6, "15"), (20, 6, "15"), (30, 6, "15"),
                (70, 6, "20"), (80, 6, "20"), (90, 6, "20"),
            ),
            events=[{"frame": 60, "type": "reid", "player_id": 6}], live_frames={6: range(1, 101)},
        ))
        self.assertEqual(labels_at(by_player, 5, [6], [0]), ["#15"])
        self.assertEqual(labels_at(by_player, 59, [6], [0]), ["#15"])
        # The far side of the break is a different person and says so.
        self.assertEqual(labels_at(by_player, 60, [6], [0]), ["#20"])

    def test_one_number_per_team_is_settled_in_the_replay_not_at_draw_time(self):
        """Backfill makes arbitration matter more, not less.

        Two identities claiming 9 on one team cannot both be right, and the
        weaker abstains. The decision belongs to the replay -- the same place
        the render made it -- so that a redraw cannot reach a different verdict
        than the run it is redrawing.
        """
        rows = reads(
            *[(f, 1, "9") for f in (10, 20, 30, 40, 50)],
            *[(f, 2, "9") for f in (80, 90, 100)],
        )
        live = {1: range(1, 121), 2: range(1, 121)}

        same = index(segment_verdicts(rows, [], live, teams={1: 0, 2: 0}))
        self.assertEqual(labels_at(same, 110, [1, 2]), ["#9", "P2"])
        # ...and the backfill inherits that, rather than reinstating the loser.
        self.assertEqual(labels_at(same, 5, [1, 2]), ["#9", "P2"])

        opposite = index(segment_verdicts(rows, [], live, teams={1: 0, 2: 1}))
        self.assertEqual(labels_at(opposite, 110, [1, 2]), ["#9", "#9"])

    def test_a_player_with_no_resolved_number_still_draws_its_id(self):
        by_player = index(segment_verdicts(
            reads((10, 4, "7"), (20, 4, "9")),
            events=[], live_frames={4: range(1, 51)},
        ))
        self.assertEqual(labels_at(by_player, 25, [4], [0]), ["P4"])

    def test_a_player_absent_from_the_segments_draws_its_id(self):
        self.assertEqual(labels_at({}, 5, [12], [0]), ["P12"])


if __name__ == "__main__":
    unittest.main()
