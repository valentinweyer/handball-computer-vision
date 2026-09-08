"""Number evidence deciding identity: arbitration, linking, and the re-ID margin.

Each test here stands for a failure seen on the 60s Melsungen clip, where the
overlay showed two players wearing 15 on the same team and a returning 15 whose
label had moved to a teammate.
"""
import unittest

import numpy as np

from handball_cv.jersey.identity import NumberVoter
from handball_cv.tracking.identity import PlayerRegistry, REID_MARGIN_MIN
from scripts.render_full_pipeline import NumberIdentityResolver


def vote(voter, identity_id, value, times):
    for _ in range(times):
        voter.observe(identity_id, value)


class ArbitrationTests(unittest.TestCase):
    """One squad number, one player per team."""

    def test_the_weaker_claim_is_withheld_not_relabelled(self):
        voter = NumberVoter()
        vote(voter, 1, "25", 20)
        vote(voter, 2, "25", 4)
        self.assertEqual(voter.arbitrate({1: 0, 2: 0}), {2: 1})

        self.assertEqual(voter.best(1)[0], "25")
        # Abstains -- it does not inherit some other guess.
        self.assertIsNone(voter.best(2)[0])

    def test_the_same_number_on_opposite_teams_is_left_alone(self):
        voter = NumberVoter()
        vote(voter, 1, "25", 10)
        vote(voter, 2, "25", 10)
        self.assertEqual(voter.arbitrate({1: 0, 2: 1}), {})
        self.assertEqual(voter.best(1)[0], "25")
        self.assertEqual(voter.best(2)[0], "25")

    def test_suppression_reverses_when_the_evidence_does(self):
        voter = NumberVoter()
        vote(voter, 1, "25", 5)
        vote(voter, 2, "25", 4)
        self.assertEqual(voter.arbitrate({1: 0, 2: 0}), {2: 1})
        vote(voter, 2, "25", 10)
        self.assertEqual(voter.arbitrate({1: 0, 2: 0}), {1: 2})
        self.assertEqual(voter.best(2)[0], "25")
        self.assertIsNone(voter.best(1)[0])

    def test_an_unresolved_identity_claims_nothing(self):
        voter = NumberVoter()
        vote(voter, 1, "25", 10)
        vote(voter, 2, "25", 1)          # below min_votes: no verdict to contest
        self.assertEqual(voter.arbitrate({1: 0, 2: 0}), {})


class ReidMarginTests(unittest.TestCase):
    """Within a team the similarity score barely ranks; abstain on a near tie."""

    def _retired(self, registry, fragment_id, embedding):
        player = registry.create(
            0, fragment_id, team=0, embedding=np.asarray(embedding, float),
            is_goalkeeper=False, confidence=0.9, quality=1.0,
        )
        registry.retire(player.player_id, 0)
        return player

    def test_two_indistinguishable_candidates_produce_no_match(self):
        registry = PlayerRegistry()
        self._retired(registry, 1, [1.0, 0.02, 0.0])
        self._retired(registry, 2, [1.0, 0.0, 0.02])
        query = np.array([1.0, 0.01, 0.01])
        self.assertIsNone(registry.reid_match(query, 5, set()))

    def test_a_clear_winner_still_matches(self):
        registry = PlayerRegistry()
        first = self._retired(registry, 1, [1.0, 0.0, 0.0])
        self._retired(registry, 2, [0.0, 1.0, 0.0])
        match = registry.reid_match(np.array([1.0, 0.05, 0.0]), 5, set())
        self.assertIs(match, first)

    def test_a_lone_candidate_has_no_runner_up_to_beat(self):
        registry = PlayerRegistry()
        only = self._retired(registry, 1, [1.0, 0.0, 0.0])
        self.assertIs(registry.reid_match(np.array([1.0, 0.1, 0.0]), 5, set()), only)

    def test_the_margin_is_the_gate_and_not_the_floor(self):
        # Both candidates clear 0.7 comfortably; only the gap decides.
        registry = PlayerRegistry(reid_margin_min=0.5)
        self._retired(registry, 1, [1.0, 0.0, 0.0])
        self._retired(registry, 2, [0.9, 0.1, 0.0])
        self.assertIsNone(registry.reid_match(np.array([1.0, 0.0, 0.0]), 5, set()))
        self.assertGreater(REID_MARGIN_MIN, 0.0)


class _Registry(PlayerRegistry):
    """A registry whose team labels are set directly, so the resolver can be
    tested without a team model."""

    def __init__(self, teams):
        super().__init__()
        self._teams = teams

    def team_by_player_id(self):
        return dict(self._teams)


class LinkingTests(unittest.TestCase):
    def _resolver(self, teams):
        voter = NumberVoter()
        registry = _Registry(teams)
        return registry, voter, NumberIdentityResolver(registry, voter)

    def test_two_halves_of_one_player_are_folded(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 0})
        resolver.begin_frame([1])          # never on screen together
        resolver.begin_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 1)
        self.assertEqual(voter.best(1), ("15", 15, 1.0))

    def test_identities_seen_together_are_never_folded(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 0})
        resolver.begin_frame([1, 2])
        vote(voter, 1, "25", 9)
        vote(voter, 2, "25", 6)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 2)
        # Not merged, so arbitration has to withhold the weaker one instead.
        self.assertEqual(voter.best(1)[0], "25")
        self.assertIsNone(voter.best(2)[0])

    def test_opposite_teams_keep_their_own_number(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 1})
        resolver.begin_frame([1])
        resolver.begin_frame([2])
        vote(voter, 1, "7", 5)
        vote(voter, 2, "7", 5)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 2)
        self.assertEqual(voter.best(1)[0], "7")
        self.assertEqual(voter.best(2)[0], "7")

    def test_a_fold_inherits_co_liveness(self):
        # 3 shared a frame with 1. After 2 folds into 1, 3 must not fold in too.
        registry, voter, resolver = self._resolver({1: 0, 2: 0, 3: 0})
        resolver.begin_frame([1, 3])
        resolver.begin_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        vote(voter, 3, "15", 4)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 1)
        self.assertEqual(registry.canonical(3), 3)

    def test_linking_is_recorded_as_an_event(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 0})
        resolver.begin_frame([1])
        resolver.begin_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        resolver.resolve(250)

        links = [e for e in registry.events if e["type"] == "link"]
        self.assertEqual(len(links), 1)
        self.assertEqual(
            {k: links[0][k] for k in ("frame", "player_id", "into_player_id")},
            {"frame": 250, "player_id": 2, "into_player_id": 1},
        )


class InheritedVerdictTests(unittest.TestCase):
    """A number earned by one fragment must not speak for the next one.

    The 60s Melsungen case, exactly: p6 settled on 15 from nine reads over
    frames 65-110, was revived onto a different player at frame 560, and read
    20, 29 and 2 off that player's shirt while still labelled 15.
    """

    def test_a_revived_identity_stops_asserting_its_old_number(self):
        voter = NumberVoter()
        vote(voter, 6, "15", 9)
        self.assertEqual(voter.best(6)[0], "15")

        voter.suspend(6)
        self.assertIsNone(voter.best(6)[0])

    def test_reads_off_a_different_shirt_never_restore_it(self):
        voter = NumberVoter()
        vote(voter, 6, "15", 9)
        voter.suspend(6)
        for value in ("20", "29", "2"):        # the reads p6 actually produced
            voter.observe(6, value)
        self.assertIsNone(voter.best(6)[0])

    def test_one_agreeing_read_vouches_for_a_correct_revival(self):
        voter = NumberVoter()
        vote(voter, 6, "15", 9)
        voter.suspend(6)
        voter.observe(6, "15")
        self.assertEqual(voter.best(6)[0], "15")

    def test_a_partial_read_of_the_same_number_also_vouches(self):
        # A crop catching only the trailing digit of 15 reads "5"; the fold rule
        # already treats that as corroboration, and so must this.
        voter = NumberVoter()
        vote(voter, 6, "15", 9)
        voter.suspend(6)
        voter.observe(6, "5")
        self.assertEqual(voter.best(6)[0], "15")

    def test_a_contradicted_tally_is_disowned_so_the_new_player_can_be_read(self):
        # Keeping the inherited votes would leave 9 stale ones in the
        # denominator, and a fresh number would need ~25 reads to clear
        # min_margin against them.
        voter = NumberVoter()
        vote(voter, 6, "15", 9)
        voter.suspend(6)
        vote(voter, 6, "29", 4)
        self.assertEqual(voter.best(6), ("29", 4, 1.0))

    def test_one_stray_misread_does_not_discard_the_tally(self):
        voter = NumberVoter()
        vote(voter, 6, "15", 9)
        voter.suspend(6)
        voter.observe(6, "29")           # a misread, not a new player
        voter.observe(6, "15")
        self.assertEqual(voter.best(6)[0], "15")
        self.assertEqual(voter._votes[6].counts["15"], 10)

    def test_an_unvouched_claim_cannot_suppress_a_first_hand_one(self):
        # p6 carries more folded votes for 15 than the real 15 does. Unless the
        # inherited claim is excluded, arbitration hands the number to the wrong
        # player and withholds it from the right one.
        voter = NumberVoter()
        vote(voter, 6, "15", 13)
        voter.suspend(6)
        vote(voter, 20, "15", 6)
        self.assertEqual(voter.arbitrate({6: 0, 20: 0}), {})
        self.assertEqual(voter.best(20)[0], "15")
        self.assertIsNone(voter.best(6)[0])

    def test_suspending_an_identity_with_no_verdict_is_a_no_op(self):
        voter = NumberVoter()
        voter.observe(9, "7")
        self.assertIsNone(voter.suspend(9))
        self.assertIsNone(voter.best(9)[0])


class ResolverSuspensionTests(unittest.TestCase):
    def test_a_reid_event_suspends_before_the_frames_reads_are_counted(self):
        voter = NumberVoter()
        registry = _Registry({6: 0})
        resolver = NumberIdentityResolver(registry, voter)
        vote(voter, 6, "15", 9)

        resolver.begin_frame([6])
        self.assertEqual(voter.best(6)[0], "15")

        registry.events.append({"frame": 560, "type": "reid", "player_id": 6})
        resolver.begin_frame([6])
        self.assertIsNone(voter.best(6)[0])

    def test_each_reid_event_is_acted_on_once(self):
        voter = NumberVoter()
        registry = _Registry({6: 0})
        resolver = NumberIdentityResolver(registry, voter)
        vote(voter, 6, "15", 9)
        registry.events.append({"frame": 560, "type": "reid", "player_id": 6})

        resolver.begin_frame([6])
        voter.observe(6, "15")                 # vouched by the next read
        resolver.begin_frame([6])              # must not re-suspend
        self.assertEqual(voter.best(6)[0], "15")


if __name__ == "__main__":
    unittest.main()
