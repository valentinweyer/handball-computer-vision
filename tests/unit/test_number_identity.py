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
        resolver.observe_frame([1])          # never on screen together
        resolver.observe_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 1)
        self.assertEqual(voter.best(1), ("15", 15, 1.0))

    def test_identities_seen_together_are_never_folded(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 0})
        resolver.observe_frame([1, 2])
        vote(voter, 1, "25", 9)
        vote(voter, 2, "25", 6)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 2)
        # Not merged, so arbitration has to withhold the weaker one instead.
        self.assertEqual(voter.best(1)[0], "25")
        self.assertIsNone(voter.best(2)[0])

    def test_opposite_teams_keep_their_own_number(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 1})
        resolver.observe_frame([1])
        resolver.observe_frame([2])
        vote(voter, 1, "7", 5)
        vote(voter, 2, "7", 5)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 2)
        self.assertEqual(voter.best(1)[0], "7")
        self.assertEqual(voter.best(2)[0], "7")

    def test_a_fold_inherits_co_liveness(self):
        # 3 shared a frame with 1. After 2 folds into 1, 3 must not fold in too.
        registry, voter, resolver = self._resolver({1: 0, 2: 0, 3: 0})
        resolver.observe_frame([1, 3])
        resolver.observe_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        vote(voter, 3, "15", 4)
        resolver.resolve(100)

        self.assertEqual(registry.canonical(2), 1)
        self.assertEqual(registry.canonical(3), 3)

    def test_linking_is_recorded_as_an_event(self):
        registry, voter, resolver = self._resolver({1: 0, 2: 0})
        resolver.observe_frame([1])
        resolver.observe_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        resolver.resolve(250)

        links = [e for e in registry.events if e["type"] == "link"]
        self.assertEqual(len(links), 1)
        self.assertEqual(
            {k: links[0][k] for k in ("frame", "player_id", "into_player_id")},
            {"frame": 250, "player_id": 2, "into_player_id": 1},
        )


if __name__ == "__main__":
    unittest.main()
