"""Number evidence deciding identity: arbitration, linking, and the re-ID margin.

Each test here stands for a failure seen in a real clip, where the overlay
showed two players wearing 15 on the same team and a returning 15 whose label had
moved to a teammate.
"""
import unittest

import numpy as np

from handball_cv.jersey.identity import IDENTITY_BREAK_EVENTS, NumberIdentityResolver, NumberVoter, causal_labels
from handball_cv.tracking.identity import PlayerRegistry, REID_MARGIN_MIN


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

    def test_offscreen_holder_releases_number_between_ocr_frames(self):
        registry, voter, resolver = self._resolver({4: 1, 13: 1})
        resolver.begin_frame([4, 13])
        vote(voter, 4, "41", 77)
        vote(voter, 13, "41", 35)
        resolver.resolve(3300)
        self.assertIsNone(voter.best(13)[0])

        # No new read or resolve call: visibility alone releases the claim.
        resolver.begin_frame([13])
        self.assertEqual(voter.best(13)[0], "41")
        self.assertEqual(registry.canonical(13), 13)  # past overlap still bars linking

        resolver.begin_frame([4, 13])
        self.assertIsNone(voter.best(13)[0])
        self.assertEqual(voter.best(4)[0], "41")

    def test_replay_arbitrates_only_currently_visible_holders(self):
        reads = ([{"frame": 0, "player_id": 4, "value": "41"}] * 77
                 + [{"frame": 0, "player_id": 13, "value": "41"}] * 35)
        timeline = causal_labels(
            reads, [], {4: {0, 6}, 13: set(range(7))}, teams={4: 1, 13: 1},
        )
        # Track 4 leaves at frame 1 and returns at 6, between OCR ticks.
        self.assertEqual(timeline[13], ((1, "41"), (6, None)))

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

    def test_reads_after_a_fold_reach_the_surviving_identity(self):
        """Measured on the overnight batch: 7% of all number evidence, and 39%
        on one clip, landed on identities that had already been folded away.

        The tracker keeps emitting the folded id -- an alias is an
        interpretation layer, not a renumbering -- so a caller filing votes
        under the raw id strands every read taken after the link, and
        contradicting evidence can never correct the surviving verdict.
        """
        registry, voter, resolver = self._resolver({1: 0, 2: 0})
        resolver.begin_frame([1])
        resolver.begin_frame([2])
        vote(voter, 1, "15", 9)
        vote(voter, 2, "15", 6)
        resolver.resolve(100)
        self.assertEqual(registry.canonical(2), 1)

        # What the render does with every subsequent read for the folded id.
        for _ in range(4):
            voter.observe(registry.canonical(2), "15")
        self.assertEqual(voter.best(1)[1], 19)      # 9 + 6 folded + 4 after
        self.assertIsNone(voter.best(2)[0])

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

    An identity settled on 15 from nine reads, was revived onto a different
    player, and read 20, 29 and 2 off that player's shirt while still labelled 15.
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
    def test_aliased_identity_invalidation_reaches_the_number_owner(self):
        for event_type in IDENTITY_BREAK_EVENTS:
            with self.subTest(event_type=event_type):
                voter = NumberVoter()
                registry = _Registry({47: 0, 52: 0, 59: 0})
                registry.alias.update({59: 52, 52: 47})
                resolver = NumberIdentityResolver(registry, voter)
                vote(voter, 47, "61", 9)
                registry.events.append({"frame": 5970, "type": event_type, "player_id": 59})
                resolver.begin_frame([59])
                self.assertIsNone(voter.best(47)[0])
                voter.observe(47, "61")  # a fresh agreeing read may restore it
                resolver.begin_frame([59])
                self.assertEqual(voter.best(47)[0], "61")

    def test_replay_invalidates_linked_number_until_fresh_evidence(self):
        for event_type in IDENTITY_BREAK_EVENTS:
            with self.subTest(event_type=event_type):
                rows = [{"frame": f, "player_id": 47, "value": "61"} for f in (0, 5, 10)]
                rows.append({"frame": 35, "player_id": 47, "value": "61"})
                timeline = causal_labels(
                    rows,
                    [{"frame": 20, "type": "link", "player_id": 59, "into_player_id": 47},
                     {"frame": 30, "type": event_type, "player_id": 59}],
                    {47: set(range(20)), 59: set(range(20, 41))},
                )
                self.assertEqual(timeline[59], ((20, "61"), (30, None), (35, "61")))

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


def _reads(track, value, n, frame=0):
    return [{"frame": frame + i, "track_id": track, "player_id": track, "value": value} for i in range(n)]


class UnbackedLinkTests(unittest.TestCase):
    def test_a_fragment_whose_own_reads_say_another_number_is_not_folded(self):
        # A #22 whose five chest frames settled "11", then fifteen clear "22" reads.
        from handball_cv.jersey.identity import unbacked_links
        reads = _reads(75, "11", 5) + _reads(75, "22", 15, frame=300) + _reads(5, "11", 40)
        link = {"frame": 10, "type": "link", "player_id": 75, "into_player_id": 5, "number": "11"}
        self.assertEqual(unbacked_links(reads, [link]), [link])

    def test_trailing_digit_reads_back_a_two_digit_number(self):
        from handball_cv.jersey.identity import unbacked_links
        reads = _reads(7, "17", 3) + _reads(7, "7", 3) + _reads(7, "4", 5) + _reads(3, "17", 20)
        link = {"frame": 10, "type": "link", "player_id": 7, "into_player_id": 3, "number": "17"}
        self.assertEqual(unbacked_links(reads, [link]), [])

    def test_only_the_folded_in_side_is_judged(self):
        # The keeper's family may hold a track the tracker switched between players.
        from handball_cv.jersey.identity import unbacked_links
        reads = _reads(46, "53", 300) + _reads(46, "11", 160, frame=400) + _reads(95, "11", 27)
        link = {"frame": 500, "type": "link", "player_id": 95, "into_player_id": 46, "number": "11"}
        self.assertEqual(unbacked_links(reads, [link]), [])

    def test_a_link_recorded_without_its_number_is_not_judged(self):
        from handball_cv.jersey.identity import unbacked_links
        reads = _reads(8, "4", 10)
        link = {"frame": 10, "type": "link", "player_id": 8, "into_player_id": 3}
        self.assertEqual(unbacked_links(reads, [link]), [])

    def test_a_dropped_link_unfolds_its_reads_and_aliases(self):
        from handball_cv.pipeline.tables import without_unbacked_links
        reads = (_reads(5, "11", 10) + _reads(75, "11", 3, frame=20)
                 + [{**r, "player_id": 5} for r in _reads(75, "22", 10, frame=40)])
        events = [{"frame": 30, "type": "link", "player_id": 75, "into_player_id": 5, "number": "11"},
                  {"frame": 31, "type": "reid", "player_id": 9}]
        run = {"number_reads": reads, "identity_events": events, "player_aliases": {"75": 5}}
        guarded = without_unbacked_links(run)
        self.assertEqual(guarded["player_aliases"], {})
        self.assertEqual([e["type"] for e in guarded["identity_events"]], ["unbacked_link", "reid"])
        self.assertEqual({r["player_id"] for r in guarded["number_reads"] if r["track_id"] == 75}, {75})
        self.assertIs(without_unbacked_links({**run, "number_reads": [{"frame": 0, "value": "1"}]})["number_reads"][0]["value"], "1")

    def test_a_backed_link_is_left_alone(self):
        from handball_cv.pipeline.tables import without_unbacked_links
        run = {"number_reads": _reads(5, "11", 10) + _reads(95, "11", 10, frame=20),
               "identity_events": [{"frame": 30, "type": "link", "player_id": 95, "into_player_id": 5,
                                    "number": "11"}],
               "player_aliases": {"95": 5}}
        self.assertIs(without_unbacked_links(run), run)

    def _misread_run(self):
        # #11 (track 5) leaves; track 75 settles "11" from three chest frames,
        # then reads "22"; #22 returns as track 9. All one team, never together.
        from handball_cv.jersey.identity import replay_number_links
        frames = {**{f: [5] for f in range(10)}, **{f: [75] for f in range(20, 100)},
                  **{f: [9] for f in range(100, 121)}}
        run = {"number_reads": _reads(5, "11", 10) + _reads(75, "11", 3, frame=20)
               + _reads(75, "22", 10, frame=40) + _reads(9, "22", 5, frame=100),
               "identity_events": [], "frame_players": {str(f): ids for f, ids in frames.items()},
               "player_teams": {"5": 0, "75": 0, "9": 0}, "ocr_every": 1}
        links, reads = replay_number_links(run)
        return {**run, "identity_events": links, "number_reads": reads,
                "player_aliases": {str(e["player_id"]): e["into_player_id"] for e in links}}

    def test_the_replay_makes_the_live_links(self):
        run = self._misread_run()
        self.assertEqual([(e["frame"], e["player_id"], e["into_player_id"], e["number"])
                          for e in run["identity_events"]], [(22, 75, 5, "11")])
        self.assertEqual({r["player_id"] for r in run["number_reads"] if r["track_id"] == 75 and r["frame"] > 22}, {5})

    def test_without_the_unbacked_link_the_misread_fragment_links_to_its_own_number(self):
        from handball_cv.pipeline.tables import with_backed_links
        relinked = with_backed_links(self._misread_run())
        self.assertEqual(relinked["player_aliases"], {"9": 75})
        self.assertEqual([(e["type"], e["player_id"], e["into_player_id"]) for e in relinked["identity_events"]],
                         [("unbacked_link", 75, 5), ("link", 9, 75)])
        self.assertEqual({r["player_id"] for r in relinked["number_reads"] if r["track_id"] == 75}, {75})

    def test_backed_links_are_not_replayed(self):
        from handball_cv.pipeline.tables import with_backed_links
        run = {"number_reads": _reads(5, "11", 10) + _reads(95, "11", 10, frame=20),
               "identity_events": [{"frame": 30, "type": "link", "player_id": 95, "into_player_id": 5,
                                    "number": "11"}],
               "player_aliases": {"95": 5}}
        self.assertIs(with_backed_links(run), run)

    def test_the_registry_records_the_linked_number(self):
        registry = PlayerRegistry()
        registry.link(7, 3, 10, "17")
        self.assertEqual(registry.events[-1]["number"], "17")


class ReadWeightTests(unittest.TestCase):
    def test_a_big_number_box_votes_twice(self):
        from handball_cv.jersey.identity import read_weight
        self.assertEqual(read_weight([0, 0, 10, 20]), 2)     # a back number
        self.assertEqual(read_weight([0, 0, 10, 12]), 1)     # a chest number
        self.assertEqual(read_weight(None), 1)

    def test_two_back_reads_outweigh_three_chest_reads(self):
        # Early chest reads said "11", later back reads "22".
        voter = NumberVoter()
        for _ in range(3):
            voter.observe(75, "11", 1)
        self.assertEqual(voter.best(75)[0], "11")
        for _ in range(5):
            voter.observe(75, "22", 2)
        self.assertEqual(voter.best(75)[0], "22")
