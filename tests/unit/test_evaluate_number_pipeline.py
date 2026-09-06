import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluate_number_pipeline import recompute_votes, score


def _report(players: dict) -> dict:
    return {"schema_version": 1, "reader": "qwen", "players": players}


class ScoreTests(unittest.TestCase):
    def _score(self, players, truth):
        with TemporaryDirectory() as raw_dir:
            p = Path(raw_dir) / "truth.json"
            p.write_text(json.dumps(truth))
            return score(_report(players), p)

    def test_counts_correct_wrong_and_no_verdict(self):
        players = {
            "1": {"voted_number": "16", "votes": 9, "margin": 1.0, "n_reads": 9,
                  "read_distribution": {"16": 9}},
            "2": {"voted_number": "8", "votes": 6, "margin": 1.0, "n_reads": 6,
                  "read_distribution": {"8": 6}},
            "3": {"voted_number": None, "votes": 1, "margin": 0.0, "n_reads": 1,
                  "read_distribution": {"5": 1}},
        }
        result = self._score(players, {"1": "16", "2": "18", "3": "7"})

        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["wrong"], 1)
        self.assertEqual(result["no_verdict"], 1)
        self.assertEqual(result["scored"], 3)

    def test_players_missing_from_truth_are_skipped_not_counted_wrong(self):
        # A tracked id the human didn't label must not be scored as an error --
        # tracking produces fragments and bench/partial identities that carry no
        # ground truth, and counting them as wrong would understate accuracy.
        players = {
            "1": {"voted_number": "7", "votes": 5, "margin": 0.8, "n_reads": 5,
                  "read_distribution": {"7": 5}},
            "99": {"voted_number": "3", "votes": 4, "margin": 0.5, "n_reads": 4,
                   "read_distribution": {"3": 4}},
        }
        result = self._score(players, {"1": "7"})

        self.assertEqual(result["scored"], 1)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["wrong"], 0)

    def test_numeric_and_string_truth_compare_equal(self):
        # truth written as {"1": 16} rather than {"1": "16"} must still match
        players = {"1": {"voted_number": "16", "votes": 5, "margin": 1.0,
                         "n_reads": 5, "read_distribution": {"16": 5}}}
        result = self._score(players, {"1": 16})

        self.assertEqual(result["correct"], 1)

    def test_empty_truth_scores_nothing(self):
        players = {"1": {"voted_number": "7", "votes": 5, "margin": 1.0,
                         "n_reads": 5, "read_distribution": {"7": 5}}}
        result = self._score(players, {})

        self.assertEqual(result["scored"], 0)


class RecomputeVotesTests(unittest.TestCase):
    def test_rebuilds_verdict_from_stored_read_distribution(self):
        # Simulates a report written before suffix-aware voting existed: stale
        # voted_number/votes/margin from the old algorithm, but the read_distribution
        # (the reader's raw output) is unaffected by the voting-code change.
        report = _report({
            "3": {
                "voted_number": None, "votes": 4, "margin": 0.0, "n_reads": 8,
                "read_distribution": {"17": 4, "7": 4},
            },
        })

        recompute_votes(report, min_votes=3, min_margin=0.2, min_promote_votes=2)

        self.assertEqual(report["players"]["3"]["voted_number"], "17")

    def test_below_threshold_after_recompute_is_no_verdict(self):
        report = _report({
            "9": {
                "voted_number": "5", "votes": 1, "margin": 1.0, "n_reads": 1,
                "read_distribution": {"5": 1},
            },
        })

        recompute_votes(report, min_votes=3, min_margin=0.2, min_promote_votes=2)

        self.assertIsNone(report["players"]["9"]["voted_number"])

    def test_respects_passed_thresholds_not_hardcoded_defaults(self):
        report = _report({
            "1": {
                "voted_number": None, "votes": 1, "margin": 1.0, "n_reads": 1,
                "read_distribution": {"5": 1},
            },
        })

        recompute_votes(report, min_votes=1, min_margin=0.0, min_promote_votes=2)

        self.assertEqual(report["players"]["1"]["voted_number"], "5")


if __name__ == "__main__":
    unittest.main()
