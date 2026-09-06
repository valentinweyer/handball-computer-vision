import unittest

from scripts.evaluate_identity_persistence import summarize


def _samples(points):
    return [{"frame": f, "allocated": a, "alive": v} for f, a, v in points]


class SummarizeTests(unittest.TestCase):
    def test_stable_tracking_projects_no_growth(self):
        # Cast established in the first third, then flat: the steady-state rate is what
        # repeats over a match, so the 60-minute projection stays at the alive ceiling.
        samples = _samples([(0, 4, 4), (250, 12, 12), (500, 12, 12), (750, 12, 12), (1000, 12, 12)])

        summary = summarize(samples, fps=50.0)

        self.assertEqual(summary["total_ids_allocated"], 12)
        self.assertEqual(summary["peak_ids_alive_at_once"], 12)
        self.assertEqual(summary["ids_allocated_per_minute_steady_state"], 0.0)
        self.assertEqual(summary["projected_ids_at_60_min"], 12)

    def test_linear_drift_projects_growth(self):
        # 10 new ids per 500 frames @50fps = 10 per 10s = 60/min, sustained.
        samples = _samples([(0, 5, 5), (500, 15, 5), (1000, 25, 5), (1500, 35, 5)])

        summary = summarize(samples, fps=50.0)

        self.assertEqual(summary["ids_allocated_per_minute_steady_state"], 60.0)
        self.assertGreater(summary["projected_ids_at_60_min"], 1000)

    def test_early_burst_is_excluded_from_steady_state(self):
        # A burst confined to the first third is roster appearing, not drift, so it must
        # not inflate the projection.
        samples = _samples([(0, 1, 1), (100, 14, 14), (600, 14, 14), (1200, 14, 14)])

        summary = summarize(samples, fps=50.0)

        self.assertEqual(summary["ids_allocated_per_minute_steady_state"], 0.0)
        self.assertGreater(summary["ids_allocated_per_minute_overall"], 0.0)

    def test_empty_samples_returns_empty_summary(self):
        self.assertEqual(summarize([], fps=50.0), {})


if __name__ == "__main__":
    unittest.main()
