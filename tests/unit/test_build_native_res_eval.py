import unittest

from scripts.build_native_res_eval import sample_frame_indices


class SampleFrameIndicesTests(unittest.TestCase):
    def test_includes_frame_zero_and_respects_stride(self):
        self.assertEqual(sample_frame_indices(20, 5), [0, 5, 10, 15])

    def test_never_reaches_or_exceeds_frame_count(self):
        indices = sample_frame_indices(249, 5)
        self.assertTrue(all(i < 249 for i in indices))
        self.assertEqual(indices[-1], 245)

    def test_stride_one_samples_every_frame(self):
        self.assertEqual(sample_frame_indices(4, 1), [0, 1, 2, 3])

    def test_stride_larger_than_frame_count_yields_just_frame_zero(self):
        self.assertEqual(sample_frame_indices(3, 100), [0])


if __name__ == "__main__":
    unittest.main()
