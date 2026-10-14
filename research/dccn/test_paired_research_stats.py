"""Small deterministic integrity tests; no experiment data are generated."""
import unittest

import numpy as np

from paired_research_stats import summarize_paired_slots, summarize_seed_deltas


class PairedStatsTests(unittest.TestCase):
    def test_exact_identity(self):
        result = summarize_paired_slots([1, 2, 0, 5], [1, 2, 0, 5], 320, 200)
        self.assertEqual(result['candidate_minus_reference_ber'], 0)
        self.assertEqual(result['conditional_slot_bootstrap_95pct_ci'], [0, 0])

    def test_constant_better(self):
        result = summarize_paired_slots([0, 1, 2], [1, 2, 3], 320, 200)
        self.assertAlmostEqual(result['candidate_minus_reference_ber'], -1 / 320)
        np.testing.assert_allclose(result['conditional_slot_bootstrap_95pct_ci'], [-1 / 320] * 2)

    def test_deterministic_and_paired(self):
        args = ([0, 5, 0, 5], [4, 0, 4, 0], 640)
        self.assertEqual(summarize_paired_slots(*args, bootstrap_reps=400),
                         summarize_paired_slots(*args, bootstrap_reps=400))
        forward = summarize_paired_slots(*args, bootstrap_reps=0)
        reverse = summarize_paired_slots(args[1], args[0], args[2], bootstrap_reps=0)
        self.assertAlmostEqual(forward['candidate_minus_reference_ber'],
                               -reverse['candidate_minus_reference_ber'])
        self.assertIsNone(forward['conditional_slot_bootstrap_95pct_ci'])

    def test_reject_invalid(self):
        for a, b, bits in [([0], [0], 320), ([0, 1], [0], 320),
                           ([0, 321], [0, 0], 320), ([0, -1], [0, 0], 320),
                           ([0, 0.5], [0, 0], 320), ([0, float('nan')], [0, 0], 320),
                           ([0, 0], [0, 0], 0)]:
            with self.assertRaises(ValueError):
                summarize_paired_slots(a, b, bits)

    def test_seed_summary(self):
        result = summarize_seed_deltas([1, 2, 3])
        self.assertEqual(result['mean_delta'], 2)
        self.assertEqual(result['sample_sd'], 1)
        with self.assertRaises(ValueError):
            summarize_seed_deltas([0, float('nan')])


if __name__ == '__main__':
    unittest.main()
