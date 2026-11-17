import unittest

import numpy as np

from trajectory_risk_control import (
    bonferroni_cp,
    cp_upper,
    fixed_sequence_cp,
    naive_empirical,
)


class RiskControlTests(unittest.TestCase):
    def test_cp_upper_zero_errors_is_positive(self):
        upper = cp_upper(0, 100, 0.05)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.05)

    def test_cp_upper_empty_sample_is_one(self):
        self.assertEqual(cp_upper(0, 0, 0.05), 1.0)

    def test_cp_upper_all_errors_is_one(self):
        self.assertEqual(cp_upper(10, 10, 0.05), 1.0)

    def test_fixed_sequence_stops_at_first_failure(self):
        scores = np.array([0.1] * 100 + [0.9] * 100)
        outcomes = np.array([0] * 100 + [1] * 100)
        thresholds = np.array([0.2, 0.5, 1.0])
        decision = fixed_sequence_cp(scores, outcomes, thresholds, 0.10, 0.05)
        self.assertEqual(decision.threshold, 0.5)

    def test_naive_can_accept_without_finite_sample_certificate(self):
        scores = np.linspace(0.0, 1.0, 100)
        outcomes = np.zeros(100, dtype=int)
        thresholds = np.array([1.0])
        naive = naive_empirical(scores, outcomes, thresholds, 0.01)
        certified = fixed_sequence_cp(scores, outcomes, thresholds, 0.01, 0.05)
        self.assertEqual(naive.threshold, 1.0)
        self.assertIsNone(certified.threshold)

    def test_bonferroni_uses_stricter_bound(self):
        scores = np.linspace(0.0, 1.0, 1000)
        outcomes = np.zeros(1000, dtype=int)
        thresholds = np.array([0.5, 1.0])
        fixed = fixed_sequence_cp(scores, outcomes, thresholds, 0.05, 0.05)
        bonf = bonferroni_cp(scores, outcomes, thresholds, 0.05, 0.05)
        self.assertTrue(
            bonf.threshold is None
            or fixed.threshold is None
            or bonf.threshold <= fixed.threshold
        )


if __name__ == "__main__":
    unittest.main()
