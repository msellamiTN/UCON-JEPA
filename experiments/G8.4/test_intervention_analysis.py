import unittest

import numpy as np

from intervention_analysis import (
    _bootstrap_max_t,
    _contrast_estimates,
    _randomization_max_t,
)


class InterventionAnalysisTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(77)
        self.values = np.empty((6, 4, 12, 2), dtype=np.float64)
        for block in range(6):
            for arm in range(4):
                self.values[block, arm, :, 0] = rng.poisson(
                    1.4 + 0.15 * arm, size=12
                )
                self.values[block, arm, :, 1] = rng.binomial(
                    1, 0.82 - 0.015 * arm, size=12
                )

    def test_block_standardized_contrasts_and_standard_errors(self):
        estimates, standard_errors = _contrast_estimates(self.values)
        self.assertEqual(estimates.shape, (6,))
        self.assertEqual(standard_errors.shape, (6,))
        self.assertTrue(np.all(np.isfinite(estimates)))
        self.assertTrue(np.all(standard_errors > 0))

    def test_episode_bootstrap_returns_finite_simultaneous_critical(self):
        estimates, _ = _contrast_estimates(self.values)
        critical, maxima = _bootstrap_max_t(
            self.values, estimates, replicates=200, seed=91, chunk_size=20
        )
        self.assertTrue(np.isfinite(critical))
        self.assertEqual(len(maxima), 200)
        self.assertGreaterEqual(critical, float(np.min(maxima)))

    def test_blocked_randomization_max_t_returns_adjusted_probabilities(self):
        estimates, standard_errors = _contrast_estimates(self.values)
        adjusted, global_p = _randomization_max_t(
            self.values, estimates, standard_errors,
            replicates=200, seed=92, chunk_size=20,
        )
        self.assertEqual(len(adjusted), 6)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted.values()))
        self.assertTrue(0.0 <= global_p <= 1.0)


if __name__ == "__main__":
    unittest.main()
