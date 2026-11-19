import unittest

import numpy as np

from policy_parameter_dataset import build_dataset
from policy_shift_action_baselines import (
    MODEL_NAMES, _calibrate, _paired_episode_bootstrap, previous_actions, run_seed,
)


class PolicyShiftActionBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset(seed=20261120, episodes_per_cell=60)

    def test_previous_action_is_episode_local_and_past_only(self):
        part = self.dataset["test_shifted"]
        previous = previous_actions(part)
        for episode_id in set(part["episode_id"].tolist()):
            indices = np.flatnonzero(part["episode_id"] == episode_id)
            indices = indices[np.argsort(part["step"][indices])]
            self.assertEqual(previous[indices[0]], -1)
            for before, current in zip(indices, indices[1:]):
                self.assertEqual(previous[current], int(np.argmax(part["action"][before])))

    def test_calibration_preserves_probability_simplex(self):
        p = np.asarray([[0.7, 0.3, 0.0], [0.1, 0.2, 0.7]])
        calibrated = _calibrate(p, 1.5)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(calibrated >= 0.0))

    def test_paired_bootstrap_uses_complete_episodes(self):
        model = {"a": {"nll": 0.1, "brier": 0.2}, "b": {"nll": 0.2, "brier": 0.3}}
        reference = {"a": {"nll": 0.2, "brier": 0.3}, "b": {"nll": 0.3, "brier": 0.4}}
        result = _paired_episode_bootstrap(model, reference, "brier", 123, 200)
        self.assertEqual(result["unit"], "whole episode")
        self.assertEqual(result["episodes"], 2)
        self.assertAlmostEqual(result["model_minus_historical_frequency_mean"], -0.1)
        self.assertTrue(result["ci95_percentile"][1] < 0.0)

    def test_small_run_has_both_shift_directions_and_all_models(self):
        result = run_seed(
            seed=20261120, episodes_per_cell=60, max_epochs=5,
            patience=2, bootstrap_replicates=100,
        )
        self.assertEqual(set(result["tests"]), {"test_shifted", "test_reverse"})
        for split in result["tests"].values():
            self.assertEqual(set(split["calibrated_metrics"]), set(MODEL_NAMES))
            self.assertEqual(set(split["rho_values"]), {0.85} if split["rho_values"] == [0.85] else {0.05})
            self.assertEqual(set(split["paired_episode_bootstrap_vs_historical_frequency"]), set(MODEL_NAMES) - {"historical_frequency"})


if __name__ == "__main__":
    unittest.main()
