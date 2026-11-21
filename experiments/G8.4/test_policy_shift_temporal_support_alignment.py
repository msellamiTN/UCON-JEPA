import unittest

import numpy as np

from policy_shift_temporal_support_alignment import (
    _apply_support_calibrators,
    _fit_support_calibrators,
    _safe_metrics,
    _stratified_episode_bootstrap,
)


class TemporalSupportAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.repeat(np.arange(4), 20)
        self.truth = np.tile(np.asarray([0, 1] * 10), 4)
        self.raw = np.where(self.truth == 1, 0.75, 0.25)
        self.episodes = np.asarray([f"ep-{i:02d}" for i in range(80)])
        self.policies = np.asarray(
            ["goal_directed"] * 40 + ["uniform"] * 40
        )
        self.families = np.asarray(
            ["iam"] * 20 + ["data"] * 20 + ["iam"] * 20 + ["data"] * 20
        )
        self.windows = {
            "episode_id": self.episodes,
            "agent_policy": self.policies,
            "family": self.families,
        }

    def test_C0_CP_CA_use_the_prespecified_supports(self):
        fitted = _fit_support_calibrators(
            self.raw, self.truth, self.positions, self.episodes
        )
        self.assertEqual(fitted["C0"]["scope"], "position_0")
        self.assertEqual(fitted["C0"]["calibration_windows"], 20)
        self.assertEqual(fitted["CA"]["calibration_windows"], 80)
        self.assertEqual(
            set(fitted["CP"]["params_by_position"]), {"0", "1", "2", "3"}
        )
        self.assertTrue(all(
            value["calibration_windows"] == 20
            for value in fitted["CP"]["params_by_position"].values()
        ))

    def test_CP_applies_only_to_matching_positions(self):
        fitted = _fit_support_calibrators(
            self.raw, self.truth, self.positions
        )
        predictions = _apply_support_calibrators(
            self.raw, self.positions, fitted
        )
        self.assertEqual(set(predictions), {"C0", "CP", "CA"})
        for position in range(4):
            mask = self.positions == position
            expected = 1.0 / (
                1.0 + np.exp(-(
                    fitted["CP"]["params_by_position"][str(position)]["intercept"]
                    + fitted["CP"]["params_by_position"][str(position)]["slope"]
                    * (
                        np.log(self.raw[mask] / (1.0 - self.raw[mask]))
                        - fitted["CP"]["params_by_position"][str(position)]["center"]
                    )
                    / fitted["CP"]["params_by_position"][str(position)]["scale"]
                ))
            )
            np.testing.assert_allclose(predictions["CP"][mask], expected)

    def test_missing_class_is_marked_unavailable_not_auc_half(self):
        result = _safe_metrics(np.ones(5, dtype=int), np.linspace(0.1, 0.9, 5))
        self.assertFalse(result["available"])
        self.assertIn("only one class", result["reason"])

    def test_bootstrap_clusters_by_episode_and_preserves_strata(self):
        loss_a = np.linspace(0.0, 0.8, 80)
        loss_b = np.linspace(0.1, 0.9, 80)
        result = _stratified_episode_bootstrap(
            loss_a, loss_b, self.windows, seed=77, replicates=200
        )
        self.assertEqual(result["unit"],
            "whole episode, stratified by behavior archetype/resource family")
        self.assertEqual(result["episodes"], 80)
        self.assertEqual(result["strata"], {
            "goal_directed/iam": 20,
            "goal_directed/data": 20,
            "uniform/iam": 20,
            "uniform/data": 20,
        })
        self.assertEqual(result["replicates"], 200)

    def test_bootstrap_rejects_empty_strata(self):
        windows = {
            "episode_id": np.asarray(["a", "b"]),
            "agent_policy": np.asarray(["goal_directed", "goal_directed"]),
            "family": np.asarray(["iam", "iam"]),
        }
        result = _stratified_episode_bootstrap(
            np.asarray([0.1, 0.2]), np.asarray([0.2, 0.3]),
            windows, seed=1, replicates=20,
        )
        self.assertEqual(result["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
