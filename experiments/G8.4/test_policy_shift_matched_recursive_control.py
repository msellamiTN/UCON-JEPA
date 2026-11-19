import inspect
import unittest

import numpy as np

from jepa_numpy import JEPAConfig, LinearActionConditionedJEPA
from policy_shift_supervised_state_dynamics import LinearSupervisedStateDynamics
from policy_shift_matched_recursive_control import _run_one_seed, _predict_e2


class MatchedRecursiveControlTests(unittest.TestCase):
    def test_supervised_state_dynamics_is_capacity_near_matched(self):
        jepa = LinearActionConditionedJEPA(
            34, 9, 34, 4, JEPAConfig(latent_dim=12, epochs=2, batch_size=8, seed=19)
        )
        supervised = LinearSupervisedStateDynamics(
            34, 9, 4, JEPAConfig(latent_dim=8, epochs=2, batch_size=8, seed=19)
        )
        ratio = supervised.parameter_count() / jepa.parameter_count()
        self.assertGreater(ratio, 0.90)
        self.assertLess(ratio, 1.0)
        self.assertEqual(supervised.W_future_decoder.shape, (8, 34))

    def test_supervised_fit_is_deterministic_and_predicts_next_context(self):
        rng = np.random.default_rng(4)
        context = rng.normal(size=(64, 7))
        action = np.eye(3)[rng.integers(0, 3, size=64)]
        future = 0.7 * context + 0.2 * rng.normal(size=(64, 7))
        security = rng.integers(0, 2, size=(64, 4)).astype(float)
        config = JEPAConfig(latent_dim=4, epochs=3, batch_size=16, seed=23)
        a = LinearSupervisedStateDynamics(7, 3, 4, config).fit(
            context, action, future, security
        )
        b = LinearSupervisedStateDynamics(7, 3, 4, config).fit(
            context, action, future, security
        )
        np.testing.assert_allclose(a.W_predictor, b.W_predictor)
        np.testing.assert_allclose(a.W_future_decoder, b.W_future_decoder)
        self.assertTrue(np.isfinite(a.history[-1]["future_mse"]))
        latent = a.predict_from_latents(
            context[:5] @ a.W_context, action[:5] @ a.W_action
        )
        self.assertEqual(latent.shape, (5, 4))
        self.assertTrue(np.isfinite(a.decode_security(latent)).all())
        predicted_future = a.predict_future_context(context[:5], action[:5])
        self.assertEqual(predicted_future.shape, (5, 7))
        self.assertTrue(np.isfinite(predicted_future).all())

    def test_e2_inference_helper_has_no_future_target_arguments(self):
        params = set(inspect.signature(_predict_e2).parameters)
        self.assertNotIn("future_observation", params)
        self.assertNotIn("security_targets", params)
        self.assertNotIn("realized_future_actions", params)

    def test_small_matched_recursive_benchmark(self):
        result = _run_one_seed(
            seed=20261120,
            episodes_per_cell=50,
            epochs=2,
            mc_samples=4,
            bootstrap_replicates=100,
            out_dir=None,
        )
        self.assertEqual(set(result["tests"]), {"test_shifted", "test_reverse"})
        counts = result["models"]
        ratio = (
            counts["supervised_E2"]["trainable_parameter_count"]
            / counts["JEPA_E2"]["trainable_parameter_count"]
        )
        self.assertGreater(ratio, 0.90)
        self.assertLess(ratio, 1.0)
        for split, horizons in result["tests"].items():
            self.assertEqual(set(horizons), {"1", "2", "4", "8"})
            for report in horizons.values():
                self.assertGreater(report["windows"], 0)
                self.assertEqual(
                    set(report["metrics"]),
                    {"train_prevalence", "JEPA_E2", "supervised_E2"},
                )
                for method in report["metrics"].values():
                    for key in ("auroc", "auprc", "brier", "ece_10"):
                        self.assertTrue(np.isfinite(method[key]))
                boot = report["paired_episode_bootstrap_JEPA_minus_supervised_Brier"]
                self.assertEqual(boot["unit"], "whole episode")
                self.assertEqual(boot["replicates"], 100)
                self.assertEqual(len(boot["ci95_percentile"]), 2)


if __name__ == "__main__":
    unittest.main()
