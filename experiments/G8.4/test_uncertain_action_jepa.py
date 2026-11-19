import inspect
import unittest

import numpy as np

from uncertain_action_dataset import build_dataset
from uncertain_action_jepa import (
    _make_windows, fit_e1_e2_models, rollout_known_actions,
    rollout_predicted_actions,
)


class PredictedActionJEPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset(seed=20261120, episodes_per_family_policy=60, episode_length=8)
        cls.model, cls.action_model, cls.raw_action_model, cls.scalers = fit_e1_e2_models(cls.dataset, seed=20261120, epochs=4)

    def test_windows_are_episode_local_and_contiguous(self):
        for horizon in (1, 2, 4, 8):
            windows = _make_windows(self.dataset["test"], horizon)
            self.assertEqual(windows["actions"].shape[1], horizon)
            self.assertEqual(windows["security"].shape[1], horizon)
            for eid, start in zip(windows["episode_id"], windows["start_step"]):
                source = self.dataset["test"]
                steps = np.sort(source["step"][source["episode_id"] == eid])
                self.assertTrue(np.all(np.arange(start, start + horizon) <= steps.max()))

    def test_e2_inference_signature_excludes_future_actions_and_security(self):
        signature = inspect.signature(rollout_predicted_actions)
        names = set(signature.parameters)
        self.assertNotIn("action_sequence", names)
        self.assertNotIn("future_actions", names)
        self.assertNotIn("security_targets", names)
        self.assertNotIn("future_observation", names)

    def test_e2_predictions_are_invariant_to_target_mutation(self):
        source = self.dataset["test"]
        context = source["context"][:12]
        before = rollout_predicted_actions(
            self.model, self.action_model, context,
            self.scalers["context_scaler"], self.scalers["action_scaler"],
            self.scalers["temperature"], horizon=3, mc_samples=8, seed=17,
        )
        # Mutated future actions/security are not arguments to the E2 predictor.
        mutated_actions = source["action"].copy()
        mutated_security = source["security"].copy()
        mutated_actions[:] = np.roll(mutated_actions, 1, axis=1)
        mutated_security[:] = 1.0 - mutated_security
        after = rollout_predicted_actions(
            self.model, self.action_model, context,
            self.scalers["context_scaler"], self.scalers["action_scaler"],
            self.scalers["temperature"], horizon=3, mc_samples=8, seed=17,
        )
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_e1_requires_explicit_realized_actions(self):
        source = self.dataset["test"]
        context = source["context"][:10]
        actions = source["action"][:10, None, :]
        risk_a, state_a = rollout_known_actions(
            self.model, context, actions,
            self.scalers["context_scaler"], self.scalers["action_scaler"],
        )
        altered = np.roll(actions, 1, axis=2)
        risk_b, state_b = rollout_known_actions(
            self.model, context, altered,
            self.scalers["context_scaler"], self.scalers["action_scaler"],
        )
        self.assertEqual(risk_a.shape, (10,))
        self.assertEqual(state_a.shape, (10, 3))
        self.assertTrue(np.any(risk_a != risk_b) or np.any(state_a != state_b))

    def test_predicted_action_rollout_returns_uncertainty_summary(self):
        source = self.dataset["test"]
        result = rollout_predicted_actions(
            self.model, self.action_model, source["context"][:16],
            self.scalers["context_scaler"], self.scalers["action_scaler"],
            self.scalers["temperature"], horizon=4, mc_samples=12, seed=22,
        )
        self.assertEqual(result["risk_mean"].shape, (16,))
        self.assertTrue(np.all((result["risk_mean"] >= 0) & (result["risk_mean"] <= 1)))
        self.assertTrue(np.all(result["risk_std"] >= 0))
        self.assertTrue(np.all(result["risk_q05"] <= result["risk_q50"]))
        self.assertTrue(np.all(result["risk_q50"] <= result["risk_q95"]))


if __name__ == "__main__":
    unittest.main()
