import unittest

import numpy as np

from intervention_dtmc import DTMCRiskModel, binary_risk_metrics


class DTMCRiskModelTests(unittest.TestCase):
    @staticmethod
    def _partition():
        # Context A transitions to DENY in both episodes; context B reaches SAFE_END.
        a = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        b = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return {
            "context": np.stack([a, b, a, b]),
            "security": np.asarray([
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]),
            "episode_id": np.asarray(["e1", "e1", "e2", "e2"]),
            "step": np.asarray([0, 1, 0, 1]),
            "family": np.asarray(["iam", "iam", "iam", "iam"]),
        }

    def test_dtmc_distinguishes_deny_and_safe_states_and_horizon_is_monotone(self):
        model = DTMCRiskModel(alpha=1.0).fit(self._partition())
        a = np.asarray([1.0, 0.0, 0.0, 1.0])
        b = np.asarray([1.0, 0.0, 0.0, 0.0])
        risk_a_h1 = model.predict_one(a, "iam", horizon=1)
        risk_b_h1 = model.predict_one(b, "iam", horizon=1)
        risk_a_h2 = model.predict_one(a, "iam", horizon=2)
        self.assertGreater(risk_a_h1, risk_b_h1)
        self.assertGreaterEqual(risk_a_h2, risk_a_h1)
        self.assertLessEqual(risk_a_h2, 1.0)

    def test_unseen_state_uses_family_level_backoff(self):
        model = DTMCRiskModel(alpha=1.0).fit(self._partition())
        unseen = np.asarray([1.0, 0.0, 0.0, 0.5])
        score = model.predict_one(unseen, "iam", horizon=1)
        self.assertAlmostEqual(score, model.family_deny_rate["iam"])

    def test_serialization_round_trip_preserves_scores(self):
        model = DTMCRiskModel(alpha=0.5).fit(self._partition())
        restored = DTMCRiskModel.from_dict(model.to_dict())
        context = np.asarray([1.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(
            model.predict_one(context, "iam", horizon=3),
            restored.predict_one(context, "iam", horizon=3),
        )
        self.assertEqual(model.metadata(), restored.metadata())

    def test_predictive_metrics_are_well_defined(self):
        metrics = binary_risk_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.2, 0.8, 0.9]),
        )
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertLess(metrics["brier"], 0.05)
        self.assertIsNotNone(metrics["ece_10"])


if __name__ == "__main__":
    unittest.main()
