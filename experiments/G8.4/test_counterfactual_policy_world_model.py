import unittest

import numpy as np

from counterfactual_policy_world_model import (
    empirical_residual_identity,
    keyed_exogenous_seed,
    residual_transport,
)


class ResidualTransportTests(unittest.TestCase):
    def test_no_action_change_reproduces_factual_outcome(self):
        observed = np.array([[1.2, -3.4], [0.0, 9.1]])
        mean = np.array([[0.2, 0.4], [2.0, -1.0]])
        np.testing.assert_allclose(residual_transport(observed, mean, mean), observed)

    def test_additive_shared_disturbance_is_exact(self):
        disturbance = np.array([
            [-2.0, 0.5], [0.0, 1.0], [1.5, -0.5], [3.0, 2.0],
        ])
        factual_action_effect = np.array([1.0, -4.0])
        alternative_action_effect = np.array([-2.5, 3.0])
        factual = factual_action_effect + disturbance
        alternative = alternative_action_effect + disturbance
        prediction = residual_transport(
            factual,
            factual.mean(axis=0),
            np.broadcast_to(alternative.mean(axis=0), factual.shape),
        )
        np.testing.assert_allclose(prediction, alternative, atol=1e-12, rtol=0)

    def test_empirical_residual_error_equals_action_effect_variance(self):
        factual = np.array([
            [0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [4.0, -1.0], [-1.0, 3.0],
        ])
        alternative = np.array([
            [1.0, 2.0], [3.0, -1.0], [2.5, 4.0], [1.0, 0.0], [-2.0, 1.0],
        ])
        result = empirical_residual_identity(factual, alternative)
        self.assertLess(result["absolute_identity_error"], 1e-12)
        self.assertAlmostEqual(
            result["residual_transport_mse"],
            result["action_effect_variance"],
            places=12,
        )

    def test_multiplicative_disturbance_is_not_assumed_exact(self):
        disturbance = np.array([[-2.0], [-0.5], [0.5], [2.0]])
        factual = 1.0 * disturbance
        alternative = 2.5 * disturbance
        result = empirical_residual_identity(factual, alternative)
        self.assertGreater(result["action_effect_variance"], 0.0)
        self.assertGreater(result["residual_transport_mse"], 0.0)

    def test_alternative_action_bank_shape(self):
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        factual_mean = np.array([[0.5, 0.5], [1.0, 1.0]])
        alternatives = np.stack([factual_mean + 1.0, factual_mean - 2.0], axis=1)
        result = residual_transport(observed, factual_mean, alternatives)
        self.assertEqual(result.shape, (2, 2, 2))
        np.testing.assert_allclose(result[:, 0], observed + 1.0)
        np.testing.assert_allclose(result[:, 1], observed - 2.0)

    def test_rejects_categorical_verdicts(self):
        with self.assertRaisesRegex(TypeError, "continuous readout"):
            residual_transport(
                np.array(["PERMIT", "DENY"]),
                np.array(["PERMIT", "PERMIT"]),
                np.array(["DENY", "PERMIT"]),
            )

    def test_rejects_shape_mismatch_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            residual_transport(np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "finite"):
            residual_transport(np.array([np.nan]), np.array([0.0]), np.array([1.0]))

    def test_keyed_exogenous_seed_is_order_independent_and_branch_shared(self):
        first = keyed_exogenous_seed(7, "episode-001", "environment-disturbance", 3, 2)
        second = keyed_exogenous_seed(7, "episode-001", "environment-disturbance", 3, 2)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            keyed_exogenous_seed(7, "episode-001", "agent-replan", 3, 2),
        )
        self.assertNotEqual(
            first,
            keyed_exogenous_seed(7, "episode-001", "environment-disturbance", 4, 2),
        )

    def test_rejects_invalid_exogenous_seed_keys(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            keyed_exogenous_seed(1, "", "env", 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            keyed_exogenous_seed(1, "episode", "env", -1)


if __name__ == "__main__":
    unittest.main()
