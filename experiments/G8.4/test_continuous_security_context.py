import unittest

import numpy as np

from continuous_security_context import (
    ACTIONS,
    exogenous_disturbance,
    initial_context,
    paired_action_branches,
    policy_verdict,
    transition,
)


class ContinuousSecurityContextTests(unittest.TestCase):
    def test_initial_context_is_deterministic_and_finite(self):
        a = initial_context(19)
        b = initial_context(19)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())

    def test_exogenous_stream_is_keyed_and_reproducible(self):
        a = exogenous_disturbance(9, "parent-01", 2)
        b = exogenous_disturbance(9, "parent-01", 2)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, exogenous_disturbance(9, "parent-01", 3)))

    def test_additive_action_and_disturbance_channels(self):
        x = np.array([0.2, 0.3, 0.4])
        eps = np.array([0.5, -1.0, 0.25])
        h = 0.2
        action = "export_sensitive"
        observed = transition(x, action, eps, step_size=h)
        expected = x + h * (
            np.array([0.82, 0.30, 0.58]) + np.array([0.18, 0.22, 0.15]) * eps
        )
        np.testing.assert_allclose(observed, expected)

    def test_paired_branches_share_identical_environment_disturbance(self):
        state = np.array([0.2, 0.3, 0.4])
        branches = paired_action_branches(
            state, "data", root_seed=3, parent_episode_id="p-1", decision_step=4
        )
        disturbances = [v["exogenous_disturbance"] for v in branches.values()]
        for value in disturbances[1:]:
            np.testing.assert_array_equal(value, disturbances[0])

    def test_same_branch_replays_bitwise(self):
        kwargs = dict(
            state=np.array([0.2, 0.3, 0.4]), family="data",
            root_seed=5, parent_episode_id="p-2", decision_step=1,
        )
        first = paired_action_branches(**kwargs)
        second = paired_action_branches(**kwargs)
        for action in ACTIONS["data"]:
            np.testing.assert_array_equal(
                first[action]["next_context"], second[action]["next_context"]
            )
            self.assertEqual(first[action]["ucon_verdict"], second[action]["ucon_verdict"])

    def test_policy_boundary_is_recomputed_per_branch(self):
        x = np.array([1.20, 0.2, 0.2])
        self.assertEqual(policy_verdict("data", "data_list", x), "PERMIT")
        self.assertEqual(policy_verdict("data", "export_sensitive", x), "DENY")
        self.assertEqual(policy_verdict("data", "export_sensitive", x, base_policy_allowed=False), "DENY")

    def test_rejects_cross_family_action_and_invalid_state(self):
        with self.assertRaisesRegex(ValueError, "supported by the task family"):
            paired_action_branches(
                np.zeros(3), "data", ("pay",),
                root_seed=1, parent_episode_id="p", decision_step=0,
            )
        with self.assertRaisesRegex(ValueError, "finite vector"):
            transition([0.0, np.nan, 0.0], "read_sensitive", [0.0, 0.0, 0.0])

    def test_rejects_invalid_interaction_coefficients(self):
        with self.assertRaisesRegex(ValueError, "coefficients must be finite"):
            transition(
                np.zeros(3), "read_sensitive", np.zeros(3),
                interaction=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
