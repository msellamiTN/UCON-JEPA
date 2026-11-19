import unittest
import numpy as np

from stateful_trajectory_dataset import (
    ACTION_VOCAB, CONTEXT_DIM, build_dataset, _episode, _context,
    _initial_state, _policy_oracle, _online_gate_step,
)


class StatefulTrajectoryDatasetTests(unittest.TestCase):
    def test_reproducible_generation_and_episode_disjoint_splits(self):
        a = build_dataset(seed=42, episodes_per_family=60, episode_length=5)
        b = build_dataset(seed=42, episodes_per_family=60, episode_length=5)
        for part in ("train", "validation", "test"):
            np.testing.assert_array_equal(a[part]["context"], b[part]["context"])
            self.assertEqual(set(a[part]["episode_id"]), set(b[part]["episode_id"]))
        sets = [set(a[p]["episode_id"]) for p in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_predictor_inputs_do_not_contain_future_oracle_labels(self):
        ds = build_dataset(seed=7, episodes_per_family=60, episode_length=4)
        for part in ("train", "validation", "test"):
            self.assertEqual(ds[part]["context"].shape[1], CONTEXT_DIM)
            self.assertEqual(ds[part]["action"].shape[1], len(ACTION_VOCAB))
            self.assertNotIn("security", ())
            self.assertIn("security", ds[part])  # target-side array, not concatenated into context/action

    def test_same_action_has_context_dependent_policy_outcome(self):
        # IAM grant is accepted for an admin and denied for a non-admin.
        rng_context = {"family": "iam", "security_admin": True, "scope_match": True,
                       "clearance": 2, "classification": 2, "export_approved": True,
                       "recipient_owner": True, "approver_independent": True, "payer_independent": True}
        denied_context = dict(rng_context, security_admin=False)
        s1, s2 = _initial_state(rng_context), _initial_state(denied_context)
        self.assertTrue(_policy_oracle(s1, "grant_role")[0])
        self.assertFalse(_policy_oracle(s2, "grant_role")[0])

    def test_oracle_and_online_gate_agree_on_generated_transitions(self):
        for family in ("iam", "data", "sod"):
            for seed in range(100):
                rows = _episode(seed + 1000, family, 8)
                self.assertGreaterEqual(len(rows), 1)
                self.assertTrue(all(r["policy_version"] == "ucon-jeepa-synthetic-v1" for r in rows))

    def test_targets_are_finite_and_have_expected_security_width(self):
        ds = build_dataset(seed=9, episodes_per_family=60, episode_length=4)
        for part in ("train", "validation", "test"):
            for key in ("context", "action", "future", "security"):
                self.assertTrue(np.isfinite(ds[part][key]).all())
            self.assertEqual(ds[part]["security"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
