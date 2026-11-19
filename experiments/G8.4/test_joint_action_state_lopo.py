import unittest

from uncertain_action_dataset import AGENT_POLICIES, build_dataset, build_lopo_fold
from joint_action_state_lopo import run_seed


class JointActionStateLOPOTests(unittest.TestCase):
    def test_every_lopo_fold_excludes_held_out_policy(self):
        dataset = build_dataset(seed=20261120, episodes_per_family_policy=60, episode_length=8)
        for policy in AGENT_POLICIES:
            fold = build_lopo_fold(dataset, policy)
            for part in ("train", "validation", "calibration"):
                self.assertNotIn(policy, set(fold[part]["agent_policy"].tolist()))
            self.assertEqual(set(fold["test"]["agent_policy"].tolist()), {policy})

    def test_small_lopo_execution_preserves_fold_identity(self):
        result = run_seed(
            seed=20261120, episodes_per_family_policy=30, epochs=2,
            action_weight=0.5, mc_samples=4, bootstrap_replicates=100,
        )
        self.assertEqual(set(result["lopo"]), set(AGENT_POLICIES))
        for policy, fold in result["lopo"].items():
            self.assertEqual(fold["test_policy"], policy)
            self.assertNotIn(policy, fold["train_policies"])
            self.assertNotIn(policy, fold["validation_policies"])
            self.assertNotIn(policy, fold["calibration_policies"])
            self.assertEqual(set(fold["horizons"]), {"1", "2", "4", "8"})


if __name__ == "__main__":
    unittest.main()
