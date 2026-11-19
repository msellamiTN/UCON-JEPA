import unittest
import numpy as np

from policy_shift_jepa_action import MODEL_NAMES, run_seed


class PolicyShiftJEPAActionTests(unittest.TestCase):
    def test_small_one_step_jeps_comparison(self):
        result = run_seed(
            seed=20261120,
            episodes_per_cell=50,
            jepa_epochs=2,
            baseline_epochs=3,
            baseline_patience=2,
            action_weight=0.5,
            bootstrap_replicates=100,
        )
        self.assertEqual(set(result["tests"]), {"test_shifted", "test_reverse"})
        for split_name, split in result["tests"].items():
            self.assertEqual(
                set(split["rho_values"]),
                {0.85} if split_name == "test_shifted" else {0.05},
            )
            self.assertEqual(set(split["calibrated_metrics"]), set(MODEL_NAMES))
            self.assertEqual(set(split["paired_episode_bootstrap"]), {
                "majority", "first_order_markov",
                "raw_context_diagnostic", "joint_action_state_jepa"
            })
            for model in MODEL_NAMES:
                for metric in ("nll", "top1_accuracy", "top2_accuracy",
                               "multiclass_brier", "top_label_ece_10bin"):
                    self.assertTrue(np.isfinite(split["calibrated_metrics"][model][metric]))
                    if metric in ("top1_accuracy", "top2_accuracy", "top_label_ece_10bin"):
                        self.assertGreaterEqual(split["calibrated_metrics"][model][metric], 0.0)
                        self.assertLessEqual(split["calibrated_metrics"][model][metric], 1.0)
                    if metric == "multiclass_brier":
                        self.assertGreaterEqual(split["calibrated_metrics"][model][metric], 0.0)


if __name__ == "__main__":
    unittest.main()
