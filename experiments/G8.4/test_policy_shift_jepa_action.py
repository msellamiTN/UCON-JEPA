import unittest

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
                "raw_context_diagnostic", "posthoc_jepa_latent", "joint_action_state_jepa"
            })
            for model in MODEL_NAMES:
                for metric in ("nll", "top1_accuracy", "top2_accuracy",
                               "multiclass_brier", "top_label_ece_10bin"):
                    self.assertTrue(
                        0.0 <= split["calibrated_metrics"][model][metric]
                        or metric in ("nll",)
                    )


if __name__ == "__main__":
    unittest.main()
