import unittest

from uncertain_action_dataset import build_dataset
from action_policy_prediction import run_experiment, _paired_episode_bootstrap
from action_policy_multiseed import summarize_runs


class MultiSeedActionValidationTests(unittest.TestCase):
    def test_episode_bootstrap_is_paired_and_deterministic(self):
        ds = build_dataset(seed=20261120, episodes_per_family_policy=60, episode_length=8)
        result_a = run_experiment(ds, bootstrap_replicates=100, bootstrap_seed=77)
        result_b = run_experiment(ds, bootstrap_replicates=100, bootstrap_seed=77)
        boot_a = result_a["test"]["paired_episode_bootstrap_softmax_minus_baseline"]
        boot_b = result_b["test"]["paired_episode_bootstrap_softmax_minus_baseline"]
        self.assertEqual(boot_a, boot_b)
        for baseline, values in boot_a.items():
            self.assertEqual(values["unit"], "whole episode")
            self.assertEqual(values["episodes"], result_a["test"]["past_only_softmax"]["raw"]["examples"] // 8)
            self.assertEqual(values["replicates"], 100)
            self.assertEqual(set(values["metrics"]), {"nll", "top1_accuracy", "top2_accuracy", "multiclass_brier"})
            for metric in values["metrics"].values():
                lo, hi = metric["ci95_percentile"]
                self.assertLessEqual(lo, hi)

    def test_summary_uses_seed_level_values(self):
        fake = {
            "1": {
                "in_policy": {"test": {m: {"raw": {k: 1.0 for k in ("nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin")}} for m in ("majority", "historical_frequency", "first_order_markov", "past_only_softmax")}},
                "lopo": {p: {"test": {m: {"raw": {k: 1.0 for k in ("nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin")}} for m in ("majority", "historical_frequency", "first_order_markov", "past_only_softmax")}} for p in ("goal_directed", "uniform", "retrying_adversary", "prerequisite_blind_planner", "denial_adaptive_explorer")},
            },
            "2": {
                "in_policy": {"test": {m: {"raw": {k: 3.0 for k in ("nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin")}} for m in ("majority", "historical_frequency", "first_order_markov", "past_only_softmax")}},
                "lopo": {p: {"test": {m: {"raw": {k: 3.0 for k in ("nll", "top1_accuracy", "top2_accuracy", "multiclass_brier", "top_label_ece_10bin")}} for m in ("majority", "historical_frequency", "first_order_markov", "past_only_softmax")}} for p in ("goal_directed", "uniform", "retrying_adversary", "prerequisite_blind_planner", "denial_adaptive_explorer")},
            },
        }
        summary = summarize_runs(fake)
        self.assertEqual(summary["in_policy_across_seeds"]["majority"]["nll"]["mean"], 2.0)
        self.assertEqual(summary["in_policy_across_seeds"]["majority"]["nll"]["sd_across_seeds"], 2**0.5)
        self.assertEqual(summary["lopo_macro_across_seeds"]["past_only_softmax"]["top1_accuracy"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
